# ourJobs.py
# One run of Part C (our own work), described by a small set of settings. The features stage
# uses this, and saves every finished run in the folder
# results/our_jobs, so a run that two stages need is only done once.
#
# settings of one job:
#   dataset, fold      which data and which of the 5 folds
#   model              gosdt, dl85, or a scikit-learn baseline (cart, cartCv, rf, gbdt, ada, logreg, knn, svm)
#   features           none / text / equality   (text and equality are for the monk data)
#   binarize           all / bucket / quantile5 / quantile10 / domain / mi
#   lbRef              reference for lower bound guessing: None, gbdt, rf, ada, extra,
#                      intersection, union, weak, random
#   lam, depth         lambda and depth limit (depth 0 = no limit)
#   lamChoice          "fixed", or "cv" = choose lambda (gosdt) / depth (dl85) by 3 fold CV on the train part
#   timeLimit
#   warmStart          (only when True) start GOSDT from a greedy tree, our small addition

import time

import numpy as np

import binarize
import dataTools
import expTools
import featureMaker
import guessLowerBound
import otherModels
import dl85Style
import step5_gosdt
import treeTools

# size of the reference model for lower bound guessing: 20 trees of depth 3
referenceSize = (20, 3)
blackBoxes = ["cartCv", "rf", "gbdt", "ada", "logreg", "knn", "svm"]
cvLambdas = [0.01, 0.005, 0.002, 0.001]
cvDepths = [2, 3, 4, 5]


def makeJob(dataset, fold, model="gosdt", features="none", binarizeMethod="quantile10",
            lbRef=None, lam=0.001, depth=5, lamChoice="fixed", timeLimit=600, warmStart=False):
    spec = {"dataset": dataset, "fold": fold, "model": model, "features": features, "binarize": binarizeMethod,
            "lbRef": lbRef, "lam": lam, "depth": depth, "lamChoice": lamChoice,
            "timeLimit": timeLimit}
    if warmStart:
        # only written when it is on, so the names of all the older saved runs stay the same
        spec["warmStart"] = True
    items = []
    for key in sorted(spec.keys()):
        items.append((key, spec[key]))
    return tuple(items)


def jobFolder():
    return expTools.makeResultFolder("our_jobs")


def runAll(jobs, settings, folder):
    return expTools.runJobs(ourJob, jobs, expTools.pickJobs(settings), folder, jobFolder())


# ============================== data ==============================

def prepareRaw(spec):
    rawTrain, yTrain, rawTest, yTest = dataTools.getRawFold(spec["dataset"], spec["fold"])
    added = []
    if spec["features"] in ["text", "equality"]:
        rawTrain = featureMaker.monkToText(rawTrain)
        rawTest = featureMaker.monkToText(rawTest)
    if spec["features"] == "equality":
        rawTrain, rawTest, added = featureMaker.equalityFeatures(rawTrain, rawTest)
    numTrain, numTest = featureMaker.oneHotText(rawTrain, rawTest)
    return numTrain, yTrain, numTest, yTest, added


def makeBinary(spec, numTrain, yTrain, numTest):
    method = spec["binarize"]
    info = {"guessSeconds": 0.0}
    if method == "mi":
        # mutual information: keep as many columns as 5 quantile bins give, but chosen by MI from all thresholds
        k = binarize.applyEncoder(binarize.makeEncoder(numTrain, yTrain, "quantile5"), numTrain)[0].shape[1]
        encoder = binarize.makeEncoder(numTrain, yTrain, "all")
        allTrain, allNames, allGroups = binarize.applyEncoder(encoder, numTrain)
        allTest = binarize.applyEncoder(encoder, numTest)[0]
        chosen = featureMaker.selectByMutualInformation(allTrain, yTrain, k)
        newPosition = {}
        for position in range(len(chosen)):
            newPosition[chosen[position]] = position
        groups = []
        for group in allGroups:
            kept = []
            for j in group:
                if j in newPosition:
                    kept.append(newPosition[j])
            if len(kept) >= 2:
                groups.append(kept)
        names = []
        for j in chosen:
            names.append(allNames[j])
        return allTrain[:, chosen], allTest[:, chosen], names, groups, info
    if method == "domain":
        encoder = binarize.makeEncoder(numTrain, yTrain, "given",
                                       givenThresholds=featureMaker.domainThresholds(spec["dataset"]))
    else:
        encoder = binarize.makeEncoder(numTrain, yTrain, method)
    Xtrain, names, groups = binarize.applyEncoder(encoder, numTrain)
    Xtest = binarize.applyEncoder(encoder, numTest)[0]
    return Xtrain, Xtest, names, groups, info


def makeReference(spec, Xtrain, yTrain):
    lbRef = spec["lbRef"]
    if lbRef is None:
        return None
    numTrees, treeDepth = referenceSize
    if lbRef in ["gbdt", "rf", "ada", "extra"]:
        return guessLowerBound.referencePredictions(Xtrain, yTrain, numTrees, treeDepth, modelKind=lbRef)
    if lbRef in ["intersection", "union"]:
        predictionList = []
        for kind in ["gbdt", "rf", "ada"]:
            predictionList.append(guessLowerBound.referencePredictions(Xtrain, yTrain, numTrees, treeDepth, modelKind=kind))
        return guessLowerBound.combineReferences(predictionList, yTrain, lbRef)
    if lbRef == "weak":
        # a deliberately weak reference: 2 stumps
        return guessLowerBound.referencePredictions(Xtrain, yTrain, 2, 1)
    if lbRef == "random":
        # a reference that learned nothing: trained on shuffled labels
        rng = np.random.RandomState(0)
        return guessLowerBound.referencePredictions(Xtrain, rng.permutation(yTrain), numTrees, treeDepth)
    raise ValueError("unknown reference " + str(lbRef))


# ============================== search ==============================

def fitTree(spec, Xtrain, yTrain, groups, reference, lam, depth, timeLimit):
    if spec["model"] == "gosdt":
        depthBudget = None
        if depth > 0:
            depthBudget = depth
        return step5_gosdt.gosdt(Xtrain, yTrain, lam, timeLimit=timeLimit, featureGroups=groups,
                                 depthBudget=depthBudget, referencePredictions=reference,
                                 warmStart=spec.get("warmStart", False))
    return dl85Style.dl85(Xtrain, yTrain, depth, timeLimit=timeLimit, referencePredictions=reference)


def chooseByInnerCv(spec, Xtrain, yTrain, groups, reference):
    # 3 fold cross validation INSIDE the training part, to choose lambda (gosdt) or depth (dl85).
    # (to keep it simple the thresholds and the reference come from the whole training part)
    folds = dataTools.makeFolds(len(yTrain), 3, seed=1)
    if spec["model"] == "gosdt":
        candidates = cvLambdas
    else:
        candidates = cvDepths
    bestScore = -1.0
    bestValue = candidates[0]
    for value in candidates:
        scores = []
        for k in range(3):
            validIdx = folds[k]
            trainParts = []
            for k2 in range(3):
                if k2 != k:
                    trainParts.append(folds[k2])
            trainIdx = np.concatenate(trainParts)
            innerReference = None
            if reference is not None:
                innerReference = reference[trainIdx]
            if spec["model"] == "gosdt":
                r = fitTree(spec, Xtrain[trainIdx], yTrain[trainIdx], groups, innerReference, value, spec["depth"],
                            spec["timeLimit"] / 4)
            else:
                r = fitTree(spec, Xtrain[trainIdx], yTrain[trainIdx], groups, innerReference, 0.0, value,
                            spec["timeLimit"] / 4)
            if r["tree"] is not None:
                scores.append(treeTools.accuracyOf(r["tree"], Xtrain[validIdx], yTrain[validIdx]))
        if len(scores) > 0 and np.mean(scores) > bestScore:
            bestScore = float(np.mean(scores))
            bestValue = value
    return bestValue


def ourJob(job):
    spec = dict(job)
    row = dict(spec)
    numTrain, yTrain, numTest, yTest, added = prepareRaw(spec)
    row["addedFeatures"] = len(added)

    if spec["model"] == "cart" or spec["model"] in blackBoxes:
        if spec["model"] == "cart":
            r = otherModels.cart(numTrain.values, yTrain, maxDepth=(spec["depth"] if spec["depth"] > 0 else None))
        else:
            r = otherModels.fitModel(spec["model"], numTrain.values, yTrain)
        row["seconds"] = r["seconds"]
        row["leaves"] = r["leaves"]
        row["trainAccuracy"] = otherModels.accuracyOfModel(r["model"], numTrain.values, yTrain)
        row["testAccuracy"] = otherModels.accuracyOfModel(r["model"], numTest.values, yTest)
        row["timedOut"] = False
        row["memoryOut"] = False
        return row

    Xtrain, Xtest, names, groups, info = makeBinary(spec, numTrain, yTrain, numTest)
    row["binaryFeatures"] = Xtrain.shape[1]
    startTime = time.time()
    reference = makeReference(spec, Xtrain, yTrain)
    row["guessSeconds"] = info["guessSeconds"] + time.time() - startTime
    if reference is not None:
        row["refWrongFraction"] = guessLowerBound.wrongFraction(reference, yTrain)

    lam = spec["lam"]
    depth = spec["depth"]
    if spec["lamChoice"] == "cv":
        chosen = chooseByInnerCv(spec, Xtrain, yTrain, groups, reference)
        if spec["model"] == "gosdt":
            lam = chosen
        else:
            depth = chosen
    row["usedLam"] = lam
    row["usedDepth"] = depth

    r = fitTree(spec, Xtrain, yTrain, groups, reference, lam, depth, spec["timeLimit"])
    row["seconds"] = r["seconds"]
    row["timedOut"] = r["timedOut"]
    row["memoryOut"] = r["memoryOut"]
    if "gap" in r:
        row["gap"] = r["gap"]
    tree = r["tree"]
    if tree is not None:
        row["leaves"] = treeTools.countLeaves(tree)
        row["treeDepth"] = treeTools.treeDepth(tree)
        row["trainAccuracy"] = treeTools.accuracyOf(tree, Xtrain, yTrain)
        row["testAccuracy"] = treeTools.accuracyOf(tree, Xtest, yTest)
        row["objective"] = treeTools.objectiveOf(tree, Xtrain, yTrain, lam)
        treeTools.addCounts(tree, Xtrain, yTrain)
        row["treeText"] = treeTools.treeToText(tree, names)
    return row


def rowsToTable(rows, folder, fileName):
    clean = []
    for row in rows:
        copy = {}
        for key in row:
            if key != "treeText":
                copy[key] = row[key]
        clean.append(copy)
    return expTools.saveTable(clean, folder, fileName)


def median(values):
    clean = []
    for v in values:
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            clean.append(v)
    if len(clean) == 0:
        return None
    return float(np.median(clean))
