# expExtension.py
# Experiments of the extension (McTavish et al. 2022, "Fast Sparse Decision Tree Optimization
# via Reference Ensembles"), done with OUR GOSDT and OUR DL8.5 style search:
#   python run.py --stage thresh    their Fig 3 (+ App C.3)  value of threshold guessing
#   python run.py --stage lbguess   their Fig 4 (+ App C.4)  value of lower bound guessing
#   python run.py --stage depth     their Fig 5 + Thm 4.2    value of depth guessing
#   python run.py --stage allguess  their Fig 1, Fig 2, Fig 7  all three guesses together
#
# All parts share one folder of finished jobs (results/extension_jobs), so a run that two
# figures need is only done once.

import os
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import binarize
import dataTools
import expTools
import plotTools
import treeTools
import otherModels
import dl85Style
import step5_gosdt
import guessThresholds
import guessLowerBound
import guessDepth

allDatasets = ["spiral", "ext_compas", "broward", "ext_fico", "takeaway", "restaurant"]
# GBDT used for guessing: (number of trees, depth of each tree), from their Sec 5
gbdtSetting = {"spiral": (20, 3), "ext_compas": (20, 3), "broward": (40, 1), "ext_fico": (40, 1),
               "takeaway": (50, 2), "restaurant": (50, 2)}
# lambda when only one is used (their Fig 1, 3, 4, 5 use 0.001)
mainLambda = {"spiral": 0.02, "ext_compas": 0.001, "broward": 0.001, "ext_fico": 0.001,
              "takeaway": 0.001, "restaurant": 0.001}
# lambdas for the sparsity figure (their config file)
lambdaLists = {"spiral": [0.1, 0.05, 0.02, 0.01], "ext_compas": [0.005, 0.002, 0.001, 0.0005, 0.0002],
               "broward": [0.005, 0.002, 0.001, 0.0005], "ext_fico": [0.005, 0.002, 0.001, 0.0005, 0.0002],
               "takeaway": [0.005, 0.002, 0.001, 0.0005], "restaurant": [0.005, 0.002, 0.001, 0.0005]}
niceName = {"spiral": "Spiral", "ext_compas": "COMPAS", "broward": "Broward", "ext_fico": "FICO",
            "takeaway": "Takeaway", "restaurant": "Restaurant"}


def jobFolder():
    return expTools.makeResultFolder("extension_jobs")


# ============================== one run ==============================

def extensionJob(job):
    # job = (dataset, fold, algorithm, useThresholdGuess, useLowerBoundGuess, lam, depth, timeLimit)
    # depth 0 means no depth limit. Returns one row for the results table.
    dataset, fold, algorithm, useThresh, useLb, lam, depth, timeLimit = job
    rawTrain, yTrain, rawTest, yTest = dataTools.getRawFold(dataset, fold)
    numTrees, treeDepth = gbdtSetting[dataset]
    row = {"dataset": dataset, "fold": fold, "algorithm": algorithm, "thresholdGuess": useThresh,
           "lowerBoundGuess": useLb, "lam": lam, "depth": depth}

    if algorithm == "gbdt" or algorithm == "cart":
        # baselines on the original data, no guessing (like the paper)
        if algorithm == "gbdt":
            model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
            model.fit(rawTrain.values, yTrain)
            row["leaves"] = None
        else:
            r = otherModels.cart(rawTrain.values, yTrain, maxDepth=depth)
            model = r["model"]
            row["leaves"] = r["leaves"]
        row["trainAccuracy"] = otherModels.accuracyOfModel(model, rawTrain.values, yTrain)
        row["testAccuracy"] = otherModels.accuracyOfModel(model, rawTest.values, yTest)
        row["seconds"] = 0.0
        row["timedOut"] = False
        row["memoryOut"] = False
        return row

    guessSeconds = 0.0
    if useThresh:
        Xtrain, Xtest, names, groups, info = guessThresholds.binarizeWithGuess(rawTrain, yTrain, rawTest, numTrees, treeDepth)
        guessSeconds = info["seconds"]
    else:
        encoder = binarize.makeEncoder(rawTrain, yTrain, "all")
        Xtrain, names, groups = binarize.applyEncoder(encoder, rawTrain)
        Xtest, names2, groups2 = binarize.applyEncoder(encoder, rawTest)
    row["binaryFeatures"] = Xtrain.shape[1]

    reference = None
    if useLb:
        startTime = time.time()
        reference = guessLowerBound.referencePredictions(Xtrain, yTrain, numTrees, treeDepth)
        guessSeconds = guessSeconds + time.time() - startTime
    row["guessSeconds"] = round(guessSeconds, 3)

    depthBudget = None
    if depth > 0:
        depthBudget = depth
    if algorithm == "gosdt":
        r = step5_gosdt.gosdt(Xtrain, yTrain, lam, timeLimit=timeLimit, featureGroups=groups,
                              depthBudget=depthBudget, referencePredictions=reference)
        row["gap"] = r["gap"]
    else:
        r = dl85Style.dl85(Xtrain, yTrain, depth, timeLimit=timeLimit, referencePredictions=reference)
    row["seconds"] = round(r["seconds"], 3)
    row["timedOut"] = r["timedOut"]
    row["memoryOut"] = r["memoryOut"]
    tree = r["tree"]
    if tree is not None:
        # if GOSDT stopped early we still look at the best tree it found (like the paper)
        row["leaves"] = treeTools.countLeaves(tree)
        row["treeDepth"] = treeTools.treeDepth(tree)
        row["trainAccuracy"] = treeTools.accuracyOf(tree, Xtrain, yTrain)
        row["testAccuracy"] = treeTools.accuracyOf(tree, Xtest, yTest)
        treeTools.addCounts(tree, Xtrain, yTrain)
        row["treeText"] = treeTools.treeToText(tree, names)
    return row


def runAll(jobs, settings, folder):
    answers = expTools.runJobs(extensionJob, jobs, expTools.pickJobs(settings), folder, jobFolder())
    return answers


def pickDatasets(settings):
    return expTools.pickList(allDatasets, ["spiral", "ext_compas"], settings, "dataset")


def numFolds(settings):
    if settings.get("quick"):
        return 2
    return 5


def median(values):
    clean = []
    for v in values:
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            clean.append(v)
    if len(clean) == 0:
        return None
    return float(np.median(clean))


def stopText(rows):
    # "TO" if some fold hit the time limit, "MO" if some hit the memory limit
    text = ""
    for r in rows:
        if r.get("memoryOut"):
            text = "MO"
        elif r.get("timedOut") and text == "":
            text = "TO"
    return text


# ============================== Fig 3: threshold guessing ==============================

def runThresh(settings):
    folder = expTools.makeResultFolder("08_threshold_guess")
    expTools.startLog(folder, "thresh", settings)
    timeLimit = expTools.pickTimeLimit(600, 30, settings)
    datasets = pickDatasets(settings)
    jobs = []
    for dataset in datasets:
        for fold in range(numFolds(settings)):
            for useThresh in [False, True]:
                jobs.append((dataset, fold, "gosdt", useThresh, False, mainLambda[dataset], 5, timeLimit))
                jobs.append((dataset, fold, "dl85", useThresh, False, mainLambda[dataset], 5, timeLimit))
    rows = runAll(jobs, settings, folder)
    table = saveRows(rows, folder)
    timeBars(table, datasets, [("gosdt", False, False, "GOSDT"), ("gosdt", True, False, "GOSDT + threshold guess"),
                               ("dl85", False, False, "DL8.5 style"), ("dl85", True, False, "DL8.5 style + threshold guess")],
             folder, "time_threshold_guess.png", "Value of threshold guessing (depth 5)", timeLimit)
    accuracyTable(table, datasets, folder, "accuracy_threshold_guess.csv")


# ============================== Fig 4: lower bound guessing ==============================

def runLbGuess(settings):
    folder = expTools.makeResultFolder("10_lowerbound_guess")
    expTools.startLog(folder, "lbguess", settings)
    timeLimit = expTools.pickTimeLimit(600, 30, settings)
    datasets = pickDatasets(settings)
    jobs = []
    for dataset in datasets:
        for fold in range(numFolds(settings)):
            for useLb in [False, True]:
                jobs.append((dataset, fold, "gosdt", True, useLb, mainLambda[dataset], 5, timeLimit))
                jobs.append((dataset, fold, "dl85", True, useLb, mainLambda[dataset], 5, timeLimit))
    rows = runAll(jobs, settings, folder)
    table = saveRows(rows, folder)
    timeBars(table, datasets, [("gosdt", True, False, "GOSDT + th"), ("gosdt", True, True, "GOSDT + th + lb"),
                               ("dl85", True, False, "DL8.5 style + th"), ("dl85", True, True, "DL8.5 style + th + lb")],
             folder, "time_lowerbound_guess.png", "Value of lower bound guessing (depth 5, with threshold guessing)", timeLimit)
    accuracyTable(table, datasets, folder, "accuracy_lowerbound_guess.csv")


# ============================== Fig 5: depth guessing ==============================

def runDepth(settings):
    folder = expTools.makeResultFolder("09_depth_guess")
    expTools.startLog(folder, "depth", settings)
    timeLimit = expTools.pickTimeLimit(600, 30, settings)
    dataset = "ext_compas"
    if settings.get("dataset") is not None:
        dataset = settings["dataset"]
    depths = expTools.pickList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0], [2, 3, 4, 0], settings, "depth")
    lam = mainLambda[dataset]
    if settings.get("lam") is not None:
        lam = settings["lam"]
    jobs = []
    for fold in range(numFolds(settings)):
        for depth in depths:
            jobs.append((dataset, fold, "gosdt", True, False, lam, depth, timeLimit))
    rows = runAll(jobs, settings, folder)
    table = saveRows(rows, folder)

    # median over the folds for every depth (0 = no limit, we draw it at the right end)
    xs = []
    labels = []
    trainMed = []
    testMed = []
    timeMed = []
    leavesMed = []
    for depth in depths:
        part = table[table["depth"] == depth]
        if len(part) == 0:
            continue
        x = depth
        if depth == 0:
            x = max(depths) + 1
        xs.append(x)
        labels.append("no limit" if depth == 0 else str(depth))
        trainMed.append(median(list(part["trainAccuracy"])))
        testMed.append(median(list(part["testAccuracy"])))
        timeMed.append(median(list(part["seconds"])))
        leavesMed.append(median(list(part["leaves"])))
        expTools.log(folder, "depth " + labels[-1] + ": train " + str(trainMed[-1]) + " test " + str(testMed[-1])
                     + " seconds " + str(timeMed[-1]) + " leaves " + str(leavesMed[-1]) + " " + stopText(part.to_dict("records")))
    panels = {"accuracy": {"train": (xs, trainMed), "test": (xs, testMed)},
              "training time (s)": {"GOSDT + threshold guess": (xs, timeMed)},
              "number of leaves": {"GOSDT + threshold guess": (xs, leavesMed)}}
    plotTools.plotGrid(panels, "depth limit (last point = no limit)", "", niceName.get(dataset, dataset)
                       + ": depth limit, lambda = " + str(lam), os.path.join(folder, "depth_effect.png"), columns=3)

    # Theorem 4.2 table
    thmRows = []
    for numTrees, weakDepth in [(10, 3), (100, 3), (20, 3), (40, 1), (50, 2), (30, 2), (100, 1)]:
        thmRows.append({"gbdtTrees": numTrees, "gbdtTreeDepth": weakDepth,
                        "depthFromThm42": guessDepth.depthToMatchEnsemble(numTrees, weakDepth)})
    expTools.saveTable(thmRows, folder, "theorem42_depths.csv")
    expTools.log(folder, "Theorem 4.2 depths: " + str(thmRows))


# ============================== all guesses: Fig 1, Fig 2, Fig 7 ==============================

def runAllGuess(settings):
    folder = expTools.makeResultFolder("11_all_guesses")
    expTools.startLog(folder, "allguess", settings)
    timeLimit = expTools.pickTimeLimit(600, 30, settings)
    folds = numFolds(settings)

    # ---- Fig 1: COMPAS, lambda 0.001, depth 3, 4, 5 (and no limit for GOSDT)
    dataset = "ext_compas"
    lam = mainLambda[dataset]
    jobs = []
    for fold in range(folds):
        jobs.append((dataset, fold, "gbdt", False, False, lam, 0, timeLimit))
        for depth in [3, 4, 5, 0]:
            jobs.append((dataset, fold, "gosdt", False, False, lam, depth, timeLimit))
            jobs.append((dataset, fold, "gosdt", True, True, lam, depth, timeLimit))
            if depth > 0:
                jobs.append((dataset, fold, "dl85", False, False, lam, depth, timeLimit))
                jobs.append((dataset, fold, "dl85", True, True, lam, depth, timeLimit))
    rows = runAll(jobs, settings, folder)
    table = saveRows(rows, folder, "fig1_runs.csv")
    gbdtPart = table[table["algorithm"] == "gbdt"]
    gbdtTrain = median(list(gbdtPart["trainAccuracy"]))
    gbdtTest = median(list(gbdtPart["testAccuracy"]))
    panels = {}
    stoppedPanels = {}
    for which in ["train", "test"]:
        for depth in [3, 4, 5, 0]:
            # every method always gets its place in the legend (empty if no result), so the
            # colors are the same in every panel
            seriesDict = {}
            stoppedDict = {}
            allTimes = []
            for algorithm, useGuess, label in [("gosdt", False, "GOSDT"), ("gosdt", True, "GOSDT + th + lb"),
                                               ("dl85", False, "DL8.5 style"), ("dl85", True, "DL8.5 style + th + lb")]:
                seriesDict[label] = ([], [])
                stoppedDict[label] = ([], [])
                part = table[(table["algorithm"] == algorithm) & (table["thresholdGuess"] == useGuess) & (table["depth"] == depth)]
                if len(part) == 0:
                    continue
                t = median(list(part["seconds"]))
                a = median(list(part[which + "Accuracy"]))
                if t is None or a is None:
                    # stopped without any tree (DL8.5 style), nothing to draw
                    continue
                allTimes.append(t)
                if stopText(part.to_dict("records")) == "":
                    seriesDict[label] = ([t], [a])
                else:
                    # stopped by the time or memory limit: we show the best tree it had, as an x
                    stoppedDict[label] = ([t], [a])
            gbdtValue = gbdtTrain if which == "train" else gbdtTest
            if len(allTimes) > 0:
                seriesDict["GBDT (100 trees, depth 3)"] = ([min(allTimes), max(allTimes)], [gbdtValue, gbdtValue])
            panelName = which + ", depth " + ("no limit" if depth == 0 else str(depth))
            panels[panelName] = seriesDict
            stoppedPanels[panelName] = stoppedDict
    plotTools.plotGrid(panels, "training time (s, log)", "accuracy",
                       "COMPAS, lambda = 0.001 (median of folds, x = stopped by a limit)",
                       os.path.join(folder, "fig1_compas_time_vs_accuracy.png"), logX=True, columns=4,
                       stoppedDict=stoppedPanels)

    # ---- Fig 2: sparsity vs accuracy on every dataset, all guesses
    datasets = pickDatasets(settings)
    jobs = []
    for d in datasets:
        lamList = lambdaLists[d]
        if settings.get("quick"):
            lamList = lamList[:2]
        for fold in range(folds):
            jobs.append((d, fold, "gbdt", False, False, 0.0, 0, timeLimit))
            for lamValue in lamList:
                jobs.append((d, fold, "gosdt", True, True, lamValue, 5, timeLimit))
                jobs.append((d, fold, "gosdt", True, True, lamValue, 0, timeLimit))
            for depth in [2, 3, 4, 5, 6]:
                jobs.append((d, fold, "dl85", True, True, 0.0, depth, timeLimit))
                jobs.append((d, fold, "cart", False, False, 0.0, depth, timeLimit))
    rows = runAll(jobs, settings, folder)
    table = saveRows(rows, folder, "fig2_runs.csv")
    for which in ["train", "test"]:
        panels = {}
        for d in datasets:
            seriesDict = {}
            for algorithm, depth, label in [("gosdt", 5, "GOSDT + th + lb (depth 5)"), ("gosdt", 0, "GOSDT + th + lb (no depth limit)"),
                                            ("dl85", None, "DL8.5 style + th + lb"), ("cart", None, "CART")]:
                part = table[(table["dataset"] == d) & (table["algorithm"] == algorithm)]
                if depth is not None:
                    part = part[part["depth"] == depth]
                    groupColumn = "lam"
                else:
                    groupColumn = "depth"
                points = []
                for value, group in part.groupby(groupColumn):
                    leaves = median(list(group["leaves"]))
                    acc = median(list(group[which + "Accuracy"]))
                    if leaves is not None and acc is not None:
                        points.append((leaves, acc))
                points.sort()
                if len(points) > 0:
                    seriesDict[label] = ([p[0] for p in points], [p[1] for p in points])
            gbdtPart = table[(table["dataset"] == d) & (table["algorithm"] == "gbdt")]
            if len(gbdtPart) > 0 and len(seriesDict) > 0:
                allLeaves = []
                for name in seriesDict:
                    allLeaves.extend(seriesDict[name][0])
                seriesDict["GBDT (100 trees)"] = ([min(allLeaves), max(allLeaves)],
                                                  [median(list(gbdtPart[which + "Accuracy"]))] * 2)
            panels[niceName[d]] = seriesDict
        plotTools.plotGrid(panels, "number of leaves", which + " accuracy",
                           which.capitalize() + " accuracy vs number of leaves (median of folds)",
                           os.path.join(folder, "fig2_sparsity_" + which + ".png"), columns=3)

    # ---- Fig 7: example trees on COMPAS, depth 3 and 5, all guesses
    text = "Paper Fig 7 (their fold 2): depth 3: 7 leaves, train 0.683, test 0.689. depth 5: 9 leaves, train 0.685, test 0.696\n\n"
    treeRows = []
    for depth in [3, 5]:
        row = extensionJobCached(("ext_compas", 2 % folds, "gosdt", True, True, 0.001, depth, timeLimit))
        treeRows.append({"depth": depth, "leaves": row.get("leaves"), "trainAccuracy": row.get("trainAccuracy"),
                         "testAccuracy": row.get("testAccuracy"), "seconds": row.get("seconds")})
        text = text + "==== GOSDT + th + lb, depth limit " + str(depth) + ": " + str(row.get("leaves")) + " leaves, train "
        text = text + str(round(row.get("trainAccuracy"), 3)) + ", test " + str(round(row.get("testAccuracy"), 3))
        text = text + ", " + str(row.get("seconds")) + " s\n" + row.get("treeText", "") + "\n"
    expTools.saveTable(treeRows, folder, "fig7_trees.csv")
    expTools.saveText(text, folder, "fig7_trees.txt")
    expTools.log(folder, text)


def extensionJobCached(job):
    answers = expTools.runJobs(extensionJob, [job], 1, None, jobFolder())
    return answers[0]


# ============================== tables and plots ==============================

def saveRows(rows, folder, fileName="results.csv"):
    clean = []
    for row in rows:
        copy = {}
        for key in row:
            if key != "treeText":
                copy[key] = row[key]
        clean.append(copy)
    return expTools.saveTable(clean, folder, fileName)


def timeBars(table, datasets, variants, folder, fileName, title, timeLimit):
    series = {}
    texts = {}
    for algorithm, useThresh, useLb, label in variants:
        values = []
        marks = []
        for d in datasets:
            part = table[(table["dataset"] == d) & (table["algorithm"] == algorithm)
                         & (table["thresholdGuess"] == useThresh) & (table["lowerBoundGuess"] == useLb)]
            if len(part) == 0:
                values.append(None)
                marks.append("")
                continue
            # total time = guessing time + search time, like their bars (guess time is the green part)
            total = []
            for r in part.to_dict("records"):
                g = r.get("guessSeconds")
                if g is None or (isinstance(g, float) and np.isnan(g)):
                    g = 0.0
                total.append(r["seconds"] + g)
            values.append(median(total))
            marks.append(stopText(part.to_dict("records")))
        series[label] = values
        texts[label] = marks
    names = []
    for d in datasets:
        names.append(niceName[d])
    plotTools.plotBars(names, series, "training time in seconds (log, median of folds)",
                       title + "   TO = time limit " + str(int(timeLimit)) + " s, MO = memory limit",
                       os.path.join(folder, fileName), logY=True, barText=texts)


def accuracyTable(table, datasets, folder, fileName):
    rows = []
    for key, part in table.groupby(["dataset", "algorithm", "thresholdGuess", "lowerBoundGuess"]):
        records = part.to_dict("records")
        rows.append({"dataset": key[0], "algorithm": key[1], "thresholdGuess": key[2], "lowerBoundGuess": key[3],
                     "binaryFeatures": median(list(part["binaryFeatures"])) if "binaryFeatures" in part else None,
                     "seconds": median(list(part["seconds"])), "leaves": median(list(part["leaves"])),
                     "trainAccuracy": median(list(part["trainAccuracy"])), "testAccuracy": median(list(part["testAccuracy"])),
                     "foldsStopped": sum(1 for r in records if r.get("timedOut")), "stop": stopText(records)})
    expTools.saveTable(rows, folder, fileName)
    for r in rows:
        expTools.log(folder, str(r))


def runStage(settings, stage):
    if stage == "thresh":
        runThresh(settings)
    elif stage == "lbguess":
        runLbGuess(settings)
    elif stage == "depth":
        runDepth(settings)
    elif stage == "allguess":
        runAllGuess(settings)
