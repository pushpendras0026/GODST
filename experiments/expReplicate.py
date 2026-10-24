# expReplicate.py   (python run.py --stage replicate --exp bucket|sparsity|trees|time|scale|all)
# Here we repeat the experiments of the GOSDT paper with our own code:
#   bucket   : Thm 3.1 / Fig 1, bucketization loses accuracy (chudi data)
#   sparsity : Fig 4, 5, 12, 13, accuracy vs number of leaves, 5 fold cross validation
#   trees    : Fig 19, 20, 21, example trees with the same number of leaves
#   time     : Fig 18, best objective over time, lambda = 0.005
#   scale    : Fig 6, 14 to 17, run time when we add more binary features or more samples
# Settings come from the paper App I.

import os
import time

import numpy as np
import pandas as pd

import binarize
import dataTools
import expTools
import plotTools
import treeTools
import otherModels
import dl85Style
import osdtStyle
import step2_dynamicProg
import step5_gosdt

# the 29 lambdas of App I.6
lambdaList29 = [0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006,
                0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003,
                0.0002, 0.0001]


# ============================== helpers ==============================

def fitOne(algorithm, setting, Xtrain, ytrain, timeLimit):
    # trains one model, returns (model, leaves, seconds, timedOut, kind, memoryOut)
    if algorithm == "cart":
        r = otherModels.cart(Xtrain, ytrain, maxDepth=setting, maxLeaves=2 ** setting)
        return r["model"], r["leaves"], r["seconds"], False, "sklearn", False
    if algorithm == "dl85":
        r = dl85Style.dl85(Xtrain, ytrain, setting, timeLimit=timeLimit)
        if r["tree"] is None:
            return None, None, r["seconds"], True, "ours", r["memoryOut"]
        return r["tree"], treeTools.countLeaves(r["tree"]), r["seconds"], r["timedOut"], "ours", r["memoryOut"]
    if algorithm == "gosdt":
        r = step5_gosdt.gosdt(Xtrain, ytrain, setting, timeLimit=timeLimit)
    else:
        r = osdtStyle.leafSetSearch(Xtrain, ytrain, setting, osdtStyle.makeObjective("acc"), timeLimit=timeLimit)
    return r["tree"], treeTools.countLeaves(r["tree"]), r["seconds"], r["timedOut"], "ours", r["memoryOut"]


def accuracyOfAny(model, kind, X, y):
    if kind == "sklearn":
        return otherModels.accuracyOfModel(model, X, y)
    return treeTools.accuracyOf(model, X, y)


# ============================== bucketization (Thm 3.1) ==============================

def runBucket(folder, settings):
    expTools.log(folder, "---- bucketization, chudi data (Thm 3.1, Fig 1)")
    rawX, y = dataTools.loadDataset("chudi")
    rows = []
    trees = {}
    for method in ["all", "bucket"]:
        encoder = binarize.makeEncoder(rawX, y, method)
        X, names, groups = binarize.applyEncoder(encoder, rawX)
        for depth in [1, 2, 3, 4]:
            r = step2_dynamicProg.dynamicProg(X, y, depth)
            correct = len(y) - r["mistakes"]
            row = {"thresholds": method, "binaryFeatures": X.shape[1], "maxDepth": depth,
                   "correct": correct, "samples": len(y), "bestTrainAccuracy": round(correct / len(y), 4),
                   "leaves": treeTools.countLeaves(r["tree"])}
            rows.append(row)
            expTools.log(folder, str(row))
            trees[(method, depth)] = (r["tree"], encoder, names)
    table = expTools.saveTable(rows, folder, "bucket_results.csv")

    # decision regions of the depth 2 trees, like Fig 1
    gridValues1 = np.linspace(-0.4, 6.4, 300)
    gridValues2 = np.linspace(-0.1, 5.6, 300)
    gridX, gridY = np.meshgrid(gridValues1, gridValues2)
    gridTable = pd.DataFrame({"f1": gridX.ravel(), "f2": gridY.ravel()})
    panels = []
    for method in ["all", "bucket"]:
        tree, encoder, names = trees[(method, 2)]
        Xgrid, names2, groups2 = binarize.applyEncoder(encoder, gridTable)
        prediction = np.array(treeTools.predictAll(tree, Xgrid)).reshape(gridX.shape)
        part = table[(table["thresholds"] == method) & (table["maxDepth"] == 2)].iloc[0]
        title = "depth 2, " + ("all thresholds" if method == "all" else "with bucketization")
        title = title + ": " + str(part["correct"]) + "/77 = " + str(round(100 * part["bestTrainAccuracy"], 1)) + "%"
        panels.append((title, rawX["f1"].values, rawX["f2"].values, y, gridX, gridY, prediction))
        expTools.saveText(treeTools.treeToText(tree, names), folder, "bucket_tree_" + method + "_depth2.txt")
    plotTools.plotRegions(panels, os.path.join(folder, "bucket_regions.png"))


# ============================== accuracy vs sparsity ==============================

def sparsityChain(job):
    # one algorithm on one fold of one dataset, going from easy to hard settings.
    # if one setting times out, the harder ones are skipped (they would time out too)
    dataset, fold, algorithm, settingList, timeLimit = job
    Xtrain, ytrain, Xtest, ytest, names, groups = dataTools.getFold(dataset, fold)
    rows = []
    gaveUp = False
    for setting in settingList:
        if gaveUp:
            rows.append({"dataset": dataset, "fold": fold, "algorithm": algorithm, "setting": setting,
                         "timedOut": True, "skipped": True})
            continue
        model, leaves, seconds, timedOut, kind, memoryOut = fitOne(algorithm, setting, Xtrain, ytrain, timeLimit)
        row = {"dataset": dataset, "fold": fold, "algorithm": algorithm, "setting": setting,
               "seconds": round(seconds, 3), "timedOut": timedOut, "memoryOut": memoryOut, "skipped": False}
        if model is not None:
            row["leaves"] = leaves
            row["trainAccuracy"] = accuracyOfAny(model, kind, Xtrain, ytrain)
            row["testAccuracy"] = accuracyOfAny(model, kind, Xtest, ytest)
        rows.append(row)
        if timedOut:
            gaveUp = True
    return rows


def summarizeFrontier(table):
    # median over the folds, with 25 and 75 percentiles (paper App I.6), only finished runs
    done = table[(table["timedOut"] == False) & (table["skipped"] == False)]
    summary = []
    for key, part in done.groupby(["dataset", "algorithm", "setting"]):
        summary.append({"dataset": key[0], "algorithm": key[1], "setting": key[2], "folds": len(part),
                        "leaves": float(np.median(part["leaves"])),
                        "trainMedian": float(np.median(part["trainAccuracy"])),
                        "trainLow": float(np.percentile(part["trainAccuracy"], 25)),
                        "trainHigh": float(np.percentile(part["trainAccuracy"], 75)),
                        "testMedian": float(np.median(part["testAccuracy"])),
                        "testLow": float(np.percentile(part["testAccuracy"], 25)),
                        "testHigh": float(np.percentile(part["testAccuracy"], 75))})
    return pd.DataFrame(summary)


def runSparsity(folder, settings):
    expTools.log(folder, "---- accuracy vs sparsity (Fig 4, 5, 12, 13)")
    datasets = expTools.pickList(["car", "compas_binary", "tictactoe", "monk1", "monk2", "monk3"],
                                 ["monk1", "monk2"], settings, "dataset")
    lams = expTools.pickList(lambdaList29, [0.1, 0.05, 0.02, 0.01, 0.005], settings, "lam")
    depths = [1, 2, 3, 4, 5, 6]
    timeLimit = expTools.pickTimeLimit(300, 20, settings)
    numFolds = 5
    jobs = []
    for dataset in datasets:
        for fold in range(numFolds):
            jobs.append((dataset, fold, "gosdt", lams, timeLimit))
            jobs.append((dataset, fold, "osdt", lams, timeLimit))
            jobs.append((dataset, fold, "dl85", depths, timeLimit))
            jobs.append((dataset, fold, "cart", depths, timeLimit))
    answers = expTools.runJobs(sparsityChain, jobs, expTools.pickJobs(settings), folder)
    rows = []
    for answer in answers:
        rows.extend(answer)
    table = expTools.saveTable(rows, folder, "sparsity_runs.csv")
    summary = summarizeFrontier(table)
    summary.to_csv(os.path.join(folder, "sparsity_summary.csv"), index=False)

    names = {"gosdt": "GOSDT (ours)", "osdt": "OSDT style (ours)", "dl85": "DL8.5 style (ours)", "cart": "CART"}
    for which in ["train", "test"]:
        panels = {}
        for dataset in datasets:
            seriesDict = {}
            for algorithm in ["gosdt", "osdt", "dl85", "cart"]:
                part = summary[(summary["dataset"] == dataset) & (summary["algorithm"] == algorithm)]
                part = part.sort_values("leaves")
                if len(part) > 0:
                    seriesDict[names[algorithm]] = (list(part["leaves"]), list(part[which + "Median"]),
                                                    list(part[which + "Low"]), list(part[which + "High"]))
            panels[dataset] = seriesDict
        plotTools.plotGrid(panels, "number of leaves", which + " accuracy",
                           which.capitalize() + " accuracy vs number of leaves (median of 5 folds)",
                           os.path.join(folder, "sparsity_" + which + ".png"), errorBars=True)
    timeouts = table[(table["timedOut"] == True) & (table["skipped"] == False)]
    expTools.log(folder, "runs that timed out (not used in the plots, like the paper): " + str(len(timeouts)))


# ============================== example trees ==============================

def gosdtWithLeaves(Xtrain, ytrain, target, timeLimit):
    # find a lambda that gives exactly "target" leaves.
    # Not every number of leaves can be reached: the lambda objective only gives trees on the
    # lower convex hull of (leaves, mistakes). If there is no exact match we return the
    # closest tree below and the closest tree above the target.
    tried = []
    for lam in lambdaList29:
        r = step5_gosdt.gosdt(Xtrain, ytrain, lam, timeLimit=timeLimit)
        leaves = treeTools.countLeaves(r["tree"])
        tried.append((lam, leaves, r))
        if leaves == target:
            return [(lam, r)]
        if leaves > target:
            break
    if len(tried) >= 2:
        # fine scan between the last two lambdas
        bigLam = tried[-2][0]
        smallLam = tried[-1][0]
        for k in range(1, 40):
            lam = bigLam - (bigLam - smallLam) * k / 40.0
            r = step5_gosdt.gosdt(Xtrain, ytrain, lam, timeLimit=timeLimit)
            leaves = treeTools.countLeaves(r["tree"])
            tried.append((lam, leaves, r))
            if leaves == target:
                return [(lam, r)]
    below = None
    above = None
    for item in tried:
        if item[1] < target and (below is None or item[1] > below[1]):
            below = item
        if item[1] > target and (above is None or item[1] < above[1]):
            above = item
    answer = []
    for item in [below, above]:
        if item is not None:
            answer.append((item[0], item[2]))
    return answer


def runTrees(folder, settings):
    expTools.log(folder, "---- example trees (Fig 19, 20, 21), fold 0")
    targets = [("monk1", 8), ("monk2", 7), ("tictactoe", 16)]
    paperNumbers = {"monk1": "GOSDT 100% train / 100% test, BinOCT 90.9% / 84.0%",
                    "monk2": "GOSDT 79.3% / 73.5%, DL8.5 76.3% / 70.6%",
                    "tictactoe": "GOSDT 90.9% / 70.8%, DL8.5 86.9% / 61.5%, BinOCT 82.4% / 34.4%"}
    timeLimit = expTools.pickTimeLimit(600, 60, settings)
    rows = []
    for dataset, target in targets:
        if settings.get("dataset") is not None and settings["dataset"] != dataset:
            continue
        Xtrain, ytrain, Xtest, ytest, names, groups = dataTools.getFold(dataset, 0)
        text = "Paper: " + paperNumbers[dataset] + "\n\n"
        found = gosdtWithLeaves(Xtrain, ytrain, target, timeLimit)
        if len(found) > 1 or treeTools.countLeaves(found[0][1]["tree"]) != target:
            text = text + "No lambda gives exactly " + str(target) + " leaves on this fold, so we show the closest GOSDT trees.\n\n"
        models = []
        for lam, r in found:
            models.append(("GOSDT (ours), lambda=" + format(lam, ".5g"), r["tree"], "ours"))
        depth = int(np.ceil(np.log2(target)))
        d = dl85Style.dl85(Xtrain, ytrain, depth, timeLimit=timeLimit)
        models.append(("DL8.5 style (ours), depth " + str(depth), d["tree"], "ours"))
        c = otherModels.cart(Xtrain, ytrain, maxLeaves=target)
        models.append(("CART, max " + str(target) + " leaves", c["model"], "sklearn"))
        for title, model, kind in models:
            if model is None:
                continue
            trainAcc = accuracyOfAny(model, kind, Xtrain, ytrain)
            testAcc = accuracyOfAny(model, kind, Xtest, ytest)
            if kind == "sklearn":
                leaves = int(model.get_n_leaves())
            else:
                leaves = treeTools.countLeaves(model)
            row = {"dataset": dataset, "targetLeaves": target, "model": title, "leaves": leaves,
                   "trainAccuracy": round(trainAcc, 4), "testAccuracy": round(testAcc, 4)}
            rows.append(row)
            expTools.log(folder, str(row))
            text = text + "==== " + title + ": " + str(leaves) + " leaves, train " + str(round(100 * trainAcc, 1))
            text = text + "%, test " + str(round(100 * testAcc, 1)) + "%\n"
            if kind == "ours":
                treeTools.addCounts(model, Xtrain, ytrain)
                text = text + treeTools.treeToText(model, names) + "\n"
        expTools.saveText(text, folder, "trees_" + dataset + ".txt")
    expTools.saveTable(rows, folder, "trees_results.csv")


# ============================== time to optimality ==============================

def timeJob(job):
    dataset, algorithm, lam, timeLimit = job
    X, y, names, groups = dataTools.getWholeData(dataset)
    if algorithm == "gosdt":
        r = step5_gosdt.gosdt(X, y, lam, timeLimit=timeLimit)
        curve = []
        for point in r["progress"]:
            curve.append((point[0], point[2], point[1]))
    elif algorithm == "gosdt big first":
        r = step5_gosdt.gosdt(X, y, lam, timeLimit=timeLimit, queueOrder="big")
        curve = []
        for point in r["progress"]:
            curve.append((point[0], point[2], point[1]))
    else:
        r = osdtStyle.leafSetSearch(X, y, lam, osdtStyle.makeObjective("acc"), timeLimit=timeLimit)
        curve = []
        for point in r["progress"]:
            curve.append((point[0], point[1], None))
    rows = []
    for point in curve:
        rows.append({"dataset": dataset, "algorithm": algorithm, "seconds": point[0],
                     "bestObjective": point[1], "lowerBound": point[2]})
    summary = {"dataset": dataset, "algorithm": algorithm, "seconds": r["seconds"], "timedOut": r["timedOut"],
               "memoryOut": r["memoryOut"], "objective": r["objective"], "gap": r["gap"]}
    return rows, summary


def runTime(folder, settings):
    expTools.log(folder, "---- time to optimality (Fig 18), lambda = 0.005")
    datasets = expTools.pickList(["bar7", "tictactoe", "car", "compas_binary", "fico_binary", "monk1", "monk2", "monk3"],
                                 ["monk1", "monk2", "car", "bar7"], settings, "dataset")
    lam = 0.005
    if settings.get("lam") is not None:
        lam = settings["lam"]
    timeLimit = expTools.pickTimeLimit(1800, 30, settings)
    jobs = []
    for dataset in datasets:
        for algorithm in ["gosdt", "gosdt big first", "osdt"]:
            jobs.append((dataset, algorithm, lam, timeLimit))
    answers = expTools.runJobs(timeJob, jobs, expTools.pickJobs(settings), folder)
    curveRows = []
    summaryRows = []
    for answer in answers:
        curveRows.extend(answer[0])
        summaryRows.append(answer[1])
    curves = expTools.saveTable(curveRows, folder, "time_curves.csv")
    expTools.saveTable(summaryRows, folder, "time_summary.csv")
    for row in summaryRows:
        expTools.log(folder, str(row))

    names = {"gosdt": "GOSDT (ours)", "gosdt big first": "GOSDT, big first queue", "osdt": "OSDT style (ours)"}
    panels = {}
    for dataset in datasets:
        seriesDict = {}
        for algorithm in ["gosdt", "gosdt big first", "osdt"]:
            part = curves[(curves["dataset"] == dataset) & (curves["algorithm"] == algorithm)]
            if len(part) > 0:
                seriesDict[names[algorithm]] = (list(part["seconds"]), list(part["bestObjective"]))
        panels[dataset] = seriesDict
    plotTools.plotStepGrid(panels, "seconds", "best objective so far",
                           "Best objective over time, lambda = " + str(lam), os.path.join(folder, "time_to_optimal.png"))


# ============================== scalability ==============================

def thinEncoder(encoder, stride):
    # keep only every stride-th threshold of each number column (paper Sec 5: "combining sets of k variables")
    newRules = []
    countOfColumn = {}
    for rule in encoder["rules"]:
        if rule["type"] != "<=":
            newRules.append(rule)
            continue
        col = rule["column"]
        if col not in countOfColumn:
            countOfColumn[col] = 0
        if countOfColumn[col] % stride == 0:
            newRules.append(rule)
        countOfColumn[col] = countOfColumn[col] + 1
    return {"rules": newRules}


# For the "more samples" runs on compas and fico we do NOT use all thresholds: with 629 or
# 1384 binary features every exact method stops at the memory limit of our python version,
# so the curve says nothing. We keep every 16th / 32nd threshold (52 and 56 features), the
# biggest settings where GOSDT still finished in the "more features" runs.
sampleRunStride = {"compas": 16, "fico": 32}


def makeScaleData(dataset, kind, level):
    rawX, y = dataTools.loadDataset(dataset)
    if kind == "samples":
        rng = np.random.RandomState(0)
        order = rng.permutation(len(y))
        keep = order[:int(round(level * len(y)))]
        rawX = rawX.iloc[keep].reset_index(drop=True)
        y = y[keep]
        encoder = binarize.makeEncoder(rawX, y, "all")
        if dataset in sampleRunStride:
            encoder = thinEncoder(encoder, sampleRunStride[dataset])
        X, names, groups = binarize.applyEncoder(encoder, rawX)
        return X, y, groups
    encoder = binarize.makeEncoder(rawX, y, "all")
    if dataset == "bar7":
        # all features are already binary here, so we take the first "level" columns
        X, names, groups = binarize.applyEncoder(encoder, rawX)
        return X[:, :level], y, []
    X, names, groups = binarize.applyEncoder(thinEncoder(encoder, level), rawX)
    return X, y, groups


def scaleChain(job):
    dataset, algorithm, kind, levels, timeLimit = job
    lam = 1.0 / 32
    rows = []
    timeouts = 0
    for level in levels:
        if timeouts >= 3:
            break
        X, y, groups = makeScaleData(dataset, kind, level)
        startTime = time.time()
        memoryOut = False
        if algorithm == "gosdt":
            r = step5_gosdt.gosdt(X, y, lam, timeLimit=timeLimit, featureGroups=groups)
            timedOut = r["timedOut"]
            memoryOut = r["memoryOut"]
        elif algorithm == "osdt":
            r = osdtStyle.leafSetSearch(X, y, lam, osdtStyle.makeObjective("acc"), timeLimit=timeLimit)
            timedOut = r["timedOut"]
            memoryOut = r["memoryOut"]
        elif algorithm == "dl85":
            r = dl85Style.dl85(X, y, 5, timeLimit=timeLimit)
            timedOut = r["timedOut"]
            memoryOut = r["memoryOut"]
        else:
            otherModels.cart(X, y, maxDepth=5, maxLeaves=32)
            timedOut = False
        seconds = time.time() - startTime
        if timedOut:
            timeouts = timeouts + 1
        rows.append({"dataset": dataset, "algorithm": algorithm, "kind": kind, "level": level,
                     "binaryFeatures": X.shape[1], "samples": X.shape[0], "seconds": round(seconds, 3),
                     "timedOut": timedOut, "memoryOut": memoryOut})
    return rows


def runScale(folder, settings):
    expTools.log(folder, "---- scalability (Fig 6, 14 to 17), lambda = 1/32 or depth 5")
    datasets = expTools.pickList(["bar7", "compas2016", "compas", "fico"], ["bar7", "compas2016"], settings, "dataset")
    timeLimit = expTools.pickTimeLimit(300, 20, settings)
    featureLevels = {"bar7": [2, 4, 6, 8, 10, 12, 14], "compas2016": [64, 32, 16, 8, 4, 2, 1],
                     "compas": [64, 32, 16, 8, 4, 2, 1], "fico": [64, 32, 16, 8, 4, 2, 1]}
    sampleLevels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if settings.get("quick"):
        sampleLevels = [0.2, 0.6, 1.0]
    jobs = []
    for dataset in datasets:
        for algorithm in ["gosdt", "osdt", "dl85", "cart"]:
            jobs.append((dataset, algorithm, "features", featureLevels[dataset], timeLimit))
            jobs.append((dataset, algorithm, "samples", sampleLevels, timeLimit))
    answers = expTools.runJobs(scaleChain, jobs, expTools.pickJobs(settings), folder)
    rows = []
    for answer in answers:
        rows.extend(answer)
    table = expTools.saveTable(rows, folder, "scale_results.csv")

    names = {"gosdt": "GOSDT (ours)", "osdt": "OSDT style (ours)", "dl85": "DL8.5 style (ours)", "cart": "CART"}
    for kind, xColumn, xLabel in [("features", "binaryFeatures", "number of binary features"),
                                  ("samples", "samples", "number of samples")]:
        timePanels = {}
        slowPanels = {}
        stoppedPanels = {}
        for dataset in datasets:
            timeSeries = {}
            slowSeries = {}
            stoppedSeries = {}
            for algorithm in ["gosdt", "osdt", "dl85", "cart"]:
                part = table[(table["dataset"] == dataset) & (table["algorithm"] == algorithm)
                             & (table["kind"] == kind)].sort_values(xColumn)
                if len(part) == 0:
                    continue
                finished = part[part["timedOut"] == False]
                stopped = part[part["timedOut"] == True]
                # slow down = time / fastest time of the same algorithm (paper App I.7), finished runs only
                seconds = list(finished["seconds"])
                slowdown = []
                if len(seconds) > 0:
                    fastest = max(min(seconds), 0.001)
                    for s in seconds:
                        slowdown.append(max(s, 0.001) / fastest)
                timeSeries[names[algorithm]] = (list(finished[xColumn]), seconds)
                slowSeries[names[algorithm]] = (list(finished[xColumn]), slowdown)
                stoppedSeries[names[algorithm]] = (list(stopped[xColumn]), list(stopped["seconds"]))
            timePanels[dataset] = timeSeries
            slowPanels[dataset] = slowSeries
            stoppedPanels[dataset] = stoppedSeries
        plotTools.plotGrid(timePanels, xLabel, "seconds", "Training time vs " + xLabel + " (x = stopped by a limit)",
                           os.path.join(folder, "scale_time_vs_" + kind + ".png"), logY=True, columns=2,
                           stoppedDict=stoppedPanels)
        plotTools.plotGrid(slowPanels, xLabel, "slow down (time / fastest time)",
                           "Slow down vs " + xLabel + " (finished runs only)",
                           os.path.join(folder, "scale_slowdown_vs_" + kind + ".png"), logY=True, columns=2)


# ============================== stage ==============================

def runStage(settings):
    folder = expTools.makeResultFolder("07_replication")
    which = settings.get("exp")
    if which is None:
        which = "all"
    expTools.startLog(folder, "replicate " + which, settings)
    parts = ["bucket", "trees", "sparsity", "time", "scale"]
    for part in parts:
        if which != "all" and which != part:
            continue
        partFolder = expTools.makeResultFolder(os.path.join("07_replication", part))
        expTools.startLog(partFolder, "replicate " + part, settings)
        startTime = time.time()
        if part == "bucket":
            runBucket(partFolder, settings)
        elif part == "sparsity":
            runSparsity(partFolder, settings)
        elif part == "trees":
            runTrees(partFolder, settings)
        elif part == "time":
            runTime(partFolder, settings)
        elif part == "scale":
            runScale(partFolder, settings)
        expTools.log(folder, part + " finished in " + str(round(time.time() - startTime, 1)) + " seconds")
