# expNewData.py   (python run.py --stage newdata)
# OUR WORK: does the method still work on datasets that NEITHER paper used?
# For every new dataset (5 fold cross validation) we run:
#   GOSDT as in the paper (all thresholds, no depth limit)        -> often too big, stops
#   GOSDT + threshold guessing (no depth limit)
#   GOSDT + all three guesses (threshold, lower bound, depth 5)
#   CART (depth 2 to 6) and GBDT (100 trees) as reference points

import os

import dataTools
import expTools
import plotTools
import ourJobs

lamList = [0.01, 0.005, 0.001]


def runStage(settings):
    folder = expTools.makeResultFolder("12_new_datasets")
    expTools.startLog(folder, "newdata", settings)
    datasets = expTools.pickList(dataTools.newDatasets, ["pima", "mushroom"], settings, "dataset")
    timeLimit = expTools.pickTimeLimit(600, 30, settings)
    folds = 2 if settings.get("quick") else 5
    jobs = []
    for d in datasets:
        for fold in range(folds):
            jobs.append(ourJobs.makeJob(d, fold, "gosdt", binarizeMethod="all", lam=0.01, depth=0, timeLimit=timeLimit))
            for lam in lamList:
                jobs.append(ourJobs.makeJob(d, fold, "gosdt", binarizeMethod="guess", lam=lam, depth=0, timeLimit=timeLimit))
                jobs.append(ourJobs.makeJob(d, fold, "gosdt", binarizeMethod="guess", lbRef="gbdt", lam=lam, depth=5,
                                            timeLimit=timeLimit))
            for depth in [2, 3, 4, 5, 6]:
                jobs.append(ourJobs.makeJob(d, fold, "cart", depth=depth, timeLimit=timeLimit))
            jobs.append(ourJobs.makeJob(d, fold, "gbdt", timeLimit=timeLimit))
    rows = ourJobs.runAll(jobs, settings, folder)
    table = ourJobs.rowsToTable(rows, folder, "runs.csv")

    # summary: one line per dataset and variant
    summary = []
    for d in datasets:
        part = table[table["dataset"] == d]
        variants = [("GOSDT (paper, all thresholds)", (part["model"] == "gosdt") & (part["binarize"] == "all")),
                    ("GOSDT + threshold guess", (part["model"] == "gosdt") & (part["binarize"] == "guess") & (part["lbRef"].isna())),
                    ("GOSDT + all guesses (depth 5)", (part["model"] == "gosdt") & (part["lbRef"] == "gbdt"))]
        for label, mask in variants:
            for lam, group in part[mask].groupby("lam"):
                records = group.to_dict("records")
                certified = 0
                stopped = 0
                for r in records:
                    if r.get("timedOut"):
                        stopped = stopped + 1
                    elif r.get("gap") is not None and r["gap"] <= 1e-9:
                        certified = certified + 1
                summary.append({"dataset": d, "variant": label, "lam": lam,
                                "binaryFeatures": ourJobs.median(list(group["binaryFeatures"])),
                                "seconds": ourJobs.median(list(group["seconds"])),
                                "leaves": ourJobs.median(list(group["leaves"])),
                                "trainAccuracy": ourJobs.median(list(group["trainAccuracy"])),
                                "testAccuracy": ourJobs.median(list(group["testAccuracy"])),
                                "foldsProvedOptimal": certified, "foldsStopped": stopped})
        gb = part[part["model"] == "gbdt"]
        summary.append({"dataset": d, "variant": "GBDT (100 trees)", "testAccuracy": ourJobs.median(list(gb["testAccuracy"])),
                        "trainAccuracy": ourJobs.median(list(gb["trainAccuracy"]))})
    expTools.saveTable(summary, folder, "summary.csv")
    for s in summary:
        expTools.log(folder, str(s))

    # test accuracy vs leaves, one panel per dataset (like their Fig 2)
    panels = {}
    for d in datasets:
        part = table[table["dataset"] == d]
        seriesDict = {}
        for label, mask, groupColumn in [
                ("GOSDT + threshold guess", (part["model"] == "gosdt") & (part["binarize"] == "guess") & (part["lbRef"].isna()), "lam"),
                ("GOSDT + all guesses (depth 5)", (part["model"] == "gosdt") & (part["lbRef"] == "gbdt"), "lam"),
                ("CART", part["model"] == "cart", "depth")]:
            points = []
            for value, group in part[mask].groupby(groupColumn):
                leaves = ourJobs.median(list(group["leaves"]))
                acc = ourJobs.median(list(group["testAccuracy"]))
                if leaves is not None and acc is not None:
                    points.append((leaves, acc))
            points.sort()
            seriesDict[label] = ([p[0] for p in points], [p[1] for p in points])
        gb = part[part["model"] == "gbdt"]
        allLeaves = []
        for name in seriesDict:
            allLeaves.extend(seriesDict[name][0])
        if len(allLeaves) > 0 and len(gb) > 0:
            value = ourJobs.median(list(gb["testAccuracy"]))
            seriesDict["GBDT (100 trees)"] = ([min(allLeaves), max(allLeaves)], [value, value])
        panels[d] = seriesDict
    plotTools.plotGrid(panels, "number of leaves", "test accuracy", "New datasets: test accuracy vs leaves (median of folds)",
                       os.path.join(folder, "new_datasets_test_accuracy.png"), columns=4)
