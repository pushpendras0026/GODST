# expFeatures.py   (python run.py --stage features)
# OUR WORK: feature engineering.
#   A. equality features for monk ("head == body"), the GOSDT paper App I.9 says its encoding
#      can not express monk1's true rule, so it needs 8 leaves
#   B. how to turn numbers into 0/1 features: all thresholds, bucketization, quantile bins,
#      mutual information selection, and medical cut points (Pima)

import os

import expTools
import plotTools
import ourJobs


def runStage(settings):
    folder = expTools.makeResultFolder("13_feature_engineering")
    expTools.startLog(folder, "features", settings)
    timeLimit = expTools.pickTimeLimit(600, 30, settings)
    folds = 2 if settings.get("quick") else 5

    # ---------- A. equality features on monk
    jobs = []
    monkSets = expTools.pickList(["monk1", "monk2", "monk3"], ["monk1"], settings, "dataset")
    for d in monkSets:
        for fold in range(folds):
            for features in ["text", "equality"]:
                for lam in [0.02, 0.01]:
                    jobs.append(ourJobs.makeJob(d, fold, "gosdt", features=features, binarizeMethod="all", lam=lam,
                                                depth=0, timeLimit=timeLimit))
    rows = ourJobs.runAll(jobs, settings, folder)
    table = ourJobs.rowsToTable(rows, folder, "equality_runs.csv")
    summary = []
    for key, group in table.groupby(["dataset", "features", "lam"]):
        summary.append({"dataset": key[0], "features": key[1], "lam": key[2],
                        "binaryFeatures": ourJobs.median(list(group["binaryFeatures"])),
                        "leaves": ourJobs.median(list(group["leaves"])),
                        "trainAccuracy": ourJobs.median(list(group["trainAccuracy"])),
                        "testAccuracy": ourJobs.median(list(group["testAccuracy"])),
                        "seconds": ourJobs.median(list(group["seconds"]))})
    expTools.saveTable(summary, folder, "equality_summary.csv")
    for s in summary:
        expTools.log(folder, str(s))
    text = ""
    for r in rows:
        if r["dataset"] == "monk1" and r["fold"] == 0 and r["lam"] == 0.01:
            text = text + "==== monk1 fold 0, lambda 0.01, features: " + r["features"] + " (" + str(r.get("leaves"))
            text = text + " leaves, test accuracy " + str(r.get("testAccuracy")) + ")\n" + r.get("treeText", "") + "\n"
    expTools.saveText(text, folder, "monk1_trees.txt")

    # ---------- B. binarization methods
    numberSets = expTools.pickList(["pima", "breast_cancer", "banknote", "titanic", "heart"],
                                   ["pima", "banknote"], settings, "dataset")
    methods = ["all", "bucket", "quantile5", "quantile10", "mi"]
    jobs = []
    for d in numberSets:
        for fold in range(folds):
            for method in methods:
                jobs.append(ourJobs.makeJob(d, fold, "gosdt", binarizeMethod=method, lam=0.005, depth=5, timeLimit=timeLimit))
            if d == "pima":
                jobs.append(ourJobs.makeJob(d, fold, "gosdt", binarizeMethod="domain", lam=0.005, depth=5, timeLimit=timeLimit))
    rows = ourJobs.runAll(jobs, settings, folder)
    table = ourJobs.rowsToTable(rows, folder, "binarize_runs.csv")
    summary = []
    for key, group in table.groupby(["dataset", "binarize"]):
        records = group.to_dict("records")
        stopped = ""
        for r in records:
            if r.get("memoryOut"):
                stopped = "MO"
            elif r.get("timedOut") and stopped == "":
                stopped = "TO"
        summary.append({"dataset": key[0], "binarize": key[1],
                        "binaryFeatures": ourJobs.median(list(group["binaryFeatures"])),
                        "seconds": ourJobs.median(list(group["seconds"])),
                        "leaves": ourJobs.median(list(group["leaves"])),
                        "trainAccuracy": ourJobs.median(list(group["trainAccuracy"])),
                        "testAccuracy": ourJobs.median(list(group["testAccuracy"])), "stopped": stopped})
    summaryTable = expTools.saveTable(summary, folder, "binarize_summary.csv")
    for s in summary:
        expTools.log(folder, str(s))

    names = {"all": "all thresholds", "bucket": "bucketization", "quantile5": "5 quantile bins",
             "quantile10": "10 quantile bins", "mi": "mutual information", "domain": "medical cut points"}
    for column, yLabel, logY in [("seconds", "seconds (median, log)", True), ("testAccuracy", "test accuracy (median)", False),
                                 ("binaryFeatures", "binary features (log)", True)]:
        series = {}
        texts = {}
        for method in methods:
            values = []
            marks = []
            for d in numberSets:
                part = summaryTable[(summaryTable["dataset"] == d) & (summaryTable["binarize"] == method)]
                if len(part) == 0:
                    values.append(None)
                    marks.append("")
                else:
                    values.append(part.iloc[0][column])
                    mark = part.iloc[0]["stopped"]
                    marks.append(mark if isinstance(mark, str) else "")
            series[names[method]] = values
            texts[names[method]] = marks
        plotTools.plotBars(numberSets, series, yLabel, "GOSDT (depth 5, lambda 0.005) with different binarizations, "
                           "TO/MO = stopped", os.path.join(folder, "binarize_" + column + ".png"), logY=logY, barText=texts)

    text = ""
    for r in rows:
        if r["dataset"] == "pima" and r["fold"] == 0 and r["binarize"] in ["domain", "quantile10"]:
            text = text + "==== Pima fold 0, " + names[r["binarize"]] + ": " + str(r.get("leaves")) + " leaves, test accuracy "
            text = text + str(r.get("testAccuracy")) + "\n" + r.get("treeText", "") + "\n"
    expTools.saveText(text, folder, "pima_trees.txt")
