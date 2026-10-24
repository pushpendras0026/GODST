# expAblation.py   (python run.py --stage ablation)
# OUR WORK: ablation of the guessing ideas, and our new reference ideas.
# Everything uses GOSDT, GBDT guessed thresholds, depth 5, lambda 0.001 (like the extension paper).
#
#   A. which reference for lower bound guessing:
#        none (exact best tree for these thresholds and depth, used to measure the loss)
#        gbdt (paper), random forest, AdaBoost, ExtraTrees
#        intersection of gbdt, rf, ada (OUR IDEA: wrong only if all are wrong -> tighter guarantee)
#        union of gbdt, rf, ada        (wrong if any is wrong -> faster, looser)
#        weak (2 stumps) and random (learned on shuffled labels): what happens with bad guesses
#      "regret" = objective of the guessed tree - objective of the exact best tree.
#      Theorem 4.3 says regret <= (rows the reference gets wrong) / n, we check it every time.
#   B. which ensemble for threshold guessing: gbdt (paper), rf, ada, extra
#   C. our greedy warm start for GOSDT, on and off

import os

import expTools
import plotTools
import ourJobs

referenceList = [None, "gbdt", "rf", "ada", "extra", "intersection", "union", "weak", "random"]
referenceNames = {None: "no guess (exact)", "gbdt": "GBDT (paper)", "rf": "random forest", "ada": "AdaBoost",
                  "extra": "ExtraTrees", "intersection": "intersection (ours)", "union": "union (ours)",
                  "weak": "weak (2 stumps)", "random": "random reference"}


def runStage(settings):
    folder = expTools.makeResultFolder("15_ablation")
    expTools.startLog(folder, "ablation", settings)
    datasets = expTools.pickList(["ext_compas", "broward", "ext_fico", "takeaway", "pima", "breast_cancer", "titanic",
                                  "german_credit"], ["pima", "ext_compas"], settings, "dataset")
    timeLimit = expTools.pickTimeLimit(600, 30, settings)
    folds = 2 if settings.get("quick") else 5
    lam = 0.001

    # ---------- A. reference for lower bound guessing
    jobs = []
    for d in datasets:
        for fold in range(folds):
            for ref in referenceList:
                jobs.append(ourJobs.makeJob(d, fold, "gosdt", binarizeMethod="guess", lbRef=ref, lam=lam, depth=5,
                                            timeLimit=timeLimit))
    rows = ourJobs.runAll(jobs, settings, folder)
    table = ourJobs.rowsToTable(rows, folder, "reference_runs.csv")

    # regret against the exact run of the same fold, and the Theorem 4.3 check
    exact = {}
    for r in rows:
        if r["lbRef"] is None and not r.get("timedOut"):
            exact[(r["dataset"], r["fold"])] = r["objective"]
    regretRows = []
    for r in rows:
        key = (r["dataset"], r["fold"])
        line = {"dataset": r["dataset"], "fold": r["fold"], "reference": referenceNames[r["lbRef"]],
                "seconds": r.get("seconds"), "timedOut": r.get("timedOut"), "leaves": r.get("leaves"),
                "testAccuracy": r.get("testAccuracy"), "refWrongFraction": r.get("refWrongFraction")}
        if key in exact and r.get("objective") is not None and not r.get("timedOut"):
            regret = r["objective"] - exact[key]
            line["regret"] = regret
            if r.get("refWrongFraction") is not None:
                line["theorem43Holds"] = regret <= r["refWrongFraction"] + 1e-9
        regretRows.append(line)
    regretTable = expTools.saveTable(regretRows, folder, "reference_regret.csv")

    summary = []
    for ref in referenceList:
        name = referenceNames[ref]
        part = regretTable[regretTable["reference"] == name]
        holds = [v for v in part.get("theorem43Holds", []) if v == v]
        summary.append({"reference": name, "seconds": ourJobs.median(list(part["seconds"])),
                        "regret": ourJobs.median(list(part["regret"])) if "regret" in part else None,
                        "maxRegret": float(part["regret"].max()) if "regret" in part and part["regret"].notna().any() else None,
                        "refWrongFraction": ourJobs.median(list(part["refWrongFraction"])),
                        "testAccuracy": ourJobs.median(list(part["testAccuracy"])),
                        "leaves": ourJobs.median(list(part["leaves"])),
                        "theorem43Checked": len(holds), "theorem43Holds": int(sum(1 for v in holds if v)),
                        "stopped": int(part["timedOut"].sum())})
    expTools.saveTable(summary, folder, "reference_summary.csv")
    for s in summary:
        expTools.log(folder, str(s))

    for column, yLabel, logY in [("seconds", "seconds (median, log)", True), ("regret", "regret = objective - exact (median)", False),
                                 ("testAccuracy", "test accuracy (median)", False)]:
        series = {}
        for ref in referenceList:
            if column == "regret" and ref is None:
                continue
            values = []
            for d in datasets:
                part = regretTable[(regretTable["dataset"] == d) & (regretTable["reference"] == referenceNames[ref])]
                if column in part:
                    values.append(ourJobs.median(list(part[column])))
                else:
                    values.append(None)
            series[referenceNames[ref]] = values
        plotTools.plotBars(datasets, series, yLabel, "Lower bound guessing with different references",
                           os.path.join(folder, "reference_" + column + ".png"), logY=logY)

    # ---------- B. ensemble used for threshold guessing
    jobs = []
    for d in datasets:
        for fold in range(folds):
            for kind in ["gbdt", "rf", "ada", "extra"]:
                jobs.append(ourJobs.makeJob(d, fold, "gosdt", binarizeMethod="guess", guessModel=kind, lam=lam, depth=5,
                                            timeLimit=timeLimit))
    rows = ourJobs.runAll(jobs, settings, folder)
    table = ourJobs.rowsToTable(rows, folder, "threshold_model_runs.csv")
    summary = []
    for key, group in table.groupby(["dataset", "guessModel"]):
        summary.append({"dataset": key[0], "guessModel": key[1],
                        "binaryFeatures": ourJobs.median(list(group["binaryFeatures"])),
                        "seconds": ourJobs.median(list(group["seconds"])),
                        "leaves": ourJobs.median(list(group["leaves"])),
                        "trainAccuracy": ourJobs.median(list(group["trainAccuracy"])),
                        "testAccuracy": ourJobs.median(list(group["testAccuracy"])),
                        "stopped": int(group["timedOut"].sum())})
    summaryTable = expTools.saveTable(summary, folder, "threshold_model_summary.csv")
    for s in summary:
        expTools.log(folder, str(s))
    for column, yLabel, logY in [("binaryFeatures", "binary features after guessing", False),
                                 ("testAccuracy", "test accuracy (median)", False), ("seconds", "seconds (median, log)", True)]:
        series = {}
        for kind in ["gbdt", "rf", "ada", "extra"]:
            values = []
            for d in datasets:
                part = summaryTable[(summaryTable["dataset"] == d) & (summaryTable["guessModel"] == kind)]
                values.append(part.iloc[0][column] if len(part) > 0 else None)
            series[kind] = values
        plotTools.plotBars(datasets, series, yLabel, "Threshold guessing with different ensembles",
                           os.path.join(folder, "threshold_model_" + column + ".png"), logY=logY)

    # ---------- C. our greedy warm start for GOSDT (all guesses, lambda 0.005)
    warmSets = list(datasets)
    if not settings.get("quick") and settings.get("dataset") is None:
        warmSets.append("spambase")
    jobs = []
    for d in warmSets:
        for fold in range(folds):
            for warm in [False, True]:
                jobs.append(ourJobs.makeJob(d, fold, "gosdt", binarizeMethod="guess", lbRef="gbdt", lam=0.005, depth=5,
                                            timeLimit=timeLimit, warmStart=warm))
    rows = ourJobs.runAll(jobs, settings, folder)
    table = ourJobs.rowsToTable(rows, folder, "warm_start_runs.csv")
    if "warmStart" not in table:
        table["warmStart"] = False
    table["warmStart"] = table["warmStart"].fillna(False)
    summary = []
    for key, group in table.groupby(["dataset", "warmStart"]):
        summary.append({"dataset": key[0], "warmStart": key[1], "seconds": ourJobs.median(list(group["seconds"])),
                        "objective": ourJobs.median(list(group["objective"])),
                        "testAccuracy": ourJobs.median(list(group["testAccuracy"])),
                        "leaves": ourJobs.median(list(group["leaves"])), "stopped": int(group["timedOut"].sum())})
    expTools.saveTable(summary, folder, "warm_start_summary.csv")
    for s in summary:
        expTools.log(folder, str(s))
