# expGosdt.py   (python run.py --stage gosdt)
# Step 5: full GOSDT (priority queue + dependency graph + bound updates + subset bound)
# compared with step 4. We also try the two queue orders, and the subset bound on and off.

import os

import dataTools
import expTools
import plotTools
import treeTools
import step4_bounds
import step5_gosdt


def runStage(settings):
    folder = expTools.makeResultFolder("05_gosdt")
    expTools.startLog(folder, "gosdt", settings)
    datasets = expTools.pickList(["monk1", "monk2", "monk3", "balance", "car", "compas_binary", "bar7",
                                  "tictactoe", "fico_binary", "chudi", "compas2016", "iris_versicolor"],
                                 ["monk2", "car", "chudi", "compas2016"], settings, "dataset")
    lams = expTools.pickList([0.05, 0.02, 0.01, 0.005], [0.02], settings, "lam")
    timeLimit = expTools.pickTimeLimit(300, 30, settings)

    methods = ["step4 (bounds)", "gosdt", "gosdt no subset bound", "gosdt big first"]
    rows = []
    progressCurves = {}
    for name in datasets:
        X, y, names, groups = dataTools.getWholeData(name)
        for lam in lams:
            answers = []
            for method in methods:
                if method == "gosdt no subset bound" and len(groups) == 0:
                    continue
                if method.startswith("step4"):
                    result = step4_bounds.boundsDP(X, y, lam, timeLimit=timeLimit)
                elif method == "gosdt":
                    result = step5_gosdt.gosdt(X, y, lam, timeLimit=timeLimit, featureGroups=groups)
                elif method == "gosdt no subset bound":
                    result = step5_gosdt.gosdt(X, y, lam, timeLimit=timeLimit, featureGroups=groups,
                                               useSubsetBound=False)
                else:
                    result = step5_gosdt.gosdt(X, y, lam, timeLimit=timeLimit, featureGroups=groups,
                                               queueOrder="big")
                row = {"dataset": name, "features": X.shape[1], "lam": lam, "method": method,
                       "seconds": round(result["seconds"], 3), "subproblems": result["subproblems"],
                       "timedOut": result["timedOut"], "objective": expTools.roundOrNone(result["objective"])}
                if "gap" in result:
                    row["lowerBound"] = expTools.roundOrNone(result["lowerBound"])
                    row["gap"] = expTools.roundOrNone(result["gap"])
                    row["pops"] = result["pops"]
                    row["subsetPrunes"] = result["subsetPrunes"]
                if result["tree"] is not None:
                    row["leaves"] = treeTools.countLeaves(result["tree"])
                    row["trainAccuracy"] = round(treeTools.accuracyOf(result["tree"], X, y), 4)
                    row["treeObjectiveCheck"] = abs(treeTools.objectiveOf(result["tree"], X, y, lam)
                                                    - result["objective"]) < 1e-9
                if not result["timedOut"]:
                    answers.append(result["objective"])
                rows.append(row)
                expTools.log(folder, str(row))
                if method == "gosdt":
                    progressCurves[(name, lam)] = result["progress"]
                    if name == "monk1" and lam == 0.01:
                        expTools.saveText(treeTools.treeToText(result["tree"], names), folder, "tree_monk1_lam0.01.txt")
            allSame = True
            for a in answers:
                if abs(a - answers[0]) > 1e-9:
                    allSame = False
            expTools.log(folder, "  all finished methods give the same objective: " + str(allSame))

    table = expTools.saveTable(rows, folder)

    showLam = 0.01 if 0.01 in lams else lams[0]
    series = {}
    for method in methods:
        values = []
        for name in datasets:
            part = table[(table["dataset"] == name) & (table["method"] == method) & (table["lam"] == showLam)]
            if len(part) > 0 and part.iloc[0]["timedOut"] == False:
                values.append(part.iloc[0]["seconds"])
            else:
                values.append(None)
        series[method] = values
    plotTools.plotBars(datasets, series, "seconds (log scale, missing = timed out or not used)",
                       "Step 4 vs GOSDT at lambda = " + str(showLam),
                       os.path.join(folder, "seconds_step4_vs_gosdt.png"), logY=True)

    # how the root bounds move over time (GOSDT can be stopped any time and still gives a gap)
    for key in progressCurves:
        curve = progressCurves[key]
        # only plot runs that took some time, short runs have nothing to see
        if len(curve) < 4 or curve[-1][0] < 1.0:
            continue
        ts = []
        lbs = []
        ubs = []
        for point in curve:
            ts.append(point[0])
            lbs.append(point[1])
            ubs.append(point[2])
        fileName = "bounds_over_time_" + key[0] + "_lam" + str(key[1]) + ".png"
        plotTools.plotStep({"upper bound (best tree so far)": (ts, ubs), "lower bound": (ts, lbs)},
                           "seconds", "objective", key[0] + " lambda=" + str(key[1]) + ": root bounds",
                           os.path.join(folder, fileName))
    expTools.log(folder, "done")
