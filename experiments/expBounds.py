# expBounds.py   (python run.py --stage bounds)
# Step 4: same objective as step 3, plus the bounds. Same answer, much less work?

import os

import dataTools
import expTools
import plotTools
import treeTools
import step3_sparsity
import step4_bounds


def runStage(settings):
    folder = expTools.makeResultFolder("04_bounds")
    expTools.startLog(folder, "bounds", settings)
    datasets = expTools.pickList(["monk1", "monk2", "monk3", "balance", "car", "compas_binary", "bar7",
                                  "tictactoe", "fico_binary"], ["monk1", "car"], settings, "dataset")
    lams = expTools.pickList([0.05, 0.02, 0.01, 0.005], [0.05, 0.01], settings, "lam")
    timeLimit = expTools.pickTimeLimit(300, 30, settings)

    rows = []
    for name in datasets:
        X, y, names, groups = dataTools.getWholeData(name)
        step3GaveUp = False
        for lam in lams:
            answers = {}
            for method in ["step3 (no bounds)", "step4 (bounds)"]:
                if method.startswith("step3") and step3GaveUp:
                    rows.append({"dataset": name, "lam": lam, "method": method, "timedOut": True,
                                 "note": "skipped, already timed out at a bigger lambda"})
                    continue
                if method.startswith("step3"):
                    result = step3_sparsity.sparseDP(X, y, lam, timeLimit=timeLimit)
                else:
                    result = step4_bounds.boundsDP(X, y, lam, timeLimit=timeLimit)
                row = {"dataset": name, "lam": lam, "method": method, "seconds": round(result["seconds"], 3),
                       "subproblems": result["subproblems"], "timedOut": result["timedOut"],
                       "objective": expTools.roundOrNone(result["objective"])}
                if not result["timedOut"]:
                    row["leaves"] = treeTools.countLeaves(result["tree"])
                    row["trainAccuracy"] = round(treeTools.accuracyOf(result["tree"], X, y), 4)
                    answers[method] = result["objective"]
                elif method.startswith("step3"):
                    step3GaveUp = True
                rows.append(row)
                expTools.log(folder, str(row))
            if len(answers) == 2:
                values = list(answers.values())
                expTools.log(folder, "  same objective: " + str(abs(values[0] - values[1]) < 1e-9))

    table = expTools.saveTable(rows, folder)

    # bar plot of time and subproblems at one lambda
    showLam = 0.01 if 0.01 in lams else lams[-1]
    for what in ["seconds", "subproblems"]:
        series = {}
        for method in ["step3 (no bounds)", "step4 (bounds)"]:
            values = []
            for name in datasets:
                part = table[(table["dataset"] == name) & (table["method"] == method) & (table["lam"] == showLam)]
                if len(part) > 0 and part.iloc[0]["timedOut"] == False:
                    values.append(part.iloc[0][what])
                else:
                    values.append(None)
            series[method] = values
        plotTools.plotBars(datasets, series, what + " (log scale, missing = timed out)",
                           "Step 3 vs step 4 at lambda = " + str(showLam),
                           os.path.join(folder, what + "_step3_vs_step4.png"), logY=True)
    expTools.log(folder, "done")
