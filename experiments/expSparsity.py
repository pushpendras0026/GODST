# expSparsity.py   (python run.py --stage sparsity)
# Step 3: the lambda objective. We show:
#   1. bigger lambda gives smaller trees
#   2. for the same number of leaves, the sparse trees are more accurate than the
#      depth limited trees of step 2 (a better accuracy vs leaves frontier)
#   3. step 3 has no bounds, so it always looks at the same number of subproblems,
#      no matter what lambda is (this is why we need step 4)

import os

import dataTools
import expTools
import plotTools
import treeTools
import step2_dynamicProg
import step3_sparsity


def runStage(settings):
    folder = expTools.makeResultFolder("03_sparsity")
    expTools.startLog(folder, "sparsity", settings)
    datasets = expTools.pickList(["monk1", "monk2", "monk3", "car", "compas_binary", "bar7", "tictactoe"],
                                 ["monk1", "monk2"], settings, "dataset")
    lams = expTools.pickList([0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001], [0.05, 0.02, 0.01], settings, "lam")
    depths = expTools.pickList([1, 2, 3, 4, 5], [1, 2, 3, 4], settings, "depth")
    timeLimit = expTools.pickTimeLimit(600, 30, settings)

    rows = []
    frontier = []
    for name in datasets:
        X, y, names, groups = dataTools.getWholeData(name)
        for lam in lams:
            result = step3_sparsity.sparseDP(X, y, lam, timeLimit=timeLimit)
            row = {"dataset": name, "lam": lam, "seconds": round(result["seconds"], 3),
                   "subproblems": result["subproblems"], "timedOut": result["timedOut"],
                   "objective": expTools.roundOrNone(result["objective"])}
            if not result["timedOut"]:
                row["trainAccuracy"] = round(treeTools.accuracyOf(result["tree"], X, y), 4)
                row["leaves"] = treeTools.countLeaves(result["tree"])
                row["depth"] = treeTools.treeDepth(result["tree"])
                frontier.append({"dataset": name, "method": "sparse (lambda)", "setting": lam,
                                 "leaves": row["leaves"], "trainAccuracy": row["trainAccuracy"]})
                if name == "monk1" and lam == 0.01:
                    expTools.saveText(treeTools.treeToText(result["tree"], names), folder, "tree_monk1_lam0.01.txt")
            rows.append(row)
            expTools.log(folder, str(row))
            if result["timedOut"]:
                expTools.log(folder, "  step 3 timed out on " + name + ", skipping smaller lambdas")
                break
        for depth in depths:
            result = step2_dynamicProg.dynamicProg(X, y, depth, timeLimit=timeLimit)
            if not result["timedOut"]:
                frontier.append({"dataset": name, "method": "depth limit (step 2)", "setting": depth,
                                 "leaves": treeTools.countLeaves(result["tree"]),
                                 "trainAccuracy": round(treeTools.accuracyOf(result["tree"], X, y), 4)})

    expTools.saveTable(rows, folder)
    frontierTable = expTools.saveTable(frontier, folder, "accuracy_vs_leaves.csv")

    for name in datasets:
        series = {}
        for method in ["sparse (lambda)", "depth limit (step 2)"]:
            part = frontierTable[(frontierTable["dataset"] == name) & (frontierTable["method"] == method)]
            part = part.sort_values("leaves")
            if len(part) > 0:
                series[method] = (list(part["leaves"]), list(part["trainAccuracy"]))
        plotTools.plotLines(series, "number of leaves", "training accuracy",
                            name + ": sparse objective vs depth limit", os.path.join(folder, "frontier_" + name + ".png"))
    expTools.log(folder, "done")
