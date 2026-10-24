# expDp.py   (python run.py --stage dp)
# Step 2: plain recursion vs dynamic programming with the support set memo.

import os

import dataTools
import expTools
import plotTools
import treeTools
import step1_bruteForce
import step2_dynamicProg


def runStage(settings):
    folder = expTools.makeResultFolder("02_dynamic_prog")
    expTools.startLog(folder, "dp", settings)
    datasets = expTools.pickList(["monk1", "monk2", "monk3", "tictactoe", "car", "compas_binary", "bar7"],
                                 ["monk1", "monk2"], settings, "dataset")
    depths = expTools.pickList([1, 2, 3, 4, 5, 6], [1, 2, 3, 4], settings, "depth")
    timeLimit = expTools.pickTimeLimit(300, 20, settings)

    rows = []
    for name in datasets:
        X, y, names, groups = dataTools.getWholeData(name)
        recursionGaveUp = False
        for depth in depths:
            bruteMistakes = None
            if depth <= 2:
                bruteMistakes = step1_bruteForce.bruteForce(X, y, depth)["mistakes"]
            for method in ["recursion", "memo"]:
                if method == "recursion" and recursionGaveUp:
                    rows.append({"dataset": name, "depth": depth, "method": method, "timedOut": True,
                                 "note": "skipped, smaller depth already timed out"})
                    continue
                result = step2_dynamicProg.dynamicProg(X, y, depth, useMemo=(method == "memo"),
                                                       timeLimit=timeLimit)
                row = {"dataset": name, "depth": depth, "method": method,
                       "seconds": round(result["seconds"], 3), "calls": result["calls"],
                       "subproblemsSaved": result["subproblems"], "timedOut": result["timedOut"],
                       "mistakes": result["mistakes"]}
                if not result["timedOut"]:
                    row["trainAccuracy"] = round(treeTools.accuracyOf(result["tree"], X, y), 4)
                    row["leaves"] = treeTools.countLeaves(result["tree"])
                    if bruteMistakes is not None:
                        row["sameAsBruteForce"] = (bruteMistakes == result["mistakes"])
                else:
                    if method == "recursion":
                        recursionGaveUp = True
                rows.append(row)
                expTools.log(folder, str(row))
                if name == "monk1" and method == "memo" and depth == 4:
                    expTools.saveText(treeTools.treeToText(result["tree"], names), folder, "tree_monk1_depth4.txt")

    table = expTools.saveTable(rows, folder)

    for what in ["seconds", "calls"]:
        series = {}
        for name in datasets:
            for method in ["recursion", "memo"]:
                part = table[(table["dataset"] == name) & (table["method"] == method) & (table["timedOut"] == False)]
                if len(part) > 0:
                    series[name + " " + method] = (list(part["depth"]), list(part[what]))
        plotTools.plotLines(series, "max depth", what, "Step 2: recursion vs memo (" + what + ")",
                            os.path.join(folder, what + "_vs_depth.png"), logY=True)
    expTools.log(folder, "done")
