# expBrute.py   (python run.py --stage brute)
# Step 1: how fast does brute force blow up?

import os

import dataTools
import expTools
import plotTools
import treeTools
import step1_bruteForce


def runStage(settings):
    folder = expTools.makeResultFolder("01_brute_force")
    expTools.startLog(folder, "brute", settings)
    datasets = expTools.pickList(["monk1", "monk2", "monk3", "tictactoe"], ["monk1"], settings, "dataset")
    depths = expTools.pickList([1, 2, 3], [1, 2], settings, "depth")
    timeLimit = expTools.pickTimeLimit(600, 30, settings)

    rows = []
    for name in datasets:
        X, y, names, groups = dataTools.getWholeData(name)
        for depth in depths:
            if depth > 3:
                expTools.log(folder, "brute force cannot do depth " + str(depth) + ", it needs "
                             + str(step1_bruteForce.countShapes(X.shape[1], depth)) + " trees")
                continue
            result = step1_bruteForce.bruteForce(X, y, depth, timeLimit)
            accuracy = treeTools.accuracyOf(result["tree"], X, y)
            estimate = None
            if result["timedOut"] and result["treesChecked"] > 0:
                # how long would all trees take, if every tree takes the same time
                estimate = result["seconds"] * result["totalTrees"] / result["treesChecked"]
            row = {"dataset": name, "features": X.shape[1], "depth": depth,
                   "totalTrees": result["totalTrees"], "treesChecked": result["treesChecked"],
                   "finished": not result["timedOut"], "seconds": round(result["seconds"], 3),
                   "estimatedSecondsForAll": expTools.roundOrNone(estimate, 1),
                   "mistakes": result["mistakes"], "trainAccuracy": round(accuracy, 4),
                   "leaves": treeTools.countLeaves(result["tree"])}
            rows.append(row)
            expTools.log(folder, str(row))
            if name == "monk1" and not result["timedOut"]:
                expTools.saveText(treeTools.treeToText(result["tree"], names), folder,
                                  "tree_monk1_depth" + str(depth) + ".txt")

    expTools.saveTable(rows, folder)

    # number of trees from Thm H.1 of the paper (no depth limit), just to show the growth
    growth = []
    for m in range(1, 21):
        growth.append({"binaryFeatures": m, "treesThmH1": step1_bruteForce.countTreesTheoremH1(m),
                       "shapesDepth2": step1_bruteForce.countShapes(m, 2),
                       "shapesDepth3": step1_bruteForce.countShapes(m, 3)})
    expTools.saveTable(growth, folder, "tree_count_growth.csv")

    # plots
    series = {}
    timeSeries = {}
    for name in datasets:
        xs = []
        ys = []
        ts = []
        for row in rows:
            if row["dataset"] == name:
                xs.append(row["depth"])
                ys.append(row["totalTrees"])
                if row["finished"]:
                    ts.append(row["seconds"])
                else:
                    ts.append(row["estimatedSecondsForAll"])
        series[name] = (xs, ys)
        timeSeries[name] = (xs, ts)
    plotTools.plotLines(series, "max depth", "number of trees to check", "Brute force: trees to check",
                        os.path.join(folder, "trees_vs_depth.png"), logY=True)
    plotTools.plotLines(timeSeries, "max depth", "seconds (estimated if timed out)", "Brute force: time",
                        os.path.join(folder, "time_vs_depth.png"), logY=True)
    gx = []
    gy = []
    for g in growth:
        gx.append(g["binaryFeatures"])
        gy.append(g["treesThmH1"])
    plotTools.plotLines({"2^M * M! * 2": (gx, gy)}, "number of binary features M", "number of trees",
                        "Thm H.1: number of trees grows like M!", os.path.join(folder, "thmH1_growth.png"),
                        logY=True)
    expTools.log(folder, "done")
