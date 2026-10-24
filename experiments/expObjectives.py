# expObjectives.py   (python run.py --stage objectives)
# Step 6: optimal trees for 6 different objectives (paper Sec 5, Fig 2, Fig 3, App I.5, Fig 11).
# FourClass, lambda = 0.01, 80% train / 20% test, same objective settings as the paper figure:
#   acc, bacc, wacc (w = 3), f1 (w = 0.9), auc, pauc (theta = 0.2)
# We also run them on the balance scale data, which is very imbalanced (49 positives of 625),
# to show why accuracy is not enough there.

import os

import dataTools
import expTools
import plotTools
import treeTools
import osdtStyle
import step6_objectives

objectiveList = [("acc", None, None), ("bacc", None, None), ("wacc", 3, None), ("f1", 0.9, None),
                 ("auc", None, None), ("pauc", None, 0.2)]
# training AUC of each tree, read from the legend of the paper Fig 2
paperTrainAuc = {"acc": 0.78, "bacc": 0.81, "wacc": 0.82, "f1": 0.79, "auc": 0.92, "pauc": 0.85}


def objectiveLabel(name, w, theta):
    if w is not None:
        return name + "(w=" + str(w) + ")"
    if theta is not None:
        return name + "(theta=" + str(theta) + ")"
    return name


def runStage(settings):
    folder = expTools.makeResultFolder("06_objectives")
    expTools.startLog(folder, "objectives", settings)
    datasets = expTools.pickList(["fourclass", "balance"], ["fourclass"], settings, "dataset")
    lams = expTools.pickList([0.01], [0.01], settings, "lam")
    timeLimit = expTools.pickTimeLimit(1800, 60, settings)

    rows = []
    for name in datasets:
        Xtrain, ytrain, Xtest, ytest, names, groups = dataTools.getTrainTestSplit(name, 0.2, seed=0)
        expTools.log(folder, name + ": " + str(len(ytrain)) + " train rows (" + str(int(ytrain.sum()))
                     + " positive), " + str(len(ytest)) + " test rows, " + str(Xtrain.shape[1]) + " binary features")
        for lam in lams:
            trainCurves = {}
            testCurves = {}
            treeTexts = ""
            for objName, w, theta in objectiveList:
                result = step6_objectives.fitObjective(Xtrain, ytrain, objName, lam, w, theta,
                                                       timeLimit=timeLimit, featureGroups=groups)
                tree = result["tree"]
                useScores = objName in ["auc", "pauc"]
                fprTrain, tprTrain = step6_objectives.rocCurve(tree, Xtrain, ytrain, useScores)
                fprTest, tprTest = step6_objectives.rocCurve(tree, Xtest, ytest, useScores)
                trainAuc = step6_objectives.areaUnderCurve(fprTrain, tprTrain)
                testAuc = step6_objectives.areaUnderCurve(fprTest, tprTest)
                label = objectiveLabel(objName, w, theta)
                trainCurves[label + " / " + str(round(trainAuc, 2))] = (fprTrain, tprTrain)
                testCurves[label + " / " + str(round(testAuc, 2))] = (fprTest, tprTest)
                objective = result["objectiveSpec"]
                row = {"dataset": name, "lam": lam, "objective": label, "method": result["method"],
                       "trainObjective": round(result["objective"], 6),
                       "testObjective": round(osdtStyle.treeObjective(tree, Xtest, ytest, lam, objective), 6),
                       "leaves": treeTools.countLeaves(tree), "trainAccuracy": round(treeTools.accuracyOf(tree, Xtrain, ytrain), 4),
                       "testAccuracy": round(treeTools.accuracyOf(tree, Xtest, ytest), 4),
                       "trainAUC": round(trainAuc, 4), "testAUC": round(testAuc, 4),
                       "seconds": round(result["seconds"], 3), "timedOut": result["timedOut"],
                       "gap": expTools.roundOrNone(result["gap"])}
                if name == "fourclass":
                    row["paperTrainAUC"] = paperTrainAuc[objName]
                rows.append(row)
                expTools.log(folder, str(row))
                treeTexts = treeTexts + "==== " + label + "  (leaves show positives/negatives of training rows)\n"
                treeTexts = treeTexts + treeTools.treeToText(tree, names) + "\n"
            expTools.saveText(treeTexts, folder, "trees_" + name + "_lam" + str(lam) + ".txt")
            plotTools.plotRoc(trainCurves, "Training ROC of " + name + " (lambda=" + str(lam) + ")",
                              os.path.join(folder, "train_roc_" + name + ".png"))
            plotTools.plotRoc(testCurves, "Test ROC of " + name + " (lambda=" + str(lam) + ")",
                              os.path.join(folder, "test_roc_" + name + ".png"))
    expTools.saveTable(rows, folder)
    expTools.log(folder, "done")
