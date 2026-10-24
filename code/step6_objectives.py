# step6_objectives.py
# STEP 6: other objectives, the "Generalized" part of GOSDT (paper Sec 2, App B, C, D).
#
# Up to step 5 the loss was plain accuracy. Real data is often imbalanced, so the paper also
# optimizes these (P = positives, N = negatives):
#   acc  : accuracy                     (FP + FN) / (P + N)
#   bacc : balanced accuracy            (FN / P + FP / N) / 2
#   wacc : weighted accuracy            (FP + w FN) / (w P + N)
#   f1   : F1 score                     (FP + FN) / (2P + FP - FN)
#   auc  : area under ROC convex hull   1 - AUCch            (Eq. 2)
#   pauc : partial area, FPR <= theta   1 - pAUCch
#
# acc, bacc and wacc are still a sum over the rows (just with a weight for positive rows and a
# weight for negative rows), so we can use our step 5 GOSDT with those weights.
# f1, auc and pauc are NOT a sum over leaves (Lemma D.1: the label of one leaf depends on the
# others), so like the paper we use the leaf set search from osdtStyle.py for them.

import osdtStyle
import step5_gosdt
import treeTools

objectiveNames = ["acc", "bacc", "wacc", "f1", "auc", "pauc"]


def fitObjective(X, y, name, lam, w=None, theta=None, timeLimit=None, featureGroups=None):
    objective = osdtStyle.makeObjective(name, w, theta)
    P = int(y.sum())
    N = len(y) - P
    if name in osdtStyle.additiveObjectives:
        posWeight, negWeight = osdtStyle.objectiveWeights(objective, P, N)
        result = step5_gosdt.gosdt(X, y, lam, timeLimit=timeLimit, posWeight=posWeight,
                                   negWeight=negWeight, featureGroups=featureGroups)
        result["method"] = "gosdt with weights"
    else:
        result = osdtStyle.leafSetSearch(X, y, lam, objective, timeLimit=timeLimit)
        result["method"] = "leaf set search"
    treeTools.addCounts(result["tree"], X, y)
    # leaf score = fraction of positives of the TRAINING rows in the leaf (used for ROC curves)
    allLeaves = []
    treeTools.collectLeaves(result["tree"], allLeaves)
    for leaf in allLeaves:
        p = leaf.counts[0]
        n = leaf.counts[1]
        if p + n > 0:
            leaf.score = p / (p + n)
    result["objectiveSpec"] = objective
    return result


def rocCurve(tree, X, y, useScores):
    # ROC points of a tree on some data.
    # useScores=False: the tree only says 0 or 1, so the curve has 3 points (paper Sec 2.1)
    # useScores=True : leaves are ranked by their score, one curve point per leaf (ROC convex hull)
    P = int(y.sum())
    N = len(y) - P
    if useScores:
        scores = treeTools.scoreAll(tree, X)
    else:
        scores = treeTools.predictAll(tree, X)
    # group rows with the same score
    groups = {}
    for i in range(len(y)):
        s = scores[i]
        if s not in groups:
            groups[s] = [0, 0]
        if y[i] == 1:
            groups[s][0] = groups[s][0] + 1
        else:
            groups[s][1] = groups[s][1] + 1
    fprList = [0.0]
    tprList = [0.0]
    tp = 0
    fp = 0
    for s in sorted(groups.keys(), reverse=True):
        tp = tp + groups[s][0]
        fp = fp + groups[s][1]
        fprList.append(fp / N)
        tprList.append(tp / P)
    return fprList, tprList


def areaUnderCurve(fprList, tprList):
    area = 0.0
    for k in range(1, len(fprList)):
        area = area + (fprList[k] - fprList[k - 1]) * (tprList[k] + tprList[k - 1]) / 2.0
    return area


if __name__ == "__main__":
    import dataTools
    Xtrain, ytrain, Xtest, ytest, names, groups = dataTools.getTrainTestSplit("fourclass")
    print("fourclass train rows", len(ytrain), "positives", int(ytrain.sum()))
    settings = [("acc", None, None), ("bacc", None, None), ("wacc", 3, None), ("f1", 0.9, None),
                ("auc", None, None), ("pauc", None, 0.2)]
    for name, w, theta in settings:
        r = fitObjective(Xtrain, ytrain, name, 0.01, w, theta, timeLimit=600)
        useScores = name in ["auc", "pauc"]
        fpr, tpr = rocCurve(r["tree"], Xtrain, ytrain, useScores)
        print(name, "objective", round(r["objective"], 4), "leaves", treeTools.countLeaves(r["tree"]),
              "train AUC", round(areaUnderCurve(fpr, tpr), 3), "seconds", round(r["seconds"], 2),
              "timedOut", r["timedOut"])
