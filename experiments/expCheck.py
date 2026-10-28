# expCheck.py   (python run.py --stage check)
# Before we trust any speed numbers we check that all our steps give the SAME best answer.
#   step 1 == step 2 (with and without memo)        for the max depth problem
#   step 3 == step 4 == step 5 (all its options)    for the lambda problem
# and that the tree each method returns really has the objective it says.
# We test on many small random datasets and on the monk datasets.

import numpy as np

import dataTools
import expTools
import treeTools
import step1_bruteForce
import step2_dynamicProg
import step3_sparsity
import step4_bounds
import step5_gosdt
import step6_objectives
import osdtStyle
import dl85Style
import bitData
import guessLowerBound
import featureMaker


def allLeafSets(rows, data, m):
    # every possible tree on these rows, written as a list of leaves (lists of row sets).
    # only for tiny data, it is the "obviously correct" answer we compare against
    answer = [[rows]]
    for f in range(m):
        leftRows = rows & data["notFeatureMask"][f]
        rightRows = rows & data["featureMask"][f]
        if leftRows == 0 or rightRows == 0:
            continue
        for leftPart in allLeafSets(leftRows, data, m):
            for rightPart in allLeafSets(rightRows, data, m):
                answer.append(leftPart + rightPart)
    return answer


def bestWithDepth(rows, depthLeft, data, lam, memo):
    # simple exact answer for: mistakes/n + lambda * leaves, with depth <= depthLeft
    key = (rows, depthLeft)
    if key in memo:
        return memo[key]
    total, prediction, leafLoss, eqLoss = bitData.nodeStats(rows, data)
    best = leafLoss + lam
    if depthLeft > 0:
        for f in range(data["m"]):
            leftRows = rows & data["notFeatureMask"][f]
            rightRows = rows & data["featureMask"][f]
            if leftRows == 0 or rightRows == 0:
                continue
            value = bestWithDepth(leftRows, depthLeft - 1, data, lam, memo)
            value = value + bestWithDepth(rightRows, depthLeft - 1, data, lam, memo)
            if value < best:
                best = value
    memo[key] = best
    return best


def exhaustiveBest(X, y, lam, objective):
    # best objective by checking every tree (leaf labels and scores like in osdtStyle)
    data = bitData.makeBitData(X, y, 1.0, 1.0)
    P = int(y.sum())
    N = len(y) - P
    best = None
    for leaves in allLeafSets(data["allRows"], data, X.shape[1]):
        pieces = []
        for rows in leaves:
            p = (rows & data["posMask"]).bit_count()
            pieces.append((p, rows.bit_count() - p))
        if objective["name"] in ["auc", "pauc"]:
            value = osdtStyle.rankingLoss(pieces, objective, P, N)
        else:
            FP = 0
            FN = 0
            for piece in pieces:
                if osdtStyle.leafLabel(piece[0], piece[1], objective, P, N) == 1:
                    FP = FP + piece[1]
                else:
                    FN = FN + piece[0]
            value = osdtStyle.lossFromErrors(FP, FN, objective, P, N)
        value = value + lam * len(leaves)
        if best is None or value < best:
            best = value
    return best


def randomDataset(rng):
    n = rng.randint(15, 60)
    m = rng.randint(2, 7)
    X = rng.randint(0, 2, size=(n, m)).astype(np.uint8)
    # labels follow a small rule plus some noise, so trees are not trivial
    y = (X[:, 0] ^ (X[:, 1 % m] & X[:, (2 % m)])).astype(int)
    for i in range(n):
        if rng.rand() < 0.15:
            y[i] = 1 - y[i]
    return X, y


def same(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-9


def runStage(settings):
    folder = expTools.makeResultFolder("00_checks")
    expTools.startLog(folder, "check", settings)
    rng = np.random.RandomState(1)
    rows = []
    numRandom = 30 if settings.get("quick") else 150
    failures = 0

    for trial in range(numRandom):
        X, y = randomDataset(rng)
        depth = 2
        brute = step1_bruteForce.bruteForce(X, y, depth)
        noMemo = step2_dynamicProg.dynamicProg(X, y, depth, useMemo=False)
        memo = step2_dynamicProg.dynamicProg(X, y, depth, useMemo=True)
        depthOk = brute["mistakes"] == noMemo["mistakes"] and brute["mistakes"] == memo["mistakes"]

        lam = [0.0, 0.01, 0.02, 0.05, 0.1][trial % 5]
        r3 = step3_sparsity.sparseDP(X, y, lam)
        r4 = step4_bounds.boundsDP(X, y, lam)
        r5a = step5_gosdt.gosdt(X, y, lam, queueOrder="small")
        r5b = step5_gosdt.gosdt(X, y, lam, queueOrder="big")
        lamOk = same(r3["objective"], r4["objective"]) and same(r3["objective"], r5a["objective"])
        lamOk = lamOk and same(r3["objective"], r5b["objective"])
        treeOk = same(treeTools.objectiveOf(r5a["tree"], X, y, lam), r5a["objective"])
        treeOk = treeOk and same(treeTools.objectiveOf(r4["tree"], X, y, lam), r4["objective"])

        ok = depthOk and lamOk and treeOk
        if not ok:
            failures = failures + 1
        rows.append({"test": "random" + str(trial), "n": X.shape[0], "m": X.shape[1], "lam": lam,
                     "bruteMistakes": brute["mistakes"], "dpMistakes": memo["mistakes"],
                     "step3": r3["objective"], "step4": r4["objective"], "step5small": r5a["objective"],
                     "step5big": r5b["objective"], "pass": ok})

    # thresholds of one made up continuous column, to test the subset bound for real
    for trial in range(10):
        n = rng.randint(30, 80)
        values = rng.rand(n)
        y = (values > 0.5).astype(int)
        for i in range(n):
            if rng.rand() < 0.2:
                y[i] = 1 - y[i]
        thresholds = sorted(set(values))[:-1]
        X = np.zeros((n, len(thresholds)), dtype=np.uint8)
        for j in range(len(thresholds)):
            X[:, j] = (values <= thresholds[j]).astype(np.uint8)
        groups = [list(range(len(thresholds)))]
        lam = 0.02
        r4 = step4_bounds.boundsDP(X, y, lam)
        r5 = step5_gosdt.gosdt(X, y, lam, featureGroups=groups, useSubsetBound=True)
        ok = same(r4["objective"], r5["objective"])
        if not ok:
            failures = failures + 1
        rows.append({"test": "threshold" + str(trial), "n": n, "m": X.shape[1], "lam": lam,
                     "step4": r4["objective"], "step5small": r5["objective"], "pass": ok})

    # step 6 objectives, leaf set search and DL8.5 style, against checking every tree
    objectiveSettings = [("acc", None, None), ("bacc", None, None), ("wacc", 3, None), ("f1", 0.9, None),
                         ("auc", None, None), ("pauc", None, 0.2)]
    numSmall = 10 if settings.get("quick") else 40
    for trial in range(numSmall):
        n = rng.randint(10, 30)
        m = rng.randint(2, 4)
        X = rng.randint(0, 2, size=(n, m)).astype(np.uint8)
        y = (X[:, 0] | X[:, 1]).astype(int)
        for i in range(n):
            if rng.rand() < 0.25:
                y[i] = 1 - y[i]
        if y.sum() == 0 or y.sum() == n:
            continue
        lam = [0.0, 0.01, 0.03][trial % 3]
        for name, w, theta in objectiveSettings:
            objective = osdtStyle.makeObjective(name, w, theta)
            expected = exhaustiveBest(X, y, lam, objective)
            search = osdtStyle.leafSetSearch(X, y, lam, objective)
            viaStep6 = step6_objectives.fitObjective(X, y, name, lam, w, theta)
            treeValue = osdtStyle.treeObjective(viaStep6["tree"], X, y, lam, objective)
            ok = same(expected, search["objective"]) and same(expected, viaStep6["objective"])
            ok = ok and same(expected, treeValue)
            if not ok:
                failures = failures + 1
            rows.append({"test": "objective_" + name + str(trial), "n": n, "m": m, "lam": lam,
                         "exhaustive": expected, "leafSetSearch": search["objective"],
                         "step6": viaStep6["objective"], "pass": ok})
        for depth in [1, 2, 3]:
            a = step2_dynamicProg.dynamicProg(X, y, depth)
            b = dl85Style.dl85(X, y, depth)
            ok = a["mistakes"] == b["mistakes"]
            if not ok:
                failures = failures + 1
            rows.append({"test": "dl85_" + str(trial), "n": n, "m": m, "depth": depth,
                         "dpMistakes": a["mistakes"], "dl85Mistakes": b["mistakes"], "pass": ok})

    # our additions: depth budget and lower bound guessing
    numExt = 15 if settings.get("quick") else 60
    for trial in range(numExt):
        X, y = randomDataset(rng)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        lam = [0.0, 0.01, 0.03][trial % 3]
        depth = [1, 2, 3][trial % 3]
        data = bitData.makeBitData(X, y)
        expected = bestWithDepth(data["allRows"], depth, data, lam, {})
        exact = step5_gosdt.gosdt(X, y, lam, depthBudget=depth)
        ok = same(expected, exact["objective"]) and treeTools.treeDepth(exact["tree"]) <= depth
        # reference = true labels: the guess is no stronger than the normal bound, same answer
        perfect = step5_gosdt.gosdt(X, y, lam, depthBudget=depth, referencePredictions=y)
        ok = ok and same(expected, perfect["objective"])
        # reference = predictions of the optimal tree: our guarantee says the guessed tree is optimal too
        bestPred = np.array(treeTools.predictAll(exact["tree"], X))
        fromBest = step5_gosdt.gosdt(X, y, lam, depthBudget=depth, referencePredictions=bestPred)
        ok = ok and same(expected, fromBest["objective"])
        # noisy reference: the guessed tree must still respect our guarantee bound
        noisy = y.copy()
        for i in range(len(noisy)):
            if rng.rand() < 0.2:
                noisy[i] = 1 - noisy[i]
        guessed = step5_gosdt.gosdt(X, y, lam, depthBudget=depth, referencePredictions=noisy)
        guessedValue = treeTools.objectiveOf(guessed["tree"], X, y, lam)
        bound = guessLowerBound.guaranteeBound(noisy, X, y, exact["tree"], lam)
        ok = ok and guessedValue <= bound + 1e-9 and treeTools.treeDepth(guessed["tree"]) <= depth
        # our greedy warm start must not change the optimal answer
        warm = step5_gosdt.gosdt(X, y, lam, depthBudget=depth, warmStart=True)
        ok = ok and same(expected, warm["objective"]) and same(treeTools.objectiveOf(warm["tree"], X, y, lam), expected)
        # DL8.5 style with a perfect reference must give the same mistakes
        d1 = dl85Style.dl85(X, y, depth)
        d2 = dl85Style.dl85(X, y, depth, referencePredictions=y)
        ok = ok and d1["mistakes"] == d2["mistakes"]
        if not ok:
            failures = failures + 1
        rows.append({"test": "guessing" + str(trial), "n": X.shape[0], "m": X.shape[1], "lam": lam, "depth": depth,
                     "exhaustiveWithDepth": expected, "gosdtDepth": exact["objective"],
                     "guessPerfect": perfect["objective"], "guessFromBest": fromBest["objective"],
                     "guessNoisy": guessedValue, "guaranteeBound": bound, "pass": ok})

    # our feature engineering helpers
    for trial in range(10):
        n = rng.randint(20, 60)
        column = rng.randint(0, 2, size=n)
        y = rng.randint(0, 2, size=n)
        plain = featureMaker.mutualInformation(column, y)
        X = np.zeros((n, 1), dtype=np.uint8)
        X[:, 0] = column
        ok = abs(plain - featureMaker.mutualInformation(list(column), list(y))) < 1e-12 and plain >= -1e-12
        # combining references: intersection wrong rows are inside every model's wrong rows,
        # union wrong rows contain every model's wrong rows
        p1 = rng.randint(0, 2, size=n)
        p2 = rng.randint(0, 2, size=n)
        inter = guessLowerBound.combineReferences([p1, p2], y, "intersection")
        union = guessLowerBound.combineReferences([p1, p2], y, "union")
        for i in range(n):
            if inter[i] != y[i] and (p1[i] == y[i] or p2[i] == y[i]):
                ok = False
            if union[i] == y[i] and (p1[i] != y[i] or p2[i] != y[i]):
                ok = False
        if not ok:
            failures = failures + 1
        rows.append({"test": "featureTools" + str(trial), "n": n, "pass": ok})
    rawTrain, yTrain, rawTest, yTest = dataTools.getRawFold("monk1", 0)
    textTrain = featureMaker.monkToText(rawTrain)
    eqTrain, eqTest, added = featureMaker.equalityFeatures(textTrain, featureMaker.monkToText(rawTest))
    # monk1 class = jacket red OR head == body, so this rule must be right on every row
    rule = ((eqTrain["jacket"] == "red") | (eqTrain["head==body"] == 1)).astype(int).values
    # (smiling==tie is also added, both columns have the values yes/no, that is fine)
    ok = "head==body" in added and bool((rule == yTrain).all())
    if not ok:
        failures = failures + 1
    rows.append({"test": "monk1EqualityFeature", "added": str(added), "pass": ok})

    for name in ["monk1", "monk2", "monk3"]:
        X, y, names, groups = dataTools.getWholeData(name)
        for lam in [0.05, 0.02, 0.01]:
            r3 = step3_sparsity.sparseDP(X, y, lam)
            r4 = step4_bounds.boundsDP(X, y, lam)
            r5 = step5_gosdt.gosdt(X, y, lam)
            ok = same(r3["objective"], r4["objective"]) and same(r3["objective"], r5["objective"])
            if not ok:
                failures = failures + 1
            rows.append({"test": name, "n": X.shape[0], "m": X.shape[1], "lam": lam,
                         "step3": r3["objective"], "step4": r4["objective"], "step5small": r5["objective"],
                         "pass": ok})

    expTools.saveTable(rows, folder)
    expTools.log(folder, "tests run: " + str(len(rows)) + "   failed: " + str(failures))
    if failures == 0:
        expTools.log(folder, "ALL CHECKS PASSED, every step gives the same optimal objective")
    else:
        expTools.log(folder, "SOME CHECKS FAILED, look at results.csv")
