# step3_sparsity.py
# STEP 3: sparsity.
# Up to step 2 we fixed a max depth and only counted mistakes. That gives big trees,
# because the depth limit does not care how many leaves we use.
# Now we change the objective to the one in the paper (Eq. 1):
#
#     R(tree) = mistakes / n  +  lambda * (number of leaves)
#
# Every leaf now costs lambda, so the search itself decides how big the tree should be.
# There is no depth limit anymore. We still use the memo from step 2 (key = rows only now).
#
#     best(rows) = min( leafLoss(rows) + lambda,
#                       min over f of best(rows with f=0) + best(rows with f=1) )
#
# This step has NO bounds yet, so it looks at every reachable support set.

import sys
import time

import bitData
import treeTools

sys.setrecursionlimit(20000)


def solveNode(rows, state):
    if rows in state["memo"]:
        return state["memo"][rows]
    state["calls"] = state["calls"] + 1
    if state["calls"] % 1000 == 0:
        bitData.checkTime(state["startTime"], state["timeLimit"])

    data = state["data"]
    lam = state["lam"]
    total, prediction, leafLoss, eqLoss = bitData.nodeStats(rows, data)
    bestObjective = leafLoss + lam
    bestTree = treeTools.makeLeaf(prediction)

    for f in range(data["m"]):
        leftRows = rows & data["notFeatureMask"][f]
        rightRows = rows & data["featureMask"][f]
        if leftRows == 0 or rightRows == 0:
            continue
        leftAnswer = solveNode(leftRows, state)
        rightAnswer = solveNode(rightRows, state)
        objective = leftAnswer[0] + rightAnswer[0]
        # tiny number so float noise does not make us pick a bigger tree
        if objective < bestObjective - 1e-12:
            bestObjective = objective
            bestTree = treeTools.makeSplit(f, leftAnswer[1], rightAnswer[1])

    answer = (bestObjective, bestTree)
    state["memo"][rows] = answer
    return answer


def sparseDP(X, y, lam, timeLimit=None, posWeight=None, negWeight=None):
    state = {}
    state["data"] = bitData.makeBitData(X, y, posWeight, negWeight)
    state["lam"] = lam
    state["memo"] = {}
    state["calls"] = 0
    state["startTime"] = time.time()
    state["timeLimit"] = timeLimit

    result = {}
    result["memoryOut"] = False
    try:
        answer = solveNode(state["data"]["allRows"], state)
        result["objective"] = answer[0]
        result["tree"] = answer[1]
        result["timedOut"] = False
    except bitData.TimeIsUp as stopReason:
        result["objective"] = None
        result["tree"] = None
        result["timedOut"] = True
        result["memoryOut"] = isinstance(stopReason, bitData.MemoryIsFull)
    result["seconds"] = time.time() - state["startTime"]
    result["subproblems"] = len(state["memo"])
    return result


if __name__ == "__main__":
    import dataTools
    X, y, names, groups = dataTools.getWholeData("monk1")
    for lam in [0.1, 0.05, 0.02, 0.01]:
        r = sparseDP(X, y, lam)
        print("lambda", lam, "objective", round(r["objective"], 4), "leaves", treeTools.countLeaves(r["tree"]),
              "subproblems", r["subproblems"], "seconds", round(r["seconds"], 2))
    print(treeTools.treeToText(r["tree"], names))
