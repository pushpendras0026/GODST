# step4_bounds.py
# STEP 4: bounds.
# Same objective as step 3 (mistakes/n + lambda * leaves), same answer, but we skip
# a lot of work using the bounds from the paper (App B, Algorithms 2, 3, 4):
#
#  upper bound (Eq. 10): just make the node a leaf
#        ub = leafLoss + lambda
#  lower bound if we split (Eq. 9): any split tree has at least 2 leaves, and the
#        equivalent points (same features, different labels) are always partly wrong (Thm B.9)
#        splitLb = eqLoss + 2 * lambda
#  must be a leaf (Alg 4, Thm B.6 and B.7, incremental progress):
#        if ub <= splitLb then splitting can never beat the leaf
#  leaf support / accuracy bound (Alg 4): if leafLoss <= lambda, even a perfect split
#        cannot pay for the extra leaf, so it is a leaf
#  sub tree bound (Thm B.3, hierarchical lower bound):
#        if lb(left) + lb(right) >= best so far, this feature cannot win, skip it
#        and if exact(left) + lb(right) >= best so far, skip solving the right side
#
# We also try the features with the smallest lower bound first, so we find a good tree early
# and the "skip" rule fires more often.

import sys
import time

import bitData
import treeTools

sys.setrecursionlimit(20000)


def nodeBounds(rows, state):
    # returns (prediction, leafObjective, lowerBound, mustBeLeaf) for these rows
    # saved in a cache so we do not count bits again
    if rows in state["boundCache"]:
        return state["boundCache"][rows]
    lam = state["lam"]
    total, prediction, leafLoss, eqLoss = bitData.nodeStats(rows, state["data"])
    leafObjective = leafLoss + lam
    splitLowerBound = eqLoss + 2 * lam
    mustBeLeaf = False
    if leafObjective <= splitLowerBound:
        mustBeLeaf = True
    if leafLoss <= lam:
        mustBeLeaf = True
    if mustBeLeaf:
        lowerBound = leafObjective
    else:
        lowerBound = min(leafObjective, splitLowerBound)
    answer = (prediction, leafObjective, lowerBound, mustBeLeaf)
    state["boundCache"][rows] = answer
    return answer


def lowerBoundOf(rows, state):
    # if we already solved this node we know the exact value, that is the best lower bound
    if rows in state["memo"]:
        return state["memo"][rows][0]
    return nodeBounds(rows, state)[2]


def solveNode(rows, state):
    if rows in state["memo"]:
        return state["memo"][rows]
    state["calls"] = state["calls"] + 1
    if state["calls"] % 1000 == 0:
        bitData.checkTime(state["startTime"], state["timeLimit"])

    data = state["data"]
    prediction, leafObjective, lowerBound, mustBeLeaf = nodeBounds(rows, state)
    bestObjective = leafObjective
    bestTree = treeTools.makeLeaf(prediction)

    if not mustBeLeaf:
        # make a list of possible splits with their lower bound
        candidates = []
        for f in range(data["m"]):
            leftRows = rows & data["notFeatureMask"][f]
            rightRows = rows & data["featureMask"][f]
            if leftRows == 0 or rightRows == 0:
                continue
            bound = lowerBoundOf(leftRows, state) + lowerBoundOf(rightRows, state)
            candidates.append((bound, f, leftRows, rightRows))
        candidates.sort()

        for candidate in candidates:
            bound = candidate[0]
            f = candidate[1]
            leftRows = candidate[2]
            rightRows = candidate[3]
            if bound >= bestObjective - 1e-12:
                # list is sorted, so all the next ones are also too bad (Thm B.3)
                state["pruned"] = state["pruned"] + 1
                break
            leftAnswer = solveNode(leftRows, state)
            if leftAnswer[0] + lowerBoundOf(rightRows, state) >= bestObjective - 1e-12:
                state["pruned"] = state["pruned"] + 1
                continue
            rightAnswer = solveNode(rightRows, state)
            objective = leftAnswer[0] + rightAnswer[0]
            if objective < bestObjective - 1e-12:
                bestObjective = objective
                bestTree = treeTools.makeSplit(f, leftAnswer[1], rightAnswer[1])

    answer = (bestObjective, bestTree)
    state["memo"][rows] = answer
    return answer


def boundsDP(X, y, lam, timeLimit=None, posWeight=None, negWeight=None):
    state = {}
    state["data"] = bitData.makeBitData(X, y, posWeight, negWeight)
    state["lam"] = lam
    state["memo"] = {}
    state["boundCache"] = {}
    state["calls"] = 0
    state["pruned"] = 0
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
    result["nodesLookedAt"] = len(state["boundCache"])
    result["pruned"] = state["pruned"]
    return result


if __name__ == "__main__":
    import dataTools
    import step3_sparsity
    for name in ["monk1", "monk2", "car"]:
        X, y, names, groups = dataTools.getWholeData(name)
        for lam in [0.05, 0.01]:
            a = step3_sparsity.sparseDP(X, y, lam, timeLimit=120)
            b = boundsDP(X, y, lam)
            print(name, "lambda", lam, "| step3", a["objective"], a["subproblems"], "subproblems",
                  round(a["seconds"], 2), "s | step4", b["objective"], b["subproblems"], "subproblems",
                  round(b["seconds"], 2), "s")
