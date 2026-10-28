# dl85Style.py
# Our own small version of DL8.5 (Aglin, Nijssen, Schaus 2020), used as a baseline.
# DL8.5 finds the tree with the least mistakes for a fixed max depth (no lambda).
# It is like our step 2 (dynamic programming with a cache) plus branch and bound:
#   - every call gets an upper bound "ub": we only care about answers better than ub
#   - left side is solved first, then the right side gets ub minus the left answer
#   - if a node can not beat its ub, we save a LOWER bound for it in the cache,
#     so next time we see it with the same or smaller ub we can stop right away
# The main difference to GOSDT (paper Sec 4): DL8.5 solves one subproblem fully before the next
# one (blocking calls), it needs a depth limit, and it does not penalize leaves.
# Our addition: with referencePredictions, the mistakes of a reference model are used as the lower
# bound of every node (lower bound guessing, the same idea as in step5_gosdt.py).

import sys
import time

import bitData
import treeTools

sys.setrecursionlimit(20000)


def leafAnswer(rows, data):
    total = rows.bit_count()
    pos = (rows & data["posMask"]).bit_count()
    neg = total - pos
    if pos > neg:
        return neg, treeTools.makeLeaf(1)
    return pos, treeTools.makeLeaf(0)


def solveNode(rows, depthLeft, ub, state):
    # returns (mistakes, tree). If the best possible is >= ub, returns (lowerBound, None)
    key = (rows, depthLeft)
    if key in state["solved"]:
        return state["solved"][key]
    if key in state["lowerBound"] and state["lowerBound"][key] >= ub:
        return state["lowerBound"][key], None

    state["calls"] = state["calls"] + 1
    if state["calls"] % 1000 == 0:
        bitData.checkTime(state["startTime"], state["timeLimit"])

    data = state["data"]
    leafMistakes, leafTree = leafAnswer(rows, data)
    if depthLeft == 0 or leafMistakes == 0:
        state["solved"][key] = (leafMistakes, leafTree)
        return leafMistakes, leafTree

    # equivalent points: these mistakes can never be removed by any tree
    eqMistakes = (rows & data["minorityPosMask"]).bit_count() + (rows & data["minorityNegMask"]).bit_count()
    floor = eqMistakes
    if state["refWrongMask"] is not None:
        # lower bound guessing: we hope for no better than the reference model
        refMistakes = (rows & state["refWrongMask"]).bit_count()
        if refMistakes > floor:
            floor = refMistakes
    if floor >= ub:
        state["lowerBound"][key] = floor
        return floor, None

    bestMistakes = leafMistakes
    bestTree = leafTree
    # we only look for splits that are strictly better than this
    target = min(ub, leafMistakes)
    for f in range(data["m"]):
        if target <= floor:
            # as good as the floor already, we stop here
            break
        leftRows = rows & data["notFeatureMask"][f]
        rightRows = rows & data["featureMask"][f]
        if leftRows == 0 or rightRows == 0:
            continue
        leftMistakes, leftTree = solveNode(leftRows, depthLeft - 1, target, state)
        if leftTree is None or leftMistakes >= target:
            continue
        rightMistakes, rightTree = solveNode(rightRows, depthLeft - 1, target - leftMistakes, state)
        if rightTree is None:
            continue
        total = leftMistakes + rightMistakes
        if total < target:
            bestMistakes = total
            bestTree = treeTools.makeSplit(f, leftTree, rightTree)
            target = total

    if bestMistakes < ub:
        # everything we skipped could not beat bestMistakes, so it is the exact best
        state["solved"][key] = (bestMistakes, bestTree)
        return bestMistakes, bestTree
    # even the leaf is not better than ub, and no split was: the best is at least ub
    state["lowerBound"][key] = ub
    return ub, None


def dl85(X, y, maxDepth, timeLimit=None, referencePredictions=None):
    state = {}
    state["data"] = bitData.makeBitData(X, y)
    state["refWrongMask"] = None
    if referencePredictions is not None:
        wrong = []
        for i in range(len(y)):
            wrong.append(1 if referencePredictions[i] != y[i] else 0)
        state["refWrongMask"] = bitData.columnToMask(wrong)
    state["solved"] = {}
    state["lowerBound"] = {}
    state["calls"] = 0
    state["startTime"] = time.time()
    state["timeLimit"] = timeLimit
    n = X.shape[0]
    result = {}
    result["memoryOut"] = False
    try:
        mistakes, tree = solveNode(state["data"]["allRows"], maxDepth, n + 1, state)
        result["tree"] = tree
        result["mistakes"] = mistakes
        result["timedOut"] = False
    except bitData.TimeIsUp as stopReason:
        result["tree"] = None
        result["mistakes"] = None
        result["timedOut"] = True
        result["memoryOut"] = isinstance(stopReason, bitData.MemoryIsFull)
    result["seconds"] = time.time() - state["startTime"]
    result["calls"] = state["calls"]
    return result


if __name__ == "__main__":
    import dataTools
    import step2_dynamicProg
    for name in ["monk2", "tictactoe", "car"]:
        X, y, names, groups = dataTools.getWholeData(name)
        for depth in [3, 4, 5]:
            a = step2_dynamicProg.dynamicProg(X, y, depth)
            b = dl85(X, y, depth)
            print(name, "depth", depth, "| step2", a["mistakes"], round(a["seconds"], 2), "s | dl85 style",
                  b["mistakes"], round(b["seconds"], 2), "s", "leaves", treeTools.countLeaves(b["tree"]))
