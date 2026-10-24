# step2_dynamicProg.py
# STEP 2: dynamic programming.
# Same problem as step 1 (least mistakes, max depth d), but solved in a smarter way.
#
# Idea 1 (paper App E): the best tree for some rows is
#     best(rows, d) = min( mistakes if we make a leaf,
#                          min over features f of  best(rows with f=0, d-1) + best(rows with f=1, d-1) )
# so we solve left and right parts on their own.
#
# Idea 2 (paper Sec 4.1): the answer only depends on WHICH rows are in the node (the support set),
# not on the path of splits that led there. Many different paths give the same rows,
# so we save every answer in a dictionary (memo) with the rows as the key, and never solve it twice.
#
# With useMemo=False we get plain recursion, so we can see how much the memo helps.

import sys
import time

import bitData
import treeTools

sys.setrecursionlimit(20000)


def solveNode(rows, depthLeft, state):
    if state["useMemo"]:
        key = (rows, depthLeft)
        if key in state["memo"]:
            return state["memo"][key]
    state["calls"] = state["calls"] + 1
    if state["calls"] % 1000 == 0:
        bitData.checkTime(state["startTime"], state["timeLimit"])

    data = state["data"]
    total = rows.bit_count()
    pos = (rows & data["posMask"]).bit_count()
    neg = total - pos
    if pos > neg:
        bestMistakes = neg
        bestTree = treeTools.makeLeaf(1)
    else:
        bestMistakes = pos
        bestTree = treeTools.makeLeaf(0)
    bestLeaves = 1

    if depthLeft > 0 and bestMistakes > 0:
        for f in range(data["m"]):
            leftRows = rows & data["notFeatureMask"][f]
            rightRows = rows & data["featureMask"][f]
            if leftRows == 0 or rightRows == 0:
                # one side is empty, so this split does nothing
                continue
            leftAnswer = solveNode(leftRows, depthLeft - 1, state)
            rightAnswer = solveNode(rightRows, depthLeft - 1, state)
            mistakes = leftAnswer[0] + rightAnswer[0]
            leaves = leftAnswer[1] + rightAnswer[1]
            if mistakes < bestMistakes or (mistakes == bestMistakes and leaves < bestLeaves):
                bestMistakes = mistakes
                bestLeaves = leaves
                bestTree = treeTools.makeSplit(f, leftAnswer[2], rightAnswer[2])

    answer = (bestMistakes, bestLeaves, bestTree)
    if state["useMemo"]:
        state["memo"][(rows, depthLeft)] = answer
    return answer


def dynamicProg(X, y, maxDepth, useMemo=True, timeLimit=None):
    state = {}
    state["data"] = bitData.makeBitData(X, y)
    state["useMemo"] = useMemo
    state["memo"] = {}
    state["calls"] = 0
    state["startTime"] = time.time()
    state["timeLimit"] = timeLimit

    result = {}
    result["memoryOut"] = False
    try:
        answer = solveNode(state["data"]["allRows"], maxDepth, state)
        result["tree"] = answer[2]
        result["mistakes"] = answer[0]
        result["timedOut"] = False
    except bitData.TimeIsUp as stopReason:
        result["tree"] = None
        result["mistakes"] = None
        result["timedOut"] = True
        result["memoryOut"] = isinstance(stopReason, bitData.MemoryIsFull)
    result["seconds"] = time.time() - state["startTime"]
    result["calls"] = state["calls"]
    result["subproblems"] = len(state["memo"])
    return result


if __name__ == "__main__":
    import dataTools
    X, y, names, groups = dataTools.getWholeData("monk1")
    for depth in [1, 2, 3, 4]:
        a = dynamicProg(X, y, depth, useMemo=False, timeLimit=60)
        b = dynamicProg(X, y, depth, useMemo=True)
        print("depth", depth, "no memo:", a["mistakes"], a["calls"], "calls", round(a["seconds"], 2), "s",
              "| memo:", b["mistakes"], b["calls"], "calls", round(b["seconds"], 2), "s")
    print(treeTools.treeToText(b["tree"], names))
