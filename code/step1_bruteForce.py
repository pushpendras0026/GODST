# step1_bruteForce.py
# STEP 1: brute force.
# We make every possible tree shape up to a max depth, then check each one on the data
# and keep the one with the least mistakes.
# We do not try all leaf labels, because for a fixed shape the best label of a leaf is
# always the majority label of the rows that reach it.
#
# Number of shapes with depth at most d and M features:
#   shapes(0) = 1            (just a leaf)
#   shapes(d) = 1 + M * shapes(d-1) * shapes(d-1)
# This grows crazy fast, which is the point of this step (see also paper Thm H.1).

import time

import bitData
import treeTools


def countShapes(numFeatures, depth):
    if depth == 0:
        return 1
    smaller = countShapes(numFeatures, depth - 1)
    return 1 + numFeatures * smaller * smaller


def countTreesTheoremH1(numFeatures, numClasses=2):
    # paper Thm H.1: number of trees with M binary features (no depth limit) is 2^M * M! * K
    factorial = 1
    for k in range(1, numFeatures + 1):
        factorial = factorial * k
    return (2 ** numFeatures) * factorial * numClasses


def makeAllShapes(numFeatures, depth):
    # a shape is "L" (a leaf) or a list [feature, leftShape, rightShape]
    if depth == 0:
        return ["L"]
    smaller = makeAllShapes(numFeatures, depth - 1)
    shapes = ["L"]
    for f in range(numFeatures):
        for leftShape in smaller:
            for rightShape in smaller:
                shapes.append([f, leftShape, rightShape])
    return shapes


def shapeLeaves(shape):
    if shape == "L":
        return 1
    return shapeLeaves(shape[1]) + shapeLeaves(shape[2])


def shapeMistakes(shape, rows, data):
    # number of wrong rows if we use this shape on these rows
    if shape == "L":
        total = rows.bit_count()
        pos = (rows & data["posMask"]).bit_count()
        neg = total - pos
        if pos < neg:
            return pos
        return neg
    f = shape[0]
    leftRows = rows & data["notFeatureMask"][f]
    rightRows = rows & data["featureMask"][f]
    return shapeMistakes(shape[1], leftRows, data) + shapeMistakes(shape[2], rightRows, data)


def shapeToTree(shape, rows, data):
    # turn the best shape into a real tree with majority labels
    if shape == "L":
        pos = (rows & data["posMask"]).bit_count()
        neg = rows.bit_count() - pos
        if pos > neg:
            return treeTools.makeLeaf(1)
        return treeTools.makeLeaf(0)
    f = shape[0]
    leftTree = shapeToTree(shape[1], rows & data["notFeatureMask"][f], data)
    rightTree = shapeToTree(shape[2], rows & data["featureMask"][f], data)
    return treeTools.makeSplit(f, leftTree, rightTree)


def bruteForce(X, y, maxDepth, timeLimit=None):
    # only depth 0 to 3 is possible here (depth 4 would need billions of shapes in memory)
    if maxDepth > 3:
        raise ValueError("brute force only works up to depth 3")
    startTime = time.time()
    data = bitData.makeBitData(X, y)
    allRows = data["allRows"]
    m = data["m"]

    bestShape = "L"
    bestMistakes = shapeMistakes("L", allRows, data)
    bestLeaves = 1
    checked = 1
    timedOut = False

    if maxDepth > 0:
        # the root is a split on some feature f, and both children are any shape of depth-1
        smallerShapes = makeAllShapes(m, maxDepth - 1)
        try:
            for f in range(m):
                for leftShape in smallerShapes:
                    bitData.checkTime(startTime, timeLimit)
                    for rightShape in smallerShapes:
                        shape = [f, leftShape, rightShape]
                        mistakes = shapeMistakes(shape, allRows, data)
                        checked = checked + 1
                        # if two trees have the same mistakes we keep the smaller one
                        if mistakes < bestMistakes:
                            bestShape = shape
                            bestMistakes = mistakes
                            bestLeaves = shapeLeaves(shape)
                        elif mistakes == bestMistakes and shapeLeaves(shape) < bestLeaves:
                            bestShape = shape
                            bestLeaves = shapeLeaves(shape)
        except bitData.TimeIsUp:
            timedOut = True

    result = {}
    result["tree"] = shapeToTree(bestShape, allRows, data)
    result["mistakes"] = bestMistakes
    result["seconds"] = time.time() - startTime
    result["treesChecked"] = checked
    result["totalTrees"] = countShapes(m, maxDepth)
    result["timedOut"] = timedOut
    return result


if __name__ == "__main__":
    # small demo: python code/step1_bruteForce.py
    import dataTools
    X, y, names, groups = dataTools.getWholeData("monk1")
    for depth in [1, 2]:
        result = bruteForce(X, y, depth)
        print("depth", depth, "trees", result["totalTrees"], "mistakes", result["mistakes"],
              "seconds", round(result["seconds"], 2))
    print(treeTools.treeToText(result["tree"], names))
