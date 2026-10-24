# treeTools.py
# A simple binary decision tree and the small functions we need for it.
# Rule we follow everywhere: at a split on feature f,
#   left child  = rows where feature f is 0
#   right child = rows where feature f is 1
# (same as Algorithm 5 in the paper)

import json


class TreeNode:
    def __init__(self):
        self.isLeaf = True
        self.prediction = 0
        self.score = None      # fraction of positives in the leaf (used for AUC objectives)
        self.counts = None     # (positives, negatives) of training rows in the leaf, for printing
        self.feature = -1
        self.left = None
        self.right = None


def makeLeaf(prediction, score=None):
    node = TreeNode()
    node.isLeaf = True
    node.prediction = prediction
    node.score = score
    return node


def makeSplit(feature, leftTree, rightTree):
    node = TreeNode()
    node.isLeaf = False
    node.feature = feature
    node.left = leftTree
    node.right = rightTree
    return node


def predictOne(tree, row):
    node = tree
    while not node.isLeaf:
        if row[node.feature] == 1:
            node = node.right
        else:
            node = node.left
    return node.prediction


def predictAll(tree, X):
    answers = []
    for i in range(X.shape[0]):
        answers.append(predictOne(tree, X[i]))
    return answers


def findLeaf(tree, row):
    node = tree
    while not node.isLeaf:
        if row[node.feature] == 1:
            node = node.right
        else:
            node = node.left
    return node


def scoreAll(tree, X):
    # real valued score of every row = score of its leaf (if a leaf has no score we use its label)
    answers = []
    for i in range(X.shape[0]):
        leaf = findLeaf(tree, X[i])
        if leaf.score is None:
            answers.append(float(leaf.prediction))
        else:
            answers.append(leaf.score)
    return answers


def addCounts(tree, X, y):
    # write how many positive and negative training rows end in each leaf
    allLeaves = []
    collectLeaves(tree, allLeaves)
    for leaf in allLeaves:
        leaf.counts = (0, 0)
    for i in range(X.shape[0]):
        leaf = findLeaf(tree, X[i])
        pos = leaf.counts[0]
        neg = leaf.counts[1]
        if y[i] == 1:
            pos = pos + 1
        else:
            neg = neg + 1
        leaf.counts = (pos, neg)


def collectLeaves(tree, answer):
    if tree.isLeaf:
        answer.append(tree)
    else:
        collectLeaves(tree.left, answer)
        collectLeaves(tree.right, answer)


def countLeaves(tree):
    if tree.isLeaf:
        return 1
    return countLeaves(tree.left) + countLeaves(tree.right)


def treeDepth(tree):
    # a single leaf has depth 0
    if tree.isLeaf:
        return 0
    leftDepth = treeDepth(tree.left)
    rightDepth = treeDepth(tree.right)
    if leftDepth > rightDepth:
        return leftDepth + 1
    return rightDepth + 1


def accuracyOf(tree, X, y):
    if len(y) == 0:
        return 0.0
    predictions = predictAll(tree, X)
    correct = 0
    for i in range(len(y)):
        if predictions[i] == y[i]:
            correct = correct + 1
    return correct / len(y)


def objectiveOf(tree, X, y, lam, posWeight=None, negWeight=None):
    # paper Eq. 1:  R(tree) = loss + lambda * number of leaves
    # loss is weighted, with weight 1/n for every row in the normal case
    n = len(y)
    if posWeight is None:
        posWeight = 1.0 / n
    if negWeight is None:
        negWeight = 1.0 / n
    predictions = predictAll(tree, X)
    loss = 0.0
    for i in range(n):
        if predictions[i] != y[i]:
            if y[i] == 1:
                loss = loss + posWeight
            else:
                loss = loss + negWeight
    return loss + lam * countLeaves(tree)


def treeToText(tree, featureNames, indent=""):
    # makes a readable text version of the tree
    if tree.isLeaf:
        text = indent + "predict " + str(tree.prediction)
        if tree.counts is not None:
            # same style as the paper figures: positives/negatives in the leaf
            text = text + "   (" + str(tree.counts[0]) + "/" + str(tree.counts[1]) + ")"
        return text + "\n"
    name = featureNames[tree.feature]
    text = indent + "if " + name + ":\n"
    text = text + treeToText(tree.right, featureNames, indent + "|   ")
    text = text + indent + "else:\n"
    text = text + treeToText(tree.left, featureNames, indent + "|   ")
    return text


def treeToDict(tree, featureNames):
    if tree.isLeaf:
        return {"predict": int(tree.prediction)}
    result = {}
    result["feature"] = featureNames[tree.feature]
    result["ifTrue"] = treeToDict(tree.right, featureNames)
    result["ifFalse"] = treeToDict(tree.left, featureNames)
    return result


def saveTree(tree, featureNames, filePath):
    with open(filePath, "w") as f:
        json.dump(treeToDict(tree, featureNames), f, indent=2)
