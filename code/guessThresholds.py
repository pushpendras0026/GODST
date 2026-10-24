# guessThresholds.py
# EXTENSION idea 1: threshold guessing (McTavish et al. 2022, Sec 4.1 "Column Elimination").
#
# With all thresholds ("all" in binarize.py) a continuous column with k values gives k-1
# binary features, and the search space grows like M! (paper Thm H.1). That is too much.
# Idea: a boosted tree model (GBDT) is accurate, and it only uses a few thresholds. So we
# only keep the thresholds that the GBDT uses:
#   1. train a GBDT on the raw data, take every (column, threshold) used in any of its trees
#   2. make 0/1 columns "x <= t" from them, train the GBDT again on these columns
#   3. remove the least important column (Gini importance), train again
#   4. stop when the training accuracy goes down (then we put the last column back)
# The GBDT comes from scikit-learn, it is only the reference model, not our tree search.

import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

import binarize


def makeGbdt(numTrees, treeDepth, seed=0, modelKind="gbdt"):
    # the paper uses gradient boosting (gbdt). In our ablation we also try other tree ensembles
    if modelKind == "rf":
        return RandomForestClassifier(n_estimators=numTrees, max_depth=treeDepth, random_state=seed)
    if modelKind == "extra":
        return ExtraTreesClassifier(n_estimators=numTrees, max_depth=treeDepth, random_state=seed)
    if modelKind == "ada":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=treeDepth),
                                  n_estimators=numTrees, random_state=seed)
    return GradientBoostingClassifier(n_estimators=numTrees, max_depth=treeDepth, random_state=seed)


def treesOf(model):
    # gradient boosting keeps its trees in a 2D table, the other ensembles in a list
    trees = []
    if isinstance(model, GradientBoostingClassifier):
        for k in range(model.estimators_.shape[0]):
            trees.append(model.estimators_[k, 0])
    else:
        for estimator in model.estimators_:
            trees.append(estimator)
    return trees


def thresholdsUsedBy(model, numColumns):
    # every (column number, threshold) used in a split of any tree of the ensemble
    found = []
    for estimator in treesOf(model):
        tree = estimator.tree_
        for node in range(tree.node_count):
            column = tree.feature[node]
            if column >= 0:
                # leaves have column -2, so only real splits get here
                pair = (int(column), float(tree.threshold[node]))
                if pair not in found:
                    found.append(pair)
    found.sort()
    return found


def makeThresholdColumns(values, pairs):
    # 0/1 matrix with one column per (column, threshold): 1 if value <= threshold
    answer = np.zeros((values.shape[0], len(pairs)), dtype=np.uint8)
    for j in range(len(pairs)):
        column = pairs[j][0]
        threshold = pairs[j][1]
        answer[:, j] = (values[:, column] <= threshold).astype(np.uint8)
    return answer


def guessThresholds(rawTrain, yTrain, numTrees, treeDepth, columnElimination=True, seed=0, modelKind="gbdt"):
    # returns a dict column name -> list of thresholds, and some info for the report
    startTime = time.time()
    values = rawTrain.values.astype(float)
    model = makeGbdt(numTrees, treeDepth, seed, modelKind)
    model.fit(values, yTrain)
    pairs = thresholdsUsedBy(model, values.shape[1])
    info = {"thresholdsFromGbdt": len(pairs)}

    thresholdColumns = makeThresholdColumns(values, pairs)
    model = makeGbdt(numTrees, treeDepth, seed, modelKind)
    model.fit(thresholdColumns, yTrain)
    startAccuracy = model.score(thresholdColumns, yTrain)
    info["gbdtTrainAccuracy"] = startAccuracy

    if columnElimination:
        while len(pairs) > 1:
            weakest = int(np.argmin(model.feature_importances_))
            removedPair = pairs[weakest]
            removedColumn = thresholdColumns[:, weakest].copy()
            pairs.pop(weakest)
            thresholdColumns = np.delete(thresholdColumns, weakest, axis=1)
            model = makeGbdt(numTrees, treeDepth, seed, modelKind)
            model.fit(thresholdColumns, yTrain)
            accuracy = model.score(thresholdColumns, yTrain)
            if accuracy < startAccuracy:
                # accuracy went down, so the last removed column was needed, put it back
                pairs.insert(weakest, removedPair)
                thresholdColumns = np.insert(thresholdColumns, weakest, removedColumn, axis=1)
                break

    givenThresholds = {}
    for pair in pairs:
        name = rawTrain.columns[pair[0]]
        if name not in givenThresholds:
            givenThresholds[name] = []
        givenThresholds[name].append(pair[1])
    info["thresholdsKept"] = len(pairs)
    info["seconds"] = time.time() - startTime
    return givenThresholds, info


def binarizeWithGuess(rawTrain, yTrain, rawTest, numTrees, treeDepth, columnElimination=True, modelKind="gbdt"):
    # the whole step: guess thresholds on the train part, then make 0/1 train and test data
    givenThresholds, info = guessThresholds(rawTrain, yTrain, numTrees, treeDepth, columnElimination,
                                            modelKind=modelKind)
    encoder = binarize.makeEncoder(rawTrain, yTrain, "given", givenThresholds=givenThresholds)
    Xtrain, names, groups = binarize.applyEncoder(encoder, rawTrain)
    Xtest, names2, groups2 = binarize.applyEncoder(encoder, rawTest)
    info["binaryFeatures"] = Xtrain.shape[1]
    return Xtrain, Xtest, names, groups, info


if __name__ == "__main__":
    import dataTools
    for name, numTrees, treeDepth in [("ext_compas", 20, 3), ("broward", 40, 1), ("spiral", 20, 3), ("takeaway", 50, 2)]:
        rawTrain, yTrain, rawTest, yTest = dataTools.getRawFold(name, 0)
        allEncoder = binarize.makeEncoder(rawTrain, yTrain, "all")
        Xall, namesAll, groupsAll = binarize.applyEncoder(allEncoder, rawTrain)
        Xtrain, Xtest, names, groups, info = binarizeWithGuess(rawTrain, yTrain, rawTest, numTrees, treeDepth)
        print(name, "all thresholds:", Xall.shape[1], "binary features | after guessing:", info["binaryFeatures"],
              "| gbdt used", info["thresholdsFromGbdt"], "| gbdt train acc", round(info["gbdtTrainAccuracy"], 4),
              "|", round(info["seconds"], 2), "s")
        print("   ", names[:8])
