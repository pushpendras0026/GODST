# guessLowerBound.py
# OUR IDEA: lower bound guessing.
#
# The bounds of GOSDT (step 4 and 5) are weak, so the search spends a lot of time proving
# that bad parts of the space are bad. Idea: a reference model (GBDT) that is accurate tells
# us which rows are "hard". For a group of rows s we guess
#
#       lbGuess(s) = (number of rows in s that the reference model gets wrong) / n + lambda
#
# and we stop working on s as soon as we have a subtree that is that good.
# The search part is inside step5_gosdt.py (referencePredictions) and dl85Style.py.
# Here we only make the reference predictions, and the guarantee we get from it:
#
#   R(guessed tree) <= ( |rows reference gets wrong| + |rows reference gets right but tree t gets wrong| ) / n
#                      + lambda * leaves(t)          for ANY tree t (with the same depth limit)
#
# (in short: the guessed tree is never much worse than any tree that agrees with the reference)

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

import treeTools


def makeReferenceModel(numTrees, treeDepth, seed=0, modelKind="gbdt"):
    # gradient boosting (gbdt) is our main choice, other tree ensembles can be tried too
    if modelKind == "rf":
        return RandomForestClassifier(n_estimators=numTrees, max_depth=treeDepth, random_state=seed)
    if modelKind == "extra":
        return ExtraTreesClassifier(n_estimators=numTrees, max_depth=treeDepth, random_state=seed)
    if modelKind == "ada":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=treeDepth),
                                  n_estimators=numTrees, random_state=seed)
    return GradientBoostingClassifier(n_estimators=numTrees, max_depth=treeDepth, random_state=seed)


def referencePredictions(Xtrain, yTrain, numTrees, treeDepth, seed=0, modelKind="gbdt"):
    # the reference model is trained on the SAME 0/1 data the tree search sees,
    # and we use its predictions on the training rows.
    # modelKind lets us try other ensembles, "gbdt" is the main choice.
    model = makeReferenceModel(numTrees, treeDepth, seed, modelKind)
    model.fit(Xtrain, yTrain)
    return model.predict(Xtrain)


def combineReferences(predictionList, y, mode):
    # OUR IDEA: use more than one reference model.
    #   "intersection": a row counts as "reference is wrong" only if ALL models are wrong on it.
    #                   The set of wrong rows is smaller, so the guarantee above is
    #                   tighter, but the lower bound is weaker, so the search is slower.
    #   "union"       : a row counts as wrong if ANY model is wrong: stronger (faster) guess,
    #                   but a looser guarantee.
    # We return fake predictions with exactly these wrong rows, so step 5 does not need to change.
    n = len(y)
    answer = np.array(y).copy()
    for i in range(n):
        wrongCount = 0
        for predictions in predictionList:
            if predictions[i] != y[i]:
                wrongCount = wrongCount + 1
        if mode == "intersection":
            wrong = (wrongCount == len(predictionList))
        else:
            wrong = (wrongCount > 0)
        if wrong:
            answer[i] = 1 - y[i]
    return answer


def wrongFraction(predictions, y):
    # |rows the reference gets wrong| / n, the main term of the guarantee
    wrong = 0
    for i in range(len(y)):
        if predictions[i] != y[i]:
            wrong = wrong + 1
    return wrong / len(y)


def guaranteeBound(referencePred, X, y, otherTree, lam):
    # right hand side of the guarantee for one tree t (for example the best tree without guessing)
    n = len(y)
    treePred = treeTools.predictAll(otherTree, X)
    count = 0
    for i in range(n):
        if referencePred[i] != y[i]:
            count = count + 1
        elif treePred[i] != y[i]:
            count = count + 1
    return count / n + lam * treeTools.countLeaves(otherTree)
