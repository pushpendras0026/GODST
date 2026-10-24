# guessLowerBound.py
# EXTENSION idea 3: lower bound guessing (McTavish et al. 2022, Sec 4.3).
#
# The bounds of GOSDT (step 4 and 5) are weak, so the search spends a lot of time proving
# that bad parts of the space are bad. Idea: a reference model (GBDT) that is accurate tells
# us which rows are "hard". For a group of rows s we guess (their Eq. 4):
#
#       lbGuess(s) = (number of rows in s that the reference model gets wrong) / n + lambda
#
# and we stop working on s as soon as we have a subtree that is that good.
# The search part is inside step5_gosdt.py (referencePredictions) and dl85Style.py.
# Here we only make the reference predictions, and the guarantee of their Theorem 4.3:
#
#   R(guessed tree) <= ( |rows reference gets wrong| + |rows reference gets right but tree t gets wrong| ) / n
#                      + lambda * leaves(t)          for ANY tree t (with the same depth limit)

import numpy as np

import guessThresholds
import treeTools


def referencePredictions(Xtrain, yTrain, numTrees, treeDepth, seed=0, modelKind="gbdt"):
    # the paper trains the reference model on the SAME 0/1 data the tree search sees
    # (after threshold guessing), and uses its predictions on the training rows.
    # modelKind lets us try other ensembles (our ablation), "gbdt" is the paper choice.
    model = guessThresholds.makeGbdt(numTrees, treeDepth, seed, modelKind)
    model.fit(Xtrain, yTrain)
    return model.predict(Xtrain)


def combineReferences(predictionList, y, mode):
    # OUR IDEA: use more than one reference model.
    #   "intersection": a row counts as "reference is wrong" only if ALL models are wrong on it.
    #                   The set of wrong rows is smaller, so by Theorem 4.3 the guarantee is
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
    # |rows the reference gets wrong| / n, the main term of Theorem 4.3
    wrong = 0
    for i in range(len(y)):
        if predictions[i] != y[i]:
            wrong = wrong + 1
    return wrong / len(y)


def theorem43Bound(referencePred, X, y, otherTree, lam):
    # right hand side of Theorem 4.3 for one tree t (for example the best tree without guessing)
    n = len(y)
    treePred = treeTools.predictAll(otherTree, X)
    count = 0
    for i in range(n):
        if referencePred[i] != y[i]:
            count = count + 1
        elif treePred[i] != y[i]:
            count = count + 1
    return count / n + lam * treeTools.countLeaves(otherTree)
