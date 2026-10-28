# featureMaker.py
# OUR feature engineering (Part C of the project).
#
# 1. oneHotText          : text columns -> one 0/1 column per value (scikit-learn models need numbers)
# 2. equalityFeatures    : for two text columns with the SAME set of values, a new 0/1 column
#                          "a==b". The GOSDT paper itself says (App I.9) that monk1's true rule is
#                          "jacket = red OR head = body", but their encoding can not say head = body,
#                          so the tree needs 8 leaves. With this feature it should need only 3.
# 3. monkToText          : the monk files are already 0/1 (k-1 columns per attribute), this puts
#                          them back to text columns so we can make equality features
# 4. domainThresholds    : thresholds a doctor would use (Pima diabetes), instead of data driven ones
# 5. mutualInformation   : a simple filter, keep the k binary columns with the most information
#                          about the label (as many columns as 5 quantile bins give, but chosen
#                          from all thresholds)

import math

import numpy as np
import pandas as pd


def oneHotText(rawTrain, rawTest):
    # values are taken from the TRAIN part only (a value only seen in test gives all zeros)
    newTrain = pd.DataFrame(index=rawTrain.index)
    newTest = pd.DataFrame(index=rawTest.index)
    for col in rawTrain.columns:
        if pd.api.types.is_numeric_dtype(rawTrain[col]):
            newTrain[col] = rawTrain[col].values
            newTest[col] = rawTest[col].values
            continue
        trainValues = rawTrain[col].astype(str).values
        testValues = rawTest[col].astype(str).values
        for value in sorted(set(trainValues)):
            name = str(col) + "=" + value
            newTrain[name] = (trainValues == value).astype(int)
            newTest[name] = (testValues == value).astype(int)
    return newTrain, newTest


def equalityFeatures(rawTrain, rawTest):
    # add "a==b" for every pair of text columns that use the same set of values
    textColumns = []
    for col in rawTrain.columns:
        if not pd.api.types.is_numeric_dtype(rawTrain[col]):
            textColumns.append(col)
    newTrain = rawTrain.copy()
    newTest = rawTest.copy()
    added = []
    for i in range(len(textColumns)):
        for j in range(i + 1, len(textColumns)):
            a = textColumns[i]
            b = textColumns[j]
            if set(rawTrain[a].astype(str)) == set(rawTrain[b].astype(str)):
                name = str(a) + "==" + str(b)
                newTrain[name] = (rawTrain[a].astype(str).values == rawTrain[b].astype(str).values).astype(int)
                newTest[name] = (rawTest[a].astype(str).values == rawTest[b].astype(str).values).astype(int)
                added.append(name)
    return newTrain, newTest, added


# the value that was left out of each monk attribute (k-1 encoding)
monkLeftOut = {"head": "octagon", "body": "octagon", "smiling": "no", "holding": "flag", "jacket": "blue", "tie": "no"}


def monkToText(binaryTable):
    # columns look like "head=round", "head=square"; if none of them is 1 the value is the left out one
    answer = pd.DataFrame(index=binaryTable.index)
    attributes = []
    for col in binaryTable.columns:
        attr = col.split("=")[0]
        if attr not in attributes:
            attributes.append(attr)
    for attr in attributes:
        values = []
        for i in range(len(binaryTable)):
            value = monkLeftOut[attr]
            for col in binaryTable.columns:
                if col.split("=")[0] == attr and binaryTable[col].values[i] == 1:
                    value = col.split("=")[1]
            values.append(value)
        answer[attr] = values
    return answer


# thresholds from medical guidelines for the Pima diabetes data (OpenML column names)
#   plas = glucose 2 hours after the test drink: below 140 normal, 140 to 199 pre-diabetes, 200+ diabetes
#   pres = diastolic blood pressure: below 80 normal, 80 to 89 stage 1, 90+ stage 2
#   mass = body mass index: 18.5 / 25 / 30 / 35 / 40 (under weight, normal, over weight, obese 1, 2, 3)
#   age  = 30 / 40 / 50 years
#   preg = number of pregnancies: 0, 1 to 4, 5 or more
domainThresholdTable = {
    "pima": {"plas": [139.5, 199.5], "pres": [79.5, 89.5], "mass": [18.45, 24.95, 29.95, 34.95, 39.95],
             "age": [29.5, 39.5, 49.5], "preg": [0.5, 4.5]},
}


def domainThresholds(datasetName):
    return domainThresholdTable[datasetName]


def mutualInformation(column, y):
    # I(x; y) = sum over a, b of p(a, b) * log( p(a, b) / (p(a) p(b)) ), x and y are 0/1
    n = len(y)
    total = 0.0
    for a in [0, 1]:
        for b in [0, 1]:
            both = 0
            countA = 0
            countB = 0
            for i in range(n):
                if column[i] == a:
                    countA = countA + 1
                if y[i] == b:
                    countB = countB + 1
                if column[i] == a and y[i] == b:
                    both = both + 1
            if both > 0:
                pAB = both / n
                total = total + pAB * math.log(pAB / ((countA / n) * (countB / n)))
    return total


def selectByMutualInformation(X, y, k):
    # the k columns of X with the highest mutual information with y (kept in their old order)
    scores = []
    for j in range(X.shape[1]):
        # same formula as above, but with numpy counting because there can be thousands of columns
        column = X[:, j]
        n = len(y)
        total = 0.0
        for a in [0, 1]:
            for b in [0, 1]:
                both = int(np.sum((column == a) & (y == b)))
                if both > 0:
                    pAB = both / n
                    pA = np.sum(column == a) / n
                    pB = np.sum(y == b) / n
                    total = total + pAB * math.log(pAB / (pA * pB))
        scores.append((total, j))
    scores.sort(reverse=True)
    chosen = []
    for k2 in range(min(k, len(scores))):
        chosen.append(scores[k2][1])
    chosen.sort()
    return chosen
