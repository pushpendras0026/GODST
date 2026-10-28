# binarize.py
# GOSDT only works with 0/1 features, so other columns are changed into 0/1 columns:
#   continuous column  -> one 0/1 column "x<=t" for each threshold t
#   text column        -> one 0/1 column "x=value" for each value, we leave one value out
#                         (same as the paper datasets, for example monk has 11 columns)
#   column already 0/1 -> copied as it is
#
# How we pick thresholds of a continuous column:
#   "all"    : mid point between every two neighbour values (paper Sec 2)
#   "bucket" : same as "all" but we skip the mid point when both neighbour values
#              have only one and the same label (this is bucketization, paper Sec 3)
#   "given"  : thresholds given from outside (for example medical cut points, featureMaker.py)
#   "quantile5", "quantile10" : thresholds at the quantiles, 5 or 10 bins with about the same rows

import numpy as np
import pandas as pd


def isZeroOne(values):
    for v in np.unique(values):
        if v != 0 and v != 1:
            return False
    return True


def allThresholds(values):
    uniqueValues = np.unique(values)
    thresholds = []
    for k in range(len(uniqueValues) - 1):
        thresholds.append((uniqueValues[k] + uniqueValues[k + 1]) / 2.0)
    return thresholds


def bucketThresholds(values, labels):
    uniqueValues = np.unique(values)
    # which labels do we see at each value
    labelsAtValue = {}
    for i in range(len(values)):
        v = values[i]
        if v not in labelsAtValue:
            labelsAtValue[v] = set()
        labelsAtValue[v].add(labels[i])
    thresholds = []
    for k in range(len(uniqueValues) - 1):
        labelsA = labelsAtValue[uniqueValues[k]]
        labelsB = labelsAtValue[uniqueValues[k + 1]]
        if len(labelsA) == 1 and len(labelsB) == 1 and labelsA == labelsB:
            # two neighbours with the same single label, bucketization does not allow a split here
            continue
        thresholds.append((uniqueValues[k] + uniqueValues[k + 1]) / 2.0)
    return thresholds


def quantileThresholds(values, numBins):
    # thresholds at the 1/numBins, 2/numBins, ... quantiles, so every bin has about the same rows
    thresholds = []
    for k in range(1, numBins):
        t = float(np.quantile(values, k / numBins))
        if t < np.max(values) and t not in thresholds:
            thresholds.append(t)
    return sorted(thresholds)


def makeEncoder(rawX, y, method="all", dropValue=None, givenThresholds=None):
    # rawX is a pandas table without the label column
    # dropValue: the category value we leave out (None means leave out the last one)
    # givenThresholds: dict column -> list of thresholds (only when method is "given")
    rules = []
    for col in rawX.columns:
        values = rawX[col].values
        if not pd.api.types.is_numeric_dtype(rawX[col]):
            kind = "category"
        elif isZeroOne(values):
            kind = "copy"
        else:
            kind = "number"

        if kind == "copy":
            # with given thresholds we only keep a 0/1 column if a threshold was given for it
            if method == "given" and col not in givenThresholds:
                continue
            rules.append({"column": col, "type": "copy", "name": str(col)})

        if kind == "category":
            categories = sorted(set(str(v) for v in values))
            if len(categories) > 1:
                if dropValue is not None and dropValue in categories:
                    categories.remove(dropValue)
                else:
                    categories = categories[:-1]
            for c in categories:
                rules.append({"column": col, "type": "==", "value": c, "name": str(col) + "=" + c})

        if kind == "number":
            if method == "all":
                thresholds = allThresholds(values)
            elif method == "bucket":
                thresholds = bucketThresholds(values, y)
            elif method == "given":
                thresholds = sorted(givenThresholds.get(col, []))
            elif method == "quantile5":
                thresholds = quantileThresholds(values, 5)
            elif method == "quantile10":
                thresholds = quantileThresholds(values, 10)
            else:
                raise ValueError("unknown binarize method " + str(method))
            for t in thresholds:
                rules.append({"column": col, "type": "<=", "value": t, "name": str(col) + "<=" + format(t, "g")})
    return {"rules": rules}


def applyEncoder(encoder, rawX):
    # returns the 0/1 numpy matrix, the feature names, and the feature groups.
    # a feature group = all "x<=t" features of one continuous column, sorted by t.
    # step 5 uses the groups for the subset bound (paper App G).
    rules = encoder["rules"]
    n = rawX.shape[0]
    X = np.zeros((n, len(rules)), dtype=np.uint8)
    names = []
    groupOfColumn = {}
    for j in range(len(rules)):
        rule = rules[j]
        values = rawX[rule["column"]].values
        if rule["type"] == "copy":
            column = (values == 1)
        elif rule["type"] == "==":
            column = np.array([str(v) == rule["value"] for v in values])
        else:
            column = (values <= rule["value"])
        X[:, j] = column.astype(np.uint8)
        names.append(rule["name"])
        if rule["type"] == "<=":
            if rule["column"] not in groupOfColumn:
                groupOfColumn[rule["column"]] = []
            groupOfColumn[rule["column"]].append(j)

    groups = []
    for col in groupOfColumn:
        if len(groupOfColumn[col]) >= 2:
            groups.append(groupOfColumn[col])
    return X, names, groups
