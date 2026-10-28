# dataTools.py
# Reading the datasets, making them 0/1, and making the 5 folds for cross validation.
# Every csv in data/ has the label in the LAST column.

import os
import numpy as np
import pandas as pd

import binarize

projectFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataFolder = os.path.join(projectFolder, "data")

# file of each dataset, plus small extra info
#   positive   : which label value counts as class 1 (for iris)
#   drop       : category value we leave out when we one hot encode
#   thresholds : fixed thresholds for number columns (FourClass, the paper uses f < 50, 100, 150)
datasetInfo = {
    "fourclass": {"file": "paper/fourclass.csv",
                  "thresholds": {"f1": [49.5, 99.5, 149.5], "f2": [49.5, 99.5, 149.5]}},
    "monk1": {"file": "paper/monk1.csv"},
    "monk2": {"file": "paper/monk2.csv"},
    "monk3": {"file": "paper/monk3.csv"},
    "tictactoe": {"file": "paper/tictactoe.csv", "drop": "b"},
    "car": {"file": "paper/car.csv"},
    "balance": {"file": "paper/balance.csv"},
    "compas_binary": {"file": "paper/compas_binary.csv"},
    "fico_binary": {"file": "paper/fico_binary.csv"},
    "bar7": {"file": "paper/bar7.csv"},
    "compas2016": {"file": "paper/compas2016.csv"},
    "compas": {"file": "paper/compas.csv"},
    "fico": {"file": "paper/fico.csv"},
    "chudi": {"file": "paper/chudi.csv"},
    "iris_setosa": {"file": "paper/iris.csv", "positive": "Iris-setosa"},
    "iris_versicolor": {"file": "paper/iris.csv", "positive": "Iris-versicolor"},
    "iris_virginica": {"file": "paper/iris.csv", "positive": "Iris-virginica"},
    # NEW datasets, not used in the paper (data/downloadNewData.py, data/sources.md)
    "breast_cancer": {"file": "new/breast_cancer.csv"},
    "pima": {"file": "new/pima.csv"},
    "banknote": {"file": "new/banknote.csv"},
    "heart": {"file": "new/heart.csv"},
    "titanic": {"file": "new/titanic.csv"},
}

newDatasets = ["breast_cancer", "pima", "banknote", "heart", "titanic"]


def loadDataset(name):
    # returns the raw table (no label) and the 0/1 label vector
    if name not in datasetInfo:
        raise ValueError("unknown dataset " + name)
    info = datasetInfo[name]
    table = pd.read_csv(os.path.join(dataFolder, info["file"]))
    labelColumn = table.columns[-1]
    rawX = table.drop(columns=[labelColumn])
    if "positive" in info:
        y = (table[labelColumn] == info["positive"]).astype(int).values
    else:
        y = table[labelColumn].astype(int).values
    return rawX, y


def makeEncoderFor(name, rawX, y, method):
    info = datasetInfo[name]
    if "thresholds" in info:
        return binarize.makeEncoder(rawX, y, "given", info.get("drop"), info["thresholds"])
    return binarize.makeEncoder(rawX, y, method, info.get("drop"))


def getWholeData(name, method="all"):
    # the full dataset as 0/1 (no train/test split)
    rawX, y = loadDataset(name)
    encoder = makeEncoderFor(name, rawX, y, method)
    X, names, groups = binarize.applyEncoder(encoder, rawX)
    return X, y, names, groups


def makeFolds(n, k=5, seed=0):
    # shuffle the row numbers and cut them in k parts
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    return np.array_split(order, k)


def splitRaw(name, trainIdx, testIdx, method="all"):
    # binarize with thresholds made from the TRAIN rows only, then apply them to the test rows
    rawX, y = loadDataset(name)
    rawTrain = rawX.iloc[trainIdx].reset_index(drop=True)
    rawTest = rawX.iloc[testIdx].reset_index(drop=True)
    encoder = makeEncoderFor(name, rawTrain, y[trainIdx], method)
    Xtrain, names, groups = binarize.applyEncoder(encoder, rawTrain)
    Xtest, names2, groups2 = binarize.applyEncoder(encoder, rawTest)
    return Xtrain, y[trainIdx], Xtest, y[testIdx], names, groups


def foldRows(n, foldNumber, k=5, seed=0):
    # row numbers of the train part and the test part of one fold
    folds = makeFolds(n, k, seed)
    testIdx = folds[foldNumber]
    trainParts = []
    for f in range(k):
        if f != foldNumber:
            trainParts.append(folds[f])
    trainIdx = np.concatenate(trainParts)
    return trainIdx, testIdx


def getFold(name, foldNumber, k=5, method="all", seed=0):
    # train and test part of one fold of k fold cross validation
    rawX, y = loadDataset(name)
    trainIdx, testIdx = foldRows(len(y), foldNumber, k, seed)
    return splitRaw(name, trainIdx, testIdx, method)


def getRawFold(name, foldNumber, k=5, seed=0):
    # same fold, but the raw columns (not 0/1 yet). Part C makes its own 0/1 columns from them.
    rawX, y = loadDataset(name)
    trainIdx, testIdx = foldRows(len(y), foldNumber, k, seed)
    rawTrain = rawX.iloc[trainIdx].reset_index(drop=True)
    rawTest = rawX.iloc[testIdx].reset_index(drop=True)
    return rawTrain, y[trainIdx], rawTest, y[testIdx]


def getTrainTestSplit(name, testFraction=0.2, seed=0, method="all"):
    # one random split with the same fraction of positives in train and test (stratified)
    rawX, y = loadDataset(name)
    rng = np.random.RandomState(seed)
    trainIdx = []
    testIdx = []
    for label in [0, 1]:
        rowsWithLabel = np.where(y == label)[0]
        rowsWithLabel = rng.permutation(rowsWithLabel)
        numTest = int(round(len(rowsWithLabel) * testFraction))
        testIdx.extend(rowsWithLabel[:numTest])
        trainIdx.extend(rowsWithLabel[numTest:])
    trainIdx = np.array(sorted(trainIdx))
    testIdx = np.array(sorted(testIdx))
    return splitRaw(name, trainIdx, testIdx, method)


def describe(name):
    # small summary like Table 2 of the paper
    rawX, y = loadDataset(name)
    X, yy, names, groups = getWholeData(name)
    return {"dataset": name, "samples": len(y), "rawFeatures": rawX.shape[1],
            "binaryFeatures": X.shape[1], "positives": int(y.sum())}
