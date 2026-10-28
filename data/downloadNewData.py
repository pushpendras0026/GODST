# downloadNewData.py
# Downloads the NEW datasets (not used in the GOSDT paper) one time and
# saves them as csv in data/new/. The csv files are already in the project, so you only need
# this if you want to see where they come from or make them again:
#       python data/downloadNewData.py
# Sources: OpenML (openml.org) through scikit-learn, and the breast cancer data that comes
# inside scikit-learn. Rows with missing values are removed (like the GOSDT paper, App I.2).
# The label is always the last column, called "label", with 1 = the class written below.

import os

import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new")
os.makedirs(folder, exist_ok=True)


def save(name, table, labelValues, positive, note):
    table = table.copy()
    table["label"] = (labelValues == positive).astype(int).values
    before = len(table)
    table = table.dropna()
    table.to_csv(os.path.join(folder, name + ".csv"), index=False)
    print(name, "rows", len(table), "(removed", before - len(table), "with missing values)",
          "columns", table.shape[1] - 1, "positives", int(table["label"].sum()), "|", note)


def fromOpenml(dataId):
    # parser="liac-arff": the default pandas parser of scikit-learn breaks on number categories with pandas 3
    data = fetch_openml(data_id=dataId, as_frame=True, parser="liac-arff")
    return data.data, data.target.astype(str)


# breast cancer Wisconsin (diagnostic), inside scikit-learn: 0 = malignant, 1 = benign
cancer = load_breast_cancer(as_frame=True)
save("breast_cancer", cancer.data, cancer.target.astype(str), "0", "1 = malignant, sklearn load_breast_cancer")

# Pima Indians diabetes, OpenML 37
X, y = fromOpenml(37)
save("pima", X, y, "tested_positive", "1 = diabetes, OpenML 37")

# banknote authentication, OpenML 1462 (4 image features of bank notes)
X, y = fromOpenml(1462)
save("banknote", X, y, "2", "1 = class 2 (forged), OpenML 1462")

# heart disease (Statlog), OpenML 53
X, y = fromOpenml(53)
save("heart", X, y, "present", "1 = heart disease present, OpenML 53")

# Titanic, OpenML 40945. We keep only columns known before the trip
# (boat and body would tell the answer, name/ticket/cabin are ids)
X, y = fromOpenml(40945)
X = X[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
save("titanic", X, y, "1", "1 = survived, OpenML 40945, 7 columns")

