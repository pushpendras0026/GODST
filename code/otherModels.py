# otherModels.py
# Baseline models from scikit-learn. We only use them to COMPARE with our own trees,
# our own algorithms never call scikit-learn.
#   cart    : greedy decision tree (Breiman et al. 1984), the paper uses it as the greedy baseline
#   cartCv  : CART where the depth is chosen by 3 fold cross validation on the train part
#   rf      : random forest, 100 trees
#   gbdt    : gradient boosting, 100 trees of depth 3 (also our reference model for lower bound guessing)
#   ada     : AdaBoost, 100 small trees
#   logreg  : logistic regression (numbers scaled first)
#   knn     : 5 nearest neighbours (numbers scaled first)
#   svm     : support vector machine with RBF kernel (numbers scaled first)

import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


def cart(Xtrain, ytrain, maxDepth=None, maxLeaves=None):
    startTime = time.time()
    model = DecisionTreeClassifier(max_depth=maxDepth, max_leaf_nodes=maxLeaves, random_state=0)
    model.fit(Xtrain, ytrain)
    result = {}
    result["model"] = model
    result["seconds"] = time.time() - startTime
    result["leaves"] = int(model.get_n_leaves())
    result["timedOut"] = False
    return result


def makeModel(kind):
    if kind == "cartCv":
        return GridSearchCV(DecisionTreeClassifier(random_state=0),
                            {"max_depth": [1, 2, 3, 4, 5, 6, 8, None]}, cv=3)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=0)
    if kind == "gbdt":
        return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
    if kind == "ada":
        return AdaBoostClassifier(n_estimators=100, random_state=0)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    if kind == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=0))
    raise ValueError("unknown model " + kind)


def fitModel(kind, Xtrain, ytrain):
    startTime = time.time()
    model = makeModel(kind)
    model.fit(Xtrain, ytrain)
    result = {"model": model, "seconds": time.time() - startTime, "leaves": None}
    if kind == "cartCv":
        result["leaves"] = int(model.best_estimator_.get_n_leaves())
    return result


def accuracyOfModel(model, X, y):
    if len(y) == 0:
        return 0.0
    predictions = model.predict(X)
    correct = 0
    for i in range(len(y)):
        if predictions[i] == y[i]:
            correct = correct + 1
    return correct / len(y)
