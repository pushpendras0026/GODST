# expOfficial.py   (python run.py --stage official)
# OPTIONAL check: compare our GOSDT (step 5) with the authors' own package.
# This script is the only place where the official package is used, our code never calls it.
# It needs:   pip install gosdt
#
# Both are exact methods, so the optimal objective must be the same (up to float error,
# the official one uses 32 bit floats). We also write both times, to see how much slower
# our pure python version is than their C++.

import numpy as np
import pandas as pd

import dataTools
import expTools
import treeTools
import step5_gosdt


def officialObjective(classifier, X, y, lam):
    # their classifier.predict uses an old scikit-learn argument that new versions removed,
    # so we call the predict of their tree object directly
    predictions = np.array(classifier.trees_[0].predict(X))
    mistakes = int((predictions != y).sum())
    model = classifier.get_result()["models_string"]
    leaves = model.count("\"prediction\"")
    return mistakes / len(y) + lam * leaves, leaves


def runStage(settings):
    folder = expTools.makeResultFolder("00_official_check")
    expTools.startLog(folder, "official", settings)
    try:
        from gosdt import GOSDTClassifier
    except ImportError:
        expTools.log(folder, "the official gosdt package is not installed, run:  pip install gosdt")
        return

    datasets = expTools.pickList(["monk1", "monk2", "monk3", "balance", "car", "compas_binary", "bar7",
                                  "tictactoe", "fico_binary", "chudi", "compas2016"],
                                 ["monk1", "monk2", "car"], settings, "dataset")
    lams = expTools.pickList([0.05, 0.02, 0.01, 0.005], [0.02], settings, "lam")
    timeLimit = expTools.pickTimeLimit(300, 30, settings)

    rows = []
    matches = 0
    for name in datasets:
        X, y, names, groups = dataTools.getWholeData(name)
        for lam in lams:
            ours = step5_gosdt.gosdt(X, y, lam, timeLimit=timeLimit, featureGroups=groups)
            classifier = GOSDTClassifier(regularization=lam, allow_small_reg=True, time_limit=int(timeLimit))
            classifier.fit(pd.DataFrame(X, columns=names), y)
            result = classifier.get_result()
            theirTreeObjective, theirLeaves = officialObjective(classifier, X, y, lam)
            theyTimedOut = "TIMEOUT" in str(result["status"])
            same = abs(ours["objective"] - theirTreeObjective) < 1e-5
            # if one of the two did not finish, we can only check that the answers do not disagree:
            # the finished optimal objective must lie between the other one's lower bound and its best tree
            if ours["timedOut"] or theyTimedOut:
                if ours["timedOut"] and theyTimedOut:
                    status = "both timed out"
                elif theyTimedOut:
                    agree = (ours["objective"] >= float(result["lower_bound"]) - 1e-5
                             and ours["objective"] <= theirTreeObjective + 1e-5)
                    status = "official timed out, ours inside their bounds" if agree else "DISAGREE"
                else:
                    agree = (theirTreeObjective >= ours["lowerBound"] - 1e-5
                             and theirTreeObjective <= ours["objective"] + 1e-5)
                    status = "ours timed out, official inside our bounds" if agree else "DISAGREE"
            elif same:
                status = "same optimal objective"
                matches = matches + 1
            else:
                status = "DISAGREE"
            row = {"dataset": name, "lam": lam,
                   "ourObjective": round(ours["objective"], 6),
                   "ourLeaves": treeTools.countLeaves(ours["tree"]),
                   "ourSeconds": round(ours["seconds"], 3), "ourTimedOut": ours["timedOut"],
                   "officialObjective": round(theirTreeObjective, 6), "officialLeaves": theirLeaves,
                   "officialUpperBound": round(float(result["upper_bound"]), 6),
                   "officialLowerBound": round(float(result["lower_bound"]), 6),
                   "officialSeconds": round(float(result["time"]), 3), "officialTimedOut": theyTimedOut,
                   "status": status}
            rows.append(row)
            expTools.log(folder, str(row))

    table = expTools.saveTable(rows, folder)
    bothFinished = 0
    for row in rows:
        if not row["ourTimedOut"] and not row["officialTimedOut"]:
            bothFinished = bothFinished + 1
    disagree = int((table["status"] == "DISAGREE").sum())
    expTools.log(folder, "runs where both finished: " + str(bothFinished) + ", same optimal objective in "
                 + str(matches) + " of them")
    expTools.log(folder, "runs that disagree: " + str(disagree))
