# expCompare.py   (python run.py --stage compare)
# OUR WORK: our sparse trees against other models, on the new datasets and the extension datasets.
#   GOSDT + all guesses (depth 5), lambda chosen by 3 fold CV inside the train part
#   DL8.5 style + guesses, depth chosen by 3 fold CV inside the train part
#   CART (depth chosen by CV), random forest, gradient boosting, AdaBoost,
#   logistic regression, k nearest neighbours, SVM  (scikit-learn, default settings)
# We report test accuracy (mean and std over 5 folds), the average rank of every model,
# and for the trees the number of leaves.

import os

import numpy as np

import dataTools
import expTools
import plotTools
import ourJobs

modelList = ["gosdt", "dl85", "cartCv", "rf", "gbdt", "ada", "logreg", "knn", "svm"]
modelNames = {"gosdt": "GOSDT + guesses (ours)", "dl85": "DL8.5 style + guesses (ours)", "cartCv": "CART",
              "rf": "random forest", "gbdt": "gradient boosting", "ada": "AdaBoost", "logreg": "logistic regression",
              "knn": "kNN", "svm": "SVM"}


def runStage(settings):
    folder = expTools.makeResultFolder("14_model_compare")
    expTools.startLog(folder, "compare", settings)
    datasets = expTools.pickList(dataTools.newDatasets + ["ext_compas", "broward", "ext_fico", "takeaway"],
                                 ["pima", "heart"], settings, "dataset")
    timeLimit = expTools.pickTimeLimit(600, 30, settings)
    folds = 2 if settings.get("quick") else 5
    jobs = []
    for d in datasets:
        for fold in range(folds):
            for model in modelList:
                if model == "gosdt" or model == "dl85":
                    jobs.append(ourJobs.makeJob(d, fold, model, binarizeMethod="guess", lbRef="gbdt", depth=5,
                                                lamChoice="cv", timeLimit=timeLimit))
                else:
                    jobs.append(ourJobs.makeJob(d, fold, model, timeLimit=timeLimit))
    rows = ourJobs.runAll(jobs, settings, folder)
    table = ourJobs.rowsToTable(rows, folder, "runs.csv")

    # mean and std of test accuracy for every dataset and model
    summary = []
    for d in datasets:
        line = {"dataset": d}
        for model in modelList:
            part = table[(table["dataset"] == d) & (table["model"] == model)]
            accs = [a for a in part["testAccuracy"] if a == a]
            if len(accs) > 0:
                line[model] = round(float(np.mean(accs)), 4)
                line[model + "_std"] = round(float(np.std(accs)), 4)
            leaves = [v for v in part["leaves"] if v == v and v is not None]
            if len(leaves) > 0:
                line[model + "_leaves"] = float(np.median(leaves))
        summary.append(line)
    summaryTable = expTools.saveTable(summary, folder, "test_accuracy.csv")

    # average rank over datasets (1 = best on that dataset)
    rankSum = {}
    for model in modelList:
        rankSum[model] = 0.0
    for line in summary:
        values = []
        for model in modelList:
            values.append((-line.get(model, -1.0), model))
        values.sort()
        for position in range(len(values)):
            rankSum[values[position][1]] = rankSum[values[position][1]] + position + 1
    names = []
    ranks = []
    for model in modelList:
        names.append(modelNames[model])
        ranks.append(rankSum[model] / len(summary))
        expTools.log(folder, modelNames[model] + ": average rank " + str(round(rankSum[model] / len(summary), 2)))
    plotTools.plotBars(names, {"average rank (lower is better)": ranks}, "average rank over datasets",
                       "Test accuracy rank of every model", os.path.join(folder, "average_rank.png"))

    # GOSDT against the best black box, dataset by dataset
    series = {"GOSDT + guesses (ours)": [], "best black box": [], "CART": []}
    groups = []
    for line in summary:
        groups.append(line["dataset"])
        best = 0.0
        for model in ["rf", "gbdt", "ada", "logreg", "knn", "svm"]:
            best = max(best, line.get(model, 0.0))
        series["GOSDT + guesses (ours)"].append(line.get("gosdt"))
        series["best black box"].append(best)
        series["CART"].append(line.get("cartCv"))
    plotTools.plotBars(groups, series, "mean test accuracy", "Sparse tree vs best black box model",
                       os.path.join(folder, "gosdt_vs_black_box.png"))

    text = "dataset | " + " | ".join(modelNames[m] for m in modelList) + "\n"
    for line in summary:
        cells = []
        for model in modelList:
            if model in line:
                cell = format(line[model], ".3f")
                if model + "_leaves" in line:
                    cell = cell + " (" + str(int(line[model + "_leaves"])) + " leaves)"
                cells.append(cell)
            else:
                cells.append("-")
        text = text + line["dataset"] + " | " + " | ".join(cells) + "\n"
    expTools.saveText(text, folder, "test_accuracy_table.txt")
    expTools.log(folder, text)
