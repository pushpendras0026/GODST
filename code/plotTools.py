# plotTools.py
# All plots are made here so every figure looks the same.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colorList = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
             "#bcbd22", "#17becf"]
markerList = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def plotLines(seriesDict, xLabel, yLabel, title, filePath, logY=False, logX=False):
    # seriesDict: name -> (xValues, yValues)
    plt.figure(figsize=(6.4, 4.4))
    k = 0
    for name in seriesDict:
        xs = seriesDict[name][0]
        ys = seriesDict[name][1]
        plt.plot(xs, ys, marker=markerList[k % len(markerList)], color=colorList[k % len(colorList)],
                 label=name, linewidth=1.6, markersize=5)
        k = k + 1
    if logY:
        plt.yscale("log")
    if logX:
        plt.xscale("log")
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filePath, dpi=150)
    plt.close()


def plotBars(groupNames, seriesDict, yLabel, title, filePath, logY=False, barText=None):
    # groupNames: names on the x axis, seriesDict: bar name -> list of values (one per group)
    # barText (optional): bar name -> list of short texts written on top of the bars (like "TO")
    plt.figure(figsize=(max(6.4, len(groupNames) * 0.9), 4.4))
    numSeries = len(seriesDict)
    width = 0.8 / max(1, numSeries)
    k = 0
    for name in seriesDict:
        values = seriesDict[name]
        xs = []
        cleanValues = []
        for g in range(len(groupNames)):
            xs.append(g - 0.4 + width / 2 + k * width)
            if values[g] is None:
                cleanValues.append(0)
            else:
                cleanValues.append(values[g])
        plt.bar(xs, cleanValues, width=width, label=name, color=colorList[k % len(colorList)])
        if barText is not None and name in barText:
            for g in range(len(groupNames)):
                text = barText[name][g]
                if text is not None and text != "" and cleanValues[g] > 0:
                    plt.text(xs[g], cleanValues[g], text, ha="center", va="bottom", fontsize=7)
        k = k + 1
    plt.xticks(range(len(groupNames)), groupNames, rotation=30, ha="right")
    if logY:
        plt.yscale("log")
    plt.ylabel(yLabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8, framealpha=0.6)
    plt.tight_layout()
    plt.savefig(filePath, dpi=150)
    plt.close()


def plotRoc(seriesDict, title, filePath):
    # seriesDict: name -> (fprList, tprList)
    plt.figure(figsize=(5.6, 4.6))
    k = 0
    for name in seriesDict:
        style = "-"
        if k % 2 == 1:
            style = "--"
        plt.plot(seriesDict[name][0], seriesDict[name][1], style, marker=markerList[k % len(markerList)],
                 color=colorList[k % len(colorList)], label=name, linewidth=1.6, markersize=5)
        k = k + 1
    plt.plot([0, 1], [0, 1], ":", color="#999999", linewidth=1)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(filePath, dpi=150)
    plt.close()


def plotGrid(panelDict, xLabel, yLabel, title, filePath, logX=False, logY=False, errorBars=False, columns=3,
             stoppedDict=None):
    # many small plots in one figure (like the paper figures with one panel per dataset)
    # panelDict: panel title -> seriesDict, seriesDict: name -> (xs, ys) or (xs, ys, yLow, yHigh)
    # stoppedDict (optional): panel title -> {name: (xs, ys)} of runs that hit the time or memory
    #   limit. They are drawn as an "x" in the same color, without a line.
    panelNames = list(panelDict.keys())
    rows = (len(panelNames) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.5 * rows), squeeze=False)
    for p in range(rows * columns):
        ax = axes[p // columns][p % columns]
        if p >= len(panelNames):
            ax.axis("off")
            continue
        seriesDict = panelDict[panelNames[p]]
        k = 0
        for name in seriesDict:
            series = seriesDict[name]
            color = colorList[k % len(colorList)]
            marker = markerList[k % len(markerList)]
            # every second line is dashed, so two lines on top of each other are both visible
            style = "-"
            if k % 2 == 1:
                style = "--"
            if errorBars and len(series) == 4:
                lower = []
                upper = []
                for i in range(len(series[1])):
                    lower.append(series[1][i] - series[2][i])
                    upper.append(series[3][i] - series[1][i])
                ax.errorbar(series[0], series[1], yerr=[lower, upper], marker=marker, color=color, linestyle=style,
                            label=name, linewidth=1.3, markersize=4, capsize=2)
            else:
                ax.plot(series[0], series[1], marker=marker, color=color, linestyle=style, label=name,
                        linewidth=1.3, markersize=4)
            if stoppedDict is not None and panelNames[p] in stoppedDict and name in stoppedDict[panelNames[p]]:
                stopped = stoppedDict[panelNames[p]][name]
                if len(stopped[0]) > 0:
                    ax.scatter(stopped[0], stopped[1], marker="x", color=color, s=40, linewidths=1.6)
            k = k + 1
        if stoppedDict is not None:
            ax.scatter([], [], marker="x", color="#444444", label="stopped (time or memory limit)")
        if logX:
            ax.set_xscale("log")
        if logY:
            ax.set_yscale("log")
        ax.set_title(panelNames[p], fontsize=10)
        ax.set_xlabel(xLabel, fontsize=9)
        ax.set_ylabel(yLabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, framealpha=0.55)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filePath, dpi=150)
    plt.close(fig)


def plotStepGrid(panelDict, xLabel, yLabel, title, filePath, columns=4):
    # like plotGrid but with step lines (for best objective over time)
    panelNames = list(panelDict.keys())
    rows = (len(panelNames) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows), squeeze=False)
    for p in range(rows * columns):
        ax = axes[p // columns][p % columns]
        if p >= len(panelNames):
            ax.axis("off")
            continue
        seriesDict = panelDict[panelNames[p]]
        k = 0
        for name in seriesDict:
            xs = seriesDict[name][0]
            ys = seriesDict[name][1]
            ax.step(xs, ys, where="post", color=colorList[k % len(colorList)], label=name, linewidth=1.5)
            k = k + 1
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_title(panelNames[p], fontsize=10)
        ax.set_xlabel(xLabel, fontsize=9)
        ax.set_ylabel(yLabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, framealpha=0.55)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filePath, dpi=150)
    plt.close(fig)


def plotRegions(panelList, filePath):
    # panelList: list of (title, xs, ys, labels, gridX, gridY, gridPrediction) for 2D datasets
    fig, axes = plt.subplots(1, len(panelList), figsize=(5.2 * len(panelList), 5.0), squeeze=False)
    for k in range(len(panelList)):
        title, xs, ys, labels, gridX, gridY, gridPrediction = panelList[k]
        ax = axes[0][k]
        ax.contourf(gridX, gridY, gridPrediction, levels=[-0.5, 0.5, 1.5], colors=["#a6cee3", "#fdbf6f"], alpha=0.8)
        for label, color in [(0, "#1f78b4"), (1, "#ff7f00")]:
            px = []
            py = []
            for i in range(len(labels)):
                if labels[i] == label:
                    px.append(xs[i])
                    py.append(ys[i])
            ax.scatter(px, py, color=color, s=22, label="label = " + str(label), edgecolors="white", linewidths=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(filePath, dpi=150)
    plt.close(fig)


def plotStep(seriesDict, xLabel, yLabel, title, filePath, logX=False):
    # for "best objective over time" plots, values change in steps
    plt.figure(figsize=(6.4, 4.4))
    k = 0
    for name in seriesDict:
        xs = seriesDict[name][0]
        ys = seriesDict[name][1]
        plt.step(xs, ys, where="post", color=colorList[k % len(colorList)], label=name, linewidth=1.6)
        k = k + 1
    if logX:
        plt.xscale("symlog", linthresh=0.01)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filePath, dpi=150)
    plt.close()
