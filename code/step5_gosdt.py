# step5_gosdt.py
# STEP 5: the full GOSDT algorithm (paper Algorithm 1, Sec 4).
#
# Step 4 solved each child fully before looking at the next one (depth first).
# GOSDT does not do that. It keeps:
#   - a dependency graph: one Problem per support set, with links to its parents (Sec 4.1)
#   - a priority queue of problems to work on (Sec 4.3)
# Every problem has a lower bound lb and an upper bound ub of its best objective.
# When we work on a problem we look at all its splits and update lb and ub (Eq. 7):
#       lb = min( leaf objective, min over splits of lb(left) + lb(right) )
#       ub = min( leaf objective, min over splits of ub(left) + ub(right) )
# If the bounds changed we tell the parents first (high priority), because a better
# bound at the parent can make whole parts of the search useless (asynchronous bound updates).
# We stop when the root has lb == ub, that is the proof that the tree is optimal.
#
# Scope: a child is only useful to a parent while  lb(child) + lb(sibling) <= bound of parent.
# So every child gets a "scope" = biggest objective that still matters for some parent.
# A child whose lb is already >= its scope is skipped, nobody needs it.
#
# Extra bounds used here, all the same as step 4:
#   Eq. 9 and 10 for the first lb and ub, equivalent points (Thm B.9),
#   incremental progress and leaf support (Alg 4), split never optimal if its lb > ub.
# New in this step:
#   subset bound (paper App G, Thm G.1) for thresholds of the same continuous column.
#
# At the end we read the tree out of the graph (Algorithm 6).
#
# OUR ADDITIONS, all off by default, so step 5 alone is the plain paper method:
#   depthBudget          : depth guessing (guessDepth.py). A problem is now (rows, depth left) and a
#                          problem with depth left 0 must be a leaf, so we minimize
#                          loss + lambda * leaves with depth <= d
#   referencePredictions : lower bound guessing (guessLowerBound.py). The mistakes of a reference
#                          model (GBDT) on the rows of a problem are used as its lower bound.
#                          This is a GUESS, not a proof: a problem counts as solved as soon as we
#                          find a subtree as good as the reference model on those rows.
#   warmStart            : build a quick greedy tree first and use its objective as the first upper
#                          bound of the root (OSDT also starts from a greedy tree). The answer is the
#                          same, but if the search must stop early we return at least this tree.

import gc
import heapq
import sys
import time

import bitData
import treeTools

sys.setrecursionlimit(20000)

EPS = 1e-9


class Problem:
    # one node of the dependency graph = one support set
    def __init__(self, rows):
        self.rows = rows
        self.support = 0.0
        self.prediction = 0
        self.leafObjective = 0.0
        self.lb = 0.0
        self.ub = 0.0
        self.depthLeft = None     # only used with a depth budget (depth guessing)
        self.mustBeLeaf = False
        self.featuresFromParent = []
        self.splitFeatures = []
        self.splits = []          # each split is a list [feature, leftProblem, rightProblem, alive]
        self.parents = []
        self.scope = -1.0
        self.expanded = False
        self.explored = False
        self.waitUpdate = False
        self.waitExplore = False

    def isSolved(self):
        return self.ub - self.lb <= EPS


class GosdtSearch:
    def __init__(self, X, y, lam, timeLimit=None, posWeight=None, negWeight=None,
                 featureGroups=None, useSubsetBound=True, queueOrder="small",
                 depthBudget=None, referencePredictions=None, warmStart=False):
        self.data = bitData.makeBitData(X, y, posWeight, negWeight)
        self.lam = lam
        self.warmStart = warmStart
        self.timeLimit = timeLimit
        self.queueOrder = queueOrder
        self.depthBudget = depthBudget
        self.useReference = referencePredictions is not None
        if self.useReference:
            # rows the reference model gets wrong, split by label (so we can use the weights)
            wrongPos = []
            wrongNeg = []
            for i in range(len(y)):
                wrongPos.append(1 if (y[i] == 1 and referencePredictions[i] == 0) else 0)
                wrongNeg.append(1 if (y[i] == 0 and referencePredictions[i] == 1) else 0)
            self.refWrongPosMask = bitData.columnToMask(wrongPos)
            self.refWrongNegMask = bitData.columnToMask(wrongNeg)
        self.featureGroups = featureGroups
        if featureGroups is None:
            self.featureGroups = []
        self.useSubsetBound = useSubsetBound
        self.graph = {}
        self.queue = []
        self.counter = 0
        self.pops = 0
        self.subsetPrunes = 0
        self.progress = []
        self.timedOut = False
        self.memoryOut = False
        self.stalled = False

    # ---------- making problems ----------

    def makeProblem(self, rows, parentFeatures, depthLeft):
        data = self.data
        lam = self.lam
        p = Problem(rows)
        p.depthLeft = depthLeft
        total, prediction, leafLoss, eqLoss = bitData.nodeStats(rows, data)
        p.support = total / data["n"]
        p.prediction = prediction
        p.leafObjective = leafLoss + lam
        p.ub = p.leafObjective

        # the smallest loss we think is possible on these rows
        lossFloor = eqLoss
        if self.useReference:
            # lower bound guessing: the reference model's mistakes on these rows are our guess
            refLoss = data["posWeight"] * (rows & self.refWrongPosMask).bit_count()
            refLoss = refLoss + data["negWeight"] * (rows & self.refWrongNegMask).bit_count()
            if refLoss > lossFloor:
                lossFloor = refLoss
        splitLowerBound = lossFloor + 2 * lam

        # note: if all rows are the same (one equivalence class) then eqLoss == leafLoss,
        # so the first rule below already makes it a leaf.
        # With the reference guess this rule becomes: "if ub <= lbguess + lambda, done"
        if p.leafObjective <= splitLowerBound:
            p.mustBeLeaf = True
        if leafLoss <= lam:
            p.mustBeLeaf = True
        if depthLeft is not None and depthLeft <= 0:
            # depth guessing: no depth left, so it has to be a leaf
            p.mustBeLeaf = True

        if p.mustBeLeaf:
            p.lb = p.leafObjective
        else:
            p.lb = splitLowerBound
            # we only remember which features MIGHT split it, the real check waits
            # until we expand it (many problems are never expanded, this saves time)
            p.featuresFromParent = parentFeatures
        return p

    def getProblem(self, rows, parentFeatures, depthLeft):
        # FIND_OR_CREATE_NODE in Algorithm 1.
        # With a depth budget the same rows with a different depth left is a different problem.
        if self.depthBudget is None:
            key = rows
        else:
            key = (rows, depthLeft)
        if key in self.graph:
            return self.graph[key]
        p = self.makeProblem(rows, parentFeatures, depthLeft)
        self.graph[key] = p
        return p

    def expand(self, p):
        # make (or find) the two children of every split, and link them to p
        data = self.data
        if not p.mustBeLeaf:
            for f in p.featuresFromParent:
                leftRows = p.rows & data["notFeatureMask"][f]
                rightRows = p.rows & data["featureMask"][f]
                if leftRows != 0 and rightRows != 0:
                    p.splitFeatures.append(f)
        childDepth = None
        if p.depthLeft is not None:
            childDepth = p.depthLeft - 1
        for f in p.splitFeatures:
            left = self.getProblem(p.rows & data["notFeatureMask"][f], p.splitFeatures, childDepth)
            right = self.getProblem(p.rows & data["featureMask"][f], p.splitFeatures, childDepth)
            left.parents.append(p)
            right.parents.append(p)
            split = [f, left, right, True]
            p.splits.append(split)
        p.expanded = True
        # not needed anymore, the children got their own copy of the useful features
        p.featuresFromParent = None

    # ---------- bounds ----------

    def subsetBound(self, p):
        # paper Thm G.1. A feature "x<=t" sends x<=t to the right and x>t to the left.
        # For two thresholds a < b of the same column:
        #   right side of a is inside right side of b, left side of b is inside left side of a,
        #   and a smaller set never has a worse best objective.
        #   if ub(left of a) <= lb(left of b)  -> split a is never worse than split b, remove b
        #   if ub(right of b) <= lb(right of a) -> split b is never worse than split a, remove a
        # small dictionary feature -> split, we make it here and do not keep it in the problem
        # (keeping it in every problem used a lot of memory when there are many features)
        splitOf = {}
        for split in p.splits:
            splitOf[split[0]] = split
        for group in self.featureGroups:
            aliveSplits = []
            for f in group:
                if f in splitOf and splitOf[f][3]:
                    aliveSplits.append(splitOf[f])
            if len(aliveSplits) < 2:
                continue
            keeper = aliveSplits[0]
            for k in range(1, len(aliveSplits)):
                other = aliveSplits[k]
                if keeper[1].ub <= other[1].lb:
                    other[3] = False
                    self.subsetPrunes = self.subsetPrunes + 1
                else:
                    keeper = other
            stillAlive = []
            for split in aliveSplits:
                if split[3]:
                    stillAlive.append(split)
            if len(stillAlive) < 2:
                continue
            keeper = stillAlive[-1]
            for k in range(len(stillAlive) - 2, -1, -1):
                other = stillAlive[k]
                if keeper[2].ub <= other[2].lb:
                    other[3] = False
                    self.subsetPrunes = self.subsetPrunes + 1
                else:
                    keeper = other

    def recompute(self, p):
        # Eq. 7 with bounds, returns True if lb or ub changed
        if not p.expanded:
            self.expand(p)
        if self.useSubsetBound and len(self.featureGroups) > 0:
            self.subsetBound(p)

        newLb = p.leafObjective
        newUb = p.leafObjective
        for split in p.splits:
            if not split[3]:
                continue
            lbSum = split[1].lb + split[2].lb
            ubSum = split[1].ub + split[2].ub
            if lbSum < newLb:
                newLb = lbSum
            if ubSum < newUb:
                newUb = ubSum

        oldLb = p.lb
        oldUb = p.ub
        if newUb < p.ub:
            p.ub = newUb
        if newLb > p.lb:
            p.lb = newLb
        if p.lb > p.ub or p.ub - p.lb <= EPS:
            p.lb = p.ub

        # a split whose lower bound is above ub can never be the best one, forget it
        for split in p.splits:
            if split[3] and split[1].lb + split[2].lb > p.ub + EPS:
                split[3] = False

        return p.lb != oldLb or p.ub != oldUb

    # ---------- queue ----------

    def push(self, level, p):
        # level 0 = bound update for a parent (worked on first), level 1 = explore a new problem
        # inside a level the order depends on queueOrder:
        #   "small" : small support first. These problems are near the leaves, finishing them
        #             gives real trees early (good upper bounds), a bit like depth first search
        #   "big"   : big support and small lb first, like the authors C++ code
        # the paper (App I.8) also says the best order depends on the dataset
        if level == 0:
            if p.waitUpdate:
                return
            p.waitUpdate = True
        else:
            if p.waitExplore:
                return
            p.waitExplore = True
        self.counter = self.counter + 1
        if self.queueOrder == "small":
            priority = p.support
        else:
            priority = p.lb - p.support
        heapq.heappush(self.queue, (level, priority, self.counter, p))

    def explore(self, child, newScope):
        if child.isSolved():
            return
        if child.explored and newScope <= child.scope + EPS:
            return
        if newScope > child.scope:
            child.scope = newScope
        child.explored = True
        self.push(1, child)

    def process(self, p):
        if p.isSolved():
            return
        if p is not self.root and p.lb >= p.scope - EPS:
            return
        oldUb = p.ub
        changed = self.recompute(p)
        if changed:
            # we only wake up the parents when it can change their decision:
            # the problem is solved, or it got out of scope (parent can drop this split),
            # or we found a better tree (parent can get a better upper bound)
            if p.isSolved() or p.lb >= p.scope - EPS or p.ub < oldUb:
                for parent in p.parents:
                    self.push(0, parent)
            if p is self.root:
                self.progress.append((time.time() - self.startTime, p.lb, p.ub))
        if p.isSolved():
            return
        boundary = p.ub
        if p.scope < boundary:
            boundary = p.scope
        for split in p.splits:
            if not split[3]:
                continue
            left = split[1]
            right = split[2]
            lbSum = left.lb + right.lb
            ubSum = left.ub + right.ub
            if lbSum <= boundary and lbSum < ubSum - EPS:
                self.explore(left, boundary - right.lb)
                self.explore(right, boundary - left.lb)

    # ---------- main loop ----------

    # ---------- greedy warm start (our addition, off by default) ----------

    def greedyTree(self, rows, depthLeft):
        # a quick greedy tree for the same objective (like CART but it pays lambda per leaf):
        # split on the feature with the smallest child loss, only if it pays for the extra leaf.
        # returns (objective, tree)
        data = self.data
        lam = self.lam
        total, prediction, leafLoss, eqLoss = bitData.nodeStats(rows, data)
        leafAnswer = (leafLoss + lam, treeTools.makeLeaf(prediction))
        if depthLeft is not None and depthLeft <= 0:
            return leafAnswer
        bestFeature = -1
        bestLoss = None
        for f in range(data["m"]):
            leftRows = rows & data["notFeatureMask"][f]
            rightRows = rows & data["featureMask"][f]
            if leftRows == 0 or rightRows == 0:
                continue
            loss = bitData.nodeStats(leftRows, data)[2] + bitData.nodeStats(rightRows, data)[2]
            if bestLoss is None or loss < bestLoss:
                bestLoss = loss
                bestFeature = f
        if bestFeature < 0 or bestLoss + 2 * lam >= leafLoss + lam:
            return leafAnswer
        childDepth = None
        if depthLeft is not None:
            childDepth = depthLeft - 1
        leftAnswer = self.greedyTree(rows & data["notFeatureMask"][bestFeature], childDepth)
        rightAnswer = self.greedyTree(rows & data["featureMask"][bestFeature], childDepth)
        if leftAnswer[0] + rightAnswer[0] < leafAnswer[0]:
            return (leftAnswer[0] + rightAnswer[0], treeTools.makeSplit(bestFeature, leftAnswer[1], rightAnswer[1]))
        return leafAnswer

    def treeObjectiveOnRows(self, tree, rows):
        # objective of a finished tree on these rows (best label in every leaf)
        if tree.isLeaf:
            return bitData.nodeStats(rows, self.data)[2] + self.lam
        leftRows = rows & self.data["notFeatureMask"][tree.feature]
        rightRows = rows & self.data["featureMask"][tree.feature]
        return self.treeObjectiveOnRows(tree.left, leftRows) + self.treeObjectiveOnRows(tree.right, rightRows)

    def run(self):
        self.startTime = time.time()
        allFeatures = list(range(self.data["m"]))
        self.root = self.getProblem(self.data["allRows"], allFeatures, self.depthBudget)
        self.root.scope = float("inf")
        self.root.explored = True
        self.warmObjective = None
        if self.warmStart:
            # a real tree we already have gives the root a small upper bound from the start, so the
            # search can drop more splits, and if we must stop early we still have at least this tree
            self.warmObjective, self.warmTree = self.greedyTree(self.data["allRows"], self.depthBudget)
            if self.warmObjective < self.root.ub:
                self.root.ub = self.warmObjective
                if self.root.lb > self.root.ub:
                    self.root.lb = self.root.ub
        self.progress.append((0.0, self.root.lb, self.root.ub))
        self.push(1, self.root)

        while not self.root.isSolved():
            if len(self.queue) == 0:
                # should not happen, it would mean a bug
                self.stalled = True
                break
            if self.pops % 200 == 0:
                reached = bitData.limitReached(self.startTime, self.timeLimit)
                if reached is not None:
                    # out of time or out of memory: stop, the best tree so far and the gap are still valid
                    self.timedOut = True
                    self.memoryOut = (reached == "memory")
                    break
            entry = heapq.heappop(self.queue)
            p = entry[3]
            if entry[0] == 0:
                p.waitUpdate = False
            else:
                p.waitExplore = False
            self.pops = self.pops + 1
            self.process(p)

        self.progress.append((time.time() - self.startTime, self.root.lb, self.root.ub))
        result = {}
        result["tree"] = self.extractTree(self.root)
        if self.warmObjective is not None:
            # the graph does not contain the greedy tree, so if nothing better was found we give it back
            if self.warmObjective < self.treeObjectiveOnRows(result["tree"], self.data["allRows"]) - EPS:
                result["tree"] = self.warmTree
        result["objective"] = self.root.ub
        result["lowerBound"] = self.root.lb
        result["gap"] = self.root.ub - self.root.lb
        result["seconds"] = time.time() - self.startTime
        result["timedOut"] = self.timedOut
        result["memoryOut"] = self.memoryOut
        result["stalled"] = self.stalled
        result["subproblems"] = len(self.graph)
        result["pops"] = self.pops
        result["subsetPrunes"] = self.subsetPrunes
        result["progress"] = self.progress
        return result

    # ---------- read the tree out (Algorithm 6) ----------

    def extractTree(self, p):
        bestSplit = None
        bestValue = p.leafObjective
        for split in p.splits:
            value = split[1].ub + split[2].ub
            if value < bestValue - EPS:
                bestValue = value
                bestSplit = split
        if bestSplit is None:
            return treeTools.makeLeaf(p.prediction)
        leftTree = self.extractTree(bestSplit[1])
        rightTree = self.extractTree(bestSplit[2])
        return treeTools.makeSplit(bestSplit[0], leftTree, rightTree)


def gosdt(X, y, lam, timeLimit=None, posWeight=None, negWeight=None, featureGroups=None,
          useSubsetBound=True, queueOrder="small", depthBudget=None, referencePredictions=None, warmStart=False):
    search = GosdtSearch(X, y, lam, timeLimit, posWeight, negWeight, featureGroups, useSubsetBound, queueOrder,
                         depthBudget, referencePredictions, warmStart)
    result = search.run()
    # the graph has links both ways (parents and children), python only frees such loops in
    # its garbage collector, so we call it now. Otherwise old graphs stay in memory when we
    # run many lambdas one after another.
    search = None
    gc.collect()
    return result


if __name__ == "__main__":
    import dataTools
    import step4_bounds
    for name in ["monk1", "monk2", "car", "tictactoe"]:
        X, y, names, groups = dataTools.getWholeData(name)
        for lam in [0.05, 0.01]:
            a = step4_bounds.boundsDP(X, y, lam, timeLimit=300)
            b = gosdt(X, y, lam, timeLimit=300)
            check = treeTools.objectiveOf(b["tree"], X, y, lam)
            print(name, "lambda", lam, "| step4", a["objective"], round(a["seconds"], 2), "s",
                  "| gosdt", b["objective"], "tree", check, round(b["seconds"], 2), "s",
                  b["subproblems"], "problems", b["pops"], "pops", "stalled" if b["stalled"] else "")
