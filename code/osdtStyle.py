# osdtStyle.py
# A second kind of search, closer to OSDT (Hu et al. 2019) and to PyGOSDT in the paper.
#
# Here a tree is a LIST OF LEAVES (paper Sec 2: "we represent a tree as a set of leaves").
# A partial tree has fixed leaves (they stay leaves) and open leaves (not decided yet).
# We always decide the FIRST open leaf: either fix it, or split it on some feature.
# So every tree is made exactly once.
#
# Why do we need this? F1, AUC and partial AUC are not a sum over leaves, so the left and right
# dynamic programming of step 5 does not work for them. The paper does the same thing
# (Sec 4: "for optimizing non-additive loss functions, we use PyGOSDT").
# It also works for accuracy, so we use it as our OSDT style baseline too.
#
# Objectives (paper Sec 2, App C, D), P = positives, N = negatives:
#   acc  : (FP + FN) / (P + N)
#   bacc : (FN / P + FP / N) / 2
#   wacc : (FP + w FN) / (w P + N)                   leaf says 0 if w p <= n
#   f1   : (FP + FN) / (2P + FP - FN)                 leaf says 0 if w p <= n  (App D)
#   auc  : 1 - area under the ROC convex hull         (Eq. 2)
#   pauc : 1 - area under the ROC convex hull up to false positive rate theta
#
# Bounds:
#   hierarchical lower bound (Thm B.1, Thm 2.1, Thm 2.2): the fixed leaves give a part of the
#       loss that never goes away, the open leaves are counted in the best case
#   equivalent points (Thm B.9, Thm C.3): an open leaf can at best be cut into its
#       equivalence classes (rows with the same features), never into smaller pieces
#   leaf permutation bound (Thm B.8): the same set of leaves found again in another order is skipped
#   incremental progress (Thm B.6, B.7): if splitting a leaf can not gain more than lambda it is only fixed
#   best first search: we always take the partial tree with the smallest lower bound. When that
#       lower bound is not better than our best full tree, the best full tree is optimal.

import gc
import heapq
import time

import bitData
import treeTools

EPS = 1e-9
additiveObjectives = ["acc", "bacc", "wacc"]


def makeObjective(name, w=None, theta=None):
    return {"name": name, "w": w, "theta": theta}


def objectiveWeights(objective, P, N):
    # for the additive objectives the loss is a weighted count of wrong rows
    # returns (weight of one wrong positive, weight of one wrong negative)
    name = objective["name"]
    if name == "bacc" or name == "f1" or name == "auc" or name == "pauc":
        # for f1, auc and pauc these weights are only used to grow the first greedy tree.
        # Balanced weights give a real tree on imbalanced data (plain accuracy often gives one leaf)
        return 1.0 / (2 * P), 1.0 / (2 * N)
    if name == "wacc":
        w = objective["w"]
        return w / (w * P + N), 1.0 / (w * P + N)
    return 1.0 / (P + N), 1.0 / (P + N)


def leafLabel(p, n, objective, P, N):
    name = objective["name"]
    if name == "bacc":
        if p / P <= n / N:
            return 0
        return 1
    if name == "wacc" or name == "f1":
        if objective["w"] * p <= n:
            return 0
        return 1
    if p > n:
        return 1
    return 0


def lossFromErrors(FP, FN, objective, P, N):
    name = objective["name"]
    if name == "acc":
        return (FP + FN) / (P + N)
    if name == "bacc":
        return 0.5 * (FN / P + FP / N)
    if name == "wacc":
        w = objective["w"]
        return (FP + w * FN) / (w * P + N)
    if name == "f1":
        return (FP + FN) / (2 * P + FP - FN)
    raise ValueError("not an error based objective: " + name)


def positiveFraction(piece):
    return piece[0] / (piece[0] + piece[1])


def rankingLoss(pieces, objective, P, N):
    # pieces is a list of (positives, negatives). We sort them by fraction of positives,
    # highest first, and walk along the ROC curve (every piece is one line of the curve)
    ordered = sorted(pieces, key=positiveFraction, reverse=True)
    if objective["name"] == "auc":
        # Eq. 2 of the paper
        total = 0.0
        positivesAbove = 0
        for piece in ordered:
            p = piece[0]
            n = piece[1]
            total = total + n * (2 * positivesAbove + p)
            positivesAbove = positivesAbove + p
        return 1.0 - total / (2.0 * P * N)

    # partial AUC: only the part of the curve with false positive rate <= theta
    limit = objective["theta"] * N
    area = 0.0
    fp = 0.0
    tp = 0.0
    for piece in ordered:
        if fp >= limit:
            break
        p = piece[0]
        n = piece[1]
        if n == 0:
            tp = tp + p
            continue
        if fp + n <= limit:
            area = area + n * (2 * tp + p) / 2.0
            fp = fp + n
            tp = tp + p
        else:
            part = limit - fp
            tpEnd = tp + p * part / n
            area = area + part * (tp + tpEnd) / 2.0
            fp = limit
            break
    return 1.0 - area / (P * N)


class LeafSetSearch:
    def __init__(self, X, y, lam, objective, timeLimit=None):
        self.lam = lam
        self.objective = objective
        self.name = objective["name"]
        self.timeLimit = timeLimit
        self.P = int(y.sum())
        self.N = len(y) - self.P
        posWeight, negWeight = objectiveWeights(objective, self.P, self.N)
        # bit data with the objective weights (for additive objectives)
        self.data = bitData.makeBitData(X, y, posWeight, negWeight)
        # bit data with weight 1 per row (for counting and for F1)
        self.countData = bitData.makeBitData(X, y, 1.0, 1.0)
        self.m = X.shape[1]

        # equivalence classes, needed for the AUC bounds (Thm C.3)
        self.classes = []
        if self.name == "auc" or self.name == "pauc":
            groups = {}
            for i in range(X.shape[0]):
                key = X[i].tobytes()
                if key not in groups:
                    groups[key] = []
                groups[key].append(i)
            for key in groups:
                column = [0] * X.shape[0]
                pos = 0
                for i in groups[key]:
                    column[i] = 1
                    if y[i] == 1:
                        pos = pos + 1
                mask = bitData.columnToMask(column)
                self.classes.append((mask, pos, len(groups[key]) - pos))
        self.classCache = {}
        self.statesSeen = 0
        self.progress = []
        self.timedOut = False
        self.memoryOut = False

    # ---------- leaves ----------

    def makeLeafRecord(self, rows, path):
        p = (rows & self.countData["posMask"]).bit_count()
        n = rows.bit_count() - p
        return (rows, path, p, n)

    def classesInside(self, rows):
        # the (p, n) of every equivalence class inside these rows
        if rows in self.classCache:
            return self.classCache[rows]
        answer = []
        for c in self.classes:
            if c[0] & rows != 0:
                answer.append((c[1], c[2]))
        self.classCache[rows] = answer
        return answer

    def canSplit(self, leaf):
        p = leaf[2]
        n = leaf[3]
        if p == 0 or n == 0:
            # a pure leaf never gets better by splitting, it only pays more lambda
            return False
        if self.name in additiveObjectives:
            total, prediction, leafLoss, eqLoss = bitData.nodeStats(leaf[0], self.data)
            if leafLoss - eqLoss <= self.lam:
                # incremental progress bound, splitting can gain at most leafLoss - eqLoss
                return False
        return True

    # ---------- bounds ----------

    def lowerBound(self, fixed, openLeaves):
        # exact objective when there are no open leaves
        numLeaves = len(fixed) + len(openLeaves)
        P = self.P
        N = self.N
        if self.name in additiveObjectives:
            loss = 0.0
            for leaf in fixed:
                total, prediction, leafLoss, eqLoss = bitData.nodeStats(leaf[0], self.data)
                loss = loss + leafLoss
            for leaf in openLeaves:
                # an open leaf is later either fixed (leafLoss) or split (at least eqLoss plus
                # one more leaf, Eq. 9). So it costs at least the smaller of the two.
                total, prediction, leafLoss, eqLoss = bitData.nodeStats(leaf[0], self.data)
                if self.canSplit(leaf):
                    loss = loss + min(leafLoss, eqLoss + self.lam)
                else:
                    loss = loss + leafLoss
            return loss + self.lam * numLeaves

        if self.name == "f1":
            # fixed leaves: errors are known. open leaves: at least the minority of each
            # equivalence class is wrong. The smallest F1 loss with FP >= a, FN >= b and
            # FP + FN >= s is s / (2P + s - 2b) (put all extra errors on the FP side)
            a = 0
            b = 0
            for leaf in fixed:
                if leafLabel(leaf[2], leaf[3], self.objective, P, N) == 1:
                    a = a + leaf[3]
                else:
                    b = b + leaf[2]
            s = a + b
            for leaf in openLeaves:
                rows = leaf[0]
                s = s + (rows & self.countData["minorityPosMask"]).bit_count()
                s = s + (rows & self.countData["minorityNegMask"]).bit_count()
            return s / (2.0 * P + s - 2.0 * b) + self.lam * numLeaves

        # auc and pauc: fixed leaves stay as they are, open leaves become their equivalence classes
        pieces = []
        for leaf in fixed:
            pieces.append((leaf[2], leaf[3]))
        for leaf in openLeaves:
            for piece in self.classesInside(leaf[0]):
                pieces.append(piece)
        return rankingLoss(pieces, self.objective, P, N) + self.lam * numLeaves

    # ---------- a first tree, so we can prune from the start ----------

    def greedyLeaves(self, leaf):
        # simple greedy splitting like CART: split if the best feature lowers the weighted
        # error by more than lambda. For F1 and AUC we use balanced accuracy here, it is only
        # a starting tree, the search will find the real optimum.
        total, prediction, leafLoss, eqLoss = bitData.nodeStats(leaf[0], self.data)
        bestFeature = -1
        bestLoss = leafLoss - self.lam
        for f in range(self.m):
            leftRows = leaf[0] & self.data["notFeatureMask"][f]
            rightRows = leaf[0] & self.data["featureMask"][f]
            if leftRows == 0 or rightRows == 0:
                continue
            leftLoss = bitData.nodeStats(leftRows, self.data)[2]
            rightLoss = bitData.nodeStats(rightRows, self.data)[2]
            if leftLoss + rightLoss < bestLoss - EPS:
                bestLoss = leftLoss + rightLoss
                bestFeature = f
        if bestFeature < 0:
            return [leaf]
        leftLeaf = self.makeLeafRecord(leaf[0] & self.data["notFeatureMask"][bestFeature],
                                       leaf[1] + ((bestFeature, 0),))
        rightLeaf = self.makeLeafRecord(leaf[0] & self.data["featureMask"][bestFeature],
                                        leaf[1] + ((bestFeature, 1),))
        return self.greedyLeaves(leftLeaf) + self.greedyLeaves(rightLeaf)

    # ---------- search ----------

    def addChild(self, fixed, openLeaves):
        # leaf permutation bound: same leaves (in any order) means same future
        fixedKey = []
        for leaf in fixed:
            fixedKey.append(leaf[0])
        openKey = []
        for leaf in openLeaves:
            openKey.append(leaf[0])
        key = (frozenset(fixedKey), frozenset(openKey))
        if key in self.seen:
            return
        self.seen.add(key)
        self.statesSeen = self.statesSeen + 1

        bound = self.lowerBound(fixed, openLeaves)
        if len(openLeaves) == 0:
            if bound < self.bestObjective - EPS:
                self.bestObjective = bound
                self.bestLeaves = fixed
                self.progress.append((time.time() - self.startTime, self.bestObjective))
            return
        if bound < self.bestObjective - EPS:
            self.counter = self.counter + 1
            heapq.heappush(self.queue, (bound, self.counter, fixed, openLeaves))

    def run(self):
        self.startTime = time.time()
        allRows = self.countData["allRows"]
        root = self.makeLeafRecord(allRows, ())
        self.bestLeaves = (root,)
        self.bestObjective = self.lowerBound((root,), ())
        greedy = tuple(self.greedyLeaves(root))
        greedyObjective = self.lowerBound(greedy, ())
        if greedyObjective < self.bestObjective:
            self.bestLeaves = greedy
            self.bestObjective = greedyObjective
        self.progress.append((0.0, self.bestObjective))
        self.queue = []
        self.seen = set()
        self.counter = 0
        self.addChild((), (root,))

        finalLowerBound = self.bestObjective
        pops = 0
        while len(self.queue) > 0:
            pops = pops + 1
            if pops % 200 == 0:
                reached = bitData.limitReached(self.startTime, self.timeLimit)
                if reached is not None:
                    # out of time or out of memory (this search keeps all partial trees, so it
                    # can use a lot of memory)
                    self.timedOut = True
                    self.memoryOut = (reached == "memory")
                    break
            entry = heapq.heappop(self.queue)
            bound = entry[0]
            if bound >= self.bestObjective - EPS:
                # best first: nothing left can beat the best tree, so it is optimal
                self.queue = []
                break
            fixed = entry[2]
            openLeaves = entry[3]
            leaf = openLeaves[0]
            rest = openLeaves[1:]

            # choice 1: this leaf stays a leaf
            self.addChild(fixed + (leaf,), rest)
            # choice 2: split it on some feature
            if self.canSplit(leaf):
                for f in range(self.m):
                    leftRows = leaf[0] & self.countData["notFeatureMask"][f]
                    rightRows = leaf[0] & self.countData["featureMask"][f]
                    if leftRows == 0 or rightRows == 0:
                        continue
                    leftLeaf = self.makeLeafRecord(leftRows, leaf[1] + ((f, 0),))
                    rightLeaf = self.makeLeafRecord(rightRows, leaf[1] + ((f, 1),))
                    self.addChild(fixed, (leftLeaf, rightLeaf) + rest)

        if self.timedOut and len(self.queue) > 0:
            finalLowerBound = min(self.queue[0][0], self.bestObjective)
        else:
            # search finished, so the best tree is optimal and the gap is 0
            finalLowerBound = self.bestObjective
        self.progress.append((time.time() - self.startTime, self.bestObjective))

        result = {}
        result["tree"] = self.buildTree(list(self.bestLeaves), 0)
        result["objective"] = self.bestObjective
        result["lowerBound"] = finalLowerBound
        result["gap"] = self.bestObjective - finalLowerBound
        result["seconds"] = time.time() - self.startTime
        result["timedOut"] = self.timedOut
        result["memoryOut"] = self.memoryOut
        result["statesSeen"] = self.statesSeen
        result["progress"] = self.progress
        result["leafCounts"] = []
        for leaf in self.bestLeaves:
            result["leafCounts"].append((leaf[2], leaf[3]))
        return result

    def buildTree(self, leaves, depth):
        # leaves: records that all share the same path up to "depth"
        if len(leaves) == 1 and len(leaves[0][1]) == depth:
            leaf = leaves[0]
            p = leaf[2]
            n = leaf[3]
            node = treeTools.makeLeaf(leafLabel(p, n, self.objective, self.P, self.N), p / (p + n))
            node.counts = (p, n)
            return node
        feature = leaves[0][1][depth][0]
        leftLeaves = []
        rightLeaves = []
        for leaf in leaves:
            if leaf[1][depth][1] == 0:
                leftLeaves.append(leaf)
            else:
                rightLeaves.append(leaf)
        return treeTools.makeSplit(feature, self.buildTree(leftLeaves, depth + 1),
                                   self.buildTree(rightLeaves, depth + 1))


def leafSetSearch(X, y, lam, objective, timeLimit=None):
    search = LeafSetSearch(X, y, lam, objective, timeLimit)
    result = search.run()
    # free the big queue and seen set right away (see the same note in step5_gosdt.py)
    search = None
    gc.collect()
    return result


def treeObjective(tree, X, y, lam, objective):
    # objective of a finished tree on some data (used for checks and for test data)
    P = int(y.sum())
    N = len(y) - P
    allLeaves = []
    treeTools.collectLeaves(tree, allLeaves)
    counts = {}
    for leaf in allLeaves:
        counts[id(leaf)] = [0, 0]
    for i in range(X.shape[0]):
        leaf = treeTools.findLeaf(tree, X[i])
        if y[i] == 1:
            counts[id(leaf)][0] = counts[id(leaf)][0] + 1
        else:
            counts[id(leaf)][1] = counts[id(leaf)][1] + 1
    if objective["name"] == "auc" or objective["name"] == "pauc":
        pieces = []
        for leaf in allLeaves:
            c = counts[id(leaf)]
            if c[0] + c[1] > 0:
                pieces.append((c[0], c[1]))
        return rankingLoss(pieces, objective, P, N) + lam * len(allLeaves)
    FP = 0
    FN = 0
    for leaf in allLeaves:
        c = counts[id(leaf)]
        if leafLabel(c[0], c[1], objective, P, N) == 1:
            FP = FP + c[1]
        else:
            FN = FN + c[0]
    return lossFromErrors(FP, FN, objective, P, N) + lam * len(allLeaves)


if __name__ == "__main__":
    import dataTools
    import step5_gosdt
    X, y, names, groups = dataTools.getWholeData("monk2")
    for lam in [0.05, 0.02]:
        a = leafSetSearch(X, y, lam, makeObjective("acc"), timeLimit=120)
        b = step5_gosdt.gosdt(X, y, lam)
        print("monk2 lambda", lam, "| leaf set search", round(a["objective"], 6), round(a["seconds"], 2), "s",
              a["statesSeen"], "states | gosdt", round(b["objective"], 6), round(b["seconds"], 2), "s")
