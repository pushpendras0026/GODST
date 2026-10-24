# bitData.py
# In this file we turn the 0/1 data matrix into "bit numbers".
#
# A python integer can have as many bits as we want. We use bit number i to say
# "row i is inside this group". So a group of rows (the paper calls it a support set,
# Sec 4.1) is just one integer. Then:
#   rows where feature j is 1  ->  rows & featureMask[j]
#   rows where feature j is 0  ->  rows & notFeatureMask[j]
#   how many rows              ->  rows.bit_count()
# These run inside python's C code, so they are much faster than a for loop over rows.

import time

# biggest memory one search may use (in MB). run.py can change it with --memlimit.
# Without this, a big search can eat all the memory and Linux kills every python process.
memoryLimitMb = 3000


class TimeIsUp(Exception):
    # we throw this when a search goes over its time limit
    pass


class MemoryIsFull(TimeIsUp):
    # a full memory also means the search has to stop, so we handle it like a time out,
    # but we still want to know which of the two it was
    pass


def memoryUsedMb():
    # memory this python process uses right now, in RAM plus in swap (Linux: /proc/self/status).
    # We must count swap too: when memory is almost full, Linux moves part of the process
    # to swap, and then the RAM number alone looks small even though the process is big.
    used = 0.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:") or line.startswith("VmSwap:"):
                    used = used + int(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return used


def limitReached(startTime, timeLimit):
    # returns "time", "memory" or None
    if timeLimit is not None and time.time() - startTime > timeLimit:
        return "time"
    if memoryLimitMb is not None and memoryUsedMb() > memoryLimitMb:
        return "memory"
    return None


def columnToMask(column):
    # column has one 0/1 value per row
    # we build a text like "0110" where the LAST char is row 0, then read it in base 2
    chars = []
    for value in column:
        if value == 1:
            chars.append("1")
        else:
            chars.append("0")
    if len(chars) == 0:
        return 0
    chars.reverse()
    return int("".join(chars), 2)


def makeBitData(X, y, posWeight=None, negWeight=None):
    # X is a numpy 0/1 matrix (rows x features), y is a numpy 0/1 vector
    # posWeight is the cost of getting one positive row wrong, negWeight same for negative rows
    # for normal accuracy both are 1/n (paper uses loss as a fraction of all rows)
    n = X.shape[0]
    m = X.shape[1]
    data = {}
    data["n"] = n
    data["m"] = m
    data["allRows"] = (1 << n) - 1

    featureMask = []
    notFeatureMask = []
    for j in range(m):
        mask = columnToMask(X[:, j])
        featureMask.append(mask)
        notFeatureMask.append(data["allRows"] ^ mask)
    data["featureMask"] = featureMask
    data["notFeatureMask"] = notFeatureMask

    data["posMask"] = columnToMask(y)
    data["negMask"] = data["allRows"] ^ data["posMask"]

    if posWeight is None:
        posWeight = 1.0 / n
    if negWeight is None:
        negWeight = 1.0 / n
    data["posWeight"] = posWeight
    data["negWeight"] = negWeight

    # equivalent points (paper Thm B.9 and Sec 4.1)
    # rows with exactly the same features always fall in the same leaf,
    # so if they have different labels, some of them must be wrong in every tree.
    # for each group we mark the cheaper label as the "minority" (q_u in the paper)
    groups = {}
    for i in range(n):
        key = X[i].tobytes()
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    minorityPos = [0] * n
    minorityNeg = [0] * n
    for key in groups:
        rowList = groups[key]
        pos = 0
        for i in rowList:
            if y[i] == 1:
                pos = pos + 1
        neg = len(rowList) - pos
        if posWeight * pos <= negWeight * neg:
            # cheaper to get the positives wrong
            for i in rowList:
                if y[i] == 1:
                    minorityPos[i] = 1
        else:
            for i in rowList:
                if y[i] == 0:
                    minorityNeg[i] = 1
    data["minorityPosMask"] = columnToMask(minorityPos)
    data["minorityNegMask"] = columnToMask(minorityNeg)
    data["numGroups"] = len(groups)
    return data


def nodeStats(rows, data):
    # numbers we need for one node (one support set)
    # leafLoss = loss if this node is a leaf with the best label
    # eqLoss   = loss that NO tree can avoid on these rows (equivalent points bound)
    total = rows.bit_count()
    pos = (rows & data["posMask"]).bit_count()
    neg = total - pos
    costIfPredictZero = data["posWeight"] * pos
    costIfPredictOne = data["negWeight"] * neg
    if costIfPredictOne < costIfPredictZero:
        prediction = 1
        leafLoss = costIfPredictOne
    else:
        prediction = 0
        leafLoss = costIfPredictZero

    eqLoss = data["posWeight"] * (rows & data["minorityPosMask"]).bit_count()
    eqLoss = eqLoss + data["negWeight"] * (rows & data["minorityNegMask"]).bit_count()
    if eqLoss > leafLoss:
        # can happen by a tiny float error only
        eqLoss = leafLoss
    return total, prediction, leafLoss, eqLoss


def checkTime(startTime, timeLimit):
    # throw TimeIsUp if we are over the time limit (timeLimit None means no limit),
    # or MemoryIsFull if we use too much memory
    reached = limitReached(startTime, timeLimit)
    if reached == "time":
        raise TimeIsUp()
    if reached == "memory":
        raise MemoryIsFull()
