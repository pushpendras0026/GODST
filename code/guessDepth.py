# guessDepth.py
# EXTENSION idea 2: depth guessing (McTavish et al. 2022, Sec 4.2).
#
# GOSDT (lambda per leaf) does not know how deep the best tree is, so it looks at all depths.
# DL8.5 style methods are fast because they have a depth limit, but they do not care about
# the number of leaves. Their idea: use BOTH (their Eq. 3)
#       minimize  loss + lambda * leaves   such that   depth <= d
# The depth limit itself is inside step5_gosdt.py (depthBudget). Here we only have
# Theorem 4.2, which says how deep a single tree must be to be as complex as a boosted model:
#
#   K weak trees, each from a class with VC dimension V, then trees of depth
#       d = ceil( log2( (K*V + K) * (3 * ln(K*V + K) + 2) ) )
#   have VC dimension at least the one of the ensemble.
#
# For V we use the number of leaves of one weak tree (2^depth of the weak tree). With this
# choice we get the same two numbers as in the paper: K=10 trees of depth 3 -> d = 11,
# K=100 trees of depth 3 -> d = 15.

import math


def depthToMatchEnsemble(numTrees, weakTreeDepth):
    vc = 2 ** weakTreeDepth
    inside = numTrees * vc + numTrees
    value = inside * (3 * math.log(inside) + 2)
    return int(math.ceil(math.log2(value)))


if __name__ == "__main__":
    print("paper examples: K=10 depth 3 ->", depthToMatchEnsemble(10, 3), " K=100 depth 3 ->", depthToMatchEnsemble(100, 3))
    for numTrees, weakDepth in [(20, 3), (40, 1), (50, 2), (30, 2), (100, 1)]:
        print("GBDT with", numTrees, "trees of depth", weakDepth, "-> depth", depthToMatchEnsemble(numTrees, weakDepth))
