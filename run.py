# run.py
# The one command for the whole project. Pick what to run with --stage.
#
#   python run.py --stage check        all steps give the same optimal answer?
#   python run.py --stage brute        step 1, brute force
#   python run.py --stage dp           step 2, dynamic programming
#   python run.py --stage sparsity     step 3, lambda objective
#   python run.py --stage bounds       step 4, bounds
#   python run.py --stage gosdt        step 5, full GOSDT
#   python run.py --stage objectives   step 6, other objectives (balanced acc, F1, AUC, ...)
#   python run.py --stage replicate    the experiments of the paper (pick one with --exp)
#   python run.py --stage features     our work: feature engineering and binarization methods
#   python run.py --stage all          everything above, one after another
#   python run.py --stage official     optional, compare with the authors package (pip install gosdt)
#
# Optional flags:
#   --quick           small and fast version of the stage (for a quick check)
#   --dataset NAME    only this dataset
#   --lam VALUE       only this lambda
#   --depth VALUE     only this depth (stages that use a depth)
#   --timelimit SEC   time limit for one run
#   --exp NAME        for --stage replicate: bucket, trees, sparsity, time, scale (default: all of them)
#   --jobs N          how many runs at the same time (replicate stage), default all cores but two
#   --memlimit MB     memory limit for one run (default 3000 MB). A run that needs more stops
#                     and is reported as "memory out", like a time out
#
# Every stage writes into its own folder in results/, so you can run different
# stages at the same time in different terminals.

import argparse
import os
import sys
import time

projectFolder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(projectFolder, "code"))
sys.path.insert(0, os.path.join(projectFolder, "experiments"))

stageList = ["check", "brute", "dp", "sparsity", "bounds", "gosdt", "objectives", "replicate",
             "features"]
# "official" is not part of "all" because it needs the extra package (pip install gosdt)
extraStages = ["official"]
laterStages = ["report"]
replicateParts = ["bucket", "trees", "sparsity", "time", "scale", "all"]


def runOneStage(stage, settings):
    startTime = time.time()
    if stage == "check":
        import expCheck
        expCheck.runStage(settings)
    elif stage == "brute":
        import expBrute
        expBrute.runStage(settings)
    elif stage == "dp":
        import expDp
        expDp.runStage(settings)
    elif stage == "sparsity":
        import expSparsity
        expSparsity.runStage(settings)
    elif stage == "bounds":
        import expBounds
        expBounds.runStage(settings)
    elif stage == "gosdt":
        import expGosdt
        expGosdt.runStage(settings)
    elif stage == "objectives":
        import expObjectives
        expObjectives.runStage(settings)
    elif stage == "replicate":
        import expReplicate
        expReplicate.runStage(settings)
    elif stage == "features":
        import expFeatures
        expFeatures.runStage(settings)
    elif stage == "official":
        import expOfficial
        expOfficial.runStage(settings)
    print("stage", stage, "finished in", round(time.time() - startTime, 1), "seconds")


def main():
    parser = argparse.ArgumentParser(description="GOSDT from scratch, ML lab project")
    parser.add_argument("--stage", required=True, help="which part to run: " + ", ".join(stageList + ["all"]))
    parser.add_argument("--quick", action="store_true", help="small fast version")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--timelimit", type=float, default=None)
    parser.add_argument("--exp", default=None, help="for replicate: " + ", ".join(replicateParts))
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--memlimit", type=float, default=3000)
    args = parser.parse_args()

    if args.exp is not None and args.exp not in replicateParts:
        print("unknown --exp", args.exp, "choose one of:", ", ".join(replicateParts))
        return

    import bitData
    bitData.memoryLimitMb = args.memlimit

    settings = {"quick": args.quick, "dataset": args.dataset, "lam": args.lam, "depth": args.depth,
                "timelimit": args.timelimit, "exp": args.exp, "jobs": args.jobs, "memlimit": args.memlimit}

    if args.stage == "all":
        for stage in stageList:
            runOneStage(stage, settings)
    elif args.stage in stageList or args.stage in extraStages:
        runOneStage(args.stage, settings)
    elif args.stage in laterStages:
        print("stage", args.stage, "is not ready yet, it comes in a later phase")
    else:
        print("unknown stage", args.stage)
        print("choose one of:", ", ".join(stageList + ["all"] + extraStages))


if __name__ == "__main__":
    main()
