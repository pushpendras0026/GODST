#!/bin/bash
# runAll.sh
# Runs everything. First the step stages, all at the same time (each one is a single process),
# then the paper experiments one after another (each one uses JOBS processes).
# What each stage prints goes to results/console/<stage>.txt
#
# use:   bash runAll.sh            normal run (takes some hours)
#        bash runAll.sh --quick    fast version (a few minutes)
#
# You can also just open more terminals and run  python run.py --stage <name>  in each one.
#
# If you want to close the terminal while it runs, start it like this,
# then the runs do not stop when the terminal closes:
#        nohup bash runAll.sh > results/console/runAll.txt 2>&1 &

# change these two to fit your machine: JOBS x MEMLIMIT (MB) should stay below your free memory
JOBS=3
MEMLIMIT=2000

cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p results/console

for stage in check brute dp sparsity bounds gosdt objectives; do
    echo "starting $stage"
    $PY run.py --stage $stage --memlimit $MEMLIMIT "$@" > results/console/$stage.txt 2>&1 &
done
wait

for part in bucket trees sparsity time scale; do
    echo "starting replicate $part"
    $PY run.py --stage replicate --exp $part --jobs $JOBS --memlimit $MEMLIMIT "$@" > results/console/replicate_$part.txt 2>&1
done

# our own stage
for stage in features; do
    echo "starting $stage"
    $PY run.py --stage $stage --jobs $JOBS --memlimit $MEMLIMIT "$@" > results/console/$stage.txt 2>&1
done

echo "all stages finished, see the results folder"
