# GOSDT from scratch (ML Lab project)

We implement the paper **"Generalized and Scalable Optimal Sparse Decision Trees"** (Lin, Zhong, Hu, Rudin, Seltzer, ICML 2020) from scratch in  Python. We build it step by step: brute force, then dynamic programming, then the sparsity objective, then bounds, then the full GOSDT algorithm. Each step can be run and checked on its own.

On top of it we add two ideas of our own to make the search faster: depth guessing (a depth limit together with the lambda objective) and lower bound guessing (use the mistakes of a boosted model as the lower bound of a subproblem). We also try some feature engineering and different ways to turn numbers into 0/1 features, on the paper datasets and 5 new ones.


## 1. Setup (one time)

```
bash setup.sh
```

This makes a virtual environment in `.venv` and installs `requirements.txt`. After that, run everything with `.venv/bin/python run.py ...` (or activate the venv first with `source .venv/bin/activate` and then just use `python run.py ...`).

## 2. One command, pick the part with `--stage`

```
python run.py --stage check       # do all steps give the same optimal answer? (fast, run this first)
python run.py --stage brute       # step 1 brute force
python run.py --stage dp          # step 2 dynamic programming with memo
python run.py --stage sparsity    # step 3 objective = loss + lambda * leaves
python run.py --stage bounds      # step 4 bounds from the paper
python run.py --stage gosdt       # step 5 full GOSDT (priority queue, graph, subset bound)
python run.py --stage objectives  # step 6 balanced accuracy, weighted accuracy, F1, AUC, partial AUC
python run.py --stage replicate   # the experiments of the paper (all parts, or one with --exp)
python run.py --stage features    # our work: feature engineering, 7 ways to binarize numbers
python run.py --stage all         # all of the above, one after another
python run.py --stage official    # optional: compare with the authors package (needs: pip install gosdt)
```

The paper experiments can be run one part at a time:

```
python run.py --stage replicate --exp bucket     # Thm 3.1 / Fig 1   bucketization loses accuracy
python run.py --stage replicate --exp trees      # Fig 19, 20, 21    example trees
python run.py --stage replicate --exp sparsity   # Fig 4, 5, 12, 13  accuracy vs number of leaves
python run.py --stage replicate --exp time       # Fig 18            best objective over time
python run.py --stage replicate --exp scale      # Fig 6, 14 to 17   time vs number of features / samples
```

Optional flags for any stage:

| flag | meaning |
|---|---|
| `--quick` | small, fast version (good for a first check, about a minute for everything) |
| `--dataset monk1` | only one dataset |
| `--lam 0.01` | only one lambda |
| `--depth 3` | only one depth (stages that use a depth) |
| `--timelimit 60` | time limit in seconds for one run |
| `--exp sparsity` | which part of the replicate stage to run |
| `--jobs 4` | how many runs at the same time (replicate stage), default is all cores but two |
| `--memlimit 3000` | memory limit for one run in MB (default 3000). A run that needs more stops and is written as "memory out" cause we run on smaller machine. |

Keep `jobs x memlimit` below the memory of your machine. Example for a 16 GB laptop: `--jobs 3 --memlimit 3000`.

The replicate parts save every finished job in `results/07_replication/<part>/checkpoints/`. Delete that `checkpoints` folder if you want to run everything again from zero (for example after changing the code).

Example: `python run.py --stage gosdt --dataset tictactoe --lam 0.005`

How long things take (8 core laptop): every stage with `--quick` takes under a minute. The full runs of stages 1 to 6 take from a few seconds (check, objectives) to about 40 minutes (gosdt). 

## 3. Running stages at the same time

Every stage writes only into its own folder in `results/`, so you can open more terminals and run different stages at the same time. Or start all of them in the background with:

```
bash runAll.sh            # or: bash runAll.sh --quick
```

What each stage prints is saved in `results/console/<stage>.txt`.

## 4. Where the results go

| stage | folder |
|---|---|
| check | `results/00_checks/` |
| official | `results/00_official_check/` |
| brute | `results/01_brute_force/` |
| dp | `results/02_dynamic_prog/` |
| sparsity | `results/03_sparsity/` |
| bounds | `results/04_bounds/` |
| gosdt | `results/05_gosdt/` |
| objectives | `results/06_objectives/` |
| replicate | `results/07_replication/<part>/` (bucket, trees, sparsity, time, scale) |
| features | `results/13_feature_engineering/` |

The features stage saves its finished runs in `results/our_jobs/`. Its default time limit is 600 seconds per run.

Every folder has a `.csv` table with all numbers, `log.txt` (what ran, with the settings), and `.png` plots. Some folders also have example trees as `.txt` files.

## 5. The steps and the code

| step | file | idea | paper |
|---|---|---|---|
| 1 | `code/step1_bruteForce.py` | try every tree shape up to depth d | Thm H.1 (number of trees grows like M!) |
| 2 | `code/step2_dynamicProg.py` | split into left and right subproblems, remember each answer by its set of rows (support set) | Sec 4.1, App E |
| 3 | `code/step3_sparsity.py` | objective = mistakes/n + lambda * leaves, no depth limit | Eq. 1 |
| 4 | `code/step4_bounds.py` | lower and upper bounds, equivalent points, incremental progress, sub tree bound | Eq. 9, 10, App B, Alg 2 to 4 |
| 5 | `code/step5_gosdt.py` | priority queue, dependency graph, bound updates go to parents first, scope, subset bound, tree extraction | Alg 1, Sec 4.2 to 4.5, App G, Alg 6 |
| 6 | `code/step6_objectives.py` | other objectives: acc, bacc, wacc use step 5 with weights, f1, auc, pauc use the leaf set search | Sec 2, App B, C, D |

Our own work (Part C), switched on with options of step 5, so step 5 alone is still the plain paper method:

| idea | file | what it does |
|---|---|---|
| depth guessing | `code/guessDepth.py` + `depthBudget` in step 5 | add a depth limit to the lambda objective (loss + lambda * leaves with depth <= d), and a VC dimension argument for how deep is enough |
| lower bound guessing | `code/guessLowerBound.py` + `referencePredictions` in step 5 and `dl85Style.py` | use the mistakes of a boosted model (GBDT) on the rows of a subproblem as its lower bound, with a guarantee on how much worse the tree can get |

Other parts of our own work:

| file | what it does |
|---|---|
| `code/featureMaker.py` | text columns to 0/1, automatic equality features ("head==body"), medical cut points for Pima, mutual information selection |
| `code/guessLowerBound.py` (`combineReferences`) | more than one reference model: "intersection" (a row is a reference mistake only if all references are wrong, tighter guarantee) and "union" (faster, looser) |
| `code/binarize.py` (`quantile5`, `quantile10`) | quantile bins, for the binarization comparison |
| `code/step5_gosdt.py` (`warmStart`) | start GOSDT from a quick greedy tree (like OSDT does). Same optimal answer, and a better tree when the search has to stop early. Off by default, tested in the check stage |
| `experiments/ourJobs.py` | one run of Part C, used by the features stage |
| `data/downloadNewData.py` | how the new datasets were downloaded (only needed once) |

Baselines we wrote ourselves (for the comparisons in the paper experiments):

| file | what it is |
|---|---|
| `code/osdtStyle.py` | leaf set branch and bound like OSDT / PyGOSDT, also used by step 6 for F1 and AUC |
| `code/dl85Style.py` | depth limited dynamic programming with upper bounds, like DL8.5 |
| `code/otherModels.py` | scikit-learn models, only used as baselines: CART, random forest, gradient boosting, AdaBoost, logistic regression, kNN, SVM |

Helper files:

| file | what it does |
|---|---|
| `code/bitData.py` | stores each set of rows as one python integer (bit i = row i), so splitting is `rows & mask` and counting is `rows.bit_count()` |
| `code/dataTools.py` | reads the datasets, makes the 5 folds |
| `code/binarize.py` | turns numbers and text columns into 0/1 columns (all thresholds, or bucketization) |
| `code/treeTools.py` | tree class, predict, accuracy, objective, print a tree |
| `code/expTools.py`, `code/plotTools.py` | result folders, logs, tables, plots |
| `experiments/exp*.py` | one file per stage, `run.py` calls them |

Each step file can also be run alone for a tiny demo, for example `python code/step3_sparsity.py`.

## 6. Data

The datasets are in `data/paper/` (GOSDT paper) and `data/new/` (our new datasets). Where each one comes from is written in `data/sources.md`.

