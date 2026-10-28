# Where the datasets come from

Every csv has one row per sample and the label in the last column.

## data/paper (datasets used in the GOSDT paper, Lin et al. 2020, Table 2)

| file | original source | what we got | notes |
|---|---|---|---|
| monk1.csv, monk2.csv, monk3.csv | UCI MONK's Problems | binary version used in the GOSDT experiments (training set only, as in paper App I.6) | we renamed the columns to readable names. We checked the names with the known true rules: monk1 = `jacket=red OR head=body` and monk2 = "exactly two attributes have their first value" fit 100% of rows, monk3 fits 95.1% (it has 5% label noise on purpose) |
| tictactoe.csv | UCI Tic-Tac-Toe Endgame | raw x / o / b values | we one hot encode it in our own code (`x` and `o` per square, `b` left out, so 18 features like Table 2) |
| car.csv | UCI Car Evaluation | binary version (15 features) | |
| balance.csv | UCI Balance Scale | binary version (16 features) | |
| compas_binary.csv | ProPublica COMPAS | binary version (12 features, from Hu et al. 2019) | |
| fico_binary.csv | FICO Explainable ML Challenge | binary version (17 features, from Hu et al. 2019) | |
| bar7.csv | UCI in-vehicle coupon recommendation (Wang et al. 2017) | binary version (14 features) | |
| compas2016.csv | ProPublica COMPAS | 2 number columns + 4 binary columns | we make the thresholds ourselves |
| compas.csv | ProPublica COMPAS | 22 number columns | we make the thresholds ourselves |
| fico.csv | FICO | 1000 rows, 23 number columns | we make the thresholds ourselves |
| iris.csv | UCI Iris | 4 number columns, 3 classes | used as 3 one-vs-rest problems |
| chudi.csv | small 2D dataset (77 points) from the GOSDT authors | 2 number columns | this is the dataset of Theorem 3.1 / Figure 1 (72/77 = 93.5% vs 71/77 = 92.2%) |
| fourclass.csv | LIBSVM datasets page (Chang and Lin 2011), file `fourclass` | 862 rows, 2 number columns, label +1 / -1 | downloaded by us (raw file in `data/raw/fourclass.libsvm`), label +1 became 1 and -1 became 0. Like the paper figures we only use the thresholds 50, 100, 150 on each column (6 binary features) |

The binary versions are the ones from the GOSDT authors' benchmark repository (the paper links it: github.com/Jimmy-Lin/TreeBenchmark), copied from the `tree-benchmark` repository which keeps the same files.

## data/new (datasets that the GOSDT paper did not use, our robustness tests)

Made by `data/downloadNewData.py` (run once, the csv files are already here). Rows with missing values are removed. The label is the last column, `label`.

| file | source | rows | columns | label = 1 means |
|---|---|---|---|---|
| breast_cancer.csv | Breast Cancer Wisconsin (Diagnostic), inside scikit-learn | 569 | 30 numbers | malignant |
| pima.csv | Pima Indians Diabetes, OpenML 37 | 768 | 8 numbers | diabetes |
| banknote.csv | Banknote Authentication, OpenML 1462 | 1372 | 4 numbers | class 2 |
| heart.csv | Heart Disease (Statlog), OpenML 53 | 270 | 13 | heart disease |
| titanic.csv | Titanic, OpenML 40945, only pclass, sex, age, sibsp, parch, fare, embarked (266 rows without age or fare removed) | 1043 | 7 | survived |

About the number of binary features: for continuous columns we make one threshold between every two neighbour values, so a column with k different values gives k-1 features. Table 2 of the paper says its count is "approximately the total cardinality", so its numbers are a bit higher (for example fico: we get 1384, the paper says 1407 = 1384 + 23 columns).
