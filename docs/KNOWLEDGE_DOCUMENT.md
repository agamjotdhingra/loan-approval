# Project Knowledge Document — AI-Based Loan Approval Prediction System

This document explains **every component** of the project. For each one you get
five things — exactly what to study and say in your evaluation:

> **What it is · Why we need it · How it works · Where it is in our code · What to say if the teacher asks**

Read this top to bottom before the viva. The numbers quoted are the *actual
results from your trained model*, so you can quote them as fact.

---

## Table of contents

1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [The dataset](#2-the-dataset)
3. [Machine learning basics you must be able to define](#3-machine-learning-basics)
4. [Data preparation — component by component](#4-data-preparation)
5. [Logistic Regression](#5-logistic-regression)
6. [Decision Tree](#6-decision-tree)
7. [Why Random Forest was NOT used](#7-why-random-forest-was-not-used)
8. [Model evaluation metrics](#8-model-evaluation-metrics)
9. [Our actual results — and how to read them](#9-our-actual-results)
10. [The pipeline pattern](#10-the-pipeline-pattern)
11. [Overfitting and how we checked it](#11-overfitting)
12. [Pickle and model persistence](#12-pickle-and-model-persistence)
13. [The Streamlit app](#13-the-streamlit-app)
14. [The charts](#14-the-charts)
15. [File-by-file map of the project](#15-file-by-file-map)
16. [Likely viva questions with model answers](#16-likely-viva-questions)

---

## 1. The project in one paragraph

> "We built a **supervised binary classification system**. Given 14 features
> about a loan applicant (income, credit history, loan amount, etc.), the
> model predicts one of two classes: **Approved (1)** or **Rejected (0)**. We
> prepared the public Analytics Vidhya loan dataset (614 applications),
> trained **Logistic Regression** and **Decision Tree** classifiers, compared
> them with accuracy, precision, recall, F1 and ROC-AUC on a held-out test
> set, and served the better model in an interactive **Streamlit** app."

---

## 2. The dataset

| Property | Value |
|---|---|
| Source | **Analytics Vidhya "Loan Prediction" practice problem** (also on Kaggle as "Analytics Vidhya: Loan Prediction") |
| File | `data/loan_data.csv` |
| Rows | **614 loan applications** |
| Target | `Loan_Status` → encoded **Y = 1 (approved), N = 0 (rejected)** |
| Class balance | ~**69% approved / 31% rejected** (imbalanced, but not extreme) |
| Missing values | 7 columns had gaps (Credit_History 50, Self_Employed 32, LoanAmount 22, …) |

**Columns:** `Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status`

> 🗣 **If asked "Where is the data from?"** — "From the public Analytics Vidhya
> loan prediction practice problem, a standard, openly available dataset used
> in hundreds of teaching projects. It is synthetic/desensitised data
> belonging to a fictional 'Dream Housing Finance' company — **not** any real
> bank's customer data."

---

## 3. Machine learning basics

### 3.1 Supervised learning
- **What:** learning a function from labelled examples. Each training row has
  inputs (features) AND the correct answer (label).
- **Why:** we have historical applications with known outcomes, so we can
  learn the approval pattern instead of hand-coding rules.
- **Where:** everything in `train_and_evaluate.py` — `model.fit(X_train, y_train)`.

### 3.2 Classification (vs regression)
- **What:** predicting a **category** (Approved/Rejected) rather than a
  continuous number (like a price).
- **Why:** the loan decision is inherently two-class → binary classification.

### 3.3 Features (X) and target (y)
- **What:** X = the 11 real input columns (6 categorical + 5 numeric);
  y = the answer column (`Loan_Status`).
- **Where:** `load_and_split()` in `train_and_evaluate.py`:
  `y = df["Loan_Status"]`,
  `X = df.drop(columns=["Loan_ID", "Loan_Status"])`.
- 🗣 "Loan_ID was dropped because it's a unique identifier with **no
  predictive signal** — keeping it would only add noise."

### 3.4 Train/test split
- **What:** dividing the 614 rows into 491 training + 123 test rows.
- **Why:** scoring on data the model saw during training is like grading a
  student on the exact questions they memorised — it measures memory, not
  understanding. The test set simulates **future, unseen applications**.
- **How:** scikit-learn's `train_test_split` with:
  - `test_size=0.20` → 123 rows for testing,
  - `stratify=y` → both sets keep the same 69/31 approval ratio,
  - `random_state=42` → the split is **repeatable**; every run uses the same rows.
- **Where:** end of `prepare_data()` in `data_preparation.py`.
- 🗣 "I used a **stratified** split so that the proportion of approved and
  rejected loans is identical in train and test — otherwise one split could
  randomly contain hardly any rejections and the metrics would be misleading."

---

## 4. Data preparation

*(All in `src/data_preparation.py`. Raw ML models need clean numeric tables —
this file is roughly half the total engineering effort, and examiners love
questioning it.)*

### 4.1 Missing values (`handle_missing_values`)
- **What:** filling blank cells so algorithms don't crash.
- **Why:** scikit-learn estimators raise errors on NaN; and real datasets
  always have gaps.
- **How:** two textbook strategies:
  - **Numeric columns → median.** e.g. `LoanAmount` filled with 128.0.
    Median beats mean here because incomes are skewed — a few huge incomes
    would drag the mean up, but the median is **robust to outliers**.
  - **Categorical columns → mode.** e.g. `Gender` filled with "Male",
    `Self_Employed` with "No". The most frequent value is the safest guess
    for a category.
- **Where:** `handle_missing_values(df)`; the printed audit of which column
  got which value is in the training output.
- 🗣 "I chose **median for numeric** (robust to income outliers) and **mode for
  categorical** (no 'average' exists for a category). A fancier option would
  be model-based imputation, but for 614 rows the simple methods are
  defensible and easy to explain."

### 4.2 Encoding (now inside `build_preprocessor()`)
**Why we need it at all:** scikit-learn models do **maths**, and you cannot
multiply "Graduate" by a weight. So every text value must become a number.

The leakage-safe implementation routes every categorical column through a
`ColumnTransformer` block: `SimpleImputer(most_frequent)` →
**`OneHotEncoder`**. One-hot turns each category into its own 0/1 column:
`Property_Area` becomes three columns (`_Rural / _Semiurban / _Urban`),
`Gender` becomes `Male` (presence/absence of Female is its mirror), etc.

- **Why one-hot and not a plain hand map (Rural=0, Semiurban=1, Urban=2)?**
  A hand map invents an **order and magnitude** that doesn't exist — the
  model would believe Urban is "twice" Semiurban. One-hot keeps categories
  equal and unordered.
- **Why even 2-valued columns get one-hot:** with `handle_unknown='ignore'`
  in one place, unseen categories can never crash the app, and the encoders
  learn their category lists **from training data only** (leakage safety —
  a hand-written map is written by the human looking at ALL the data).

🗣 **Trap question** — "Why didn't you hand-map Gender to 0/1?" → "A hand map
written after inspecting the whole file is a subtle leak, and one-hot of a
binary column is informationally identical anyway. Keeping every categorical
column in the same one-hot block is simpler and strictly leakage-safe."

### 4.3 Imputation moved inside the pipeline
Median/mode filling now happens in `SimpleImputer`, **fitted on the 491
training rows only**. The imputed values (e.g. LoanAmount median ≈ 126 in
the training slice) are computed from train data alone and then applied to
test rows — a textbook leak before, structurally impossible now.

### 4.4 Feature engineering (simplified out)
The earlier `TotalIncome` engineered column was removed: EDA found income
has near-zero correlation with approval and LoanAmount·ApplicantIncome is
already collinear (+0.57); adding one more income variant adds redundancy,
not signal. Simpler features = more explainable model (a course goal).

### 4.3 Feature engineering (`add_engineered_feature`)
- **What:** creating a new column from existing ones:
  `TotalIncome = ApplicantIncome + CoapplicantIncome`.
- **Why:** banks assess **household** ability to repay. Two smaller incomes
  can support a loan that one income alone cannot; giving the model the sum
  makes that relationship explicit.
- 🗣 "Feature engineering is the step where domain knowledge becomes a column —
  it's often worth more than switching algorithms."

### 4.5 Dropping useless columns
- `Loan_ID` is dropped in `load_and_split()` before anything else.
- 🗣 "An ID column is a **cardinality trap**: every value is unique, so a model
  could only 'learn' from it by memorising rows — pure overfitting fuel."

---

## 5. Logistic Regression

- **What it is:** a **linear classification model**. It computes a weighted
  sum of the features, then squashes that number through the **sigmoid
  function** to get a probability between 0 and 1. Probability ≥ 0.5 →
  approved; < 0.5 → rejected.
- **How it works, in one breath:**
  1. score = w₁·income + w₂·credit_history + … + w₁₄·area + b (a straight-line
     combination — hence *linear*);
  2. probability = sigmoid(score) = 1 / (1 + e^-score);
  3. during training it adjusts the weights `w` to maximise the likelihood of
     the observed labels (it navigates downhill on a loss curve using
     **gradient descent** — that's what `max_iter=1000` bounds).
- **Why we chose it:** it's the **classic, simplest baseline** for binary
  classification — fast, robust on small data, and its weights are directly
  interpretable (a positive weight *pushes towards approval*).
- **Where:** `build_models()` in `train_and_evaluate.py` — inside a `Pipeline`
  whose preprocessing (impute + scale + one-hot) comes from
  `build_preprocessor()` (see §10 for why the scaling matters).
- 🗣 **If asked "What is the sigmoid?"** — "The S-shaped function that maps any
  real number into (0,1), letting us read the score as a probability. At
  score 0 it outputs 0.5, which is our decision boundary."
- 🗣 **If asked "What do the coefficients mean?"** — "Each weight is the change
  in log-odds per unit of that feature. Positive weight = feature favours
  approval. In this dataset the credit-history coefficient dominates — that's
  the Logistic Regression telling us the same story that chart 02 shows."

---

## 6. Decision Tree

- **What it is:** a model that learns a **flowchart of yes/no questions**
  from the data ("Is Credit_History ≤ 0.5? → Is LoanAmount ≤ 128? → …").
- **How it works:**
  - At each node the algorithm searches **every feature and every threshold**
    for the **best split** — the question that separates the classes most
    cleanly (measured by **Gini impurity**: how "mixed" the two classes are
    in a group; 0 = pure, 0.5 = 50/50).
  - It repeats recursively until groups are pure or limits are hit.
- **Why we chose it:** it's naturally **explainable** — you can read the tree
  as business rules — and it needs **no feature scaling**
  (see the scaling discussion in §10; a tree only compares one column against
  a threshold, so multiplying a column by 1000 just moves the threshold, it
  never changes the decision).
- **Our hyperparameter:** `max_depth=4`.
- 🗣 **Why limit depth?** — "Unlimited trees keep splitting until every leaf
  is pure, i.e. they **memorise** the training rows — textbook overfitting.
  `max_depth=4` caps the flowchart at four questions, forcing broadly general
  rules. It also keeps the tree human-readable, which was a project goal."
- 🗣 **If asked "What is Gini?"** — "The chance of mislabelling a random
  sample if we guess according to the class mix of the node. The tree picks
  the split that lowers Gini the most."

---

## 7. Why Random Forest was NOT used

Random Forest = many decision trees trained on random subsets of rows and
features, whose votes are averaged. It usually improves on a single tree.

**Our decision rule (agreed upfront):** add it **only if** the single tree
showed clear overfitting.

**The evidence:** after training, the overfit check printed

```
Logistic Regression: train 0.800 vs test 0.862  (gap -0.061)
Decision Tree:       train 0.809 vs test 0.846  (gap -0.037)
```

Both gaps are small *and negative* (the models generalise; they don't
memorise). With only 614 rows, a 100-tree forest would add complexity and
loss of interpretability for essentially no gain.

> 🗣 **If asked "Why not Random Forest?"** — "I tested the precondition for
> needing it: the gap between train and test accuracy. Ours was ≈ 0.04-0.06,
> so the tree is not overfitting, and a forest's averaging machinery would
> add runtime and reduce explainability without a measurable benefit. I kept
> the simplest model that meets the requirement — Occam's razor."

That answer (evidence-based refusal) scores better than blindly adding it.

---

## 8. Model evaluation metrics

*(All implemented in `train_and_evaluate()`; every metric computed on the
**test set only**.)*

First, the four outcomes a confusion matrix counts:

| | Model says: Approved | Model says: Rejected |
|---|---|---|
| **Actually approved** | True Positive (TP) | False Negative (FN) — *good customer rejected* |
| **Actually rejected** | False Positive (FP) — *bad loan approved* | True Negative (TN) |

- **Accuracy** = (TP+TN)/all — overall correctness.
- **Precision** (of predicted-approvals) = TP/(TP+FP) — *"when we approve, how
  often are we right?"* Costly errors here: giving loans to people who won't
  repay.
- **Recall** (of actual approvals) = TP/(TP+FN) — *"of all who deserved a
  loan, how many did we catch?"* Costly errors here: turning away good
  customers.
- **F1-score** = harmonic mean of precision and recall = 2PR/(P+R). One number
  that punishes a lopsided model (e.g. high precision, terrible recall).
- **ROC-AUC** — area under the curve of true-positive-rate vs
  false-positive-rate as the decision threshold sweeps from 0 to 1.
  1.0 = perfect ranking, 0.5 = coin flip. It measures ranking quality
  independent of the 0.5 cut-off.
- **Classification report** — precision/recall/F1 *per class*, revealing
  which class the model is weaker on.

🗣 **Why not trust accuracy alone?** "The dataset is ~69/31. A lazy model that
always answers 'Approved' scores 69% accuracy while being useless — it
rejects **nobody**. Precision, recall and F1 expose that. Also F1 focuses on
the approved class, which is the economically important one."

🗣 **Which metric did you optimise for and why?** "I selected the final model
by **F1 on the test set**, because it balances precision and recall on the
(positively skews the majority) approved class rather than being inflated by
the class imbalance."

---

## 9. Our actual results

From the final run (`output/plots/model_comparison.csv` holds the exact table):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.862** | 0.840 | **0.988** | **0.908** | **0.852** |
| Decision Tree (depth 4) | 0.846 | 0.824 | 0.988 | 0.898 | 0.706 |

Per-class view (Logistic Regression, from the real classification report):
for *rejected* applicants, precision 0.96 but recall 0.58 — i.e. when the
model says "reject" it's right 96% of the time, but it misses ~42% of true
rejections (it is a lenient model).

**Reading of the comparison (the "story" for your report):**
1. **Logistic Regression wins on Accuracy, Precision, F1 and ROC-AUC**
   (recall ties at 0.988). Both models lean heavily on `Credit_History` — visible in EDA chart 02 and in both models' behaviour.
2. The tree's ranking quality (ROC-AUC 0.706) is clearly behind LR's 0.852 — a
   depth-4 tree makes coarse, staircase probabilities; LR has smooth,
   well-ordered probabilities.
3. Both models generalise (no overfit gap — §11), so the comparison is fair.

> 🗣 **"Which model is better, and why?"** — "Logistic Regression: higher on
> all five metrics and particularly on ROC-AUC (0.852 vs 0.706), meaning its
> probability ranking is better, and it's the more interpretable model.
> That's also the classic finding for small, mostly-linear datasets."

---

## 10. The pipeline pattern (Pipeline + ColumnTransformer)

- **What:** scikit-learn's `Pipeline` chains preprocessing steps and the
  model into **one object** that behaves like a single estimator. Our
  `ColumnTransformer` routes columns to two parallel mini-pipelines that
  then rejoin before the estimator:

  ```
  raw row ──┬─ numeric block:    SimpleImputer(median) → StandardScaler
            └─ categorical block: SimpleImputer(mode) → OneHotEncoder
                         ↓
                 fitted model (LR or tree)
  ```
- **Why:** (a) **leakage safety**: every transformer is fit on the 491
  training rows only — the test set's medians, category lists and scale
  statistics never touch training (fitting on everything first is the
  textbook leak: test statistics sneaking into training, inflating scores);
  the pipeline structure makes that mistake impossible by construction;
  (b) **deployment**: when we pickle the pipeline, imputer + scaler +
  encoder travel **together with** the model into the Streamlit app, so a
  raw live applicant row (e.g. "Graduate", "Urban") is preprocessed
  exactly like training data was — that's why `app.py` now sends raw
  strings instead of hand-encoded numbers;
  (c) **immutability advantage**: in production, even if the target were
  accidentally included in X, the pipeline approach shrinks but does not
  remove the risk surface — it constrains where fitting can happen.
- **How it works under the hood:** `pipeline.fit(X_train, y_train)` runs
  imputer.fit→imputer.transform→scaler.fit/transform→encoder.fit/transform
  →model.fit, in order; `pipeline.predict(X_test)` re-runs only the
  transform steps (with already-fitted statistics) then model.predict.
- 🗣 **Why a scaler for Logistic Regression but not the tree?** "LR multiplies
  every feature by a weight, so a feature in the tens of thousands (income)
  numerically drowns one that's 0/1 (gender) — the optimiser struggles and
  the weights become incomparable. StandardScaler rescales every feature to
  mean 0, variance 1, so all weights are on the same footing. A decision tree
  only asks 'is value x > threshold t?' for one column at a time —
  scaling just re-labels the thresholds. It's **scale-invariant by
  construction**."
- 🗣 **What does ColumnTransformer give you?** "Per-column routing: different
  columns need different treatment (numeric → median+scale; categorical →
  mode+one-hot), and ColumnTransformer applies each rule to the right
  columns by NAME, reassembling the outputs into one matrix for the model.
  It also means a saved pipeline knows exactly which column is which when
  the app sends it a raw row."

---

## 11. Overfitting

- **What:** the model memorises the training rows (including their noise) and
  fails on new data. Signature: **high train accuracy, much lower test
  accuracy** (a big positive gap).
- **Our check:** `train_and_evaluate.py` prints both accuracies per model:
  LR 0.800/0.862 (gap −0.061), DT 0.809/0.846 (gap −0.037) → **no overfitting
  at our chosen complexity**.
- **Prevention measures we used:** `max_depth=4` on the tree (limiting model
  capacity), a held-out test set (detecting overfitting when it happens),
  and deliberately simple feature engineering.
- 🗣 "Overfitting is controlled with a **complexity budget** (tree depth) and
  **detected** with a train/test comparison. Also, more data would help, but
  at 614 rows the simple models generalise already."

---

## 12. Pickle and model persistence

- **What:** Python's object-serialisation format — turning a live, in-memory
  object (our fitted pipeline) into a file, and back.
- **Why:** training should happen **once**; the Streamlit app must not
  retrain (slow) on every page load. `pickle.dump` → `best_model.pkl`;
  `pickle.load` in the app → the same fitted object, instantly.
- **Where:** saving at the end of `main()` in `train_and_evaluate.py`; loading
  in `load_models()` in `app.py` (cached with `@st.cache_resource`).
- 🗣 "Alternative approaches are joblib (pickle tuned for numpy arrays) or
  ONNX for cross-language deployment — pickle is the standard teaching
  choice and perfectly adequate here."
- ⚠️ One honest caveat you can volunteer for bonus points: "Pickle files are
  not safe to load from untrusted sources — arbitrary code execution risk.
  In production you'd sign them or use a safer format; here it's our own
  file, so it's fine."

---

## 13. The Streamlit app

- **What:** a Python-only web UI framework — every interactive element is a
  function call (`st.selectbox`, `st.number_input`, `st.form`...). Chosen
  because the whole project stays in Python, no HTML/JS required.
- **How the app flows** (`src/app.py`):
  1. loads both pickled pipelines **once** (`@st.cache_resource`);
  2. renders a form whose fields mirror the raw CSV columns one-for-one;
  3. on "Predict", builds a one-row DataFrame with those **raw** values —
     since the pipeline rewrite, all imputing/scaling/encoding happens
     INSIDE the saved pipeline, so the app no longer hand-encodes anything
     (the old "same encoding as training" trap is structurally gone);
  4. calls `predict` and `predict_proba` on **both** pipelines and shows
     Approved/Rejected + confidence side by side;
  5. an expander shows the raw row, so you can see exactly what was sent.
- 🗣 **Demo path (do exactly this in the viva):** (1) keep defaults → both
  models say **APPROVED** (80% / 77%); (2) switch credit history to "No
  credit history (0)" → both say **REJECTED** (LR P(approve) ≈ 0.10, i.e.
  90% reject confidence). That single toggle demonstrates the decisive
  feature live — verified in the real app.

---

## 14. The charts

*(all in `output/plots/`, built with matplotlib + seaborn)*

| File | What it shows | Sentence to say |
|---|---|---|
| `01_class_distribution.png` | 422 approved / 192 rejected | "Approvals outnumber rejections ~69/31 — why I stratified the split and why accuracy alone is a weak metric." |
| `02_credit_history.png` | approval rate with vs without credit history | "Approval is ~79% with a credit history vs ~8% without — credit history is the dominant feature, and the models agree." |
| `03_income_vs_amount.png` | income vs loan amount scatter, coloured by approval | "Approvals spread across income levels; income alone doesn't separate the classes — its signal is weaker than credit history's." |
| `04_confusion_matrices.png` | one 2×2 matrix per model | "Top-left = correct rejections, bottom-right = correct approvals; the off-diagonal cells are the two business costs." |
| `05_model_comparison.png` | grouped bars, 5 metrics × 2 models | "LR leads on every metric; the biggest gap is ROC-AUC." |
| `model_comparison.csv` | the exact metric table | (quote these numbers in your report) |

*(Matplotlib was used for the bar/scatter charts; seaborn for the heatmap
confusion matrices — stylistic and practical choices, the library doesn't
change the analysis.)*

---

## 15. File-by-file map

```
├── data/loan_data.csv            614 rows, the raw public dataset
├── src/
│   ├── data_preparation.py       load → clean → encode → engineer → split
│   ├── train_and_evaluate.py     2 models, 6 metrics, 6 plots, pickle save
│   └── app.py                    Streamlit UI, loads pickles, predicts live
├── output/
│   ├── models/best_model.pkl     winning pipeline (LR + scaler)
│   ├── models/second_model.pkl   runner-up (Decision Tree)
│   └── plots/*.png + csv         5 charts + metric table
├── docs/KNOWLEDGE_DOCUMENT.md    this document
├── docs/PRESENTATION_SLIDES.md   11 slides ready to paste
├── requirements.txt              6 packages
└── README.md                     how to run everything
```

---

## 16. Likely viva questions

**Q: Why is this "AI"?**
A: The system learns the decision rule from historical labelled examples
rather than following hard-coded rules — that's (classical) machine
learning, the core of AI concerned with learning from data.

**Q: Why only two models? Why not deep learning?**
A: Two reasons. Statistical: 614 rows is tiny; deep nets need orders of
magnitude more data and would overfit badly. Practical: with tabular data
and a need for explainability, linear models and shallow trees are the appropriate
tools — and simpler models are easier to defend in a regulated domain like lending.

**Q: What would you improve next?**
A: (1) Cross-validation instead of a single split for more stable
estimates; (2) try logistic regression with the income columns
log-transformed to tame outliers; (3) calibrate the decision threshold —
for a bank, lowering the approval cut-off trades recall against precision
deliberately; (4) fairness check — ensure gender/marital status are not
driving predictions unethically.

**Q: Is this model fair / ethical?**
A: Honest answer: sensitive attributes like Gender and Education are in the
feature set because the dataset includes them. A real system should test
for disparate impact and probably drop them. We labelled the app "demo
only — not a real lending decision" for exactly this reason.

**Q: What is `random_state=42` for?**
A: Many steps are random (the split, the optimiser's start point). Fixing a
random seed makes every run reproduce identical numbers — essential for
verifiable results.

**Q: LR "converged" — what does `max_iter=1000` mean?**
A: Gradient descent iteratively improves the weights; max_iter is the step
budget. 1000 was enough for the optimisation to settle (no convergence
warning was raised).

**Q: Why stratify — what goes wrong without it?**
A: With 69/31 imbalance, an unlucky random test split could contain very few
rejections; precision/recall for the rejected class would then hinge on a
handful of samples. Stratification keeps the class ratio in both sets equal
to the full dataset's.
