# Project Knowledge Document: Loan Approval Prediction

The complete study guide for the project. For each component you get:

> **What it is · Why we need it · How it works · Where it is in the code · What to say if asked**

Every number here comes from running the project's own scripts
(`src/eda.py`, `src/train_and_evaluate.py`) and the saved models, so you can
quote them as fact. If you retrain with changes, the numbers in
`output/plots/model_comparison.csv` are the source of truth.

Companion documents:
- [FEATURES_EXPLAINED.md](FEATURES_EXPLAINED.md): every input and engineered feature in detail.
- [PRESENTATION_SLIDES.md](PRESENTATION_SLIDES.md): the text and speaker notes of the deck.

---

## Table of contents

1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [The dataset](#2-the-dataset)
3. [Machine learning basics](#3-machine-learning-basics)
4. [Exploratory data analysis](#4-exploratory-data-analysis)
5. [Feature engineering](#5-feature-engineering)
6. [Preprocessing and the pipeline](#6-preprocessing-and-the-pipeline)
7. [Logistic Regression](#7-logistic-regression)
8. [Decision Tree](#8-decision-tree)
9. [Random Forest](#9-random-forest)
10. [Cross-validation and model selection](#10-cross-validation-and-model-selection)
11. [Evaluation metrics](#11-evaluation-metrics)
12. [Our actual results](#12-our-actual-results)
13. [Feature importance](#13-feature-importance)
14. [Overfitting](#14-overfitting)
15. [Pickle and model persistence](#15-pickle-and-model-persistence)
16. [The Streamlit app](#16-the-streamlit-app)
17. [Charts](#17-charts)
18. [File-by-file map](#18-file-by-file-map)
19. [Limitations and future scope](#19-limitations-and-future-scope)
20. [Likely viva questions](#20-likely-viva-questions)
21. [One-paragraph summary to recite](#21-one-paragraph-summary-to-recite)

---

## 1. The project in one paragraph

> "We built a **supervised binary classification system** that predicts
> whether a loan application will be **Approved** or **Rejected**. It uses
> the public Analytics Vidhya loan dataset (614 applications, 11 input
> columns). We added **5 engineered affordability features** (household
> income, EMI, income left after EMI, loan-to-income ratio, log loan amount).
> We trained **Logistic Regression, a Decision Tree and a Random Forest**
> inside leakage-safe scikit-learn pipelines and chose the best one with
> **5-fold cross-validation**. The Random Forest won narrowly with 87.8% test
> accuracy and F1 0.918. A **Streamlit** app shows applicants a simple
> decision while keeping the model internals hidden."

---

## 2. The dataset

| Property | Value |
|---|---|
| Source | **Analytics Vidhya "Loan Prediction" practice problem** (also on Kaggle) |
| File | `data/loan_data.csv` |
| Shape | **614 rows × 13 columns** |
| Target | `Loan_Status`: **Y = approved (422, 68.7%)**, **N = rejected (192, 31.3%)** |
| Missing values | **149 cells in 7 columns**: Credit_History 50, Self_Employed 32, LoanAmount 22, Dependents 15, Loan_Amount_Term 14, Gender 13, Married 3 |
| Duplicates | none |

**Columns:** `Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status`

- Incomes are **₹ per month**. `LoanAmount` is in **₹ thousands**.
  `Loan_Amount_Term` is in **months** (360 = 30 years; 512 of 614 rows).
- `Loan_ID` is dropped. A unique ID carries no signal and could only be memorised.

> 🗣 **"Where is the data from?"** "The public Analytics Vidhya practice
> dataset for a fictional company, 'Dream Housing Finance'. It is not any real
> bank's customer data."

---

## 3. Machine learning basics

### 3.1 Supervised learning
- **What:** learning a function from **labelled** examples. Each training row
  has the inputs *and* the correct answer.
- **Why:** we have past applications with known outcomes, so the model can
  learn the approval pattern instead of us hand-coding rules.
- **Where:** `pipeline.fit(X_train, y_train)` in `train_and_evaluate.py`.
  The "supervision" is `y_train`.

### 3.2 Classification vs regression
- Classification predicts a **category**; regression predicts a **number**.
- Ours is **binary classification**: the models' `classes_` are `['N', 'Y']`.
- 🗣 "Approve or not is a yes/no question, so this is classification. The
  models also give a probability through `predict_proba`."

### 3.3 Features (X) and target (y)
- **X** = 11 raw columns (6 categorical + 5 numeric). The pipeline then adds
  5 engineered columns, giving **16 columns before encoding**.
- **y** = `Loan_Status`.
- **Where:** `load_and_split()`:
  `y = df["Loan_Status"]`, `X = df.drop(columns=["Loan_ID", "Loan_Status"])`.

### 3.4 Train/test split
- **491 training rows / 123 test rows** (80/20).
- `stratify=y` keeps the 69/31 ratio in both halves (test set: 69.1% approved).
- `random_state=42` makes the split identical on every run.
- 🗣 "Scoring a model on rows it trained on measures memory, not learning. The
  test set stands in for future applications and is used **once**, at the end."

---

## 4. Exploratory data analysis

**What:** looking at the data with charts and statistics *before* modelling.
**Where:** `src/eda.py` writes 6 charts to `outputs/`.

| Finding (real numbers) | What it changed |
|---|---|
| **Credit history:** 80% approved with a history vs **8% without** (10×) | We expected it to dominate, and it does (§13) |
| **Class imbalance:** 422 vs 192 (≈ 2.2 : 1) | Stratified split; F1 instead of accuracy for selection |
| **Income skew = 6.54:** median ₹3,812, mean ₹5,403, max ₹81,000 | Median (not mean) imputation; log of loan amount |
| **Property area:** Semiurban 77% > Urban 66% > Rural 61% | A mild signal |
| **Correlation:** Credit_History ↔ approval +0.56, income ↔ approval ≈ 0, income ↔ LoanAmount +0.57 | Raw income is weak **alone**, so we built **ratio** features (§5) |

> 🗣 **"If income barely correlates with approval, why use it?"** "Correlation
> only sees straight-line, one-column relationships. What matters to a lender
> is income *relative to the loan*. That's why we engineered loan-to-income and
> income-after-EMI, and the Random Forest ranks those right after credit history."

---

## 5. Feature engineering

**What:** creating new columns from existing ones using domain knowledge.
**Where:** `src/features.py` → `add_engineered_features()`.

| New feature | Formula | Meaning |
|---|---|---|
| `TotalIncome` | ApplicantIncome + CoapplicantIncome | Household income |
| `EMI` | LoanAmount × 1000 ÷ Loan_Amount_Term | Approx. monthly instalment in ₹ (no interest) |
| `BalanceIncome` | TotalIncome − EMI | Money left each month |
| `LoanToIncome` | LoanAmount × 1000 ÷ TotalIncome | Loan size in months of income |
| `LoanAmountLog` | log(1 + LoanAmount) | Tames the long tail of loan sizes |

**Evidence it helped:**
- Logistic Regression's test **ROC-AUC rose from 0.852 to 0.870** with the
  features added. Nothing else changed.
- In the Random Forest, **LoanToIncome (0.078)** and **BalanceIncome (0.061)**
  are the #2 and #3 columns, after Credit_History (0.415).

**Why it's leakage-safe:** the formulas use only the applicant's own row and
learn nothing from the dataset. They run as a `FunctionTransformer`, the
first step of every pipeline.

> 🗣 **"Why put feature engineering inside the pipeline?"** "So the saved
> model computes it automatically. The app sends raw form values, and training
> and prediction can never compute the features differently."

> 🗣 **"Why is it in a separate file?"** "Pickle saves a *reference* to the
> function, `features.add_engineered_features`. The app must be able to import
> it, so it can't live in a script that runs as `__main__`."

---

## 6. Preprocessing and the pipeline

```
raw row (11 columns)
  └─ FunctionTransformer(add_engineered_features)        → 16 columns
       └─ ColumnTransformer
            ├─ 10 numeric:    SimpleImputer(median) → StandardScaler
            └─ 6 categorical: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
                 └─ 25-column matrix → model (LR / DT / RF)
```

**Where:** `build_preprocessor()` and `make_pipeline()` in `train_and_evaluate.py`.

### 6.1 Missing values (imputation)
- **Numeric → median.** Learned on the 491 training rows: ApplicantIncome
  3,859, CoapplicantIncome 1,032, LoanAmount 128, Term 360, Credit_History 1.0.
- **Categorical → mode** (Male, Yes, 0 dependents, Graduate, not self-employed).
- 🗣 "Median because incomes are skewed (6.54). A few huge incomes drag the
  mean up; the median stays typical. There's no 'average gender', so
  categories use the most frequent value."
- Engineered features built from a missing value are also missing, and the
  same imputer fills them.

### 6.2 Encoding
- `OneHotEncoder` gives each category its own 0/1 column, e.g.
  `Property_Area` → `_Rural / _Semiurban / _Urban`.
- 🗣 "Mapping Rural=0, Semiurban=1, Urban=2 would invent an order and size
  that doesn't exist. One-hot keeps categories equal."
- `handle_unknown="ignore"`: an unseen category becomes all zeros instead of
  crashing the app.

### 6.3 Scaling
- `StandardScaler` rescales each numeric column to mean 0, variance 1.
- 🗣 "Logistic Regression multiplies features by weights, so income (thousands)
  would drown a 0/1 flag. Trees only compare one column to a threshold, so
  scaling doesn't affect them."

### 6.4 Data leakage, and why the pipeline prevents it
- **Leakage** = information from the test set sneaking into training and
  inflating scores. Example: computing the median on all 614 rows before splitting.
- In a `Pipeline`, every imputer, scaler and encoder is fit only when
  `pipeline.fit(X_train)` runs. Test rows are only **transformed**, never fit on.
- 🗣 "Our design makes leakage impossible by construction, not just unlikely."

---

## 7. Logistic Regression

- **What:** a linear classifier. Score = weighted sum of features + bias.
  Probability = sigmoid(score) = 1 / (1 + e^−score). Approve if ≥ 0.5.
- **Training:** adjusts weights to fit the labels (`max_iter=1000`; it
  converged in 36 iterations). Default L2 regularisation, C = 1.0.
- **Our learned weights (scaled space):** **Credit_History +1.254** (by far
  the largest), Semiurban +0.466, BalanceIncome +0.421, Married=Yes +0.277;
  Dependents=1 −0.397, Rural −0.316, TotalIncome −0.315.
- **Test results:** accuracy 0.862, F1 0.908, **best ROC-AUC (0.870)**.
- 🗣 **Trap question: "Why is TotalIncome's weight negative?"** "TotalIncome
  and BalanceIncome are almost the same number, because the EMI is small. When
  two features are highly correlated, linear weights split between them in odd
  ways. Look at the pair together, not one weight alone. This is called
  multicollinearity."
- 🗣 **"What is the sigmoid?"** "An S-curve that maps any number into 0 to 1,
  so the score reads as a probability. Score 0 gives 0.5, the decision boundary."

---

## 8. Decision Tree

- **What:** a flowchart of yes/no questions learned from data. At each node it
  picks the feature and threshold that best separate the classes, measured by
  **Gini impurity** (0 = pure, 0.5 = 50/50 mix).
- **Ours:** `max_depth=4` (depth 4, **11 leaves**).
- 🗣 **"Why limit the depth?"** "An unlimited tree splits until every leaf is
  pure, which means memorising training rows (overfitting). Depth 4 forces
  general rules and keeps the tree readable."
- **Weakness seen in results:** lowest ROC-AUC (0.802). A small tree gives
  coarse, step-like probabilities, so it ranks applicants less finely.
- It has the best rejected-class recall (0.66), so it catches more risky
  applicants, but it also misses 4 good ones.

---

## 9. Random Forest

- **What:** an **ensemble** of many decision trees whose votes are averaged.
  - **Bagging:** each tree trains on a random bootstrap sample of the rows.
  - **Random feature subsets:** each split considers only some of the features.
  - Different trees make different mistakes, and averaging cancels many of them.
- **Ours:** `RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)`.
  The depth and leaf-size limits keep each tree small, so the forest doesn't overfit.
- **Why we added it:** it is the natural next step beyond a single tree, and it
  gives a built-in feature importance that shows our engineered features individually.
- **Result:** best CV F1, best test accuracy (0.878) and best test F1 (0.918),
  so it is the **selected model**.
- 🗣 **"Isn't a forest a black box?"** "Less explainable than one tree, yes. We
  compensate with feature importance (§13), and the app explains decisions in
  plain language."

---

## 10. Cross-validation and model selection

- **What:** 5-fold **stratified** cross-validation on the **491 training rows**.
  Split them into 5 parts, train on 4 and validate on the 5th, five times. Report the mean ± standard deviation.
- **Why:** a single 123-row test set is one lucky-or-unlucky draw. CV gives a
  steadier estimate and shows the spread.
- **Selection rule:** the highest **mean CV F1** wins. The test set never
  influences the choice; it only reports the final score.
- **Where:** `cross_validate_one()` in `train_and_evaluate.py`.

| Model | CV F1 (mean ± std) | CV accuracy | CV ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.864 ± 0.026 | 0.790 | 0.723 |
| Decision Tree | 0.863 ± 0.030 | 0.788 | 0.691 |
| **Random Forest** | **0.867 ± 0.028** | **0.794** | **0.773** |

> 🗣 **Be honest:** "The three models are within one standard deviation of each
> other, so the forest wins narrowly. Logistic Regression is a fair
> alternative: it's simpler and has the best test ROC-AUC."

> 🗣 **"Why are CV scores lower than test scores?"** "Each CV fold trains on
> only about 393 rows and validates on about 98. That's less data and more
> noise. The 123-row test set happened to be a slightly easier draw. The CV
> number is the more conservative estimate."

---

## 11. Evaluation metrics

| | Predicted Approved | Predicted Rejected |
|---|---|---|
| **Actually approved** | True Positive (TP) | False Negative (FN): good customer turned away |
| **Actually rejected** | False Positive (FP): risky loan approved | True Negative (TN) |

- **Accuracy** = (TP+TN) / all. Misleading with imbalance: "always approve" scores 69%.
- **Precision** = TP / (TP+FP). *When we approve, how often are we right?* Protects the bank.
- **Recall** = TP / (TP+FN). *Of all good applicants, how many did we approve?* Protects customers.
- **F1** = 2PR / (P+R). High only if both are high. **Our selection metric.**
- **ROC-AUC** = how well the model *ranks* applicants across all thresholds
  (0.5 = coin flip, 1 = perfect). Computed from `predict_proba`.

---

## 12. Our actual results

Test set (123 rows), from `output/plots/model_comparison.csv`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.862 | 0.840 | **0.988** | 0.908 | **0.870** |
| Decision Tree (depth 4) | 0.862 | **0.862** | 0.953 | 0.905 | 0.802 |
| **Random Forest** ⭐ | **0.878** | 0.857 | **0.988** | **0.918** | 0.862 |

**Confusion matrices** (rows = truth N, Y; columns = predicted N, Y):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 22 | 16 | 1 | 84 |
| Decision Tree | 25 | 13 | 4 | 81 |
| **Random Forest** | 24 | 14 | 1 | 84 |

**Check a number yourself (examiners love this):** RF precision =
84 / (84 + 14) = 0.857 ✔. Recall = 84 / (84 + 1) = 0.988 ✔.

**The story:**
1. All three models catch almost every good applicant (recall 0.95 to 0.99).
2. Their weakness is the **rejected** class. The RF rejects only 24 of 38
   risky applicants (recall 0.63), so it is **lenient**.
3. That's partly because the data has twice as many approvals as rejections.

---

## 13. Feature importance

### 13.1 Permutation importance (selected Random Forest)
**How:** shuffle one input column of the test set and measure how much ROC-AUC
drops. A big drop means the model relied on that column. This works for any
model. Chart: `output/plots/06_feature_importance.png`.

| Raw input | Drop in ROC-AUC |
|---|---|
| **Credit_History** | **0.292** |
| CoapplicantIncome | 0.079 |
| ApplicantIncome | 0.052 |
| LoanAmount | 0.041 |
| Property_Area | 0.005 |
| Married, Self_Employed, Gender, Education, Dependents, Term | ≈ 0 |

Shuffling a raw income column also scrambles the engineered features built
from it, so each bar is that input's **total** contribution.

### 13.2 Random Forest built-in (Gini) importance
Shows every model column, including the engineered ones
(`07_forest_importance.png`): Credit_History 0.415, **LoanToIncome 0.078,
BalanceIncome 0.061**, ApplicantIncome 0.061, TotalIncome 0.061,
LoanAmountLog 0.059, EMI 0.056 …

> 🗣 **Conclusion:** "The model decides on **credit history + affordability**.
> Demographic columns contribute almost nothing, which is good for fairness,
> and they could be dropped."

---

## 14. Overfitting

- **What:** the model memorises training rows and fails on new data. The sign
  is train accuracy much higher than test accuracy.
- **Our check** (printed by `train_and_evaluate.py`):

| Model | Train acc | Test acc | Gap |
|---|---|---|---|
| Logistic Regression | 0.800 | 0.862 | −0.061 |
| Decision Tree | 0.827 | 0.862 | −0.035 |
| Random Forest | 0.815 | 0.878 | −0.063 |

- All gaps are **negative**, so there's no overfitting. The limits that keep
  it so: tree depth 4; forest depth 6 with at least 5 rows per leaf; LR's L2 penalty.
- 🗣 "Overfitting is **prevented** with complexity limits and **detected** by
  comparing train and test scores. Cross-validation is a second check."

---

## 15. Pickle and model persistence

- **What:** Python's way to save a live object to a file and load it back.
- **Why:** train once, then load instantly in the app. The **whole pipeline**
  is saved (feature engineering + imputer + scaler + encoder + model), so the
  app processes inputs exactly like training did.
- **Files:** `output/models/logistic_regression.pkl`, `decision_tree.pkl`,
  `random_forest.pkl`. Each stores `{"model_name", "model", "is_best"}`.
- **Loading:** `load_models()` in `app.py`, cached with `@st.cache_resource`.
- ⚠️ Honest caveat: "Never unpickle files from untrusted sources, because it
  can run arbitrary code. Ours are files we created ourselves."

---

## 16. The Streamlit app

**Run:** `./.venv/bin/streamlit run src/app.py` → http://localhost:8501

**What applicants see** (`src/app.py`):
1. A form in three sections: *About you*, *Income & credit*, *Loan details*.
   Credit history is a plain Yes/No question, and the term is chosen in years.
2. One decision from the **selected model**: "Likely to be approved" or "Not
   likely", with a coarse likelihood (High / Moderate / Low / Very low)
   instead of an exact probability.
3. *Your loan at a glance*: household income, estimated EMI, money left after EMI.
4. *What influenced this result*: plain-language reasons based on the factors
   the importance analysis found (credit history, loan size vs income,
   co-applicant income).

**What applicants do NOT see:** model names, the three-model comparison, exact
probabilities, the raw row sent to the pipeline. These are internals a real
lender wouldn't expose.

**Examiner view:** add `?examiner=1` to the URL
(http://localhost:8501/?examiner=1). The page then also shows every model's
decision and P(approved), the engineered features and the raw row.

**Demo path (verified):**
1. Keep the defaults → **Likely to be approved (High)**. Examiner view: LR 82%, RF 79%, DT 75%.
2. Answer **No** to repaying past loans → **Not likely (Very low)**. Examiner view: LR 11%, RF 24%, DT 0%.

> 🗣 "One answer flips every model, the same thing the feature-importance
> chart shows. The app sends raw values; all processing happens inside the
> saved pipeline."

Styling comes from `.streamlit/config.toml` (green theme) plus a little CSS in `app.py`.

---

## 17. Charts

| File | Shows | What to say |
|---|---|---|
| `outputs/1_approval_vs_rejection.png` | 422 vs 192 | "Imbalanced, so we stratify and don't trust accuracy alone." |
| `outputs/2_applicant_income_distribution.png` | skewed income | "Median ₹3,812 vs mean ₹5,403, so we impute with the median." |
| `outputs/3_loan_amount_distribution.png` | skewed loan amount | "Median 128k, max 700k, hence the log feature." |
| `outputs/4_credit_history_vs_approval.png` | 80% vs 8% | "The single strongest signal." |
| `outputs/5_property_area_vs_approval.png` | 77 / 66 / 61% | "A mild effect." |
| `outputs/6_correlation_heatmap.png` | correlations | "Income alone ≈ 0, which motivated the ratio features." |
| `output/plots/04_confusion_matrices.png` | 3 confusion matrices | "Rows = truth, columns = prediction; off-diagonal = the two costly errors." |
| `output/plots/05_model_comparison.png` | 5 metrics × 3 models | "Close race; the RF leads on accuracy and F1." |
| `output/plots/06_feature_importance.png` | permutation importance | "Credit history, then affordability." |
| `output/plots/07_forest_importance.png` | RF Gini importance | "Engineered features rank #2 and #3." |

---

## 18. File-by-file map

```
├── data/loan_data.csv              614 rows, the public dataset
├── src/
│   ├── eda.py                      6 EDA charts → outputs/
│   ├── features.py                 the 5 engineered features
│   ├── train_and_evaluate.py       pipelines, 3 models, CV, test metrics,
│   │                               importance charts, pickles
│   ├── app.py                      Streamlit app (applicant view + examiner view)
│   └── build_presentation.py       builds the PowerPoint from real results
├── output/
│   ├── models/*.pkl                one fitted pipeline per model
│   └── plots/                      04–07 charts + model_comparison.csv
├── outputs/                        EDA charts 1–6
├── docs/
│   ├── Loan_Approval_Presentation.pptx   14 slides + speaker notes
│   ├── PRESENTATION_SLIDES.md      text version of the deck
│   ├── FEATURES_EXPLAINED.md       every feature explained
│   └── KNOWLEDGE_DOCUMENT.md       this document
├── .streamlit/config.toml          app theme
├── requirements.txt
└── README.md
```

---

## 19. Limitations and future scope

| Limitation | Future improvement |
|---|---|
| **Lenient:** about 1 in 3 risky applicants still approved (RF rejected-class recall 0.63) | Class weights (`class_weight="balanced"`) or a higher decision threshold |
| Small (614 rows), public, not from a real lender | Larger, real-world data; monitor for drift |
| Models are close; the winner is within CV noise | Hyperparameter tuning (GridSearchCV); try gradient boosting (XGBoost) |
| Sensitive attributes (Gender, Married, Dependents) still in the inputs | They contribute ≈ 0, so drop them and run a fairness audit |
| EMI ignores interest | Use the real amortised EMI formula with an interest rate |

---

## 20. Likely viva questions

**Q: Why is this "AI"?**
A: The system learns its decision rule from labelled historical data instead
of following hand-written rules. That's machine learning, the core of modern AI.

**Q: What did you modify / add to the base project?**
A: (1) Five engineered affordability features. (2) A Random Forest as a third
model. (3) 5-fold cross-validation for model selection instead of picking on
the test set. (4) Feature-importance analysis. (5) A redesigned app that hides
model internals from applicants.

**Q: Why not deep learning?**
A: 614 rows is far too small; a neural network would overfit. For small
tabular data, linear models and tree ensembles are the right tools, and
they're easier to explain in a regulated area like lending.

**Q: Why F1 and not accuracy for choosing the model?**
A: With a 69/31 imbalance, accuracy rewards always saying "approve". F1 needs
both precision and recall to be high.

**Q: Why choose the model on cross-validation instead of the test set?**
A: If we choose using the test set, the test score is no longer an unbiased
estimate. CV uses only training rows, so the test set stays a fair final exam.

**Q: How do you know the engineered features helped?**
A: LR's ROC-AUC went from 0.852 to 0.870 with only that change, and the
forest ranks two of them right after credit history.

**Q: Your model approves 14 of 38 risky applicants. Is that acceptable?**
A: Not for a real bank. It's our main limitation. The fix is class weights or
raising the approval threshold, trading some recall for precision.

**Q: Is the model fair?**
A: Gender, marital status and dependents are in the data, but permutation
importance shows they contribute ≈ 0. A real system should drop them and test
for disparate impact. The app is labelled as an academic demo.

**Q: Why does the app hide probabilities from users?**
A: Exact model scores are internal risk information. Applicants get an
understandable decision and reasons; examiners can see the details with `?examiner=1`.

**Q: What is `random_state=42`?**
A: A fixed random seed, so the split, the forest and CV folds come out the
same every run. Results are reproducible.

**Q: What does `stratify=y` protect against?**
A: An unlucky split where the test set has very few rejections, which would
make rejected-class metrics depend on a handful of rows.

---

## 21. One-paragraph summary to recite

> "614 public loan applications, 11 raw features plus 5 engineered
> affordability features, target Loan_Status. All preprocessing (feature
> engineering, median/mode imputation, scaling, one-hot encoding) sits inside a
> scikit-learn Pipeline fit only on the 491 training rows. We compared Logistic
> Regression, a depth-4 Decision Tree and a 300-tree Random Forest with 5-fold
> cross-validation. The Random Forest had the best CV F1 (0.867 ± 0.028). On
> the untouched 123-row test set it scored 87.8% accuracy, 0.857 precision,
> 0.988 recall, 0.918 F1 and 0.862 ROC-AUC. Credit history is by far the most
> important feature, followed by income relative to loan size. The main
> limitation is leniency: about one in three risky applicants is still
> approved. The Streamlit app gives applicants a simple decision with reasons
> and hides model internals behind an examiner view."
