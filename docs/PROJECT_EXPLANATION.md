# Project Explanation — AI-Based Loan Approval Prediction System

A beginner-friendly explanation of **every component** of this project. Each
section follows the same pattern:

> **Definition → Why we use it → Small example → Where it is in the code →
> Teacher questions with technically correct answers**

Every number in this document was computed from the actual project artifacts
(`model_comparison.csv`, the saved pipelines, the EDA script output). Nothing
is invented, and features we did not implement ( Random Forest, cross-
validation, hyperparameter tuning grids…) are explicitly marked as *not
implemented* — so you never accidentally claim them.

**Files in the project (actual inventory):**

| File | Role | Status |
|---|---|---|
| `data/loan_data.csv` | The public dataset (614 rows) | used by everything |
| `src/eda.py` | 6 EDA charts into `outputs/` | active |
| `src/train_and_evaluate.py` | leakage-safe training + evaluation + pickles | **the core ML script** |
| `src/app.py` | Streamlit app that loads saved pipelines | active |
| `src/build_presentation.py` | generates `docs/Loan_Approval_Presentation.pptx` | active, offline use |
| `output/models/best_model.pkl` | fitted Logistic Regression pipeline | loaded by the app |
| `output/models/second_model.pkl` | fitted Decision Tree pipeline | loaded by the app |
| `output/plots/*.png` | confusion matrices + comparison chart (training) | presentation use |
| `outputs/*.png` | EDA charts | presentation use |
| `docs/PRESENTATION_SLIDES.md`, `KNOWLEDGE_DOCUMENT.md` | earlier study notes | background reading |
| `src/data_preparation.py` | **legacy** manual preprocessing script | superseded — see §6 box |

---

## 1. Problem statement and objectives

**Definition.** The project predicts one binary outcome — *loan approved* vs
*loan rejected* — for a new application, using historical labelled
applications as its teacher.

**Why we use it (the motivation).** Screening thousands of applications by
hand is slow and inconsistent, and the two possible mistakes have very
different costs:
- **False approval** → the bank lends money that never comes back (expensive).
- **False rejection** → a creditworthy customer walks away (lost business).

A model makes these decisions consistently, and we can *measure* how often it
makes each kind of mistake.

**Small example.** Application: income ₹5,000, credit history good, loan
₹1,30,000/360 days → model outputs `Y (approved), P = 0.80`.

**Where it lives.** The whole pipeline: `src/train_and_evaluate.py` (build)
→ `src/app.py` (use). Objectives framing is also on slide 3 of the
presentation.

**Teacher questions & answers:**

- **Q: What exactly is your problem statement?**
  A: *Learn from 614 historical loan applications to predict `Loan_Status`
  for a new application, and serve that prediction in an interactive app,
  with explainable classical ML models.*

- **Q: Which mistake is worse, and how does your system reflect that?**
  A: *For the bank, a false approval (bad loan) is usually costlier. Our
  metric choice reflects this concern: we report precision (approval
  correctness, 0.840 for LR) alongside recall, rather than accuracy alone.*

- **Q: What are your project objectives?**
  A: *(1) end-to-end pipeline; (2) handle missing values/skew/imbalance;
  (3) train LR + Decision Tree fairly on identical preprocessing; (4)
  measure with five metrics; (5) deploy in Streamlit; (6) keep it
  explainable — no deep learning.*

- **Q: Did you achieve all objectives?**
  A: *Yes, with one honest caveat: our model is lenient — recall on the
  rejected class is only 0.58 — so it catches approvals very well but also
  approves some applications it should reject. We state this in the
  limitations.*

---

## 2. AI, Machine Learning and supervised learning

**Definition.**
- **AI** — making machines do things that would need intelligence if done by
  people. Machine learning is the branch that gets there by *learning from
  data* instead of being programmed rule by rule.
- **Machine learning** — algorithms that improve at a task with experience
  (E), measured by performance (P), on task (T).
- **Supervised learning** — the training data is *labelled*: every example
  comes with the correct answer.

**Why we use it.** Writing "if income > X and credit_aggregate > Y then
approve" rules by hand is brittle; approval patterns live in the data, and
614 labelled examples are literally historical decisions we can learn from.

**Small example.**
- Not ML: `if CREDIT_SCORE >= 700: approve` — fixed rule, never improves.
- ML: show the algorithm 491 past applications **with outcomes**; it
  discovers which combination of features separates Y from N, weights and
  all.
- *Not* supervised learning (so not used here): e.g. clustering customers
  into groups without labels (unsupervised).

**Where it lives.** `train_and_evaluate.py` → `pipeline.fit(X_train,
y_train)`: X = the features, y = the labels ("Y"/"N"). The "supervision" is
the `y_train` argument.

**Teacher questions & answers:**

- **Q: Where is the "learning" in your code?**
  A: *In `pipeline.fit(X_train, y_train)`. For Logistic Regression, fit
  adjusts 20 weights by gradient descent (it converged in 26 iterations —
  that's the learned `n_iter_`); for the tree, fit recursively chooses split
  questions; both end up storing fitted parameters, which we pickle.*

- **Q: Is your model AI or ML?**
  A: *It's ML, which is the practical core of AI: it learned a decision
  function from labelled historical examples rather than being
  hand-programmed.*

- **Q: What would unsupervised learning be here instead?**
  A: *Clustering applicants into risk groups with no `Loan_Status` labels.
  It wouldn't fit our goal, which is predicting a known, binary outcome —
  that's the definition of needing supervision.*

---

## 3. Classification and why it suits this problem

**Definition.** **Classification** predicts a *category*; **regression**
predicts a *number on a scale* (e.g. price). Ours is **binary
classification** — exactly two categories.

**Why we use it.** `Loan_Status` has exactly two values (Y/N → the model's
`classes_` array literally contains `['N', 'Y']`), and the question "approve
this application?" has no meaningful numeric middle ground.

**Small example.**
- Classification: *Will this loan be approved?* → {Y, N}. ✔ our case.
- Regression (wrong tool here): *How much loan should we offer?* → ₹ amount.
- Multiclass (not needed): risk grade in {A, B, C, D}.

**Where it lives.** Target selection in `load_and_split()`:
`y = df["Loan_Status"]`. The metrics used (`precision_score(pos_label="Y")`
etc.) and `OneHotEncoder`'s handling of the categorical target side all
assume classification.

**Teacher questions & answers:**

- **Q: Why is this classification and not regression?**
  A: *The label is a category (Y/N), not a quantity. A regression model
  would output numbers like 0.73 — we'd then have to invent a cut-off;
  classification models (ours) output the class plus its probability
  directly via `predict_proba`.*

- **Q: Binary or multiclass?**
  A: *Binary — the pipeline's `classes_` are `['N','Y']`. The metrics are
  configured accordingly, e.g. precision with `pos_label='Y'`.*

- **Q: Is your probability output still classification?**
  A: *Yes — the class plus a confidence. Logistic Regression is
  specifically a classification model that arrives at the class through a
  probability; the tree is a classification model that counts votes in
  leaves.*

---

## 4. Dataset, features and target variable

**Definition.** The dataset is a table: 614 rows (applications) × 13 columns
(information per application). **Target** = the column we learn to predict:
`Loan_Status`. **Features (X)** = the 11 other informative columns (after
dropping `Loan_ID`).

**Why we use it.** (public dataset; not proprietary bank data!) We need
real-but-legal, reproducible training data. The dataset is the **Analytics
Vidhya "Loan Prediction" practice problem** (also mirrored on Kaggle) — a
public teaching dataset for a fictional "Dream Housing Finance" company. It
is *not* any bank's proprietary customer data, and we label it that way
everywhere.

**Actual dataset facts (verified):**

| Property | Value |
|---|---|
| Shape | 614 × 13 |
| Class split | 422 Y (68.7%) / 192 N (31.3%) |
| Missing cells | 149 across 7 columns (Credit_History 50, Self_Employed 32, LoanAmount 22, Dependents 15, Term 14, Gender 13, Married 3) |
| Duplicates | 0 rows, 0 Loan_IDs |
| dtypes | 8 object (text), 5 numeric |

**Features list (exactly what the pipelines were trained on — 11 columns):**

- Categorical (6): `Gender`, `Married`, `Dependents`, `Education`,
  `Self_Employed`, `Property_Area`
- Numeric (5): `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`
  (in ₹ thousands), `Loan_Amount_Term` (days), `Credit_History` (1/0)

Dropped: `Loan_ID` (identifier). Excluded from features: `Loan_Status`
(target).

**Small example.** Row `LP001003`: Male, Married=Yes, Dependents=1,
Graduate, Not self-employed, income 4583 (+ coapplicant 1508), loan 128k,
term 360, credit history 1, Rural → **N**. That one row is one training
example.

**Where it lives.** `data/loan_data.csv`; loading and X/y construction in
`load_and_split()`; dataset inspection script from the earlier session
printed the missing/duplicate/dtype audit.

**Teacher questions & answers:**

- **Q: What is your target variable and how is it encoded?**
  A: *`Loan_Status`. In training we keep the strings 'Y'/'N' (our pipeline's
  `classes_` are literally `['N','Y']`); in evaluation we mark 'Y' as the
  positive class (`pos_label='Y'`).*

- **Q: Why did you drop Loan_ID?**
  A: *It is unique per row — it carries zero generalisable signal, and
  letting a model see it would only invite memorisation (overfitting).*

- **Q: Is `Credit_History` numeric or categorical?**
  A: *It is a categorical flag stored as a number (1/0). We keep it in the
  numeric block, which is fine — the imputer fills it with its training
  median (1.0) and the scale barely matters for a 0/1 column.*

- **Q: Whose data is this?**
  A: *Analytics Vidhya's public practice dataset for a fictional company —
  not proprietary data of any bank.*

---

## 5. Exploratory Data Analysis

**Definition.** EDA = inspecting data with statistics and charts **before**
modelling, to find patterns, problems, and the features worth attention.

**Why we use it.** Three findings from our real EDA directly drove the ML
design:
1. **Class imbalance** (69/31) → stratified splitting + not trusting
   accuracy alone.
2. **Skewed income** (skew = 6.54; median ₹3,812 vs mean ₹5,403; max
   ₹81,000) → median (not mean) imputation.
3. **Credit history dominates** (80% approval with history vs 8% without —
   a 10× gap) → expected the models to lean on it; they did.

**Actual EDA results (from `src/eda.py`, all charts saved under
`outputs/`):**

| # | Chart | Key finding |
|---|---|---|
| 1 | `1_approval_vs_rejection.png` | 422 vs 192 — imbalance ~2.2:1 |
| 2 | `2_applicant_income_distribution.png` | right-skewed, outliers to ₹81,000 |
| 3 | `3_loan_amount_distribution.png` | median 128k, mean 146k, also right-skewed |
| 4 | `4_credit_history_vs_approval.png` | 80% vs 8% approval — dominant feature |
| 5 | `5_property_area_vs_approval.png` | mild: Semiurban 77% > Urban 66% > Rural 61% |
| 6 | `6_correlation_heatmap.png` | Credit_History↔approval +0.56; income↔LoanAmount +0.57; income↔approval ≈ 0.00 |

**Small example.** Chart 4 is a two-bar chart: 8% vs 80%. That single visual
tells you credit history will be the model's strongest signal — and it was
(the learned LR weight for `Credit_History` is +1.247, by far the largest).

**Where it lives.** `src/eda.py` (Pandas + Matplotlib + Seaborn), run:
`./.venv/bin/python src/eda.py`.

**Teacher questions & answers:**

- **Q: What did EDA change in your design?**
  A: *Three concrete decisions: median-based imputation (income skew 6.54),
  stratified splitting (69/31 imbalance), and the interpretation of results
  (we expected credit history to dominate — it did).*

- **Q: If income↔approval correlation is ≈ 0, why keep income at all?**
  A: *Correlations capture only linear, single-feature relationships; a
  model can still use income in combination with other features. We kept it
  (standard practice for tabular data), and the trained model confirms it
  stayed a minor contributor.*

- **Q: Why is the Semiurban approval rate (77%) higher than Urban (66%)?**
  A: *That's what the data shows (233/202/179 applications per area); it
  may reflect loan sizes or applicant mixes in those areas rather than a
  causal effect. We present it as a mild pattern, not a rule.*

---

## 6. Data preprocessing

**Definition.** Preprocessing = transforming raw, human-readable records
into clean, numeric, model-ready matrices — *imputation, scaling, encoding,
splitting* — without changing the science, only the format.

**Why we use it.** The raw CSV mixes text ("Graduate"), blank cells, and
values with wildly different ranges (₹ incomes vs 0/1 flags). Logistic
Regression needs numbers on comparable scales; the encoder and imputer must
learn their vocabulary/statistics from data.

> **Box — what changed and why (be ready for this):** In an earlier version
> we preprocessed manually in `src/data_preparation.py` (a script that
> still exists but is **superseded** and now unused). The current design
> wraps all preprocessing inside the training pipeline itself:
> `ColumnTransformer` + nested `Pipeline` steps in
> `build_preprocessor()` in `src/train_and_evaluate.py`. That removed the
> manually engineered `TotalIncome` column and the hand-written encoding
> maps. If asked: *the pipeline approach is the correct, leakage-safe,
> deployment-friendly implementation; the legacy file is kept only for
> step-by-step teaching of what each preprocessing step does.*

**The exact pipeline (as implemented):**

```
raw row →─┬─ numeric block (5 cols):     SimpleImputer(median) → StandardScaler
          └─ categorical block (6 cols): SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
                    ↓
              combined 20-feature matrix → estimator (LR or DT)
```

Expanding the one-hot output (15 categorical columns from the 6 text
columns): `Gender_Female/Male, Married_No/Yes, Dependents_0/1/2/3+,
Education_Graduate/Not Graduate, Self_Employed_No/Yes, Property_Area_
Rural/Semiurban/Urban` (verified with `get_feature_names_out()`).

**Small example.** Applicant "Graduate, Urban": after encoding this becomes
`Education_Graduate=1, Education_Not Graduate=0, Property_Area_Urban=1,
Property_Area_Semiurban=0, Property_Area_Rural=0, ...`.

**Where it lives.** `build_preprocessor()` in `src/train_and_evaluate.py`.

**Teacher questions & answers:**

- **Q: What is a ColumnTransformer?**
  A: *It routes different column groups to different sub-pipelines by name:
  our 5 numeric columns to median-impute + scale, our 6 categorical columns
  to mode-impute + one-hot, then concatenates the outputs into one matrix
  for the estimator. Using column names means no positional mistakes when
  the app sends a row.*

- **Q: Why is preprocessing inside the pipeline?**
  A: *Two reasons. Leakage safety: transformers are fit only when
  `pipeline.fit(X_train)` runs, so no test-set statistics can leak into
  training. Deployment: pickling the pipeline carries imputer stats, scaler
  means and encoder vocabulary together with the model, so the app can send
  raw values.*

- **Q: Did you do any feature engineering?**
  A: *We deliberately removed the earlier engineered `TotalIncome` column:
  the EDA showed income alone has ≈0 correlation with approval, and adding
  another income variant was redundant. Simpler features = more explainable
  model.*

---

## 7. Missing-value handling and categorical encoding

**Definition.** **Imputation** = replacing blank cells with an educated
guess computed from the data. **Encoding** = turning text categories into
numbers a model can multiply.

**Why we use it.** scikit-learn estimators raise errors on NaN and cannot do
arithmetic on "Graduate" — so both steps are mandatory before fitting.

**How ours works (verified values learned from the 491 training rows):**

- Numeric medians learned by `SimpleImputer(strategy="median")`:
  `ApplicantIncome` ₹3,859 · `CoapplicantIncome` ₹1,032 · `LoanAmount` 128 ·
  `Loan_Amount_Term` 360 · `Credit_History` 1.0
- Categorical most-frequent values: Gender Male, Married Yes, Dependents 0,
  Education Graduate, Self_Employed No.
- **Why median not mean?** Our EDA measured income skew = 6.54: applicant
  incomes range up to ₹81,000, the mean (5,403) gets dragged up, the median
  (3,812) stays representative of the typical applicant.
- **Why mode for categories?** "Average gender" is meaningless; the most
  common category is the honest default guess.
- `OneHotEncoder(handle_unknown="ignore")` creates the 15 binary columns —
  one per category — so no fake ordering is invented (Rural/Semiurban/Urban
  are not "0/1/2").

**Small example.** An application with a missing `LoanAmount` becomes
`LoanAmount = 128.0` (the training median) *before* scaling — and the same
in the app. An application with `Property_Area = Urban` becomes three
0/1 values `Rural=0, Semiurban=0, Urban=1`.

**Where it lives.** Nested in `build_preprocessor()` (numeric block and
categorical block). In-app, this happens automatically inside the pickled
pipeline when `app.py` submits raw values.

**Teacher questions & answers:**

- **Q: What if a value is missing at prediction time (in the app)?**
  A: *The same imputer inside the pipeline fills it with the median/mode
  learned during training — which is precisely why preprocessing must
  travel with the model in the pickle.*

- **Q: What does `handle_unknown='ignore'` do?**
  A: *If a live input contains a category never seen in training (say a new
  area value), the encoder emits all-zero columns for it instead of
  crashing. For 0/1 binary columns this is the neutral representation; it
  trades a small information loss for robustness.*

- **Q: Why not drop rows with missing values instead of imputing?**
  A: *Only 149 of our cells are missing, but they scatter across 149
  different rows in the worst case — dropping rows would throw away
  perfectly good applications, and our training data is small (491 rows) as
  it is.*

---

## 8. Training and testing data

**Definition.** **Train set** = rows the model learns from (80%). **Test
set** = rows held out and used only to measure performance (20%).

**Why we use it.** Scoring a model on the rows it trained on measures
memory, not learning. Held-out data simulates tomorrow's applications. Our
split: **491 train / 123 test**, `random_state=42` (reproducible),
`stratify=y` (keeps the 69/31 approval ratio in both — measured test share
69.1% vs 68.7% overall).

**What "data leakage" means concretely in our project.** The imputer
statistics, scaler means and one-hot vocabulary are all learned from the
491 training rows only. Verified: scaler means (train) = [5530, 1569.5,
146.5, 341.7, 0.9] — these are *train-set* numbers, then applied to the
test rows unchanged.

**Small example.** Missing-value filling *before* splitting (the classic
mistake): the median would then be computed from test rows too — test
information leaking into training, inflating scores. Our implementation
cannot do this: the imputer lives inside `pipeline.fit(X_train)`.

**Where it lives.** `load_and_split()` in `src/train_and_evaluate.py`
(`train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)`).

**Teacher questions & answers:**

- **Q: Why 80/20 and not k-fold cross-validation?**
  A: *80/20 with a fixed seed gives a reproducible baseline for a small
  dataset in a teaching context. k-fold CV would be the better estimator
  (more stable), and it's listed in our future scope — we didn't implement
  it yet and say so rather than claiming it.*

- **Q: What does `stratify=y` protect you from?**
  A: *From an unlucky split in which the 123 test rows would contain
  disproportionate rejections, making recall/precision for the rejected
  class depended on split luck. Stratification keeps both halves at ~69/31.*

- **Q: Why the same `random_state=42` everywhere?**
  A: *So every run of `train_and_evaluate.py` produces the same split, same
  learning trajectory (that's also why `n_iter_ = 26` is stable for LR), and
  the same metrics — results can be quoted as repeatable facts.*

---

## 9. Logistic Regression

**Definition.** A **linear classification model**: it computes a weighted
sum of (preprocessed) features, squashes the score through the **sigmoid**
to get P(approved), and classifies as approved when that probability ≥ 0.5.

**Why we use it.** On small tabular datasets it is the classic baseline:
fast, stable, and interpretable — every feature has one weight, positive
pushes toward approval, negative toward rejection. In our comparison it
turned out to be the better model on every metric (accuracy 0.862, F1 0.908,
ROC-AUC 0.852).

**How training actually worked here (fitted facts from the saved model):**
`LogisticRegression(max_iter=1000, random_state=42)` converged in 26
iterations (stored `n_iter_`), with L2 penalty. Top learned weights
(scaled space): **`Credit_History` +1.247** (dominant), `Semiurban` +0.464,
`Married=Yes` +0.276, `Rural` −0.311, `Dependents=1` −0.387. No strong
convergence issue; no additional regularisation tuning was done (not
implemented — say so if asked about hyperparameter tuning).

**Small example.** After scaling, suppose a test applicant's
credit-history-scaled value is +2.5 (well above the train mean): the +1.247
weight contributes +3.1 to the score — a strong push toward approval.
Whether they're finally approved still depends on the sum of everything.

**Where it lives.** `build_models()` in `src/train_and_evaluate.py`; the
fitted object is `output/models/best_model.pkl`.

**Teacher questions & answers:**

- **Q: What does the sigmoid do, exactly?**
  A: *σ(z) = 1/(1+e^−z) maps any real score into (0,1) so it reads as a
  probability; σ(0) = 0.5 is the decision boundary. `predict_proba` returns
  it; `predict` selects the class.*

- **Q: Why must features be scaled for this model but not the tree?**
  A: *LR is an optimisation over weighted sums: a feature in the tens of
  thousands (income, mean 5,530 in train) numerically dominates a 0/1 flag
  and slows/destabilises gradient descent. StandardScaler (mean 0, variance
  1, fitted on train only) puts all features on the same footing. A tree
  only asks "is x > t?" per column, which is scale-invariant.*

- **Q: How would you explain a prediction to a rejected applicant?**
  A: *With the weights: which inputs pushed the score down (e.g. no credit
  history) and by how much in log-odds. This interpretability was an
  explicit reason to choose LR over a heavier model.*

- **Q: What regularisation are you using?**
  A: *scikit-learn's default L2 (ridge) penalty with C=1.0 — a mild brake on
  extreme weights. We did not tune C; that's future scope.*

---

## 10. Decision Tree

**Definition.** A model that learns a **flowchart of yes/no questions** from
the data: at each node it picks the feature + threshold that best separates
the classes (lowest Gini impurity), repeating recursively.

**Why we use it.** It's the most human-readable counterpart to LR — its
decisions are literal rules — and it is scale-invariant, so it demonstrates
that the scaler matters for LR but is inert for the tree. Capped at
**`max_depth=4`**.

**Fitted facts (from the saved tree):** depth exactly 4, **10 leaves**,
trained on the same 20 preprocessed features. Nodes at depth 0 test
credit-history-type questions first (tree feature-importance places
`Credit_History` on top, consistent with EDA and the LR weights).

**Small example.** The kind of rule depth-4 trees produce (illustrative
shape, not a quote of a specific leaf): "Credit_History ≤ 0.5 and
Dependents ≤ 0.5 → mostly N … Credit_History > 0.5 and (Scaled
ApplicantIncome irrelevant) → mostly Y." The tree's predicted probability
for a leaf = the fraction of training positives that fell in it. (For our
"no credit history" test applicant it outputs P(Y) = 0.02 — that leaf was
~2% approvals in training.)

**Where it lives.** `build_models()`; fitted tree in
`output/models/second_model.pkl`; you can feel the rule behaviour live in
the app (flip credit history → both models reject).

**Teacher questions & answers:**

- **Q: Why `max_depth=4`?**
  A: *An unlimited tree keeps splitting until leaves are pure — memorising
  the training rows (overfitting). Depth 4 caps the flowchart at four
  questions, keeps 10 broad leaves, and stays human-readable. Our overfit
  check with this cap: train 0.809 vs test 0.846 — no overfit gap.*

- **Q: What is Gini impurity?**
  A: *The probability of mislabelling a random sample if we label by the
  node's class mix (0 = pure, 0.5 = 50/50). The algorithm picks the split
  that reduces weighted Gini the most.*

- **Q: Why does the tree need no scaling?**
  A: *Each split compares one column to a threshold; multiplying a column
  by 1000 just shifts the threshold. Scale-invariant by construction —
  which is also why we left the same preprocessing in place for fairness
  of comparison rather than tailoring per model. (Ours is a fair
  head-to-head with identical preprocessing for both.)*

- **Q: Why not a deeper tree or a Random Forest?**
  A: *Deeper would raise train accuracy without improving test (and erode
  readability). Random Forest was considered and deliberately not added:
  our precondition for it — measurable overfitting of the single tree —
  was not met (test gap nearly −0.04). With 614 rows, the simpler model
  wins on explainability at equal or better performance.*

---

## 11. Evaluation metrics and confusion matrix

**Definition.** The **confusion matrix** counts the four outcomes; the
metrics summarise them from different points of view. For 'Y' as positive:

- **Accuracy** = (TP+TN)/all
- **Precision** = TP/(TP+FP) — *when we approve, how often are we right?*
- **Recall** = TP/(TP+FN) — *of all deserving approvals, how many did we
  catch?*
- **F1** = 2PR/(P+R) — harmonic mean; punishes lopsided behaviour. (Our
  model-selection criterion.)
- **ROC-AUC** = area under the TPR-vs-FPR curve as the 0.5 threshold
  sweeps; measures *ranking* quality, threshold-independent (computed from
  `predict_proba`).

**Our actual numbers (test set, 123 rows; verified against the saved
pipelines and `model_comparison.csv`):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | **0.862** | 0.840 | 0.988 | **0.908** | **0.852** |
| Decision Tree | 0.846 | 0.824 | 0.988 | 0.898 | 0.706 |

Confusion matrices (rows = truth N,Y; cols = prediction N,Y):

- **LR: [[22, 16], [1, 84]]** → 22 true rejections, **16 bad-loan approvals
  (FP)**, 1 good loan missed (FN), 84 correct approvals.
- **DT: [[20, 18], [1, 84]]** → 20 / 18 / 1 / 84.

**Small example (recompute a metric from the matrix).** Take LR's precision:
the matrix says 84 correct approvals and 16 bad approvals, so precision =
84/(84+16) = 0.840 ✔. Recall for approvals = 84/(84+1) = 0.988 ✔. Any
number you quote must survive this kind of recomputation — teachers love
the check, and doing it live in the viva demonstrates you understand what
the metrics mean rather than having memorised them.

**Selection.** Best model picked by measured test F1 → **Logistic
Regression**; overfit gaps: LR train 0.800/test 0.862 (−0.061), DT 0.809/
0.846 (−0.037) → both generalise.

**Where it lives.** `evaluate_one()` + the comparison-table/assembling code
in `src/train_and_evaluate.py`; charts `output/plots/04_confusion_matrices.png`
and `05_model_comparison.png`; table `output/plots/model_comparison.csv`.

**Teacher questions & answers:**

- **Q: Why not trust accuracy alone?**
  A: *With 69/31 imbalance, a lazy "always approve" model scores 69% while
  rejecting nobody — accuracy hides that. Precision (0.840) and recall
  (0.988) plus F1 give the real picture; they're why we selected the model
  by F1.*

- **Q: Your recall is 0.988 but is that unambiguously good?**
  A: *It means we miss almost no deserving approvals (1 FN of 85), but the
  price is visible in the rejected class: recall there is 0.58 (LR catches
  only 22 of 38 true rejections, letting 16 through). That leniency is a
  stated limitation and a future-scope item (threshold calibration).*

- **Q: Why is the tree's ROC-AUC much lower than LR's?**
  A: *A depth-4 tree outputs coarse, staircase probabilities tied to 10
  leaves; LR outputs smooth, continuously varying probabilities. Ranking
  applicants well across the whole threshold range (which ROC-AUC measures)
  favours smoothness: 0.852 vs 0.706.*

- **Q: Show me the confusion matrix reading.**
  A: *Rows = truth, columns = prediction. LR: 22 correctly rejected, 16
  wrongly approved (bank's expensive error), 1 wrongly rejected (customer's
  loss), 84 correctly approved. The 16 FPs are exactly what precision
  penalises.*

---

## 12. Project architecture

**Definition.** The end-to-end arrangement of components and data flow —
from raw CSV to a live app — so each stage has one clear job.

**Why we structure it this way.** Separation of *build* (train offline) from
*serve* (load in app) makes the app instant and reproducible; the pipeline
carries preprocessing inside the model artefact so build-time and app-time
data treatment are identical by construction.

```
data/loan_data.csv
        │  (offline, run once)
        ▼
src/train_and_evaluate.py  ──►  leakage-safe Pipeline + ColumnTransformer
        │                            ├─ model 1: Logistic Regression
        │                            └─ model 2: Decision Tree (depth 4)
        │                            evaluate → metrics table, confusion charts
        ▼
output/models/best_model.pkl (LR), second_model.pkl (DT)   ← pickled, fitted
output/plots/… + model_comparison.csv                      ← presentation assets
        │  (online, every request)
        ▼
src/app.py (Streamlit) ── raw form row ──► pipeline.predict/predict_proba
        ▼
      YES/NO decision + confidence shown to the user
```

**Small example.** Trace form → model: the app builds a 1-row DataFrame
(raw values e.g. "Graduate") → `pipeline.predict(input_df)` runs
imputer → scaler → encoder (all with train-fitted parameters) → LR →
`"Y"` with P = 0.80.

**Where it lives.** Everything above; the architecture diagram also appears
as slide 7 and the seven-step flow figure in `src/build_presentation.py`.

**Teacher questions & answers:**

- **Q: Why pickle rather than saving just weights?**
  A: *Because the fitted transformer statistics (medians, scaler means,
  encoder vocabulary) must travel with the estimator. Pickling the whole
  pipeline preserves the exact object that was fitted and verified.*

- **Q: Where could this architecture fail?**
  A: *Honest list: a pipeline trained with an older scikit-learn version
  may not unpickle with a newer one (we pin versions in
  `requirements.txt`); a category never seen in training maps to all-zeros
  (handled by `handle_unknown='ignore'` with information loss); and pickle
  loading of untrusted files is unsafe (ours are files we created).*

- **Q: What is not in this architecture?**
  A: *No model registry, no CI/CD, no monitoring — those belong to a
  production deployment, not a college project. We keep only what the
  problem needs.*

---

## 13. Code walkthrough

**Definition.** A guided tour of the actual code, function by function.

**(a) `src/train_and_evaluate.py` — the core script (offline training):**

- `CATEGORICAL` / `NUMERIC` constants — the 6 + 5 feature-column lists,
  defined once.
- `load_and_split()` — reads the CSV, drops `Loan_ID`, separates y, splits
  491/123 stratified with seed 42.
- `build_preprocessor()` — numeric block (median impute → StandardScaler),
  categorical block (mode impute → OneHotEncoder ignore-unknown),
  assembled in a `ColumnTransformer`.
- `build_models()` — two full pipelines sharing that preprocessor; differs
  only in the estimator: `LogisticRegression(max_iter=1000, random_state=42)`
  vs `DecisionTreeClassifier(max_depth=4, random_state=42)`.
- `evaluate_one()` — computes the 5 metrics from `predict` (class) and
  `predict_proba` (for ROC-AUC).
- `main()` — trains both, prints classification reports and the train vs
  test overfit check, writes `model_comparison.csv`, selects best by F1,
  pickles both pipelines, saves confusion-matrix and comparison charts.

**(b) `src/eda.py` — six chart functions + `main()` with `sns.set_theme` —
loads the CSV, adds a temporary `Approved_flag` (Y→1) purely for
rate/correlation maths (the CSV itself is untouched), writes the 6 PNGs to
`outputs/`.**

**(c) `src/app.py` — `load_models()` (`@st.cache_resource`: loads pickles
once, so no retrain at startup, and no re-read on each interaction) →
Streamlit form mirroring the 11 features → validation (non-negative
incomes, loan ≥ 1, term ≥ 12) → builds 1-row DataFrame in `FEATURE_ORDER`
→ `predict` / `predict_proba` per pipeline → success/error boxes, plus a
demo expander showing the raw row.**

**(d) `src/build_presentation.py` — offline generator for the 12-slide
PPTX, embedding the verified metric table and saved charts.**

**Small example.** The exact path of one app prediction (with real values
observed): defaults {Male, Yes, 0 deps, Graduate, No self-emp, 5000, 1500,
130k, 360, CH=1, Urban} → LR P(Y) = 0.80 → "APPROVED (80% confidence)".

**Teacher questions & answers:**

- **Q: Walk me through what happens on one click of Predict.**
  A: *The form submits → validation → 1-row DataFrame → both cached
  pipelines predict → `predict` returns 'Y'/'N' and `predict_proba[:, 1]`
  returns P(approved) → Streamlit renders green "APPROVED (80%
  confidence)" for LR and "APPROVED (77%)" for DT (the numbers we measured
  live).*

- **Q: Why is there a `data_preparation.py` you call "legacy"?**
  A: *It was the manual preprocessing stage of an earlier iteration (hand
  written encoding maps + a TotalIncome feature). The pipeline approach in
  `train_and_evaluate.py` replaced it — strictly better for leakage safety
  and deployment consistency. It now serves only as a teaching aid for
  what each step does.*

- **Q: What part of the code are you least sure about?**
  A: *(Honesty helps here.)* *The imputer/scaler internals we trust
  because scikit-learn handles them; our own logic, e.g. validation
  thresholds, are simple and explicit, and every metric we present is
  recomputable from `model_comparison.csv`.*

---

## 14. Streamlit application workflow

**Definition.** The runtime path from a browser form to a displayed
decision, using the fitted artefacts only (no training at runtime).

**Why this design.** Requirement-mapped: cached loads (no retrain), feature-
mirrored inputs, validation, clear Y/N display, probability display, and
graceful failure when model files are absent.

**Actual workflow (all verified live):**

1. `load_models()` reads both pickles once (`@st.cache_resource`); if
   missing, an error message renders and `st.stop()` prevents a crash.
2. The user fills the form (11 fields with sensible defaults).
3. On **Predict**, inputs are validated (negative incomes etc.).
4. The 1-row raw DataFrame goes to both pipelines; the in-pipeline
   imputer/scaler/encoder run on it; both models answer.
5. Result: green "APPROVED" or red "REJECTED" with confidence; an expander
   shows the raw row; a caption carries the provenance disclaimer ("public
   dataset… academic demonstration only — not an official HDFC Bank
   decision").

**Small example (measured live):** defaults → LR APPROVED 80%, DT APPROVED
77%. Credit history toggled to 0 → LR REJECTED (P(Y)=0.10), DT REJECTED
(P(Y)=0.02) — the decisive-feature demo in one toggle.

**Where it lives.** `src/app.py`; run with
`./.venv/bin/streamlit run src/app.py`.

**Teacher questions & answers:**

- **Q: How do you guarantee the app never retrains?**
  A: *The app contains no `fit` call at all — it only loads pickles, and
  the load is cached by Streamlit so it happens once per process, not per
  interaction.*

- **Q: What does `@st.cache_resource` do?**
  A: *It memoises the unpickled pipeline object in memory, so none of the
  11 form re-renders re-read the 2 pickle files, keeping predictions
  instant.*

- **Q: What happens with invalid or out-of-range input?**
  A: *Validation collects errors (e.g. income < 0) and displays them; the
  prediction step does not run until the inputs pass. Categorical inputs
  are selects, so impossible values can't even be typed.*

- **Q: Why send raw strings rather than encoding in the app?**
  A: *Because the saved pipeline owns the encoding (it was fit on train
  data). Sending raw values removes the classic bug of dev- and train-side
  encodings drifting apart.*

---

## 15. Model limitations and future improvements

**Definition.** An honest statement of what the model doesn't do well, and
what would fix it — evidence-based, not decorative.

**Our limitations (stated, not hidden):**

1. **Leniency.** LR recall on the rejected class is only 0.58 (22 of 38
   true rejections caught; 16 bad-loan approvals). As deployments go, that
   is the opposite of prudent for a bank's default risk.
2. **Small, public, non-bank data.** 614 rows of a practice dataset; not
   representative of any real lender's loan book.
3. **No threshold calibration.** We use the standard 0.5 cut-off; a bank
   would tune it deliberately to trade precision vs recall.
4. **No cross-validation yet** — a single stratified split (seeded, but
   still one split).
5. **Sensitive attributes present.** `Gender` and `Education` remain in the
   feature set (the dataset includes them); a real system must test for
   disparate impact and likely drop them.
6. **Not implemented (do not claim!):** Random Forest (rejected with
   evidence — no overfit gap to fix), k-fold CV, hyperparameter grids,
   SHAP-style explanations, API deployment.

**Future improvements (matching each limitation):**

1. Tune the decision threshold (e.g. raise the approval cut-off to 0.6 to
   trade recall for precision) — cheapest first step.
2. k-fold cross-validation for stable estimates.
3. Fairness audit on Gender/Education; drop if they drive predictions.
4. Larger and more recent data; monitoring for drift once deployed.
5. Optional: add Random Forest *if* a deeper-features version of the
   problem shows overfitting (our earlier decision rule).

**Small example.** A concrete failure case of our leniency: of the 38
actually-rejected test applications, LR correctly rejects 22 and wrongly
approves 16 (with a decent 0.96 precision *among its own rejections*, but a
0.58 recall *across all true rejections*). A bank plugging this model in
would let 16 of 38 defaulting profiles through the door — which is exactly
why §15 lists threshold calibration as the first future step.

**Where it lives.** These are discussed in the summary part of the summary
slides (slide 12) and the "future scope" bullets keep our stated intentions
aligned with reality. No hidden gaps.

**Teacher questions & answers:**

- **Q: How do you know your model isn't overfitting?**
  A: *We measure it: train vs test accuracy prints 0.800/0.862 for LR and
  0.809/0.846 for DT — negative gaps. Also note the depth-4 cap on the
  tree as the complexity budget.*

- **Q: Why didn't you add Random Forest? It usually improves trees.**
  A: *Our agreed precondition was measurable overfitting of the single
  tree; the measured gap is ≈ −0.04, so the averaging machinery of a forest
  would add runtime and interpretability cost for no measurable gain here.
  We kept the simplest model that meets the requirement — Occam's razor,
  and our report says it explicitly rather than quietly dropping an idea.*

- **Q: If you deployed this at an actual bank, what's the first thing that
  must change?**
  A: *Validate on real data from that bank, retrain/monitor on their loan
  book, apply fairness testing (sensitive attributes), and re-tune the
  decision threshold to their risk appetite. The demo app already says
  "not a lending decision".*

---

## One-page summary to recite

> "614 public loan applications, 11 features, target Loan_Status. We tested
> leakage-safe preprocessing (median/mode imputation, scaling, one-hot
> encoding) inside a sklearn Pipeline+ColumnTransformer fit strictly on the
> 491 training rows. We trained Logistic Regression and a depth-4 Decision
> Tree on identical preprocessing, and measured on a held-out stratified
> 123-row test set: LR = 0.862 accuracy, 0.840 precision, 0.988 recall,
> 0.908 F1, 0.852 ROC-AUC, confusion matrix [[22,16],[1,84]]; DT = 0.846,
> 0.824, 0.988, 0.898, 0.706 with [[20,18],[1,84]]. Selected Logistic
> Regression by F1. Known limitation: leniency (recall 0.58 on rejections);
> Random Forest was deliberately not added (no overfitting measured). The
> Streamlit app loads the saved pipelines, sends raw form values, and
> correctly flips to rejection when credit history is removed — verified
> live."
