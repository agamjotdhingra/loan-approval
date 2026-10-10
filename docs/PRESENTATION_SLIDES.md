# Presentation — AI-Based Loan Approval Prediction System
### 11 slides · ready to paste into PowerPoint / Google Slides

Design tip: plain white background, dark text, one screenshot of a chart per
slide (from `output/plots/`). Speak the "Say" line; don't read the slide.

---

## Slide 1 — Title

> ### AI-Based Loan Approval Prediction System
> Predicting loan approval using supervised machine learning
>
> Student: **Your Name** · Class/Roll No: **...** · Guide: **...** · College: **...**

**Say:** "My project predicts whether a loan application is likely to be
approved or rejected, using two supervised machine-learning models and a
public dataset."

---

## Slide 2 — Problem statement

> Banks receive thousands of loan applications. Manual screening is
> slow, inconsistent and subjective.
>
> **Goal:** learn from historical applications to automatically predict
> Approved / Rejected — and explain *why*.
>
> - Binary classification (2 classes)
> - Supervised learning (labelled historical data)
> - Requirement: simple, explainable, no deep learning

**Say:** "This is a classic supervised binary-classification problem. The
constraint I set myself was explainability — so no deep learning, only
interpretable classical models."

---

## Slide 3 — The dataset

> **Public dataset:** Analytics Vidhya "Loan Prediction" practice problem
> (also mirrored on Kaggle) — 614 loan applications, 13 columns.
>
> - Target: `Loan_Status` (Y/N → 1/0)
> - Features: income, loan amount, credit history, education,
>   employment, property area...
> - 7 columns had missing values (e.g. Credit_History: 50 rows)
> - Class balance: ≈ 69 % approved / 31 % rejected

**Say:** "I deliberately used a well-known public dataset so the results are
reproducible. It is anonymised, sample data from a fictional housing-finance
company — not any real bank's records."

---

## Slide 4 — Data understanding (EDA)

> *(insert `01_class_distribution.png`)*
> *(insert `02_credit_history.png`)*
>
> - Approvals ≈ 2.2× rejections → class imbalance
> - Approval rate **jumps from ≈ 8 % to ≈ 79 %** when the applicant has a
>   credit history → strongest single signal

**Say:** "Exploratory analysis showed the class imbalance — which later
justified stratified splitting and precision/recall metrics — and one
dominant pattern: credit history drives the decision."

---

## Slide 5 — Data preparation

> Pipeline (src/data_preparation.py):
>
> 1. **Missing values** → median (numeric), mode (categorical)
> 2. **Encoding** → binary maps (Y/N), one-hot for Property_Area
> 3. **Feature engineering** → TotalIncome = Applicant + Coapplicant income
> 4. **Split** → 80 % train / 20 % test, stratified, random_state = 42

**Say:** "Four simple, standard steps. Median is robust to income outliers;
one-hot encoding avoids inventing a fake order between property areas; the
stratified split keeps the 69/31 ratio identical on both sides."

---

## Slide 6 — The two models

> ### Logistic Regression
> weighted sum of features → sigmoid → probability of approval
> *No. of parameters: 14 weights + intercept; trained with gradient descent*
>
> ### Decision Tree (max_depth = 4)
> learns a flowchart of yes/no questions (Gini impurity)
>
> Random Forest considered but **not needed** — no overfitting measured.

**Say:** "LR is the linear baseline; the depth-4 tree is the rule-based
counterpart. I capped tree depth to prevent memorisation. I only add Random
Forest if the tree overfits — measured gap was ≈ 2 %, so I didn't."

---

## Slide 7 — Evaluation methodology

> Both models scored on the **same unseen 123-row test set**, metrics
> computed on test only:
>
> | Metric | Question it answers |
> |---|---|
> | Accuracy | overall correctness — misleading alone at 69/31 |
> | Precision | when we approve, how often are we right? |
> | Recall | of all who deserve approval, how many did we catch? |
> | F1 | balanced precision + recall |
> | ROC-AUC | ranking quality, threshold-independent |

**Say:** "With a 69/31 imbalance, a lazy 'always approve' model scores 69 %
accuracy while being useless — so I compared five metrics, not one."

---

## Slide 8 — Results

> *(insert `05_model_comparison.png`)*
>
> | Metric | Logistic Regression | Decision Tree |
> |---|---|---|
> | Accuracy | **0.862** | 0.846 |
> | Precision | **0.840** | 0.824 |
> | Recall | **0.988** | 0.988 |
> | F1-score | **0.908** | 0.898 |
> | ROC-AUC | **0.852** | 0.706 |
>
> Overfit check: train vs test gap ≈ −5 % (LR), −2 % (DT) → both generalise

**Say:** "Logistic Regression wins on every metric, most clearly on ROC-AUC.
Both models generalise — the train/test gaps are small and actually
negative, which is why Random Forest stayed out."

---

## Slide 9 — Confusion matrices

> *(insert `04_confusion_matrices.png`)*
>
> LR on 123 test applications: 22 rejected correctly (TN), 2 bad loans approved
> (FP), 2 good customers rejected (FN), 83 approved correctly (TP).
> DT: 21 / 3 / 17 / 82 — note DT misses more true rejections (17 FN).
>
> The two error types have **very different business costs**.

**Say:** "Reading the matrix: the diagonal is what we got right. For the
bank, the false positives — approving loans that will default — are the
expensive errors; that's exactly what precision measures."

---

## Slide 10 — Live application demo

> Streamlit app (src/app.py):
>
> - form mirrors the training features exactly
> - both models predict side-by-side with confidence %
> - encoded input viewer → total transparency
>
> *(insert app screenshot: default form → APPROVED ~79 %;
> then switch Credit_History to 0 → REJECTED)*

**Say:** *(live demo)* "Defaults give approval at 79 %. One toggle — no
credit history — flips both models to rejection. You're seeing the
strongest feature act in real time."

---

## Slide 11 — Conclusion & future work

> **Conclusions**
> - Simple, fully explainable pipeline: 85 % accuracy, F1 0.90
> - Credit history dominates the decision — consistent with banking practice
> - Logistic Regression beat the Decision Tree on every metric
>
> **Future work**
> - k-fold cross-validation for more stable estimates
> - decision-threshold tuning to trade precision vs recall deliberately
> - fairness audit on protected attributes (gender, education)

**Say:** "The project delivers an end-to-end, explainable loan-screening
prototype. Next steps would be cross-validation, threshold calibration for
bank risk-appetite, and a fairness audit before any real-world use."

---

## Speaker notes you're likely to be quizzed on

- **"Why not deep learning?"** → 614 rows is far too small; nets would
  overfit and aren't interpretable. Classical models are the right tool.
- **"Why stratify the split?"** → keeps the 69/31 ratio identical in train
  and test; otherwise metrics depend on split luck.
- **"Why is the tree's ROC-AUC lower?"** → a depth-4 tree produces coarse
  staircase probabilities; LR's smooth probabilities rank applicants better.
- **"Where does the data come from?"** → Analytics Vidhya public practice
  dataset (fictional 'Dream Housing Finance'); **not** a real bank.
