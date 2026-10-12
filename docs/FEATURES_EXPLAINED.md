# Features Used in the Loan Approval Model

The model predicts **`Loan_Status`** (Y = approved, N = rejected). It uses
**11 raw input columns** from the dataset, plus **5 engineered columns**
built from them in [`src/features.py`](../src/features.py).
`Loan_ID` is dropped. It is a unique ID, so it carries no signal, and a
model could only memorise it.

Every number below comes from running `src/eda.py` and
`src/train_and_evaluate.py` on the 614-row dataset.

---

## 1. Raw input features (11)

| # | Feature | Type | What it means | Missing | What we learned |
|---|---|---|---|---|---|
| 1 | **Credit_History** | binary (1/0) | 1 = past loans repaid on time | 50 | **The strongest signal by far.** 80% approved with a history vs 8% without (10×). Top in every importance chart. |
| 2 | **ApplicantIncome** | numeric (₹/month) | Main applicant's monthly income | 0 | Very right-skewed (median ₹3,812, mean ₹5,403, max ₹81,000). On its own it barely separates approved from rejected. |
| 3 | **CoapplicantIncome** | numeric (₹/month) | Income of a co-applicant, e.g. a spouse (0 if none) | 0 | Second most important input for the Random Forest. Household income matters. |
| 4 | **LoanAmount** | numeric (₹ thousands) | Amount requested | 22 | Median 128k, max 700k. Only meaningful when compared with income (see engineered features). |
| 5 | **Loan_Amount_Term** | numeric (months) | Repayment period (mostly 360 = 30 years) | 14 | Almost always 360, so little information alone. Used to compute the EMI. |
| 6 | **Property_Area** | category (Urban / Semiurban / Rural) | Where the property is | 0 | Mild effect: Semiurban 77% > Urban 66% > Rural 61% approval. |
| 7 | **Married** | binary | Applicant married? | 3 | Weak signal. |
| 8 | **Dependents** | category (0 / 1 / 2 / 3+) | Number of people supported | 15 | Weak signal; treated as a category because of "3+". |
| 9 | **Education** | binary (Graduate / Not Graduate) | Education level | 0 | Weak signal. |
| 10 | **Self_Employed** | binary | Self-employed vs salaried | 32 | Weak signal. |
| 11 | **Gender** | binary | Applicant gender | 13 | ~zero importance. A **sensitive attribute**, kept only for the study (see fairness note). |

---

## 2. Engineered features (5)

A loan officer asks: *"Can this household afford the monthly instalment?"*
The raw columns don't answer that directly, so we compute it:

| Feature | Formula | Why it helps |
|---|---|---|
| **TotalIncome** | ApplicantIncome + CoapplicantIncome | The bank lends to the *household*, not one person. |
| **LoanAmountLog** | log(1 + LoanAmount) | Compresses the long right tail so a few huge loans don't dominate a linear model. |
| **EMI** | LoanAmount × 1000 ÷ Loan_Amount_Term | Approximate monthly instalment in ₹ (interest ignored for simplicity). |
| **BalanceIncome** | TotalIncome − EMI | Money left each month after paying the EMI: a direct affordability measure. |
| **LoanToIncome** | LoanAmount × 1000 ÷ TotalIncome | How many months of income the loan equals. Higher means riskier. |

**Evidence that they help:**
- In the Random Forest's built-in importance, **LoanToIncome (0.078) and
  BalanceIncome (0.061) rank 2nd and 3rd**, right after Credit_History (0.415).
- Logistic Regression's test ROC-AUC rose from **0.852 → 0.870** once these
  features were added. Everything else was unchanged.

**Why this is leakage-safe:** the formulas use only the applicant's own row
and learn nothing from the dataset. They run *inside* the scikit-learn
pipeline as a `FunctionTransformer`, before imputation and scaling.

---

## 3. How the features are processed (inside the pipeline)

```
raw row (11 columns)
   └─ FunctionTransformer(add_engineered_features)   → 16 columns
        └─ ColumnTransformer
             ├─ 10 numeric     → SimpleImputer(median) → StandardScaler
             └─ 6 categorical  → SimpleImputer(mode)   → OneHotEncoder
                  └─ model (LR / Decision Tree / Random Forest)
```

- **Median** for numbers, because incomes are skewed (skew = 6.54).
- **Mode** for categories, because there is no "average gender".
- **One-hot encoding**, so "Urban" is never treated as "bigger" than "Rural".
- **Scaling**, because Logistic Regression needs features on a similar scale.
- All of these are **fitted on the 491 training rows only**.

---

## 4. Which features actually matter? (measured)

**Permutation importance** (`output/plots/06_feature_importance.png`):
shuffle one column of the test set and measure how much ROC-AUC drops.
Results for the selected Random Forest:

| Feature | Drop in ROC-AUC |
|---|---|
| Credit_History | **0.292** |
| CoapplicantIncome | 0.079 |
| ApplicantIncome | 0.052 |
| LoanAmount | 0.041 |
| Property_Area | 0.005 |
| Married, Self_Employed, Gender, Education, Dependents, Loan_Amount_Term | ≈ 0 |

**Conclusion:** the model decides on **credit history + affordability
(income vs loan size)**. The personal/demographic columns contribute almost
nothing.

---

## 5. Fairness note (good viva point)

Gender, Married and Dependents are sensitive attributes. Real lenders are
often legally barred from using some of them. Our importance analysis shows
they add ≈ 0 to the predictions, so a natural next step is to **drop them**
and confirm the scores stay the same.
