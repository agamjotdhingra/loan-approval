# AI-Based Loan Approval Prediction System

A supervised machine-learning project that predicts whether a loan
application will be **Approved** or **Rejected**. It compares Logistic
Regression, a Decision Tree and a Random Forest, and includes an interactive
Streamlit app.

> Dataset: public [Analytics Vidhya "Loan Prediction" practice problem]
> (https://www.analyticsvidhya.com/blog/2021/11/loan-prediction-practice-problem/)
> (614 applications, also mirrored on Kaggle). Fictional "Dream Housing
> Finance" sample data — **not** any real bank's records.

## Features

11 raw inputs plus 5 engineered **affordability** features
(`TotalIncome`, `EMI`, `BalanceIncome`, `LoanToIncome`, `LoanAmountLog`).
Every feature is explained in
[docs/FEATURES_EXPLAINED.md](docs/FEATURES_EXPLAINED.md).

## Results (test set, 123 applications)

| Metric | Logistic Regression | Decision Tree (depth 4) | Random Forest |
|---|---|---|---|
| Accuracy | 0.862 | 0.862 | **0.878** |
| Precision | 0.840 | **0.862** | 0.857 |
| Recall | **0.988** | 0.953 | **0.988** |
| F1-score | 0.908 | 0.905 | **0.918** |
| ROC-AUC | **0.870** | 0.802 | 0.862 |
| 5-fold CV F1 (train) | 0.864 ± 0.026 | 0.863 ± 0.030 | **0.867 ± 0.028** |

The model is chosen by **5-fold cross-validated F1 on the training rows**
(Random Forest), so the test set never influences the choice. The three
models are close: their CV scores are within one standard deviation of each
other. Credit history is by far the most important feature, followed by the
income/loan-size (affordability) features.

All preprocessing (feature engineering, imputation, scaling, one-hot
encoding) lives inside scikit-learn `Pipeline` + `ColumnTransformer` objects.
These are **fitted on the 80% training rows only** and then simply applied to
the test rows and to live app input.

## Project structure

```
├── data/loan_data.csv            the public dataset (614 rows)
├── src/
│   ├── eda.py                    6 exploratory charts -> outputs/
│   ├── features.py               the 5 engineered features
│   ├── train_and_evaluate.py     3 models, CV + test metrics, plots, pickles
│   ├── app.py                    Streamlit app (predict with all 3 models)
│   └── build_presentation.py     generates docs/Loan_Approval_Presentation.pptx
├── output/
│   ├── models/                   one .pkl pipeline per model
│   └── plots/                    confusion matrices, comparison, feature importance
├── docs/
│   ├── Loan_Approval_Presentation.pptx   14-slide deck with speaker notes
│   └── FEATURES_EXPLAINED.md     every feature explained
└── requirements.txt
```

## How to run

```bash
# 1. one-time setup (creates a virtual environment)
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip install -r requirements.txt

# 2. train + evaluate (also generates all charts and saved models)
./.venv/bin/python src/train_and_evaluate.py

# 3. (optional) rebuild the presentation from the latest results
./.venv/bin/python src/build_presentation.py

# 4. run the app
./.venv/bin/streamlit run src/app.py          # opens at http://localhost:8501
```

`src/data_preparation.py` can also be run on its own to see the cleaning/
encoding steps printed step by step.

## Learning documents

- **[docs/KNOWLEDGE_DOCUMENT.md](docs/KNOWLEDGE_DOCUMENT.md)** — the complete
  study guide: every component (dataset, encodings, models, metrics, pipeline,
  pickle, app) with *What / Why / How / Where in code / what to say in the
  viva*, plus 15 likely viva questions with model answers.
- **[docs/PRESENTATION_SLIDES.md](docs/PRESENTATION_SLIDES.md)** — 11 slides
  with speaker notes.
