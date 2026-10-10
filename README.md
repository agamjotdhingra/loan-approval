# AI-Based Loan Approval Prediction System

A supervised machine-learning project that predicts whether a loan
application will be **Approved** or **Rejected**, with a comparison of
Logistic Regression vs Decision Tree and an interactive Streamlit app.

> Dataset: public [Analytics Vidhya "Loan Prediction" practice problem]
> (https://www.analyticsvidhya.com/blog/2021/11/loan-prediction-practice-problem/)
> (614 applications, also mirrored on Kaggle). Fictional "Dream Housing
> Finance" sample data — **not** any real bank's records.

## Results (test set, 123 applications)

| Metric | Logistic Regression | Decision Tree (depth 4) |
|---|---|---|
| Accuracy | **0.862** | 0.846 |
| Precision | **0.840** | 0.824 |
| Recall | **0.988** | 0.988 |
| F1-score | **0.908** | 0.898 |
| ROC-AUC | **0.852** | 0.706 |

Logistic Regression wins on every metric; both models generalise (no
overfitting: train - test gap is negative for both), so Random Forest was
deliberately **not** added.

Preprocessing is leakage-safe: missing-value imputation, scaling and one-hot
encoding live inside scikit-learn `Pipeline` + `ColumnTransformer` objects
that are **fitted on the 80% training rows only** and merely applied to the
test rows and to live app input.

## Project structure

```
├── data/loan_data.csv            the public dataset (614 rows)
├── src/
│   ├── data_preparation.py       load → clean → encode → engineer → split
│   ├── train_and_evaluate.py     train 2 models, compute 6 metrics, make 5 plots, save pickles
│   └── app.py                    Streamlit app (predict with both models)
├── output/
│   ├── models/                   best_model.pkl (LR), second_model.pkl (DT)
│   └── plots/                    EDA charts, confusion matrices, comparison chart
├── docs/
│   ├── KNOWLEDGE_DOCUMENT.md     every component explained (What/Why/How/Where/viva answers)
│   └── PRESENTATION_SLIDES.md    11 slides, ready to paste
└── requirements.txt
```

## How to run

```bash
# 1. one-time setup (creates a virtual environment)
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip install -r requirements.txt

# 2. train + evaluate (also generates all charts and saved models)
./.venv/bin/python src/train_and_evaluate.py

# 3. run the app
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
