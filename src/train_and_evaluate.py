"""
train_and_evaluate.py - THE MACHINE LEARNING STAGE
==================================================
PART A: preprocessing (leakage-safe: fitted on TRAIN data only)
PART B: train Logistic Regression + Decision Tree
PART C: evaluate both on the held-out test set, pick the real winner
        (no invented numbers - everything below is computed live)

Run from the project root:
    python src/train_and_evaluate.py

THE ONE BIG IDEA (say this in the viva):
Every preprocessing step (imputing, scaling, encoding) is wrapped in a
scikit-learn Pipeline, so it is FIT on the 80% training rows and then
APPLIED unchanged to the 20% test rows. Nothing from the test set (means,
medians, category lists...) ever flows backwards into training. That is
what "preventing data leakage" means. Unlike a manual approach, it is
impossible to leak even by accident, because the transformer only ever
sees training data when .fit() is called.
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "loan_data.csv")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "output", "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "output", "models")

# ===========================================================================
# PART A - PREPROCESSING
# ===========================================================================

# Column groups (defined ONCE here, reused everywhere below).
# Kept in sync with the dataset inspection: 6 categorical + 5 numeric.
CATEGORICAL = ["Gender", "Married", "Dependents", "Education",
               "Self_Employed", "Property_Area"]
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
           "Loan_Amount_Term", "Credit_History"]


def load_and_split(test_size: float = 0.20, seed: int = 42):
    """PART A steps 1-2, 5-7: load, drop the ID, split 80/20 stratified.

    WHY stratify=y? The target is 69%/31%. Stratifying keeps that exact
    ratio in both the train and the test set - our metric numbers then
    don't depend on split luck.
    WHY a fixed random_state? It makes the split reproducible: rerunning
    this script (and the app demo) always yields the same 123 test rows,
    so results can be quoted as stable facts.
    """
    df = pd.read_csv(DATA_PATH)

    # --- step 1: X (features) and y (target) ---------------------------
    y = df["Loan_Status"]                     # what we predict: Y/N
    # --- step 2: drop the identifier -----------------------------------
    # Loan_ID is unique per row: a model could only 'learn' from it by
    # memorising rows (= overfitting), so it must never be a feature.
    X = df.drop(columns=["Loan_ID", "Loan_Status"])

    # --- steps 5-7: 80:20 split, fixed seed, stratified ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,       # 20% held out for honest evaluation
        random_state=seed,         # reproducible split
        stratify=y,                # keep 69/31 ratio in both halves
    )
    print(f"[SPLIT] train={len(X_train)} rows, test={len(X_test)} rows, "
          f"test approval rate={(y_test == 'Y').mean():.1%} "
          f"(full data: {(y == 'Y').mean():.1%})")
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """PART A steps 3-4, 8-9: the leakage-safe preprocessing block.

    ColumnTransformer routes each column to its own mini-pipeline:

      NUMERIC columns   -> SimpleImputer(median) -> StandardScaler
      CATEGORICAL cols  -> SimpleImputer(mode)   -> OneHotEncoder

    WHY median for numeric? (from our own EDA: income skew = 6.54)
    The income distribution is heavily right-skewed - a few applicants
    earn 81,000 while the typical applicant earns ~3,800. The mean gets
    dragged up by those extremes; the median stays representative.
    WHY mode for categorical? There is no 'average gender'. The most
    frequent category is the honest default guess.

    WHY OneHotEncoder for Dependents too? It is text ('0','1','2','3+').
    One-hot keeps '3+' from pretending to be 'twice as much' as '1+' -
    categories stay unordered and equal.

    WHY handle_unknown='ignore'? If the app someday receives a category
    never seen in training, the encoder emits all-zeros for that row
    rather than crashing. (Test categories seen below are all known.)

    LEAKAGE SAFETY (step 8): not one of these .fit() calls runs on test
    data. We only fit the preprocessor as part of pipeline.fit(X_train).
    fitted on train data only. When pipeline.predict(X_test) runs, the
    already-fitted transformers merely TRANSFORM the test rows.
    """
    numeric_block = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),          # needed by Logistic Regression
    ])
    categorical_block = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_block, NUMERIC),
            ("cat", categorical_block, CATEGORICAL),
        ]
    )


# ===========================================================================
# PART B - THE TWO MODELS (each wrapped in its own full pipeline)
# ===========================================================================
def build_models() -> "dict[str, Pipeline]":
    """Both pipelines start with the SAME preprocessing, differing only in
    the final estimator - a fair head-to-head.

    LogisticRegression(max_iter=1000):
      max_iter raises the gradient-descent step budget; 1000 converges
      comfortably (no warning is raised).

    DecisionTreeClassifier(max_depth=4):
      depth is the tree's complexity budget. Unbounded trees keep splitting
      until leaves are pure = memorising the training rows (overfitting).
      Four questions deep also stays human-readable.
    """
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ]),
    }


# ===========================================================================
# PART C - EVALUATION (every number below is computed, never assumed)
# ===========================================================================

def evaluate_one(name: str, model, X_test, y_test) -> dict:
    """Score ONE fitted model on the unseen test rows.

    predict()        -> the 0/1 decision used for accuracy/precision/recall/F1
    predict_proba()  -> the probability of approval, needed for ROC-AUC
                        (ROC-AUC judges the RANKING of applicants, which is
                        threshold-independent, so it uses probabilities)
    Each metric in one line (for the viva):
      Accuracy  - share of all 123 test cases decided correctly.
      Precision - of the ones we approved, how many were truly safe?
                  (protects the bank from bad loans: punishes FPs)
      Recall    - of all truly approvable loans, how many did we catch?
                  (protects the customer: punishes FNs)
      F1        - harmonic mean: only high if BOTH are high.
      ROC-AUC   - ranking quality across all thresholds; 0.5=coin flip, 1=perfect.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="Y"),
        "Recall":    recall_score(y_test, y_pred, pos_label="Y"),
        "F1-score":  f1_score(y_test, y_pred, pos_label="Y"),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),   # requires proba
    }


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---------------- PART A ----------------
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = build_preprocessor()

    # ---------------- PART B ----------------
    models = build_models()
    results, confusion = {}, {}

    for name, pipeline in models.items():
        print(f"\n=== Training {name} ===")
        pipeline.fit(X_train, y_train)            # fit = learn, on TRAIN only
        results[name] = evaluate_one(name, pipeline, X_test, y_test)
        confusion[name] = confusion_matrix(y_test, y_pred=pipeline.predict(X_test), labels=["N", "Y"])

        print(classification_report(y_test, pipeline.predict(X_test),
                                    labels=["N", "Y"], target_names=["Rejected (N)", "Approved (Y)"]))
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = results[name]["Accuracy"]
        print(f"Train accuracy = {train_acc:.3f} | Test accuracy = {test_acc:.3f} "
              f"| overfit gap = {train_acc - test_acc:+.3f}")

    # ---------------- PART C ----------------
    # comparison table, rounded for display (full precision under the hood)
    comparison = pd.DataFrame(results).T
    print("\n=== COMPARISON TABLE (computed on the test set) ===")
    print(comparison.round(3).to_string())
    comparison.round(3).to_csv(os.path.join(PLOTS_DIR, "model_comparison.csv"))

    # model selection by ACTUAL results: highest F1 wins (F1 balances
    # precision and recall, unlike accuracy which a 69/31 imbalance inflates)
    best_name = comparison["F1-score"].idxmax()
    print(f"\n=> Selected model (highest test F1): {best_name}")

    # save BOTH pipelines so the Streamlit app can load and compare them
    for filename, name in [("best_model.pkl", best_name),
                           ("second_model.pkl",
                            next(n for n in results if n != best_name))]:
        with open(os.path.join(MODELS_DIR, filename), "wb") as f:
            pickle.dump({"model_name": name, "model": models[name]}, f)
        print(f"Saved {name} -> {filename}")

    # confusion matrix heatmaps (one panel per model)
    fig, axes = plt.subplots(1, len(confusion), figsize=(9, 3.8))
    for ax, (name, cm) in zip(np.atleast_1d(axes), confusion.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred N", "Pred Y"],
                    yticklabels=["True N", "True Y"])
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "04_confusion_matrices.png"), dpi=150)
    plt.close(fig)

    # grouped metric bars
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, row) in enumerate(results.items()):
        bars = ax.bar(x + (i - 0.5) * width, [row[m] for m in metrics], width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "05_model_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nCharts refreshed: 04_confusion_matrices.png, 05_model_comparison.png")


if __name__ == "__main__":
    main()
