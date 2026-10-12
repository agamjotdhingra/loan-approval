"""
train_and_evaluate.py - THE MACHINE LEARNING STAGE
==================================================
PART A: preprocessing (leakage-safe: fitted on TRAIN data only)
PART B: train Logistic Regression + Decision Tree + Random Forest
        (all three see the 11 raw features + 5 engineered ones, features.py)
PART C: 5-fold cross-validation on the training rows picks the winner,
        then every model is scored once on the held-out test set
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

from features import ENGINEERED, add_engineered_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "loan_data.csv")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "output", "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "output", "models")

# ===========================================================================
# PART A - PREPROCESSING
# ===========================================================================

# Column groups (defined ONCE here, reused everywhere below).
# 6 categorical + 5 raw numeric + 5 engineered numeric (see features.py).
CATEGORICAL = ["Gender", "Married", "Dependents", "Education",
               "Self_Employed", "Property_Area"]
RAW_NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
               "Loan_Amount_Term", "Credit_History"]
NUMERIC = RAW_NUMERIC + ENGINEERED
RAW_FEATURES = CATEGORICAL + RAW_NUMERIC


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
def make_pipeline(estimator) -> Pipeline:
    """feature engineering -> preprocessing -> estimator, as ONE object.

    The engineering step is a stateless FunctionTransformer (it learns
    nothing), so placing it inside the pipeline is leakage-safe and means
    the app can keep sending plain raw rows.
    """
    return Pipeline([
        ("engineer", FunctionTransformer(add_engineered_features)),
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])


def build_models() -> "dict[str, Pipeline]":
    """All pipelines share the SAME feature engineering and preprocessing,
    differing only in the final estimator - a fair head-to-head.

    LogisticRegression(max_iter=1000):
      max_iter raises the gradient-descent step budget; 1000 converges
      comfortably (no warning is raised).

    DecisionTreeClassifier(max_depth=4):
      depth is the tree's complexity budget. Unbounded trees keep splitting
      until leaves are pure = memorising the training rows (overfitting).
      Four questions deep also stays human-readable.

    RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5):
      an ENSEMBLE of 300 trees, each grown on a random bootstrap sample of
      rows and a random subset of features; the forest votes. Averaging
      many different trees cancels out their individual mistakes, which
      usually beats a single tree. depth/leaf limits keep each tree small.
    """
    return {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=1000, random_state=42)),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(max_depth=4, random_state=42)),
        "Random Forest": make_pipeline(
            RandomForestClassifier(n_estimators=300, max_depth=6,
                                   min_samples_leaf=5, random_state=42)),
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


def cross_validate_one(model, X_train, y_train, folds: int = 5) -> dict:
    """5-fold stratified cross-validation on the TRAINING rows only.

    WHY? One 123-row test set is a single, slightly lucky-or-unlucky draw.
    CV splits the 491 training rows into 5 parts, trains on 4 and scores
    on the 5th, five times over. The MEAN is a steadier estimate and the
    STD shows how much the score wobbles. The test set stays untouched.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_validate(model, X_train, (y_train == "Y").astype(int), cv=cv,
                            scoring=["accuracy", "f1", "roc_auc"])
    return {
        "CV Accuracy": scores["test_accuracy"].mean(),
        "CV F1": scores["test_f1"].mean(),
        "CV F1 std": scores["test_f1"].std(),
        "CV ROC-AUC": scores["test_roc_auc"].mean(),
    }


def plot_feature_importance(model, name, X_test, y_test) -> pd.Series:
    """Permutation importance of each RAW input column for one model.

    HOW: shuffle one column of the test set, re-score the model, and see
    how much ROC-AUC drops. A big drop = the model relied on that column.
    It works for ANY model and is measured in the units of the metric.
    (Shuffling ApplicantIncome also breaks TotalIncome, EMI... because the
    engineered columns are rebuilt inside the pipeline - so each bar is
    the total value of that raw column, direct + via engineered features.)
    """
    result = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                    n_repeats=30, random_state=42)
    importance = pd.Series(result.importances_mean,
                           index=X_test.columns).sort_values()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#1565c0" if v > 0 else "#9e9e9e" for v in importance.values]
    ax.barh(importance.index, importance.values, color=colors)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Drop in test ROC-AUC when the column is shuffled")
    ax.set_title(f"Which inputs matter? (permutation importance, {name})")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "06_feature_importance.png"), dpi=150)
    plt.close(fig)
    return importance.sort_values(ascending=False)


def plot_forest_importance(forest_pipeline) -> pd.Series:
    """Built-in (Gini) importance of the Random Forest, per MODEL column -
    this view shows the engineered features individually."""
    names = forest_pipeline.named_steps["preprocess"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]       # drop 'num__' / 'cat__'
    importance = pd.Series(
        forest_pipeline.named_steps["model"].feature_importances_, index=names
    ).sort_values().tail(12)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#ef6c00" if n in ENGINEERED else "#0e7c7b" for n in importance.index]
    ax.barh(importance.index, importance.values, color=colors)
    ax.set_xlabel("Share of total impurity reduction")
    ax.set_title("Random Forest: top 12 model columns (orange = engineered)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "07_forest_importance.png"), dpi=150)
    plt.close(fig)
    return importance.sort_values(ascending=False)


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---------------- PART A ----------------
    X_train, X_test, y_train, y_test = load_and_split()

    # ---------------- PART B ----------------
    models = build_models()
    results, confusion = {}, {}

    for name, pipeline in models.items():
        print(f"\n=== Training {name} ===")
        cv_scores = cross_validate_one(pipeline, X_train, y_train)
        print(f"5-fold CV on train: F1 = {cv_scores['CV F1']:.3f} "
              f"± {cv_scores['CV F1 std']:.3f}")
        pipeline.fit(X_train, y_train)            # fit = learn, on TRAIN only
        results[name] = {**evaluate_one(name, pipeline, X_test, y_test), **cv_scores}
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

    # model selection uses the CROSS-VALIDATED F1 (training rows only), so
    # the test set is never used to choose - it only reports. F1 balances
    # precision and recall, unlike accuracy which a 69/31 imbalance inflates.
    best_name = comparison["CV F1"].idxmax()
    print(f"\n=> Selected model (highest 5-fold CV F1): {best_name}")

    # save EVERY pipeline so the Streamlit app can compare them side by side
    for name, pipeline in models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        with open(os.path.join(MODELS_DIR, filename), "wb") as f:
            pickle.dump({"model_name": name, "model": pipeline,
                         "is_best": name == best_name}, f)
        print(f"Saved {name} -> {filename}")

    importance = plot_feature_importance(models[best_name], best_name, X_test, y_test)
    print(f"\n=== Permutation importance ({best_name}, drop in ROC-AUC) ===")
    print(importance.round(3).to_string())
    forest = plot_forest_importance(models["Random Forest"])
    print("\n=== Random Forest built-in importance (top 12) ===")
    print(forest.round(3).to_string())

    # confusion matrix heatmaps (one panel per model)
    fig, axes = plt.subplots(1, len(confusion), figsize=(12, 3.8))
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
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (name, row) in enumerate(results.items()):
        bars = ax.bar(x + (i - 1) * width, [row[m] for m in metrics], width, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model comparison on the test set")
    ax.legend(loc="upper center", ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "05_model_comparison.png"), dpi=150)
    plt.close(fig)
    print("\nCharts refreshed: 04_confusion_matrices.png, 05_model_comparison.png, "
          "06_feature_importance.png, 07_forest_importance.png")


if __name__ == "__main__":
    main()
