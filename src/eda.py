"""
eda.py - Exploratory Data Analysis on the loan dataset
======================================================
PURPOSE: before building any model, we LOOK at the data. Charts answer
questions like "which features separate approved from rejected loans?"

Run from the project root:
    python src/eda.py

All 6 charts are saved into the folder  outputs/  and key numbers are
printed so every observation we quote comes from the real dataset.
NOTHING here trains a model - this is pure looking.

Beginner note: each chart is a grouped set of matplotlib/seaborn calls:
    plt.subplots()  -> create a blank canvas (figure) with an axes area
    sns./ax.        -> draw or decorate (bars, labels, titles)
    fig.savefig()   -> write the finished image to disk as a PNG
"""

import os

import matplotlib
matplotlib.use("Agg")            # render to files, not a screen window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "loan_data.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# seaborn's built-in look: subtle grid + faster default styles
sns.set_theme(style="whitegrid")


def load_and_prepare():
    """Load the CSV and add two tiny helper columns used by the charts.

    We make a TEMPORARY numeric copy of the target (Y->1, N->0) only so we
    can compute approval rates and correlations. The CSV itself is untouched.
    """
    df = pd.read_csv(DATA_PATH)

    # For counting approvals we only need 1/0 values:
    df["Approved_flag"] = (df["Loan_Status"] == "Y").astype(int)

    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns "
          f"({(df['Loan_Status']=='Y').mean():.1%} approved)")
    return df


# ===========================================================================
# CHART 1: approval vs rejection counts
# ===========================================================================
def chart_1_class_counts(df):
    counts = df["Loan_Status"].value_counts()          # Y: 422, N: 192 (real)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Approved (Y)", "Rejected (N)"], counts.values,
                  color=["#2e7d32", "#c62828"], width=0.55)
    # put the exact number above every bar - avoids guessing from height
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 6, str(value),
                ha="center", fontsize=12, fontweight="bold")

    ax.set_title("Loan Approval vs Rejection (614 applications)", fontsize=13)
    ax.set_xlabel("Loan status (the thing we want to predict)")
    ax.set_ylabel("Number of applications")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_approval_vs_rejection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    n_y, n_n = counts.get("Y", 0), counts.get("N", 0)
    print(f"[1] Approved={n_y} ({n_y/(n_y+n_n):.1%}), Rejected={n_n} "
          f"({n_n/(n_y+n_n):.1%})")
    return path


# ===========================================================================
# CHART 2: applicant income distribution (histogram)
# ===========================================================================
def chart_2_income_distribution(df):
    incomes = df["ApplicantIncome"].dropna()

    fig, (hist_ax, box_ax) = plt.subplots(1, 2, figsize=(11, 4))

    # Left: histogram - how many applicants fall in each income range.
    hist_ax.hist(incomes, bins=40, color="#1565c0", edgecolor="white")
    median_income = incomes.median()
    mean_income = incomes.mean()
    # vertical lines show WHY we prefer the median later during cleaning
    hist_ax.axvline(median_income, color="#ef6c00", linestyle="--", linewidth=2,
                    label=f"Median = {median_income:,.0f}")
    hist_ax.axvline(mean_income, color="#c62828", linestyle=":", linewidth=2,
                    label=f"Mean = {mean_income:,.0f}")
    hist_ax.set_title("Distribution of Applicant Monthly Income")
    hist_ax.set_xlabel("Applicant income (per month)")
    hist_ax.set_ylabel("Number of applicants")
    hist_ax.legend(fontsize=9)

    # Right: boxplot - the dots beyond the whiskers are the outliers.
    box_ax.boxplot(incomes, vert=False)
    box_ax.set_title("Same Income Data as a Boxplot")
    box_ax.set_xlabel("Applicant income (per month)")
    box_ax.set_yticks([])

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "2_applicant_income_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"[2] Income  median={median_income:,.0f}  mean={mean_income:,.0f}  "
          f"max={incomes.max():,.0f}  skew={incomes.skew():.2f}")
    return path


# ===========================================================================
# CHART 3: loan amount distribution
# ===========================================================================
def chart_3_loan_amount(df):
    amounts = df["LoanAmount"].dropna()   # 22 values are missing - dropnan for the plot only

    fig, (hist_ax, box_ax) = plt.subplots(1, 2, figsize=(11, 4))
    hist_ax.hist(amounts, bins=30, color="#00695c", edgecolor="white")
    hist_ax.axvline(amounts.median(), color="#ef6c00", linestyle="--", linewidth=2,
                    label=f"Median = {amounts.median():,.0f}k")
    hist_ax.axvline(amounts.mean(), color="#c62828", linestyle=":", linewidth=2,
                    label=f"Mean = {amounts.mean():,.0f}k")
    hist_ax.set_title("Distribution of Loan Amount (in thousands)")
    hist_ax.set_xlabel("Loan amount (thousand)")
    hist_ax.set_ylabel("Number of applications")
    hist_ax.legend(fontsize=9)

    box_ax.boxplot(amounts, vert=False)
    box_ax.set_title("Same Loan Amounts as a Boxplot")
    box_ax.set_xlabel("Loan amount (thousand)")
    box_ax.set_yticks([])

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "3_loan_amount_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"[3] LoanAmount median={amounts.median():.0f}k  mean={amounts.mean():.0f}k  "
          f"max={amounts.max():.0f}k  missing=22")
    return path


# ===========================================================================
# CHART 4: credit history vs approval (grouped bars)
# ===========================================================================
def chart_4_credit_history(df):
    # contingency table: rows = credit history (0/1), columns = Y/N counts
    table = pd.crosstab(df["Credit_History"], df["Loan_Status"])
    fig, (rate_ax, count_ax) = plt.subplots(1, 2, figsize=(11, 4))

    # Left: approval RATE per credit-history group - the striking contrast.
    rate = df.groupby("Credit_History")["Approved_flag"].mean() * 100
    bars = rate_ax.bar(["No credit history (0)", "Has credit history (1)"],
                       rate.reindex([0.0, 1.0]).values,
                       color=["#ef6c00", "#1565c0"], width=0.5)
    for bar, value in zip(bars, rate.reindex([0.0, 1.0]).values):
        rate_ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%",
                     ha="center", fontsize=12, fontweight="bold")
    rate_ax.set_title("Approval Rate by Credit History")
    rate_ax.set_xlabel("Credit history group")
    rate_ax.set_ylabel("Approved (%)")
    rate_ax.set_ylim(0, 100)

    # Right: raw counts behind those percentages (context of group sizes).
    table.plot(kind="bar", ax=count_ax, color=["#c62828", "#2e7d32"])
    count_ax.set_title("Applications by Credit History and Outcome")
    count_ax.set_xlabel("Credit history (0 = none, 1 = good)")
    count_ax.set_ylabel("Number of applications")
    count_ax.set_xticklabels(["No history (0)", "Good history (1)"],
                             rotation=0)
    count_ax.legend(["Rejected (N)", "Approved (Y)"])

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "4_credit_history_vs_approval.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    r0 = rate.loc[0.0]
    r1 = rate.loc[1.0]
    print(f"[4] Approval rate: with history={r1:.0f}%  without history={r0:.0f}%  "
          f"ratio={r1/r0:.1f}x   (counts: {table.to_dict()})")
    return path


# ===========================================================================
# CHART 5: property area vs approval (grouped bars)
# ===========================================================================
def chart_5_property_area(df):
    order = ["Urban", "Semiurban", "Rural"]
    rate = df.groupby("Property_Area")["Approved_flag"].mean().reindex(order) * 100
    counts = df["Property_Area"].value_counts().reindex(order)

    fig, (rate_ax, count_ax) = plt.subplots(1, 2, figsize=(11, 4))

    bars = rate_ax.bar(order, rate.values,
                       color=["#1565c0", "#6a1b9a", "#ef6c00"], width=0.5)
    for bar, value in zip(bars, rate.values):
        rate_ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}%",
                     ha="center", fontsize=12, fontweight="bold")
    rate_ax.set_title("Approval Rate by Property Area")
    rate_ax.set_xlabel("Property area")
    rate_ax.set_ylabel("Approved (%)")
    rate_ax.set_ylim(0, 100)

    bars = count_ax.bar(order, counts.values,
                        color=["#1565c0", "#6a1b9a", "#ef6c00"], width=0.5)
    for bar, value in zip(bars, counts.values):
        count_ax.text(bar.get_x() + bar.get_width() / 2, value + 4, str(value),
                      ha="center", fontsize=12, fontweight="bold")
    count_ax.set_title("Applications per Property Area")
    count_ax.set_xlabel("Property area")
    count_ax.set_ylabel("Number of applications")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "5_property_area_vs_approval.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"[5] Approval: Urban={rate['Urban']:.0f}%  Semiurban={rate['Semiurban']:.0f}%  "
          f"Rural={rate['Rural']:.0f}%   (counts: {counts.to_dict()})")
    return path


# ===========================================================================
# CHART 6: correlation heatmap of the numeric columns
# ===========================================================================
def chart_6_heatmap(df):
    """Correlation = a number between -1 and +1 saying how two columns move
    together. +0.9 means 'they rise together strongly'; 0 means 'no pattern'.

    We TEMPORARILY map the target Y->1 / N->0 so the heatmap can also show
    which numeric feature relates to approval. Gender/Married etc. stay out -
    they are text labels with no numeric meaning (yet; they get encoded in
    the training step)."""
    numeric = df[["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                  "Loan_Amount_Term", "Credit_History", "Approved_flag"]]
    corr = numeric.corr().round(2)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.6,
                cbar_kws={"label": "Correlation coefficient (-1 to +1)"},
                ax=ax)
    ax.set_title("Correlation Between Numeric Features\n"
                 "(Approved_flag = Loan_Status, Y=1 / N=0)", fontsize=12)
    # rotate labels so nothing overlaps
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "6_correlation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    target_corr = corr["Approved_flag"].drop("Approved_flag").sort_values(ascending=False)
    print("[6] Correlation with approval (real values):")
    for feature, value in target_corr.items():
        print(f"      {feature:<18} {value:+.2f}")
    # strongest non-target pair, quoted in the explanation below
    pairs = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    print(f"      strongest feature pair: {pairs.idxmax()} = {corr.loc[pairs.idxmax()]:.2f}")
    return path


# ===========================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_prepare()

    paths = [
        chart_1_class_counts(df),
        chart_2_income_distribution(df),
        chart_3_loan_amount(df),
        chart_4_credit_history(df),
        chart_5_property_area(df),
        chart_6_heatmap(df),
    ]

    print("\nAll 6 charts saved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
