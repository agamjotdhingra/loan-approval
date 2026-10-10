"""
data_preparation.py
===================
Step 1 of the pipeline: turn the raw CSV into clean, numeric, model-ready data.

What this file does (in order):
  1. LOAD    - read the public loan dataset CSV into a pandas DataFrame.
  2. CLEAN   - fill missing values (median for numbers, mode for categories).
  3. ENCODE  - convert text categories into numbers (ML models need numbers).
  4. SPLIT   - divide rows into a training set (80%) and a test set (20%).

Run it directly to see a printed summary of every step:
    python src/data_preparation.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths - we figure out the project root no matter where the script is run
# from, so "python src/data_preparation.py" works from the project folder
# while editors that run from "src/" also work.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "loan_data.csv")


def load_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """READ the raw loan dataset into a DataFrame.

    WHY: pandas turns the CSV file into a table (DataFrame) that we can
    inspect, clean and feed to scikit-learn. Nothing more exotic needed.
    """
    df = pd.read_csv(csv_path)
    print(f"[LOAD] {len(df)} rows, {len(df.columns)} columns loaded from {csv_path}")
    return df


def report_missing(df: pd.DataFrame) -> pd.Series:
    """Count missing values per column and print the non-zero ones.

    WHY: ML algorithms cannot handle blank cells. We must first FIND the
    holes (this function) and then FILL them (handle_missing_values).
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        print("[CLEAN] No missing values found.")
    else:
        print("[CLEAN] Missing values found:")
        for column, count in missing.items():
            print(f"        - {column:<20} {count} missing")
    return missing


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """FILL the blank cells so the table is complete.

    HOW: two very simple, textbook strategies -
      * Numeric columns  -> fill with the MEDIAN of that column.
        (Median is preferred over mean because a few very high incomes
        would pull the mean up, while the median is robust to outliers.)
      * Categoric columns -> fill with the MODE (the most frequent value).
        (For a category like 'Gender' there is no 'middle' value, so the
        most common value is the safest guess.)
    """
    df = df.copy()

    for column in df.columns:
        if df[column].isnull().any():
            if df[column].dtype == "object":          # text/categorical column
                mode_value = df[column].mode()[0]     # most frequent value
                df[column] = df[column].fillna(mode_value)
                print(f"        filled '{column}' with mode = {mode_value!r}")
            else:                                     # numeric column
                median_value = df[column].median()    # middle value
                df[column] = df[column].fillna(median_value)
                print(f"        filled '{column}' with median = {median_value}")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """CONVERT text categories into numbers (called 'encoding').

    HOW: three small tricks, each easy to explain in a viva -
      * Binary text columns (2 values: yes/no type)
            -> plain mapping, e.g. {'Y': 1, 'N': 0}.
      * 'Dependents' contains the text '3+'
            -> we strip the '+' so it becomes the number 3.
      * 'Property_Area' has 3 values (Urban/Semiurban/Rural)
            -> one-hot encoding: one new 0/1 column per area so the model
               never thinks 'Urban' is 'bigger' than 'Rural' (they are
               categories, not quantities).
    """
    df = df.copy()

    # --- 1. Binary columns: a simple dictionary mapping -------------------
    binary_maps = {
        "Loan_Status":    {"N": 0, "Y": 1},   # 1 = approved  (our TARGET)
        "Gender":         {"Male": 1, "Female": 0},
        "Married":        {"No": 0, "Yes": 1},
        "Education":      {"Not Graduate": 0, "Graduate": 1},
        "Self_Employed":  {"No": 0, "Yes": 1},
    }
    for column, mapping in binary_maps.items():
        df[column] = df[column].map(mapping)
        print(f"        encoded '{column}' -> {mapping}")

    # --- 2. Dependents: '3+' is text, we need the number 3 ----------------
    df["Dependents"] = df["Dependents"].str.replace("+", "", regex=False).astype(int)
    print("        encoded 'Dependents': '3+' -> 3")

    # --- 3. Credit_History is already numeric (1.0 / 0.0) -----------------
    df["Credit_History"] = df["Credit_History"].astype(int)
    print("        encoded 'Credit_History': float -> int")

    # --- 4. Property_Area: one-hot encoding (3 category columns) ----------
    # prefix keeps the column names readable, e.g. 'Property_Area_Urban'
    df = pd.get_dummies(df, columns=["Property_Area"], prefix="Property_Area")
    print("        one-hot encoded 'Property_Area' -> 3 new 0/1 columns")

    return df


def add_engineered_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create TotalIncome = ApplicantIncome + CoapplicantIncome.

    WHY: a bank cares about the HOUSEHOLD income, not just the applicant's.
    Combining both incomes into one column is called 'feature engineering'
    and usually helps the model see the real financial picture.
    """
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    print("[FEATURE] added 'TotalIncome' = ApplicantIncome + CoapplicantIncome")
    return df


def prepare_data(csv_path: str = DATA_PATH, verbose: bool = True):
    """Run the whole preparation pipeline and return the split datasets.

    Returns four objects exactly as scikit-learn convention expects:
        X_train, X_test  : feature tables  (the inputs)
        y_train, y_test  : target vector   ('Loan_Status': 1 = approved)
    """
    df = load_data(csv_path)

    if verbose:
        missing = report_missing(df)
        if len(missing):
            print("[CLEAN] Filling missing values:")
            df = handle_missing_values(df)
            # `df.isnull().sum().sum()` is a sanity check: total missing = 0
            assert df.isnull().sum().sum() == 0
        print("[ENCODE] Encoding text columns as numbers:")
    else:
        df = handle_missing_values(df)

    df = encode_features(df)
    df = add_engineered_feature(df)

    # -------------------------------------------------------------------
    # SPLIT into features (X) and target (y), then into train and test.
    #
    # WHY a separate test set? If we score the model on the same rows it
    # learned from, we cannot tell whether it *learned the pattern* or just
    # *memorised the answers*. The test set acts like unseen future
    # applications.
    #
    # stratify=y keeps the approved/rejected ratio the same in both sets,
    # and random_state=42 makes the split repeatable (same rows every run).
    # -------------------------------------------------------------------
    X = df.drop(columns=["Loan_ID", "Loan_Status"])  # Loan_ID is an ID, not a signal
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# When run directly, print a short walkthrough so you can SEE each step.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()

    print("\n================ SUMMARY ================")
    print(f"Training set: {X_train.shape[0]} rows, {X_train.shape[1]} feature columns")
    print(f"Test set    : {X_test.shape[0]} rows, {X_test.shape[1]} feature columns")
    print("\nFeature columns:", list(X_train.columns))
    print("\nClass balance (train): ")
    print(y_train.value_counts(normalize=True).rename("fraction"))
