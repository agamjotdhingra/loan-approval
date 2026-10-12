"""
features.py - FEATURE ENGINEERING (new columns built from the raw ones)
======================================================================
The raw dataset gives incomes, loan amount and loan term as separate
columns. A loan officer, however, thinks in terms of AFFORDABILITY:
"How big is the monthly instalment compared to what the household earns?"
This file turns that intuition into 5 extra numeric columns.

    TotalIncome   = ApplicantIncome + CoapplicantIncome
                    -> the household's combined monthly income
    LoanAmountLog = log(1 + LoanAmount)
                    -> squashes the long right tail of loan amounts
    EMI           = LoanAmount * 1000 / Loan_Amount_Term
                    -> approx. monthly instalment (EMI) in rupees
                       (LoanAmount is in thousands, term is in months)
    BalanceIncome = TotalIncome - EMI
                    -> money left each month after paying the EMI
    LoanToIncome  = LoanAmount * 1000 / TotalIncome
                    -> how many months of income the loan is worth

WHY a separate file? The function is used INSIDE the saved scikit-learn
pipeline (as a FunctionTransformer step). pickle stores only a reference
"features.add_engineered_features", so the app must be able to import
this module - it cannot live in a script that runs as __main__.

Missing inputs (e.g. a blank LoanAmount) simply produce NaN here; the
SimpleImputer that runs next in the pipeline fills those with the median
learned from the TRAINING rows, so no leakage is introduced.
"""

import numpy as np
import pandas as pd

ENGINEERED = ["TotalIncome", "LoanAmountLog", "EMI", "BalanceIncome", "LoanToIncome"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with the 5 engineered columns appended."""
    X = X.copy()
    total = X["ApplicantIncome"] + X["CoapplicantIncome"]
    emi = X["LoanAmount"] * 1000 / X["Loan_Amount_Term"]

    X["TotalIncome"] = total
    X["LoanAmountLog"] = np.log1p(X["LoanAmount"])
    X["EMI"] = emi
    X["BalanceIncome"] = total - emi
    # a zero household income would divide by zero -> treat as missing
    X["LoanToIncome"] = X["LoanAmount"] * 1000 / total.replace(0, np.nan)
    return X
