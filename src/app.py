"""
app.py - Streamlit front-end for the Loan Approval Prediction System
====================================================================
HOW TO RUN (from the project root):
    streamlit run src/app.py

WHAT THE APP DOES
-----------------
1. A form where you enter an applicant's details exactly as a bank officer
   would (income, credit history, property area...).
2. Both trained pipelines (saved by train_and_evaluate.py) predict
   Approved / Rejected for those details, with a confidence percentage.
3. Both models' answers appear side by side - the live version of the
   project's "Logistic Regression vs Decision Tree" comparison.

A KEY SIMPLIFICATION since the Pipeline rewrite: the app now sends RAW,
human-readable values (e.g. "Graduate", "Urban") straight into the saved
pipeline. Every encoding/scaling step happens INSIDE the pipeline, exactly
as it did during training - one more benefit of wrapping preprocessing and
model in a single object.
"""

import os
import pickle

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "output", "models")

# Columns in the order the training script presents them. Order doesn't
# strictly matter for ColumnTransformer (it selects columns by NAME), but
# a consistent order keeps the preview table readable.
FEATURE_ORDER = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]


@st.cache_resource(show_spinner="Loading trained models...")
def load_models() -> dict:
    """Load both pickled pipelines ONCE (cache = no re-read on every click)."""
    loaded = {}
    for file_name in ["best_model.pkl", "second_model.pkl"]:
        path = os.path.join(MODELS_DIR, file_name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded[file_name] = pickle.load(f)
    return loaded


models = load_models()
if not models:
    st.error("No trained models found. Run first:\n\n`python src/train_and_evaluate.py`")
    st.stop()

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")
st.title("🏦 AI-Based Loan Approval Prediction System")
st.caption("Predict whether a loan application is likely to be Approved or Rejected.")

# ---------------------------------------------------------------------------
# Input form - the RAW dataset values, same options as the CSV columns
# ---------------------------------------------------------------------------
with st.form("applicant_form"):
    st.subheader("Applicant details")
    col_a, col_b = st.columns(2)

    with col_a:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self employed", ["No", "Yes"])
        credit_history = st.selectbox(
            "Credit history (did they repay past loans on time?)",
            ["Has credit history (1)", "No credit history (0)"],
        )

    with col_b:
        applicant_income = st.number_input("Applicant monthly income",
                                           min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Co-applicant monthly income",
                                             min_value=0, value=1500, step=100)
        loan_amount = st.number_input("Loan amount (in thousands)",
                                      min_value=1, value=130, step=10)
        loan_term = st.number_input("Loan term (days)",
                                    min_value=12, value=360, step=12)
        property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

if submitted:
    # Validate inputs
    validation_errors = []
    if applicant_income < 0:
        validation_errors.append("Applicant income cannot be negative.")
    if coapplicant_income < 0:
        validation_errors.append("Co-applicant income cannot be negative.")
    if loan_amount < 1:
        validation_errors.append("Loan amount must be at least 1 (thousand).")
    if loan_term < 12:
        validation_errors.append("Loan term must be at least 12 days.")
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        st.stop()
    
    # ONE raw row, exactly like one row of the CSV (missing-value handling,
    # scaling and one-hot encoding all happen INSIDE the pipeline).
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,           # stays '3+' - trained that way
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1.0 if credit_history.startswith("Has") else 0.0,
        "Property_Area": property_area,
    }])[FEATURE_ORDER]

    st.subheader("Model predictions")
    col_lr, col_dt = st.columns(2)

    for column, (file_name, bundle) in zip([col_lr, col_dt], sorted(models.items())):
        pipeline = bundle["model"]
        name = bundle["model_name"]
        prediction = pipeline.predict(input_df)[0]            # 'Y' or 'N'
        prob_y = pipeline.predict_proba(input_df)[0][1]       # P(approved)

        with column:
            if prediction == "Y":
                st.success(f"✅ {name}: **APPROVED** ({prob_y:.0%} confidence)")
            else:
                st.error(f"❌ {name}: **REJECTED** ({1 - prob_y:.0%} confidence)")

    with st.expander("Show the raw applicant row sent to the pipelines"):
        st.dataframe(input_df)

st.markdown("---")
st.caption(
    "Trained on the public Analytics Vidhya / Kaggle loan prediction dataset "
    "(614 applications). **Academic demonstration only — not an official "
    "HDFC Bank decision.**"
)
