"""
app.py - Streamlit front-end for the Loan Approval Prediction System
====================================================================
HOW TO RUN (from the project root):
    streamlit run src/app.py

WHAT THE APP DOES
-----------------
1. A customer-style form: personal details, income, and the loan requested.
2. The SELECTED model (best 5-fold CV F1, saved by train_and_evaluate.py)
   decides Approved / Not approved. The applicant sees only the decision,
   a plain-language likelihood, simple reasons and their estimated EMI.
3. Model internals (which models exist, exact probabilities, the raw row
   sent to the pipeline) are NOT shown to applicants. For the project demo
   they appear in an examiner view opened with the URL parameter:
       http://localhost:8501/?examiner=1

A KEY SIMPLIFICATION since the Pipeline rewrite: the app sends RAW,
human-readable values (e.g. "Graduate", "Urban") straight into the saved
pipeline. Every encoding/scaling step happens INSIDE the pipeline, exactly
as it did during training.
"""

import glob
import os
import pickle
import sys

import pandas as pd
import streamlit as st

# the saved pipelines reference features.py, so src/ must be importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import add_engineered_features  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "output", "models")

# Columns in the order the training script presents them. Order doesn't
# strictly matter for ColumnTransformer (it selects columns by NAME), but
# a consistent order keeps the examiner preview table readable.
FEATURE_ORDER = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]

st.set_page_config(page_title="LoanSense · Loan Eligibility Check",
                   page_icon="🏦", layout="centered")

# ---------------------------------------------------------------------------
# Styling - same green/gold palette as the project presentation
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  #MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
  .block-container {padding-top: 2rem; max-width: 860px;}
  .hero {background: #0F3D3E; color: #fff; border-radius: 16px;
         padding: 28px 32px; margin-bottom: 1.5rem;}
  .hero .tag {color: #E3A72F; font-weight: 700; font-size: 0.8rem;
              letter-spacing: 0.08em;}
  .hero h1 {color: #fff; font-size: 2rem; margin: 0.2rem 0 0.4rem; padding: 0;}
  .hero p {color: #EAF3F0; margin: 0; font-size: 1rem;}
  .result {border-radius: 16px; padding: 24px 28px; margin: 1rem 0;}
  .result.ok {background: #E6F4EA; border: 1px solid #9BD3AE;}
  .result.no {background: #FBEAEA; border: 1px solid #E8A9A4;}
  .result h2 {margin: 0 0 0.3rem; padding: 0; font-size: 1.6rem;}
  .result.ok h2 {color: #1F6F5C;}
  .result.no h2 {color: #B5443C;}
  .result p {margin: 0; color: #1E2B2B;}
  .pill {display: inline-block; padding: 2px 12px; border-radius: 999px;
         font-weight: 600; font-size: 0.85rem; margin-top: 0.6rem;
         background: #0F3D3E; color: #fff;}
  .reason {padding: 10px 14px; border-radius: 10px; background: #F4F7F6;
           margin-bottom: 8px; color: #1E2B2B;}
  .muted {color: #5B6B6B; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading...")
def load_models() -> dict:
    """Load every pickled pipeline ONCE (cache = no re-read on every click).
    The best model (chosen by cross-validation) is listed first."""
    loaded = {}
    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*.pkl"))):
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        loaded[bundle["model_name"]] = bundle
    return dict(sorted(loaded.items(), key=lambda kv: not kv[1].get("is_best")))


def likelihood_label(prob_approved: float) -> str:
    """Turn the probability into a coarse label - applicants never see the
    exact model probability."""
    if prob_approved >= 0.75:
        return "High"
    if prob_approved >= 0.5:
        return "Moderate"
    if prob_approved >= 0.3:
        return "Low"
    return "Very low"


def applicant_reasons(row: pd.Series) -> "list[str]":
    """Plain-language guidance based on the factors that matter most in our
    feature-importance analysis (credit history, then affordability)."""
    reasons = []
    if row["Credit_History"] == 1:
        reasons.append("✅ You have a record of repaying past loans on time.")
    else:
        reasons.append("⚠️ No record of repaying past loans. This is the "
                       "biggest factor in loan decisions.")
    if row["LoanToIncome"] <= 30:
        reasons.append("✅ The loan amount is reasonable for your household income.")
    else:
        reasons.append("⚠️ The loan amount is high compared with your household "
                       "income. A smaller amount or a co-applicant could help.")
    if row["CoapplicantIncome"] > 0:
        reasons.append("✅ A co-applicant's income adds to your repayment capacity.")
    return reasons


models = load_models()
if not models:
    st.error("The service is not ready yet. Please try again later.")
    st.stop()

examiner_mode = st.query_params.get("examiner") == "1"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="tag">AI LOAN ELIGIBILITY CHECK</div>
  <h1>🏦 LoanSense</h1>
  <p>Answer a few questions to see how likely your home-loan application is to be approved.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input form - grouped like a real application
# ---------------------------------------------------------------------------
with st.form("applicant_form", border=True):
    st.markdown("#### 👤 About you")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Male", "Female"])
    married = c2.selectbox("Married", ["Yes", "No"])
    dependents = c3.selectbox("Dependents", ["0", "1", "2", "3+"])
    c1, c2 = st.columns(2)
    education = c1.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = c2.selectbox("Self-employed", ["No", "Yes"])

    st.markdown("#### 💰 Income & credit")
    c1, c2 = st.columns(2)
    applicant_income = c1.number_input("Your monthly income (₹)",
                                       min_value=0, value=5000, step=500)
    coapplicant_income = c2.number_input("Co-applicant's monthly income (₹)",
                                         min_value=0, value=1500, step=500,
                                         help="Enter 0 if there is no co-applicant.")
    credit_history = st.radio(
        "Have you repaid all your previous loans on time?",
        ["Yes", "No / never had a loan"], horizontal=True)

    st.markdown("#### 🏠 Loan details")
    c1, c2, c3 = st.columns(3)
    loan_amount = c1.number_input("Loan amount (₹ thousands)",
                                  min_value=1, value=130, step=10)
    loan_term = c2.selectbox("Repayment period", [360, 300, 240, 180, 120, 84, 60, 36, 12],
                             format_func=lambda m: f"{m // 12} years ({m} months)")
    property_area = c3.selectbox("Property location", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Check my eligibility", type="primary",
                                      use_container_width=True)

if submitted:
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
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": 1.0 if credit_history == "Yes" else 0.0,
        "Property_Area": property_area,
    }])[FEATURE_ORDER]

    if applicant_income + coapplicant_income == 0:
        st.warning("Please enter at least one monthly income.")
        st.stop()

    best_name, best = next(iter(models.items()))
    prob_y = best["model"].predict_proba(input_df)[0][1]     # P(approved)
    approved = best["model"].predict(input_df)[0] == "Y"
    engineered = add_engineered_features(input_df).iloc[0]

    if approved:
        st.markdown(f"""
        <div class="result ok">
          <h2>✅ Likely to be approved</h2>
          <p>Based on the details you entered, your application looks eligible.</p>
          <span class="pill">Approval likelihood: {likelihood_label(prob_y)}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result no">
          <h2>❌ Not likely to be approved</h2>
          <p>Based on the details you entered, your application may be declined.</p>
          <span class="pill">Approval likelihood: {likelihood_label(prob_y)}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### Your loan at a glance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Household income / month", f"₹{engineered['TotalIncome']:,.0f}")
    m2.metric("Estimated EMI / month", f"₹{engineered['EMI']:,.0f}",
              help="Loan amount ÷ number of months. Interest is not included.")
    m3.metric("Left after EMI / month", f"₹{engineered['BalanceIncome']:,.0f}")

    st.markdown("#### What influenced this result")
    for reason in applicant_reasons(engineered):
        st.markdown(f'<div class="reason">{reason}</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Examiner view - only with ?examiner=1, never shown to applicants
    # -----------------------------------------------------------------------
    if examiner_mode:
        st.divider()
        st.markdown("#### 🔍 Examiner view (model internals)")
        rows = []
        for name, bundle in models.items():
            p = bundle["model"].predict_proba(input_df)[0][1]
            rows.append({"Model": name + (" ⭐ selected" if bundle.get("is_best") else ""),
                         "Decision": "Approved" if p >= 0.5 else "Rejected",
                         "P(approved)": f"{p:.1%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Applicant decision comes from {best_name} "
                   "(highest 5-fold cross-validated F1).")
        st.markdown("**Engineered features**")
        st.dataframe(engineered[["TotalIncome", "EMI", "BalanceIncome",
                                 "LoanToIncome", "LoanAmountLog"]]
                     .to_frame("value").T.round(2), hide_index=True)
        st.markdown("**Raw row sent to the pipeline**")
        st.dataframe(input_df, hide_index=True)

st.markdown(
    '<p class="muted" style="text-align:center;margin-top:2rem;">'
    "This is an indicative check by an AI model trained on a public sample "
    "dataset, for an academic project. It is not a real lending decision."
    "</p>", unsafe_allow_html=True)
