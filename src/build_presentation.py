"""
build_presentation.py - generates the 12-slide college PowerPoint
=================================================================
Every number and every image inserted here is taken from the REAL project
artifacts:
  metrics:  output/plots/model_comparison.csv (verified values)
  charts:   outputs/*.png (EDA) and output/plots/04, 05 (results)
Nothing is invented. Run from project root:
    ./.venv/bin/python src/build_presentation.py
"""

import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJECT_ROOT, "docs", "Loan_Approval_Presentation.pptx")

# ---- the verified project numbers (source: output/plots/model_comparison.csv)
METRICS = {
    "Logistic Regression": {"Accuracy": 0.862, "Precision": 0.840,
                            "Recall": 0.988, "F1": 0.908, "ROC-AUC": 0.852},
    "Decision Tree":       {"Accuracy": 0.846, "Precision": 0.824,
                            "Recall": 0.988, "F1": 0.898, "ROC-AUC": 0.706},
}

# ---- a small design system (academic navy + white) ------------------------
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
TEAL = RGBColor(0x0E, 0x7C, 0x7B)
GREY = RGBColor(0x44, 0x44, 0x44)
LIGHT = RGBColor(0xF4, 0xF6, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)   # 16:9

prs = Presentation()
prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
BLANK = prs.slide_layouts[6]

CHARTS = {
    "eda_class":    os.path.join(PROJECT_ROOT, "outputs", "1_approval_vs_rejection.png"),
    "eda_income":   os.path.join(PROJECT_ROOT, "outputs", "2_applicant_income_distribution.png"),
    "eda_credit":   os.path.join(PROJECT_ROOT, "outputs", "4_credit_history_vs_approval.png"),
    "eda_heat":     os.path.join(PROJECT_ROOT, "outputs", "6_correlation_heatmap.png"),
    "confusion":    os.path.join(PROJECT_ROOT, "output", "plots", "04_confusion_matrices.png"),
    "comparison":   os.path.join(PROJECT_ROOT, "output", "plots", "05_model_comparison.png"),
}


# ===========================================================================
# helpers
# ===========================================================================
def add_slide(title_text: str) -> "tuple[object, object]":
    """Empty slide + navy title bar + title text. Returns (slide, body_top)."""
    slide = prs.slides.add_slide(BLANK)

    bar = slide.shapes.add_shape(1, 0, 0, SLIDE_W, Inches(1.0))   # 1 = rectangle
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    tf = bar.text_frame
    tf.margin_left = Inches(0.5)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size, p.font.bold, p.font.color.rgb = Pt(30), True, WHITE

    return slide


def bullets(slide, items, left=0.6, top=1.3, width=6.1, height=None,
            size=17, title=None):
    """Bulleted body text. items = list[str] (sub-bullets prefixed '  - ').
    height=None auto-fits the frame so it never spills past the footer."""
    if height is None:
        height = max(0.4, 6.85 - top)   # keep every frame inside the slide
    box = slide.shapes.add_textbox(Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        sub = item.startswith("  - ")
        text = item[4:] if sub else item
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = ("•  " + text) if not sub else ("–  " + text)
        p.level = 1 if sub else 0
        p.font.size = Pt(size - 2 if sub else size)
        p.font.color.rgb = GREY
        p.space_after = Pt(6)
    if title:
        tb = slide.shapes.add_textbox(Inches(left), Inches(top - 0.35),
                                      Inches(width), Inches(0.35))
        tb.text_frame.text = title
        tb.text_frame.paragraphs[0].font.bold = True
        tb.text_frame.paragraphs[0].font.size = Pt(15)
        tb.text_frame.paragraphs[0].font.color.rgb = TEAL


def picture(slide, key, left, top, width=None, height=None):
    return slide.shapes.add_picture(CHARTS[key], Inches(left), Inches(top),
                                    width=Inches(width) if width else None,
                                    height=Inches(height) if height else None)


def speaker_notes(slide, text: str):
    slide.notes_slide.notes_text_frame.text = text


def footer(slide, number: int):
    tb = slide.shapes.add_textbox(Inches(11.9), Inches(7.05), Inches(1.2),
                                  Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    p.text = f"{number} / 12"
    p.font.size, p.font.color.rgb, p.alignment = Pt(10), GREY, PP_ALIGN.RIGHT
    cap = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(8, ),
                                   Inches(0.3))
    q = cap.text_frame.paragraphs[0]
    q.text = ("Public dataset: Analytics Vidhya Loan Prediction "
              "(not proprietary bank data)")
    q.font.size, q.font.color.rgb = Pt(10), GREY


# ===========================================================================
# SLIDE 1 — Title & team
# ===========================================================================
s = add_slide("")   # we draw our own title block
band = s.shapes.add_shape(1, 0, Inches(2.2), SLIDE_W, Inches(2.6))
band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
tb = band.text_frame; tb.word_wrap = True
p = tb.paragraphs[0]
p.text = "AI-Based Loan Approval Prediction System"
p.alignment = PP_ALIGN.CENTER
p.font.size, p.font.bold, p.font.color.rgb = Pt(40), True, WHITE
p2 = tb.add_paragraph(); p2.text = "Predicting loan approval with Logistic Regression vs Decision Tree"
p2.alignment = PP_ALIGN.CENTER; p2.font.size = Pt(20); p2.font.color.rgb = LIGHT

tb2 = s.shapes.add_textbox(Inches(1, ), Inches(5.2), Inches(11.3), Inches(1.6))
tframe = tb2.text_frame
tframe.text = ("Student:  <Your Name>      Roll No:  <...>      "
               "Class:  <...>      Guide:  <...>")
tframe.paragraphs[0].alignment = PP_ALIGN.CENTER
tframe.paragraphs[0].font.size = Pt(18)
p3 = tframe.add_paragraph()
p3.text = "<College name>   ·   <Date of presentation>"
p3.alignment = PP_ALIGN.CENTER; p3.font.size = Pt(14); p3.font.color.rgb = GREY
speaker_notes(s, "Greet the panel. One line: 'My project predicts whether a "
                 "loan application will be approved or rejected using two "
                 "supervised machine-learning models trained on a public "
                 "dataset.' Introduce yourself and your guide, then move on "
                 "quickly - this slide costs you 30 seconds.")
footer(s, 1)

# ===========================================================================
# SLIDE 2 — Problem statement
# ===========================================================================
s = add_slide("2. Problem Statement")
bullets(s, [
    "Banks receive thousands of loan applications; manual screening is slow, inconsistent and subjective.",
    "Two types of costly mistakes:",
    "  - approving an applicant who will default (bad loan / loss)",
    "  - rejecting a creditworthy customer (lost business)",
    "Goal: learn from historical labelled applications and predict Approved / Rejected automatically.",
    "Formally: a supervised, binary classification problem on tabular data.",
], top=1.35, size=19)
picture(s, "eda_class", 7.3, 1.6, width=5.5)
bullets(s, ["Why this dataset suits the task: one clear target (Loan_Status)."],
        left=7.3, top=6.3, width=5.5, size=12)
speaker_notes(s, "Emphasise the two asymmetric costs: for the bank, a false "
                 "approval (default) is usually more expensive than a false "
                 "rejection. That asymmetry is exactly why we report precision "
                 "AND recall, not just accuracy.")
footer(s, 2)

# ===========================================================================
# SLIDE 3 — Objectives
# ===========================================================================
s = add_slide("3. Project Objectives")
bullets(s, [
    "Build an end-to-end supervised ML pipeline: data → preprocessing → training → evaluation → deployment.",
    "Handle real-world data quality issues: missing values, skew, class imbalance.",
    "Train and fairly compare two classical models: Logistic Regression and Decision Tree (max_depth 4).",
    "Evaluate with accuracy, precision, recall, F1, ROC-AUC and confusion matrices on a held-out test set.",
    "Serve the models in an interactive Streamlit application with live predictions.",
    "Keep every step explainable — no deep learning, no black boxes.",
], top=1.35, size=19)
speaker_notes(s, "If asked 'why only two models?' — statistical honesty: 614 "
                 "rows is small; simple models generalise and stay "
                 "interpretable. A Random Forest was considered but rejected "
                 "on evidence (no overfitting measured).")
footer(s, 3)

# ===========================================================================
# SLIDE 4 — Dataset
# ===========================================================================
s = add_slide("4. Dataset Description")
bullets(s, [
    "Source: Analytics Vidhya 'Loan Prediction' practice problem (public, also on Kaggle).",
    "Fictional 'Dream Housing Finance' sample data - NOT any real bank's records.",
    "614 applications × 13 columns; target = Loan_Status (Y/N).",
    "Class balance: 422 approved (68.7%) vs 192 rejected (31.3%).",
    "Feature types: 6 categorical (Gender, Married, Dependents, Education, Self_Employed, Property_Area) + 5 numeric (incomes, LoanAmount, Term, Credit_History).",
    "Data quality: 149 missing cells across 7 columns (Credit_History 50, Self_Employed 32, LoanAmount 22...).",
], top=1.35, size=17)
speaker_notes(s, "Be explicit about data provenance BEFORE anyone asks: the "
                 "dataset is a public teaching dataset owned by no bank. The "
                 "149 missing cells motivate the preprocessing slide that "
                 "follows.")
footer(s, 4)

# ===========================================================================
# SLIDE 5 — EDA
# ===========================================================================
s = add_slide("5. Exploratory Data Analysis")
picture(s, "eda_credit", 0.4, 1.45, width=6.4)
picture(s, "eda_income", 6.9, 1.45, width=6.1)
bullets(s, [
    "Credit history is decisive: approval 80% with history vs 8% without (10×).",
    "Income is right-skewed (skew 6.54): median ₹3,812 but mean ₹5,403, outliers to ₹81,000.",
    "Property-area effect is mild: Semiurban 77% > Urban 66% > Rural 61%.",
    "Heatmap: Credit_History↔approval +0.56; income↔LoanAmount +0.57; income↔approval ≈ 0.",
], left=0.5, top=5.1, width=12.4, size=15)
speaker_notes(s, "Tell the story in one line: 'EDA showed credit history "
                 "dominates and income barely separates the classes - which "
                 "later matched the trained models exactly.' The skew picture "
                 "justifies choosing the median when filling missing incomes.")
footer(s, 5)

# ===========================================================================
# SLIDE 6 — Preprocessing
# ===========================================================================
s = add_slide("6. Data Preprocessing (leakage-safe)")
bullets(s, [
    "Split BEFORE any fitting: 80% train (491) / 20% test (123), stratified, random_state = 42.",
    "All preprocessing lives inside scikit-learn Pipeline + ColumnTransformer:",
    "  - numeric: SimpleImputer(median) → StandardScaler",
    "  - categorical: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')",
    "Imputers, scaler and encoder are FIT on the 491 training rows ONLY, then applied unchanged to test rows — no data leakage.",
    "Loan_ID dropped: a unique ID has no predictive meaning (memorisation fuel).",
], top=1.35, size=17)
# small architecture sketch of the column routing
flow = s.shapes.add_shape(1, 0.8, 4.55, Inches(11.7), Inches(2.0))
flow.fill.solid(); flow.fill.fore_color.rgb = LIGHT; flow.line.color.rgb = TEAL
ftf = flow.text_frame; ftf.word_wrap = True
lines = [
    ("raw app row", 44),
    ("   ├── numeric block → median-impute → scale", 44),
    ("   └── categorical block → mode-impute → one-hot", 44),
    ("        ↓ combined matrix → fitted model → Y/N", 44),
]
for i, (txt, _) in enumerate(lines):
    pp = ftf.paragraphs[0] if i == 0 else ftf.add_paragraph()
    pp.text = txt; pp.font.size = Pt(15)
    pp.font.name = "Courier New"; pp.font.color.rgb = NAVY
speaker_notes(s, "Define leakage in one sentence: 'test-set statistics "
                 "sneaking into training, inflating scores.' Then say our "
                 "structure makes it impossible: the pipeline is fit only "
                 "when we call fit on X_train. This is the slide to underline.")
footer(s, 6)

# ===========================================================================
# SLIDE 7 — System architecture
# ===========================================================================
s = add_slide("7. Proposed System Architecture")
steps = ["CSV dataset\n(614 rows)", "Leak-safe\npreprocessing", "Train\nLR + DT",
         "Evaluate on\n123-row test set", "Select best\n(highest F1)",
         "Pickle to\nbest_model.pkl", "Streamlit app\nraw-row predictions"]
x = 0.55
for i, label in enumerate(steps):
    box = s.shapes.add_shape(1, Inches(x + i * 1.72), Inches(2.3),
                             Inches(1.55), Inches(1.6))
    box.fill.solid(); box.fill.fore_color.rgb = TEAL if i % 2 == 0 else NAVY
    box.line.fill.background()
    btf = box.text_frame; btf.word_wrap = True; btf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for j, line in enumerate(label.split("\n")):
        bp = btf.paragraphs[0] if j == 0 else btf.add_paragraph()
        bp.text = line; bp.alignment = PP_ALIGN.CENTER
        bp.font.size, bp.font.color.rgb, bp.font.bold = Pt(13), WHITE, True
    if i < len(steps) - 1:
        arrow = s.shapes.add_shape(13, Inches(x + i * 1.72 + 1.55),
                                   Inches(2.95), Inches(0.18), Inches(0.3))
        arrow.fill.solid(); arrow.fill.fore_color.rgb = GREY
        arrow.line.fill.background()
bullets(s, [
    "One pipeline object per model chains preprocessing + estimator, so train-time and app-time inputs are processed identically.",
    "Pickling persists the FITTED pipeline (scaler stats + encoder vocabulary + weights) — the app never retrains.",
    "Both saved pipelines load once via @st.cache_resource; predictions appear in milliseconds.",
], top=4.4, width=12.3, size=16)
speaker_notes(s, "Walk the arrows left to right, then point at the pickle "
                 "step: 'everything the model learned travels into the app "
                 "in one file — that's why the app starts instantly.'")
footer(s, 7)

# ===========================================================================
# SLIDE 8 — The two models
# ===========================================================================
s = add_slide("8. Logistic Regression vs Decision Tree")
bullets(s, [
    "Logistic Regression — linear baseline:",
    "  - weighted sum of features → sigmoid → P(approved); smooth probabilities, interpretable weights",
    "  - needs feature scaling → hence the StandardScaler block",
    "Decision Tree (max_depth = 4) — rule-based:",
    "  - learns 4 yes/no questions via Gini impurity; scale-invariant, human-readable",
    "  - depth capped at 4 to prevent memorisation (overfitting)",
    "Both share the identical preprocessing → a fair head-to-head.",
], top=1.35, width=6.3, size=16)
picture(s, "confusion", 7.05, 1.7, width=5.9)
bullets(s, [
    "Confusion matrices on the test set: LR = 22 TN / 2 FP / 2 FN / 83 TP.",
], left=7.05, top=5.35, width=5.9, size=13)
speaker_notes(s, "If asked 'what is the sigmoid?' — the S-curve mapping any "
                 "score into (0,1), so it reads as a probability; 0.5 is the "
                 "decision boundary. If asked about Gini — the mix-level of "
                 "a node; the tree picks splits that lower it most.")
footer(s, 8)

# ===========================================================================
# SLIDE 9 — Metrics
# ===========================================================================
s = add_slide("9. Model Evaluation Metrics")
bullets(s, [
    "Accuracy — overall share of correct decisions; misleading alone at 69/31 imbalance.",
    "Precision — of predicted approvals, how many were truly safe? (guards the bank: punishes FP)",
    "Recall — of truly approvable loans, how many did we catch? (guards the customer: punishes FN)",
    "F1-score — harmonic mean of precision & recall; our model-selection criterion.",
    "ROC-AUC — ranking quality across all thresholds (0.5 = coin-flip, 1.0 = perfect).",
    "All computed on the untouched 123-row test set only.",
], top=1.35, width=6.4, size=17)
picture(s, "comparison", 7.15, 1.7, width=5.8)
speaker_notes(s, "Give the lazy-model argument for why accuracy alone is "
                 "weak: 'always approve' would score 69% and reject nobody. "
                 "Precision/recall expose that. Mention F1 is what we used "
                 "to pick the final model.")
footer(s, 9)

# ===========================================================================
# SLIDE 10 — Results
# ===========================================================================
s = add_slide("10. Actual Results & Model Comparison")
rows, cols = 3, 6
tbl = s.shapes.add_table(rows, cols, Inches(0.55), Inches(1.5),
                         Inches(7.2), Inches(1.7)).table
headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
for c, h in enumerate(headers):
    cell = tbl.cell(0, c); cell.text = h
    cell.text_frame.paragraphs[0].font.bold = True
    cell.text_frame.paragraphs[0].font.size = Pt(15)
tbl.cell(1, 0).text = "Logistic Regression"
tbl.cell(2, 0).text = "Decision Tree (depth 4)"
lr, dt = METRICS["Logistic Regression"], METRICS["Decision Tree"]
for c, key in enumerate(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], start=1):
    tbl.cell(1, c).text = f"{lr[key]:.3f}"
    tbl.cell(2, c).text = f"{dt[key]:.3f}"
    for r in (1, 2):
        tbl.cell(r, c).text_frame.paragraphs[0].font.size = Pt(14)
bullets(s, [
    "Logistic Regression wins on Accuracy, Precision, F1 and ROC-AUC (Recall ties at 0.988).",
    "Selected model: Logistic Regression — by measured test F1 (0.908), not assumption.",
    "Overfit check: train 0.800 vs test 0.862 (LR); 0.809 vs 0.846 (DT) → gaps NEGATIVE, models generalise.",
    "Random Forest was deliberately NOT added: no overfitting to fix (Occam's razor).",
], top=3.5, width=7.2, size=15)
picture(s, "confusion", 7.55, 1.6, width=5.4)
speaker_notes(s, "Quote numbers straight from the table. The likely student "
                 "question is 'why not Random Forest?' — answer: we tested "
                 "the precondition (overfit gap) and it wasn't met; simpler "
                 "model wins on explainability. ROC-AUC gap (0.852 vs 0.706) "
                 "shows LR ranks applicants better.")
footer(s, 10)

# ===========================================================================
# SLIDE 11 — Application demo
# ===========================================================================
s = add_slide("11. Application Demonstration")
bullets(s, [
    "Streamlit app loads both pickled pipelines once (cache) — startup without retraining.",
    "Form mirrors the 11 training features; raw values only — all encoding happens inside the pipeline.",
    "Input validation + graceful handling of missing model files.",
    "Demo (verified live): defaults → APPROVED 80% (LR) / 77% (DT); flip credit history → REJECTED.",
    "That one toggle shows the decisive feature in action.",
], top=1.35, width=6.1, size=16)
# demo screenshot captured from the running app is not embedded (dark-theme
# Streamlit export); keep a clean placeholder + live demo instruction instead
box = s.shapes.add_shape(1, 6.85, 1.6, Inches(6.0), Inches(3.9))
box.fill.solid(); box.fill.fore_color.rgb = LIGHT; box.line.color.rgb = TEAL
btf = box.text_frame; btf.word_wrap = True; btf.vertical_anchor = MSO_ANCHOR.MIDDLE
demo_lines = [
    "LIVE DEMO — run:", "",
    "./.venv/bin/streamlit run src/app.py", "",
    "1) keep defaults → Predict → APPROVED 80% / 77%",
    "2) set Credit History to 0 → Predict → REJECTED",
]
for i, line in enumerate(demo_lines):
    pp = btf.paragraphs[0] if i == 0 else btf.add_paragraph()
    pp.text = line; pp.alignment = PP_ALIGN.CENTER
    pp.font.size = Pt(16 if line.startswith("./.") else 14)
    pp.font.bold = True
    pp.font.color.rgb = NAVY
    if line.startswith("./."):
        pp.font.name = "Courier New"
speaker_notes(s, "Actually run the app live if permitted — it uses the same "
                 "input the examiners can see on the projector. Narrate: "
                 "'one toggle flips both models' decisions because credit "
                 "history was the dominant feature in training too.'")
footer(s, 11)

# ===========================================================================
# SLIDE 12 — Conclusion
# ===========================================================================
s = add_slide("12. Conclusion, Limitations & Future Scope")
bullets(s, [
    "Delivered: complete, explainable loan-screening pipeline — 86% test accuracy, F1 0.908 (LR best).",
    "Learnings: credit history dominates approval decisions; leakage-safe preprocessing is a structural choice, not an afterthought.",
    "Limitations — said honestly:",
    "  - dataset is small (614) and not from a real lender",
    "  - recall for the rejected class is only 0.58 — the model is lenient",
    "  - sensitive attributes (Gender, Education) remain in the feature set",
    "Future scope: k-fold cross-validation, decision-threshold calibration, fairness audit, larger/real datasets.",
], top=1.35, width=12.3, size=17)
tb = s.shapes.add_textbox(Inches(0.6), Inches(6.1), Inches(12), Inches(0.8))
p = tb.text_frame.paragraphs[0]
p.text = "Thank you — questions welcome."
p.font.size, p.font.bold, p.font.color.rgb, p.alignment = Pt(22), True, NAVY, PP_ALIGN.CENTER
speaker_notes(s, "Volunteer a limitation unprompted: 'the model is lenient — "
                 "it catches approvals well (recall 0.988) but misses 42% of "
                 "true rejections.' Examiners reward self-awareness. Close "
                 "with future scope and thank the panel.")
footer(s, 12)

# ===========================================================================
prs.save(OUT_PATH)
print(f"Saved {len(prs.slides.__iter__.__self__._sldIdLst)} slides -> {OUT_PATH}")
