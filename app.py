import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent  # works from any folder

st.set_page_config(page_title="AquaCheck", page_icon="💧", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(BASE / "water_quality_model.joblib")
    meta = json.load(open(BASE / "meta.json"))
    results = pd.read_csv(BASE / "results.csv", index_col=0)
    return model, meta, results


model, meta, results = load()
FEATURES = meta["features"]

HELP = {
    "ph": "How acidic or alkaline the water is (WHO guideline: 6.5 to 8.5)",
    "Hardness": "Dissolved calcium and magnesium (mg/L)",
    "Solids": "Total dissolved solids (ppm)",
    "Chloramines": "Disinfectant level (ppm)",
    "Sulfate": "Sulfate concentration (mg/L)",
    "Conductivity": "Ability to conduct electricity (uS/cm)",
    "Organic_carbon": "Organic carbon (ppm)",
    "Trihalomethanes": "Disinfection by-products (ug/L)",
    "Turbidity": "How cloudy the water is (NTU)",
}

st.title("💧 AquaCheck – Smart Water Quality Predictor")
st.caption("EVS project · Machine Learning on the Kaggle Water Potability dataset (3,276 samples)")

tab_predict, tab_models, tab_data, tab_about = st.tabs(
    ["🔎 Predict", "📊 Model comparison", "🧪 Data insights", "ℹ️ About"])

# ------------------------------------------------------------------ Predict
with tab_predict:
    st.subheader("Enter the water-quality values")
    c1, c2, c3 = st.columns(3)
    values = {}
    for i, f in enumerate(FEATURES):
        lo, hi, med = meta["ranges"][f]
        with [c1, c2, c3][i % 3]:
            values[f] = st.number_input(
                f.replace("_", " ").title() if f != "ph" else "pH",
                min_value=0.0, max_value=float(hi * 1.5),
                value=float(round(med, 2)), step=float(max(hi / 500, 0.01)),
                help=HELP[f], key=f)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([values], columns=FEATURES)
        pred = int(model.predict(row)[0])
        proba = float(model.predict_proba(row)[0][1])
        if pred == 1:
            st.success(f"✅ Predicted: POTABLE  (model confidence for potable: {proba:.0%})")
        else:
            st.error(f"❌ Predicted: NOT POTABLE  (model confidence for potable: {proba:.0%})")
        st.progress(proba, text="Probability of potable")
        st.warning("This is a statistical prediction from historical data. "
                   "It is NOT a certified drinking-water safety test.")

# ------------------------------------------------------------------ Models
with tab_models:
    st.subheader("Five algorithms, same data, same 80/20 split")
    st.dataframe(results.style.format("{:.1%}").highlight_max(axis=0, color="#cdeccd"),
                 width='stretch')
    st.image(str(BASE / "charts/model_comparison.png"))
    st.info(f"Saved for the website: **{meta['best_model']}** (highest F1-score). "
            f"Train = {meta['n_train']} samples, Test = {meta['n_test']} samples.")
    st.markdown(
        "**Reading the table honestly:** accuracy looks moderate because only ~39% of samples "
        "are potable, so a model that always says *Not Potable* already scores ~61%. "
        "That is why Precision, Recall and F1 are shown too. Logistic Regression behaves "
        "exactly like that on this split (it never predicts Potable).")
    st.image(str(BASE / "charts/confusion_matrix.png"), width=380)

# ------------------------------------------------------------------ Data
with tab_data:
    a, b = st.columns(2)
    a.image(str(BASE / "charts/feature_importance.png"))
    b.image(str(BASE / "charts/class_balance.png"), width=380)
    st.image(str(BASE / "charts/correlation.png"), width=520)
    st.caption(f"{meta['n_missing']} missing values (pH, Sulfate, Trihalomethanes) were filled "
               "with the median of the training data.")

# ------------------------------------------------------------------ About
with tab_about:
    st.markdown("""
**Pipeline:** data collection → median imputation → scaling → 80/20 split →
train 5 models → evaluate (accuracy, precision, recall, F1) → save best model → this website.

**Limitations**
- Historical dataset, not a live sensor feed from any local water supply.
- Even the best model is only moderately accurate on this dataset.
- Never use it to decide whether real water is safe to drink; use a certified lab test.
""")
