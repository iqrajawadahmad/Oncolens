
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="OncoLens",
    page_icon="🧬",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "oncolens_random_forest.joblib")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.joblib")
    return model, list(feature_columns)

st.title("🧬 OncoLens")
st.subheader("Gene Expression Cancer-Type Classification")
st.write(
    "Upload a CSV containing gene-expression features to obtain "
    "research-prototype predictions."
)

st.warning(
    "Research prototype only — not intended for clinical diagnosis "
    "or medical decision-making."
)

try:
    model, feature_columns = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload gene-expression CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        if data.columns.duplicated().any():
            st.error("The CSV contains duplicate column names.")
            st.stop()

        missing = [
            col for col in feature_columns
            if col not in data.columns
        ]

        if missing:
            st.error(
                f"Missing {len(missing)} required gene-expression "
                "columns. Please upload data with the same feature "
                "schema used to train the model."
            )
            st.write("Example required columns:", feature_columns[:10])
            st.stop()

        X = data.loc[:, feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )

        if X.isna().any().any():
            st.error(
                "Gene-expression features contain missing or "
                "non-numeric values. Please clean the CSV first."
            )
            st.stop()

        if not np.isfinite(X.to_numpy()).all():
            st.error("The data contains infinite values.")
            st.stop()

        st.write("### Uploaded data preview")
        st.dataframe(data.head())

        if st.button("Run prediction", type="primary"):
            predictions = model.predict(X)

            results = pd.DataFrame({
                "Sample": range(1, len(X) + 1),
                "Predicted cancer type": predictions
            })

            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X)
                results["Maximum model score (%)"] = (
                    scores.max(axis=1) * 100
                ).round(2)

            st.success(f"Processed {len(results)} samples.")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download predictions CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="oncolens_predictions.csv",
                mime="text/csv"
            )

            st.caption(
                "Model scores are not necessarily calibrated "
                "probabilities and must not be interpreted as "
                "clinical certainty."
            )

    except Exception as e:
        st.error(f"Could not process the uploaded CSV: {e}")
