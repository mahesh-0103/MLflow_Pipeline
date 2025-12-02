# app_modules/4_model_compare.py
import streamlit as st
import pandas as pd
import json

# --- MAPPING FOR VERBOSE DISPLAY ---
# This map ensures the technical key (e.g., 'rf_clf') is converted to a clear name (e.g., 'Random Forest Classifier')
VERBOSE_MODEL_MAP = {
    "LinearReg": "Linear Regression", "RidgeReg": "Ridge Regression", "LassoReg": "Lasso Regression",
    "KNNReg": "KNN Regressor", "SVR": "Support Vector Reg.", "RandomForestReg": "Random Forest Regressor", 
    "ExtraTreesReg": "Extra Trees Regressor", "GradientBoostReg": "Gradient Boost Reg.",
    "HistGradientBoostReg": "Hist Gradient Boost Reg.", "AdaBoostReg": "AdaBoost Regressor",
    "MLPReg": "MLP Regressor", "XGBReg": "XGBoost Regressor", "LGBMReg": "LGBM Regressor", 
    "CatBoostReg": "CatBoost Regressor",
    
    "LogisticReg": "Logistic Regression", "KNNClf": "KNN Classifier", "SVC": "Support Vector Clf.",
    "RandomForestClf": "Random Forest Classifier", "ExtraTreesClf": "Extra Trees Classifier",
    "GradientBoostClf": "Gradient Boost Clf.", "HistGradientBoostClf": "Hist Gradient Boost Clf.",
    "AdaBoostClf": "AdaBoost Classifier", "MLPClf": "MLP Classifier", "XGBClf": "XGBoost Classifier", 
    "LGBMClf": "LGBM Classifier", "CatBoostClf": "CatBoost Classifier",
}

# Helper function for user-friendly name display
def get_display_name(key: str) -> str:
    if key is None or key == "":
        return "N/A (No Best Model Selected)"
    
    # Use the verbose map, otherwise default to replacing underscores
    return VERBOSE_MODEL_MAP.get(key, key.replace('_', ' ').title())

def app():
    st.subheader("🏆 Comparative Results and Registration")
    st.markdown("---")
    runs = st.session_state.get("train_runs")
    if not runs:
        st.info("No training runs available in the session. Please run the **Run Process** page first.")
        return

    st.subheader("All Training Runs")
    
    # --- Dynamic Metric Display ---
    runs_data = [r.get("metrics", {}) for r in runs]
    all_metrics = set().union(*(d.keys() for d in runs_data))
    
    metadata_cols = ["Model"]
    display_cols = metadata_cols + sorted(list(all_metrics))
    
    df = pd.DataFrame([
        {
            "Model": get_display_name(r.get("model_key") or r.get("model", "")), 
            **r.get("metrics", {})
        } 
        for r in runs
    ])
    
    df = df.reindex(columns=display_cols, fill_value='—') 
    
    # Ensure the dataframe width is stretched to show full names
    st.dataframe(df, width='stretch')

    # --- END Dynamic Metric Display ---

    st.markdown("---")

    st.subheader("Best Model Summary")
    
    best_metric = st.session_state.get("best_metric_name", "N/A")
    best_value = st.session_state.get("best_metric_value", None)
    best_model_key = st.session_state.get("best_model_key", None)

    # Convert best_value for display
    display_value = "—"
    if best_value is not None:
        try:
            display_value = f"{best_value:.4f}"
        except TypeError:
            display_value = str(best_value)

    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("Best Metric", best_metric)
    col_2.metric("Best Value", display_value)
    col_3.metric("Best Model", get_display_name(best_model_key)) # Displays the full verbose name

    st.markdown("---")

    st.subheader("Downloads")
    st.download_button(
        label="⬇️ Download Runs JSON", 
        data=json.dumps(runs, indent=4, default=str), 
        file_name="runs.json",
        mime="application/json",
        help="Download all metadata for the training runs."
    )