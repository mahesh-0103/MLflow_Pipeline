# app_modules/1_upload.py
import streamlit as st
import pandas as pd
from pathlib import Path
import io
import json
import pickle
import joblib

SUPPORTED = [".csv", ".pkl", ".pickle", ".joblib", ".json", ".xlsx"]


def _read_uploaded(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in (".pkl", ".pickle"):
        uploaded_file.seek(0)
        return pickle.load(uploaded_file)
    if suffix == ".json":
        uploaded_file.seek(0)
        return pd.read_json(uploaded_file)
    if suffix == ".xlsx":
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)
    if suffix == ".joblib":
        uploaded_file.seek(0)
        return joblib.load(uploaded_file)
    # fallback to pandas csv
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def app():
    st.subheader("Data Upload and Verification")
    st.markdown("---")

    # Use a single column layout now that the server path is removed
    col1, col2 = st.columns([1, 1]) 

    with col1:
        st.subheader("📁 File Uploader")
        st.caption("Upload CSV / PKL / JSON / XLSX / joblib.")
        uploaded = st.file_uploader(
            "Upload dataset", type=[s.lstrip('.') for s in SUPPORTED], accept_multiple_files=False
        )
    
    # col2 is now used for informational display, removing the old explicit tag
    with col2:
        pass # Intentionally empty

    df = None
    
    # Uploaded via browser (Deployment Path)
    if uploaded is not None:
        try:
            df = _read_uploaded(uploaded)
            if isinstance(df, (dict, list)):
                df = pd.DataFrame(df)
            
            # --- FIX: Explicitly cast boolean-like object columns to bool to prevent ArrowTypeError ---
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        temp = df[col].replace({'True': True, 'False': False, 1: True, 0: False})
                        if temp.isin([True, False]).all() or temp.isna().all():
                            df[col] = temp.astype(pd.BooleanDtype())
                    except Exception:
                        pass
            # --- END FIX ---

            st.success("✅ File successfully parsed into DataFrame")
        except Exception as e:
            st.error(f"🚨 Failed to parse upload: {e}")

    st.markdown("---")
    
    if df is not None:
        st.session_state["df"] = df
        st.session_state["upload_file_name"] = uploaded.name if uploaded else None
        
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head(200), width='stretch')
        
        col_info_1, col_info_2 = st.columns(2)
        with col_info_1:
            st.metric("Rows", df.shape[0])
        with col_info_2:
            st.metric("Columns", df.shape[1])
            
        st.markdown(f"**Column Names:** `{', '.join(list(df.columns))}`")
        
    else:
        st.info("Awaiting file load. Please upload a file via the browser.")