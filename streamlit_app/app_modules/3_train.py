# app_modules/3_train.py
import streamlit as st
import pandas as pd
import traceback
import json
import math
import numpy as np

# Simple detection logic used for UI defaulting
def _is_ui_regression(df: pd.DataFrame, target_col: str) -> bool:
    """Detects if a target column is likely continuous for UI purposes."""
    if target_col not in df.columns:
        return False
        
    y = df[target_col].dropna()
    
    if not pd.api.types.is_numeric_dtype(y):
        return False
        
    if pd.api.types.is_float_dtype(y):
        return True
        
    if y.nunique() <= 20:
        return False 
    
    return True

# --- CRITICAL FIX: The entire application logic starts here ---
def app():
    st.subheader("⚙️ Run Process Configuration") 
    st.markdown("---")

    df = st.session_state.get("df")
    if df is None:
        st.warning("Upload dataset first on the **⬆️ Upload Data** page.")
        return

    # Implementation of Toggle Mode using st.expander
    with st.expander("Pipeline Configuration", expanded=True):
        st.write("**Define parameters for the training pipeline.**")
        col_target, col_metric = st.columns(2)
        
        # 1. Target Column Selection & Type Detection
        with col_target:
            target_col = st.selectbox("Target Column", list(df.columns), key="train_target_col", help="Column to predict.")
            st.session_state["target_col"] = target_col
            
            is_regression = _is_ui_regression(df, target_col)
            
            if is_regression:
                default_metric = "rmse"
                metric_options = ["rmse", "mae", "r2", "mse", "mape"]
                default_maximize = False
                st.info("Target detected as **Regression**.")
            else:
                default_metric = "f1"
                metric_options = ["f1", "accuracy", "precision", "recall"]
                default_maximize = True
                st.info("Target detected as **Classification**.")

        # 2. Metric Selection (Toggle/Select Box)
        with col_metric:
            current_metric = st.session_state.get("metric_name", default_metric)
            if current_metric not in metric_options:
                current_metric = default_metric
                
            st.session_state["metric_name"] = st.selectbox(
                "Metric for Best Model Selection", 
                metric_options, 
                index=metric_options.index(current_metric),
                key="train_metric_name",
                help="Metric to select the best model."
            )
            
            st.session_state["maximize"] = st.checkbox(
                "Higher is better for this metric?", 
                value=st.session_state.get("maximize", default_maximize), 
                key="train_maximize",
                help="Check for accuracy/f1/r2; uncheck for rmse/mae/mse."
            )
            
        st.session_state["model_name"] = st.text_input("Registered MLflow Model Name", value=st.session_state.get("model_name", "battery_model"), key="train_model_name", help="Name used to register the final model in MLflow.")
        
        st.markdown("---")
        st.session_state["run_prefect"] = st.checkbox("✅ Use Prefect Flow", value=st.session_state.get("run_prefect", False), key="train_run_prefect", help="If checked, runs the pipeline via Prefect tasks.")


    if st.button("🚀 Run Pipeline", type="primary"):
        target = st.session_state["target_col"]
        metric = st.session_state["metric_name"]
        maximize = st.session_state["maximize"]
        model_name = st.session_state["model_name"]
        run_prefect = st.session_state["run_prefect"]
        
        try:
            if run_prefect:
                # --- Deployment-Ready Prefect Flow Logic (Passes DF) ---
                try:
                    from workflows.pipeline import full_pipeline_streamlit 
                except Exception as e:
                    st.warning(f"⚠️ Prefect flow import failed. Is Prefect installed? Error: {str(e)}")
                    full_pipeline_streamlit = None
                    
                if full_pipeline_streamlit:
                    with st.spinner("⏳ Running Prefect flow..."):
                        res = full_pipeline_streamlit(
                            df=df, 
                            target_col=target, 
                            metric_name=metric, 
                            maximize=maximize, 
                            model_name=model_name
                        )
                        st.success("✅ Prefect pipeline finished.")
                        if isinstance(res, dict):
                            st.json(res)
                            st.session_state["train_runs"] = res.get("runs", [])
                else:
                    st.error("❌ Cannot run Prefect flow; please uncheck 'Use Prefect Flow'.")
            else:
                # --- Local Training Fallback Logic ---
                try:
                    from src.training.train import train_and_log
                    from src.data_pipeline.preprocess import preprocess
                except Exception as e:
                    st.error("❌ Could not import training/preprocess functions. Check imports and packages.")
                    st.exception(e)
                    return

                st.info("Running local training with selected models...")
                
                try:
                    with st.spinner("⏳ Preprocessing data..."):
                        X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df, target)
                    st.success("✅ Preprocessing successful.")
                except Exception as e:
                    st.error("❌ Preprocess failed.")
                    st.exception(e)
                    return

                # Training Block
                try:
                    with st.spinner("⏳ Training models..."):
                        res = train_and_log(
                            X_train, y_train, X_val, y_val, 
                            experiment_name="mlops_demo", 
                            metric_name=metric, 
                            maximize=maximize
                        )
                except Exception as e:
                    st.error("🚨 Training failed during model fitting.")
                    st.exception(e)
                    return
                
                st.success("✅ Training done. Results logged to MLflow.")
                st.json(res)
                
                runs = res.get("runs") or []
                st.session_state["train_runs"] = runs
                for k in ["best_model_key", "best_run_id", "best_metric_value", "best_metric_name"]:
                    if k in res:
                        st.session_state[k] = res[k]

        except Exception as e:
            st.error("🚨 Pipeline failed during execution:")
            st.exception(e)

    # quick helper to show stored runs
    runs = st.session_state.get("train_runs")
    if runs:
        st.markdown("---")
        st.subheader("📚 Recent Runs Summary")
        st.write(f"Found **{len(runs)}** runs stored in session.")
        runs_df = pd.DataFrame([{"model_key": r.get("model_key") or r.get("model"), **r.get("metrics", {})} for r in runs])
        st.dataframe(runs_df, width='stretch')
        
        st.write("")
        if st.button("🗑️ Clear Stored Runs"):
            st.session_state.pop("train_runs", None)
            st.experimental_rerun()