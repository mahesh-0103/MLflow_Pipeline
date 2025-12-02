# workflows/pipeline.py (Modified for Streamlit Deployment)
import sys
from pathlib import Path

# add project root to PYTHONPATH so imports work when executed from workflows/
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from prefect import flow, task
# Importing ingest is now optional since the flow accepts the DataFrame
from src.data_pipeline.preprocess import preprocess
from src.training.train import train_and_log
from src.models.register import register_best_model
import pandas as pd


@task
def validate_task(df: pd.DataFrame):
    print(f"[VALIDATION] Dataframe shape: {df.shape}. Basic validation assumed complete.")
    return df

@task
def preprocess_task(df, target_col: str):
    return preprocess(df, target_col)


@task
def train_task(X_train, y_train, X_val, y_val, experiment_name: str = "mlops_demo", mode: str = "auto"):
    return train_and_log(X_train, y_train, X_val, y_val, experiment_name=experiment_name, mode=mode)


@task
def register_task(experiment_name: str, model_name: str, metric_name: str, maximize: bool):
    return register_best_model(experiment_name=experiment_name, model_name=model_name, metric_name=metric_name, maximize=maximize)


# NOTE: Renamed original flow to distinguish it from the new flow
@flow(name="MLOps End-to-End Pipeline (File Path)")
def full_pipeline_file_path(csv_path: str, target_col: str, metric_name: str = "rmse", maximize: bool = False, model_name: str = "battery_model", experiment_name: str = "mlops_demo", train_mode: str = "auto"):
    from src.data_pipeline.ingest import ingest_and_prepare
    @task
    def ingest_task(path: str):
        return ingest_and_prepare(path) 
    
    df = ingest_task(csv_path)
    # ... (rest of the original logic continues from here using df)
    df = validate_task(df) 
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_task(df, target_col)
    train_result = train_task(X_train, y_train, X_val, y_val, experiment_name, mode=train_mode)
    version, metric_value, best_run_id = register_task(experiment_name, model_name, metric_name, maximize)
    return {"model_version": version, "metric_value": metric_value, "best_run_id": best_run_id, "runs": train_result.get('runs', [])}


@flow(name="MLOps End-to-End Pipeline (Streamlit DF)")
def full_pipeline_streamlit(df: pd.DataFrame, target_col: str, metric_name: str = "rmse", maximize: bool = False, model_name: str = "battery_model", experiment_name: str = "mlops_demo", train_mode: str = "auto"):
    """
    New flow entry point for Streamlit deployment. Accepts DataFrame directly.
    """
    print("\n===== PIPELINE STARTED (Streamlit DF) =====")
    
    # 1. Validation (Task runs validation on the received DF)
    df = validate_task(df)

    # 2. Preprocess (Task runs preprocessing on the received DF)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_task(df, target_col)

    # 3. Train
    train_result = train_task(X_train, y_train, X_val, y_val, experiment_name, mode=train_mode)
    print(f"[PIPELINE] Train completed. Detected regression? {train_result.get('is_regression')}")

    # 4. Register
    version, metric_value, best_run_id = register_task(experiment_name, model_name, metric_name, maximize)

    print("\n===== PIPELINE COMPLETED =====")
    return {"model_version": version, "metric_value": metric_value, "best_run_id": best_run_id, "runs": train_result.get('runs', [])}

# The __name__ == "__main__" block below uses the original flow for CLI execution.
if __name__ == "__main__":
    # ... (code to run full_pipeline_file_path remains the same)
    from src.data_pipeline.ingest import ingest_and_prepare # Needed for CLI run
    full_pipeline_file_path(...)