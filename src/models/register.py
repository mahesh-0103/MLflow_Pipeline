# src/models/register.py
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException # Added for clarity

# ------------------------------------------------------------
# REGISTER BEST MODEL USING USER METRIC
# ------------------------------------------------------------
def register_best_model(
    experiment_name: str,
    model_name: str,
    metric_name: str,
    maximize: bool = False
):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    exp_id = exp.experiment_id
    runs = client.search_runs(exp_id, order_by=[f"metrics.{metric_name} DESC" if maximize else f"metrics.{metric_name} ASC"])

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    # --- FIX 1: Handle NULL best_model_key/best_run_id (Use top run if best is NULL) ---
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    
    # Get the key used for the model artifact from the run's tags
    model_key = best_run.data.tags.get('mlflow.runName')

    if model_key is None:
        # Fallback if runName tag is missing (shouldn't happen with our train.py)
        raise MlflowException("Best run does not have a valid 'mlflow.runName' tag.")
    # --- END FIX 1 ---

    metric_value = best_run.data.metrics.get(metric_name)

    # --- CRITICAL FIX 2: Use the model_key as the artifact path ---
    # The artifact path must match the name used in train_and_log (which is the model_key)
    mv = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/{model_key}", 
        name=model_name
    )
    # --- END CRITICAL FIX 2 ---

    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production"
    )

    print(f"[REGISTRY] Best run: {best_run_id}")
    print(f"[REGISTRY] Best model key: {model_key}")
    print(f"[REGISTRY] Best {metric_name}: {metric_value}")
    print(f"[REGISTRY] Model '{model_name}' version {mv.version} promoted to PRODUCTION")

    return mv.version, metric_value, best_run_id