# src/training/train.py
import math
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# core models (always available)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier, ExtraTreesRegressor, ExtraTreesClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier

# optional model libs (import safely)
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except Exception:
    CatBoostRegressor = CatBoostClassifier = None

# For Keras models (optional)
try:
    from tensorflow import keras
except Exception:
    keras = None


# --------------------------
# Helpers: detection & metrics
# --------------------------
def _is_regression_target(y: pd.Series) -> bool:
    """ 
    FIXED DETECTION LOGIC: Strongly prioritizes float type for Regression, 
    which aligns with UI behavior for continuous variables.
    """
    if y.dtype == 'object' or y.dtype.name == 'category' or pd.api.types.is_string_dtype(y):
        return False
        
    # --- CRITICAL FIX: Prioritize float type ---
    if pd.api.types.is_float_dtype(y):
        return True
    # --- END CRITICAL FIX ---
    
    if y.nunique() <= 20:
        return False 
        
    return True

def _regression_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse) if not math.isnan(mse) else float("nan")
    mae = mean_absolute_error(y_true, y_pred)
    try: r2 = r2_score(y_true, y_pred)
    except Exception: r2 = float("nan")
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2), "mape": float(mape)}

def _classification_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


# --------------------------
# Build model candidate dictionaries (Parallelization Implemented)
# --------------------------
def _get_regression_models():
    models = {
        "LinearReg": LinearRegression(), "RidgeReg": Ridge(), "LassoReg": Lasso(),
        "KNNReg": KNeighborsRegressor(), "SVR": SVR(),
        "RandomForestReg": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), 
        "ExtraTreesReg": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoostReg": GradientBoostingRegressor(random_state=42),
        "HistGradientBoostReg": HistGradientBoostingRegressor(random_state=42),
        "AdaBoostReg": AdaBoostRegressor(random_state=42),
        "MLPReg": MLPRegressor(max_iter=500, random_state=42),
    }
    if xgb is not None: models["XGBReg"] = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, random_state=42, n_jobs=-1)
    if lgb is not None: models["LGBMReg"] = lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if CatBoostRegressor is not None: models["CatBoostReg"] = CatBoostRegressor(verbose=0, random_state=42)
    return models

def _get_classification_models():
    models = {
        "LogisticReg": LogisticRegression(max_iter=500), "KNNClf": KNeighborsClassifier(), "SVC": SVC(probability=False),
        "RandomForestClf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "ExtraTreesClf": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoostClf": GradientBoostingClassifier(random_state=42),
        "HistGradientBoostClf": HistGradientBoostingClassifier(random_state=42),
        "AdaBoostClf": AdaBoostClassifier(random_state=42),
        "MLPClf": MLPClassifier(max_iter=500, random_state=42),
    }
    if xgb is not None: models["XGBClf"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
    if lgb is not None: models["LGBMClf"] = lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if CatBoostRegressor is not None: models["CatBoostClf"] = CatBoostClassifier(verbose=0, random_state=42)
    return models


# --------------------------
# Main training + logging function
# --------------------------
def train_and_log(
    X_train, y_train, X_val, y_val,
    experiment_name: str = "mlops_demo",
    candidate_models: list | None = None,
    metric_name: str | None = None, maximize: bool | None = None, mode: str = "auto"
):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    is_reg = _is_regression_target(y_train) if mode == "auto" else (mode == "regression")
    
    # Target Label Encoding Check (Runs only if classification)
    if not is_reg and not pd.api.types.is_integer_dtype(y_train):
        print("[TRAIN] Converting non-integer classification target to LabelEncoders.")
        le = LabelEncoder()
        y_train_encoded = pd.Series(le.fit_transform(y_train), name=y_train.name)
        try:
            y_val_encoded = pd.Series(le.transform(y_val), name=y_val.name)
        except ValueError as e:
            print(f"[TRAIN WARNING] Validation set contains labels unseen in training: {e}. Skipping prediction.")
            y_val_encoded = None
        y_train = y_train_encoded
        y_val = y_val_encoded

    reg_models = _get_regression_models()
    clf_models = _get_classification_models()
    models_to_try = reg_models if is_reg else clf_models
    
    # Set input example for MLflow
    input_example = X_train.head(1) 

    if mode == "all": models_to_try = {**reg_models, **clf_models}
    if candidate_models: models_to_try = {k: v for k, v in models_to_try.items() if k in candidate_models}

    # FIX: Ensure selection metric is an available metric
    selection_metric = metric_name 
    available_metrics = ("rmse", "mae", "r2", "mse", "mape") if is_reg else ("f1", "accuracy", "precision", "recall")
    
    # If user-selected metric is not available (e.g., rmse on classification), fall back to default
    if selection_metric not in available_metrics:
        fallback = "rmse" if is_reg else "f1"
        print(f"[TRAIN WARNING] Requested metric '{selection_metric}' is not available. Falling back to '{fallback}'.")
        selection_metric = fallback
    
    if maximize is None: 
        maximize = True if selection_metric in ("accuracy", "precision", "recall", "f1", "r2") else False

    runs = []
    best_metric, best_run_id, best_model_key = None, None, None

    for key, model in models_to_try.items():
        if not is_reg and y_val is None:
             print(f"[TRAIN] Skipping {key} because validation set encoding failed.")
             continue
             
        with mlflow.start_run(run_name=key):
            try: model.fit(X_train, y_train)
            except Exception as e:
                mlflow.log_param("fit_error", str(e))
                print(f"[TRAIN] Skipping {key} due to fit error: {e}")
                continue

            try: preds = model.predict(X_val)
            except Exception as e:
                mlflow.log_param("predict_error", str(e))
                print(f"[TRAIN] Predict failed for {key}: {e}")
                continue

            if is_reg: metrics = _regression_metrics(y_val, preds)
            else: metrics = _classification_metrics(y_val, preds)

            for m_k, m_v in metrics.items():
                if isinstance(m_v, float) and math.isnan(m_v): mlflow.log_metric(m_k, -1.0)
                else: mlflow.log_metric(m_k, float(m_v))

            mlflow.log_param("model_key", key)
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path=key, # Used by register.py
                registered_model_name=None,
                input_example=input_example
            )

            run_id = mlflow.active_run().info.run_id
            runs.append({"model_key": key, "run_id": run_id, "metrics": metrics})

            cur_val = metrics.get(selection_metric)
            if cur_val is not None:
                if best_metric is None or \
                   (maximize and cur_val > best_metric) or \
                   (not maximize and cur_val < best_metric):
                    best_metric = cur_val
                    best_run_id = run_id
                    best_model_key = key

    result = {
        "is_regression": is_reg, "runs": runs, "best_model_key": best_model_key,
        "best_run_id": best_run_id, "best_metric_name": selection_metric,
        "best_metric_value": best_metric,
    }
    print(f"[TRAIN] Completed. Best: {best_model_key} | {selection_metric}={best_metric}")
    return result