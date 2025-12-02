# main.py
import argparse
import logging
from pathlib import Path

# ensure imports from project root work when running main.py
import sys
sys.path.append(str(Path(__file__).resolve().parents[0]))

from workflows.pipeline import full_pipeline  # your prefect flow (already updated)
# optionally import direct pipeline function if you want non-prefect run:
# from src.workflows.simple_pipeline import run_pipeline_direct

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlops_main")

def run_cli(csv_path: str, target_col: str, metric_name: str, maximize: bool, model_name: str):
    logger.info("Starting full pipeline (Prefect flow)...")
    result = full_pipeline(csv_path=csv_path, target_col=target_col, metric_name=metric_name,
                           maximize=maximize, model_name=model_name)
    logger.info(f"Pipeline finished. Result: {result}")
    return result

def create_arg_parser():
    p = argparse.ArgumentParser("MLOps full pipeline launcher")
    p.add_argument("--csv", "-c", type=str, default=r"C:\Users\Kira\Desktop\battery_mileage.csv",
                   help="Path to input file (CSV / PKL / JOBLIB / JSON / XLSX supported)")
    p.add_argument("--target", "-t", type=str, default="mileage", help="Target column name")
    p.add_argument("--metric", "-m", type=str, default="rmse", help="Metric to select best model")
    p.add_argument("--maximize", action="store_true", help="If set, larger metric is better (accuracy/f1/r2)")
    p.add_argument("--model-name", type=str, default="battery_model", help="Registered model name")
    return p

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    run_cli(csv_path=args.csv, target_col=args.target, metric_name=args.metric,
            maximize=args.maximize, model_name=args.model_name)

if __name__ == "__main__":
    main()
