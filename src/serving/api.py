from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Battery Mileage Prediction API")

# Load the production model from MLflow
MODEL_NAME = "battery_model"

print("[API] Loading Production model...")

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/Production"
)

print("[API] Model loaded successfully.")


@app.get("/")
def home():
    return {"message": "Battery Mileage Prediction API is running."}


@app.post("/predict")
def predict(features: dict):
    """
    Expects JSON input:
    {
        "feature1": value,
        "feature2": value,
        ...
    }
    """

    df = pd.DataFrame([features])
    preds = model.predict(df)

    return {"prediction": preds[0]}
