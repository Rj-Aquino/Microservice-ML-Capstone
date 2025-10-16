import os
import json
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import pandas as pd

# Base folder for models
BASE_MODEL_DIR = "saved_models"
DEPARTMENTS = ["Bus_Operations", "Bus_Finance", "Bus_Inventory", "Bus_HR"]

# Ensure folders exist
for dept in DEPARTMENTS:
    os.makedirs(os.path.join(BASE_MODEL_DIR, dept), exist_ok=True)

def get_model_path(department: str, model_type: str, ext: str):
    if department not in DEPARTMENTS:
        raise ValueError(f"Unknown department: {department}")
    return os.path.join(BASE_MODEL_DIR, department, f"model_{model_type}.{ext}")

def train_model(department: str, model_type: str, data: dict):
    if model_type == "linear":
        X = np.array(data["X"]).reshape(-1, 1)
        y = np.array(data["y"])
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, get_model_path(department, model_type, "pkl"))
        return {"message": f"Linear Regression trained successfully for {department}!"}

    elif model_type == "arima":
        series = pd.Series(data["values"])
        model = ARIMA(series, order=(2, 1, 2))
        model_fit = model.fit()
        model_fit.save(get_model_path(department, model_type, "pkl"))
        return {"message": f"ARIMA model trained successfully for {department}!"}

    elif model_type == "prophet":
        df = pd.DataFrame({
            "ds": pd.to_datetime(data["dates"]),
            "y": data["values"]
        })
        model = Prophet()
        model.fit(df)
        with open(get_model_path(department, model_type, "json"), "w") as f:
            f.write(model_to_json(model))
        return {"message": f"Prophet model trained successfully for {department}!"}

    else:
        raise ValueError("Unsupported model type")

def predict(department: str, model_type: str, input_data: dict):
    if model_type == "linear":
        model = joblib.load(get_model_path(department, model_type, "pkl"))
        X = np.array(input_data["X"]).reshape(-1, 1)
        return model.predict(X).tolist()

    elif model_type == "arima":
        from statsmodels.tsa.arima.model import ARIMAResults
        model_fit = ARIMAResults.load(get_model_path(department, model_type, "pkl"))
        steps = input_data.get("steps_ahead", 5)
        return model_fit.forecast(steps=steps).tolist()

    elif model_type == "prophet":
        with open(get_model_path(department, model_type, "json"), "r") as f:
            model = model_from_json(json.load(f))
        future = model.make_future_dataframe(periods=input_data.get("steps_ahead", 5))
        forecast = model.predict(future)
        return forecast["yhat"].tail(input_data.get("steps_ahead", 5)).tolist()

    else:
        raise ValueError("Unsupported model type")
