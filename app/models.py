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
            model_json = f.read()  # read as string
            model = model_from_json(model_json)
        future = model.make_future_dataframe(periods=input_data.get("steps_ahead", 5))
        forecast = model.predict(future)
        return forecast["yhat"].tail(input_data.get("steps_ahead", 5)).tolist()

    else:
        raise ValueError("Unsupported model type")


MODEL_INFO = {
    "linear": {
        "description": "Linear Regression models the relationship between one or more input variables (X) and a continuous output (y).",
        "use_cases": [
            "Predicting cost or expenses based on measurable factors",
            "Forecasting numeric values when relationships are roughly linear"
        ],
        "inputs": {
            "training": {
                "X": "List[List[float]] - multiple input features per sample",
                "y": "List[float] - target output"
            },
            "prediction": {
                "X": "List[List[float]] - same number of features used in training"
            }
        },
        "output": "Predicted numeric values for each input sample",
        "notes": "Works best for static, numeric data with linear relationships. Sensitive to unscaled or unnormalized inputs.",
        "examples": {
            "training": {
                "department": "Bus_Finance",
                "model_type": "linear",
                "data": {
                    "X": [[1], [2], [3], [4]],
                    "y": [10, 20, 30, 40]
                }
            },
            "prediction": {
                "department": "Bus_Finance",
                "model_type": "linear",
                "input": {
                    "X": [[5], [6]]
                }
            }
        }
    },
    "arima": {
        "description": "ARIMA (AutoRegressive Integrated Moving Average) predicts future values in a time series based on its past values.",
        "use_cases": [
            "Predicting passenger counts, revenue, or demand over time",
            "Forecasting trends with consistent seasonal or temporal patterns"
        ],
        "inputs": {
            "training": {"values": "List[float] - sequential numeric data"},
            "prediction": {"steps_ahead": "int - number of future steps to forecast"}
        },
        "output": "List[float] - predicted future values",
        "notes": "Univariate only. Ensure the time series is stationary or properly differenced.",
        "examples": {
            "training": {
                "department": "Bus_Operations",
                "model_type": "arima",
                "data": {
                    "values": [100, 120, 140, 160, 180]
                }
            },
            "prediction": {
                "department": "Bus_Operations",
                "model_type": "arima",
                "input": {
                    "steps_ahead": 3
                }
            }
        }
    },
    "prophet": {
        "description": "Prophet is a time-series forecasting model developed by Facebook, designed for data with trends, seasonality, and holiday effects.",
        "use_cases": [
            "Predicting revenue, ridership, or resource usage over time",
            "Data with daily/weekly/monthly seasonality or irregular trends"
        ],
        "inputs": {
            "training": {
                "dates": "List[str] - in YYYY-MM-DD format",
                "values": "List[float] - observed values corresponding to each date"
            },
            "prediction": {"steps_ahead": "int - number of future days to forecast"}
        },
        "output": "List[float] - forecasted values for the given future dates",
        "notes": "Can include additional regressors for external factors (e.g., weather, events).",
        "examples": {
            "training": {
                "department": "Bus_HR",
                "model_type": "prophet",
                "data": {
                    "dates": ["2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04"],
                    "values": [30, 32, 34, 36]
                }
            },
            "prediction": {
                "department": "Bus_HR",
                "model_type": "prophet",
                "input": {
                    "steps_ahead": 3
                }
            }
        }
    }
}