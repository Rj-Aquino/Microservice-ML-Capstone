from fastapi import HTTPException

import os
import json
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import pandas as pd

from scaler import load_department_scalers, scalers_X, scalers_y
from preprocess import inverse_transform_y

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# === Directory setup ===
BASE_MODEL_DIR = "saved_models"
DEPARTMENTS = ["Bus_Operations", "Bus_Finance", "Bus_Inventory", "Bus_HR"]

os.makedirs(BASE_MODEL_DIR, exist_ok=True)
for dept in DEPARTMENTS:
    os.makedirs(os.path.join(BASE_MODEL_DIR, dept), exist_ok=True)

# === PATH HELPERS ===
def get_model_path(department: str, model_type: str, ext: str):
    if department not in DEPARTMENTS:
        raise ValueError(f"Unknown department: {department}")
    return os.path.join(BASE_MODEL_DIR, department, f"model_{model_type}.{ext}")

# === TRAINING ===
def train_model(department: str, model_type: str, data: dict):
    import os
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.arima.model import ARIMA
    from prophet import Prophet
    from prophet.serialize import model_to_json

    dept_folder = os.path.join(BASE_MODEL_DIR, department)
    os.makedirs(dept_folder, exist_ok=True)

    model_type = model_type.lower()

    # === LINEAR REGRESSION ===
    if model_type == "linear":
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float).ravel()

        # Replace NaN or Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit scalers on all data
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1,1)).ravel()

        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y_scaled)

        # Save model and scalers
        joblib.dump(model, get_model_path(department, "linear", "pkl"))
        joblib.dump(x_scaler, os.path.join(dept_folder, "x_scaler.pkl"))
        joblib.dump(y_scaler, os.path.join(dept_folder, "y_scaler.pkl"))

        return {
            "message": f"Linear Regression model trained successfully for {department}!"
        }

    # === ARIMA ===
    elif model_type == "arima":
        series = pd.Series(data["values"])
        model = ARIMA(series, order=(2, 1, 2))
        model_fit = model.fit()
        model_fit.save(get_model_path(department, "arima", "pkl"))
        return {"message": f"ARIMA model trained successfully for {department}!"}

    # === PROPHET ===
    elif model_type == "prophet":
        df = pd.DataFrame({
            "ds": pd.to_datetime(data["dates"]),
            "y": data["values"]
        })
        model = Prophet()
        model.fit(df)
        with open(get_model_path(department, "prophet", "json"), "w") as f:
            f.write(model_to_json(model))
        return {"message": f"Prophet model trained successfully for {department}!"}

    else:
        raise ValueError("Unsupported model type")

# === PREDICTION ===
def predict(department: str, model_type: str, input_data: dict):
    import os
    import joblib
    import numpy as np
    from fastapi import HTTPException
    from statsmodels.tsa.arima.model import ARIMAResults
    from prophet import Prophet
    from prophet.serialize import model_from_json

    model_type = model_type.lower()

    # === LINEAR REGRESSION ===
    if model_type == "linear":
        model_path = get_model_path(department, "linear", "pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"No trained Linear model found for {department}")
        model = joblib.load(model_path)

        # Load scalers from same folder as model
        dept_folder = os.path.dirname(model_path)
        x_scaler_path = os.path.join(dept_folder, "x_scaler.pkl")
        y_scaler_path = os.path.join(dept_folder, "y_scaler.pkl")
        if not (os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path)):
            raise HTTPException(status_code=400, detail=f"Missing scalers for department '{department}'")
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)

        # Preprocess input
        X = np.array(input_data.get("X"), dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.size == 0 or np.isnan(X).any() or np.isinf(X).any():
            raise HTTPException(status_code=400, detail="Invalid input: contains NaN, Inf, or empty values.")

        # Scale, predict, inverse-transform
        X_scaled = x_scaler.transform(X)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        return {"department": department, "model_type": "linear", "predictions": y_pred.tolist()}

    # === ARIMA ===
    elif model_type == "arima":
        model_path = get_model_path(department, "arima", "pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"No trained ARIMA model found for {department}")
        model_fit = ARIMAResults.load(model_path)
        steps = input_data.get("steps_ahead", 5)
        return {"department": department, "model_type": "arima", "predictions": model_fit.forecast(steps=steps).tolist()}

    # === PROPHET ===
    elif model_type == "prophet":
        model_path = get_model_path(department, "prophet", "json")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"No trained Prophet model found for {department}")
        with open(model_path, "r") as f:
            model_json = f.read()
        model = model_from_json(model_json)
        steps = input_data.get("steps_ahead", 5)
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        return {"department": department, "model_type": "prophet", "predictions": forecast["yhat"].tail(steps).tolist()}

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

# === MODEL INFO METADATA ===
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
                "data": {"X": [[1], [2], [3], [4]], "y": [10, 20, 30, 40]}
            },
            "prediction": {
                "department": "Bus_Finance",
                "model_type": "linear",
                "input": {"X": [[5], [6]]}
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
                "data": {"values": [100, 120, 140, 160, 180]}
            },
            "prediction": {
                "department": "Bus_Operations",
                "model_type": "arima",
                "input": {"steps_ahead": 3}
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
                "input": {"steps_ahead": 3}
            }
        }
    }
}
