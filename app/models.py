import os
import numpy as np
import pandas as pd
import joblib
from fastapi import HTTPException
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.metrics import r2_score

from scaler import save_department_scalers, scalers_X, scalers_y
from preprocess import validate_and_normalize, inverse_transform_y

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
    model_type = model_type.lower()
    dept_folder = os.path.join(BASE_MODEL_DIR, department)
    os.makedirs(dept_folder, exist_ok=True)

    # --- LINEAR ---
    if model_type == "linear":
        processed = validate_and_normalize(model_type, data, is_training=True, department=department)
        X_scaled = np.array(processed["X"], dtype=float)
        y_scaled = np.array(processed["y"], dtype=float).ravel()

        model = LinearRegression(fit_intercept=False)
        model.fit(X_scaled, y_scaled)

        model_path = get_model_path(department, "linear", "pkl")
        joblib.dump(model, model_path)

        # Save scalers consistently
        save_department_scalers(department)

        print(f"[DEBUG] Linear model and scalers saved for {department}")
        return {"message": f"Linear Regression trained for {department}"}

    # --- ARIMA ---
    elif model_type == "arima":
        series = pd.Series(data["values"])
        model = ARIMA(series, order=(2, 1, 2))
        model_fit = model.fit()
        model_fit.save(get_model_path(department, "arima", "pkl"))
        return {"message": f"ARIMA model trained successfully for {department}!"}

    # --- PROPHET ---
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
    model_type = model_type.lower()

    # --- LINEAR ---
    if model_type == "linear":
        model_path = get_model_path(department, "linear", "pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"No model found for {department}")
        model = joblib.load(model_path)

        # Scale input X using saved scalers
        processed = validate_and_normalize(model_type, input_data, is_training=False, department=department)
        X_scaled = np.array(processed["X"], dtype=float)

        y_pred_scaled = model.predict(X_scaled).reshape(-1, 1)
        y_pred = inverse_transform_y(y_pred_scaled, department).ravel()

        return {
            "department": department,
            "model_type": "linear",
            "predictions": y_pred.tolist()
        }

    # --- ARIMA ---
    elif model_type == "arima":
        model_path = get_model_path(department, "arima", "pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"No trained ARIMA model found for {department}")
        model_fit = ARIMAResults.load(model_path)
        steps = input_data.get("steps_ahead", 5)
        return {"department": department, "model_type": "arima", "predictions": model_fit.forecast(steps=steps).tolist()}

    # --- PROPHET ---
    elif model_type == "prophet":
        model_path = get_model_path(department, "prophet", "json")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"No trained Prophet model found for {department}")
        with open(model_path, "r") as f:
            model = model_from_json(f.read())
        steps = input_data.get("steps_ahead", 5)
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        return {"department": department, "model_type": "prophet", "predictions": forecast["yhat"].tail(steps).tolist()}

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

# === VALIDATION ===
def validate(department: str, model_type: str, test_data: dict):
    model_type = model_type.lower()
    try:
        if model_type == "linear":
            X_test = np.array(test_data.get("X", []), dtype=float)
            y_true = np.array(test_data.get("y", []), dtype=float)
            if X_test.size == 0 or y_true.size == 0:
                raise HTTPException(status_code=400, detail="Missing test data for X or y.")
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            if y_true.ndim > 1:
                y_true = y_true.ravel()

            prediction_response = predict(department, "linear", {"X": X_test.tolist()})
            y_pred = np.array(prediction_response["predictions"], dtype=float)
            r2 = r2_score(y_true, y_pred)

            return {"department": department, "model_type": "linear", "R2_score": float(r2),
                    "y_true": y_true.tolist(), "y_pred": y_pred.tolist()}

        elif model_type == "arima":
            series = np.array(test_data.get("values", []), dtype=float)
            if len(series) < 3:
                raise HTTPException(status_code=400, detail="ARIMA requires at least 3 values for testing.")
            prediction_response = predict(department, "arima", {"steps_ahead": len(series)})
            y_pred = np.array(prediction_response.get("predictions", []), dtype=float)
            return {"department": department, "model_type": "arima",
                    "y_true": series.tolist(), "y_pred": y_pred.tolist()}

        elif model_type == "prophet":
            dates = pd.to_datetime(test_data.get("dates", []))
            values = np.array(test_data.get("values", []), dtype=float)
            if len(dates) != len(values):
                raise HTTPException(status_code=400, detail="'dates' and 'values' must have same length.")
            prediction_response = predict(department, "prophet", {"steps_ahead": len(values)})
            y_pred = np.array(prediction_response.get("predictions", []), dtype=float)
            return {"department": department, "model_type": "prophet",
                    "y_true": values.tolist(), "y_pred": y_pred.tolist()}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model_type: {model_type}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# === MODEL INFO METADATA ===
MODEL_INFO = {
    "linear": {
        "description": "Linear Regression models the relationship between input X and output y.",
        "use_cases": ["Predict numeric values", "Cost forecasting"],
        "inputs": {"training": {"X": "List[List[float]]", "y": "List[float]"},
                   "prediction": {"X": "List[List[float]]"}},
        "output": "Predicted numeric values",
        "notes": "Requires scaled inputs",
    },
    "arima": {
        "description": "ARIMA predicts future values in a time series.",
        "use_cases": ["Forecasting demand or revenue"],
        "inputs": {"training": {"values": "List[float]"}, "prediction": {"steps_ahead": "int"}},
        "output": "Predicted future values",
        "notes": "Univariate only",
    },
    "prophet": {
        "description": "Prophet forecasts time series with seasonality.",
        "use_cases": ["Revenue or ridership prediction"],
        "inputs": {"training": {"dates": "List[str]", "values": "List[float]"},
                   "prediction": {"steps_ahead": "int"}},
        "output": "Forecasted values",
        "notes": "Handles trends and seasonal effects",
    }
}
