import os
import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

# === Local imports ===
from preprocess import (
    clean_training_data,
    validate_linear_data,
    validate_arima_data,
    validate_prophet_data,
    clean_prediction_data
)
from scaler import get_scaler_path

# === Setup ===
BASE_MODEL_DIR = "saved_models"
SCALER_DIR = "scalers"
DEPARTMENTS = ["Operations", "Finance", "Inventory", "HR"]
UPLOAD_DIR = "uploaded_datasets"

def load_uploaded_dataset(dataset_id: str) -> dict:
    """Load uploaded dataset by ID (CSV or JSON) and return as dict."""
    csv_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    json_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.json")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(json_path):
        df = pd.read_json(json_path)
    else:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")

    if df.empty:
        raise HTTPException(status_code=400, detail=f"Dataset {dataset_id} is empty.")

    # Optional: drop nulls, duplicates
    df = df.dropna().drop_duplicates()

    return df.to_dict(orient="list")

os.makedirs(BASE_MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
for dept in DEPARTMENTS:
    os.makedirs(os.path.join(BASE_MODEL_DIR, dept), exist_ok=True)

def get_model_path(department: str, model_type: str, ext: str = "pkl") -> str:
    """
    Return the file path for a department-specific model.
    Example: get_model_path("Finance", "linear", "pkl") => "saved_models/Finance/model_linear.pkl"
    """
    if department not in DEPARTMENTS:
        raise ValueError(f"Unknown department: {department}")

    model_type_clean = model_type.lower()
    filename = f"model_{model_type_clean}.{ext}"
    return os.path.join(BASE_MODEL_DIR, department, filename)

# ============================================================
# === TRAINING FUNCTIONS =====================================
# ============================================================

def train_model(department: str, model_type: str, cleaned_data: dict):
    """
    Trains a model using cleaned (but not normalized) data.
    Handles normalization, model fitting, and model/scaler saving.

    Assumes:
      - Data is already cleaned and validated in the API layer.
      - Database saving is already done in the endpoint.
    """
    model_type = model_type.lower()

    if department not in DEPARTMENTS:
        raise HTTPException(status_code=400, detail=f"Unknown department: {department}")

    print(f"\n🚀 Training {model_type.upper()} model for '{department}' department")

    # ============================
    # LINEAR MODEL
    # ============================
    if model_type == "linear":
        X = np.array(cleaned_data["X"], dtype=float)
        y = np.array(cleaned_data["y"], dtype=float)

        # --- Normalize (for training only)
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        print(f"📊 Normalized Linear data: X={X_scaled.shape}, y={y_scaled.shape}")

        # --- Train model
        model = LinearRegression(fit_intercept=True)
        model.fit(X_scaled, y_scaled)
        print("🧠 Linear Regression trained successfully")

        # --- Save model + scalers
        joblib.dump(model, get_model_path(department, "linear", "pkl"))
        joblib.dump(scaler_X, get_scaler_path(department, "X"))
        joblib.dump(scaler_y, get_scaler_path(department, "y"))
        print("💾 Linear model and scalers saved")

        return {"message": f"Linear model trained for {department}"}

    # ============================
    # ARIMA MODEL
    # ============================
    elif model_type == "arima":
        values = np.array(cleaned_data["values"], dtype=float)

        if len(values) < 3:
            raise HTTPException(status_code=400, detail="ARIMA requires ≥ 3 samples")

        print(f"📊 ARIMA input size: {len(values)} samples")

        # ⚙️ Use safer ARIMA settings for small or unstable data
        try:
            model = ARIMA(
                values,
                order=(2, 1, 2),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit()
        except Exception as e:
            print(f"⚠️ ARIMA fitting failed with order (2,1,2): {e}")
            # fallback to a simpler, more stable configuration
            model = ARIMA(
                values,
                order=(1, 1, 1),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit()

        # ✅ Save model
        model_fit.save(get_model_path(department, "arima", "pkl"))
        print("🧠 ARIMA model trained and saved")

        return {"message": f"ARIMA model trained for {department}", "samples": len(values)}

    # ============================
    # PROPHET MODEL
    # ============================
    elif model_type == "prophet":
        df = pd.DataFrame({
            "ds": pd.to_datetime(cleaned_data["dates"]),
            "y": cleaned_data["values"]
        })
        print(f"📊 Prophet input: {len(df)} rows")

        model = Prophet()
        model.fit(df)
        with open(get_model_path(department, "prophet", "json"), "w") as f:
            f.write(model_to_json(model))
        print("🧠 Prophet model trained and saved")

        return {"message": f"Prophet model trained for {department}", "samples": len(df)}

    # ============================
    # UNSUPPORTED
    # ============================
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

# ============================================================
# === PREDICTION =============================================
# ============================================================

def predict(department: str, model_type: str, input_data: dict):
    """
    Predicts using a trained model. Assumes model and scalers are saved.
    Input data should already be cleaned (not normalized).
    """
    model_type = model_type.lower()

    if model_type == "linear":
        print(f"\n🔮 Predicting with Linear model for '{department}'")

        model_path = get_model_path(department, "linear", "pkl")
        scaler_X_path = get_scaler_path(department, "X")
        scaler_y_path = get_scaler_path(department, "y")

        # --- Check all assets exist
        if not all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_y_path]):
            raise HTTPException(status_code=404, detail="Linear model or scalers not found.")

        # --- Load assets
        model = joblib.load(model_path)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)

        # --- Validate and prepare data
        cleaned = clean_prediction_data(input_data, "linear")

        X = np.array(cleaned["X"], dtype=float)
        X_scaled = scaler_X.transform(X)
        print(f"📊 Prediction input shape: {X.shape}")

        # --- Predict and inverse-transform
        preds_scaled = model.predict(X_scaled).reshape(-1, 1)
        preds = scaler_y.inverse_transform(preds_scaled).flatten()
        print(f"✅ Linear predictions done ({len(preds)} samples)")

        return {
            "department": department,
            "model_type": "linear",
            "predictions": preds.tolist()
        }

    elif model_type == "arima":
        print(f"\n🔮 Predicting with ARIMA model for '{department}'")

        model_path = get_model_path(department, "arima", "pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="ARIMA model not found.")

        steps = input_data.get("steps_ahead")
        if not isinstance(steps, int) or steps <= 0:
            raise HTTPException(status_code=400, detail="'steps_ahead' must be positive int.")

        model_fit = ARIMAResults.load(model_path)
        preds = model_fit.forecast(steps=steps)
        print(f"✅ ARIMA predictions generated ({steps} steps ahead)")

        return {
            "department": department,
            "model_type": "arima",
            "predictions": preds.tolist()
        }

    elif model_type == "prophet":
        print(f"\n🔮 Predicting with Prophet model for '{department}'")

        model_path = get_model_path(department, "prophet", "json")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Prophet model not found.")

        steps = input_data.get("steps_ahead")
        if not isinstance(steps, int) or steps <= 0:
            raise HTTPException(status_code=400, detail="'steps_ahead' must be positive int.")

        with open(model_path, "r") as f:
            model = model_from_json(f.read())

        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        preds = forecast["yhat"].tail(steps).tolist()
        print(f"✅ Prophet predictions generated ({steps} future points)")

        return {
            "department": department,
            "model_type": "prophet",
            "predictions": preds
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

# ============================================================
# === VALIDATION / EVALUATION ================================
# ============================================================

def validate(department: str, model_type: str, test_data: dict):
    """
    Validates a trained model using test data.
    Cleans data, performs prediction, and computes performance metrics.
    """
    model_type = model_type.lower()
    print(f"\n🧪 Validating {model_type.upper()} model for '{department}'")

    if model_type == "linear":
        cleaned = clean_training_data(test_data, "linear")
        validate_linear_data(cleaned)

        X = np.array(cleaned["X"], dtype=float)
        y_true = np.array(cleaned["y"], dtype=float)

        print(f"📊 Validation data shape: X={X.shape}, y={y_true.shape}")

        preds = predict(department, "linear", {"X": X.tolist()})["predictions"]

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        r2 = r2_score(y_true, preds)
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))

        print(f"✅ Validation complete → R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

        return {
            "department": department,
            "model_type": "linear",
            "metrics": {"R2": r2, "MAE": mae, "RMSE": rmse},
            "y_true": y_true.tolist(),
            "y_pred": preds,
        }

    elif model_type == "arima":
        cleaned = clean_training_data(test_data, "arima")
        validate_arima_data(cleaned)

        values = np.array(cleaned["values"], dtype=float)
        preds = predict(department, "arima", {"steps_ahead": len(values)})["predictions"]

        print(f"✅ ARIMA validation complete ({len(values)} samples)")

        return {
            "department": department,
            "model_type": "arima",
            "y_true": values.tolist(),
            "y_pred": preds,
        }

    elif model_type == "prophet":
        cleaned = clean_training_data(test_data, "prophet")
        validate_prophet_data(cleaned)

        values = np.array(cleaned["values"], dtype=float)
        preds = predict(department, "prophet", {"steps_ahead": len(values)})["predictions"]

        print(f"✅ Prophet validation complete ({len(values)} samples)")

        return {
            "department": department,
            "model_type": "prophet",
            "y_true": values.tolist(),
            "y_pred": preds,
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

# ============================================================
# === MODEL INFO METADATA ====================================
# ============================================================

MODEL_INFO = {
    "linear": {
        "description": "Linear Regression models relationships between input X and output y.",
        "use_cases": ["Predict numeric values", "Cost or performance forecasting"],
        "inputs": {"training": {"X": "List[List[float]]", "y": "List[float]"},
                   "prediction": {"X": "List[List[float]]"}},
        "output": "Predicted numeric values",
        "notes": "Inputs are normalized internally before fitting.",
    },
    "arima": {
        "description": "ARIMA predicts future values in a time series.",
        "use_cases": ["Forecasting demand or revenue"],
        "inputs": {"training": {"values": "List[float]"},
                   "prediction": {"steps_ahead": "int"}},
        "output": "Future values",
        "notes": "Univariate time-series only.",
    },
    "prophet": {
        "description": "Prophet forecasts time series with seasonality and trend adjustments.",
        "use_cases": ["Revenue or ridership prediction"],
        "inputs": {"training": {"dates": "List[str]", "values": "List[float]"},
                   "prediction": {"steps_ahead": "int"}},
        "output": "Forecasted values",
        "notes": "Automatically handles missing data and seasonality.",
    },
}
