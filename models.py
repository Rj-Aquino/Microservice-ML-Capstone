import gzip
import pickle
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
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
from database import (
    SessionLocal,
    TrainedModel,
    save_file_to_db,
    read_file_from_db,
    save_model_to_db,
    load_model_from_db,
    delete_file_by_type
)

from scaler import (
   scalers_y,
   scalers_X,
   feature_dims
)


DEPARTMENTS = ["operations", "finance", "inventory", "hr"]

# ============================================================
# === TRAINING FUNCTIONS =====================================
# ============================================================

def train_model(department: str, model_type: str, cleaned_data: dict):
    """
    Train a model (linear, ARIMA, or Prophet) for a department.
    Old models/scalers are only deleted after successful training.
    """
    model_type = model_type.lower()
    department = department.lower()
    print(f"\n🚀 Training {model_type.upper()} model for '{department}' department")

    # ============================================
    # STEP 1: Train model in memory (safe section)
    # ============================================
    trained_model = None
    files_to_save = []

    if model_type == "linear":
        X = np.array(cleaned_data["X"], dtype=float)
        y = np.array(cleaned_data["y"], dtype=float)

        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        model = LinearRegression(fit_intercept=True)
        model.fit(X_scaled, y_scaled)

        print("🧠 Linear Regression trained successfully")

        trained_model = model
        files_to_save.append(("scaler_X", gzip.compress(pickle.dumps(scaler_X))))
        files_to_save.append(("scaler_y", gzip.compress(pickle.dumps(scaler_y))))

        # Save in-memory refs
        scalers_X[department] = scaler_X
        scalers_y[department] = scaler_y
        feature_dims[department] = X.shape[1]

    elif model_type == "arima":
        values = np.array(cleaned_data["values"], dtype=float)
        if len(values) < 3:
            raise HTTPException(status_code=400, detail="ARIMA requires ≥ 3 samples")

        try:
            trained_model = ARIMA(values, order=(2,1,2),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False).fit()
        except Exception as e:
            print(f"⚠️ ARIMA(2,1,2) failed, fallback (1,1,1): {e}")
            trained_model = ARIMA(values, order=(1,1,1),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False).fit()

    elif model_type == "prophet":
        df = pd.DataFrame({
            "ds": pd.to_datetime(cleaned_data["dates"]),
            "y": cleaned_data["values"]
        })
        trained_model = Prophet()
        trained_model.fit(df)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_type}")

    # ============================================
    # STEP 2: Replace DB entries atomically
    # ============================================
    db = SessionLocal()
    try:
        # --- Delete old entries ---
        if model_type in ["linear", "arima"]:
            old_model = db.query(TrainedModel).filter(
                TrainedModel.department == department,
                TrainedModel.model_type == model_type
            ).first()
            if old_model:
                db.delete(old_model)
                db.commit()
                print(f"[DB] 🗑️ Deleted previous {model_type} model for {department}")

        if model_type == "linear":
            for scaler_type in ["scaler_X", "scaler_y"]:
                delete_file_by_type(department, scaler_type)
                print(f"[DB] 🗑️ Deleted previous {scaler_type} for {department}")

        elif model_type == "prophet":
            delete_file_by_type(department, "prophet_model")
            print(f"[DB] 🗑️ Deleted previous Prophet model for {department}")

        # --- Save new model ---
        if model_type == "linear":
            save_model_to_db(trained_model, "linear", department)
            for ftype, fdata in files_to_save:
                save_file_to_db(f"{department}_{ftype}.pkl", fdata, ftype, department)
            print("💾 Linear model + scalers saved to DB")

        elif model_type == "arima":
            save_model_to_db(trained_model, "arima", department)
            print("💾 ARIMA model saved to DB")

        elif model_type == "prophet":
            save_file_to_db(
                f"{department}_prophet_model.json",
                model_to_json(trained_model).encode("utf-8"),
                "prophet_model",
                department
            )
            print("💾 Prophet model saved to DB")

        return {"message": f"{model_type.upper()} model retrained for {department}"}

    finally:
        db.close()

# ============================================================
# === PREDICTION =============================================
# ============================================================

def predict(department: str, model_type: str, input_data: dict):
    model_type = model_type.lower()
    department = department.lower()
    if department not in DEPARTMENTS:
        raise HTTPException(status_code=400, detail=f"Unknown department: {department}")

    # ============================
    # LINEAR
    # ============================
    if model_type == "linear":
        print(f"\n🔮 Predicting with Linear model for '{department}'")
        model = load_model_from_db(department=department, model_type="linear")

        # Assuming you saved scalers using the department name in file_id
        scaler_X = pickle.loads(gzip.decompress(read_file_from_db(filename=f"{department}_scaler_X.pkl", department=department)))
        scaler_y = pickle.loads(gzip.decompress(read_file_from_db(filename=f"{department}_scaler_y.pkl", department=department)))

        cleaned = clean_prediction_data(input_data, "linear")
        X = np.array(cleaned["X"], dtype=float)
        X_scaled = scaler_X.transform(X)
        preds_scaled = model.predict(X_scaled).reshape(-1, 1)
        preds = scaler_y.inverse_transform(preds_scaled).flatten()
        return {"department": department, "model_type": "linear", "predictions": preds.tolist()}

    # ============================
    # ARIMA
    # ============================
    elif model_type == "arima":
        print(f"\n🔮 Predicting with ARIMA model for '{department}'")
        model_fit = load_model_from_db(department=department, model_type="arima")

        steps = input_data.get("steps_ahead")
        if not isinstance(steps, int) or steps <= 0:
            raise HTTPException(status_code=400, detail="'steps_ahead' must be positive int.")
        preds = model_fit.forecast(steps=steps)
        return {"department": department, "model_type": "arima", "predictions": preds.tolist()}

    # ============================
    # PROPHET
    # ============================
    elif model_type == "prophet":
        print(f"\n🔮 Predicting with Prophet model for '{department}'")
        # Prophet is stored as a file
        model_json = read_file_from_db(filename=f"{department}_prophet_model.json", department=department)
        model = model_from_json(model_json.decode())

        steps = input_data.get("steps_ahead")
        if not isinstance(steps, int) or steps <= 0:
            raise HTTPException(status_code=400, detail="'steps_ahead' must be positive int.")
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        preds = forecast["yhat"].tail(steps).tolist()
        return {"department": department, "model_type": "prophet", "predictions": preds}

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

# ============================================================
# === VALIDATION / EVALUATION ================================
# ============================================================

def validate(department: str, model_type: str, test_data: dict):
    model_type = model_type.lower()
    department = department.lower()
    print(f"\n🧪 Validating {model_type.upper()} model for '{department}'")

    if model_type == "linear":
        cleaned = clean_training_data(test_data, "linear")
        validate_linear_data(cleaned)
        X = np.array(cleaned["X"], dtype=float)
        y_true = np.array(cleaned["y"], dtype=float)
        preds = predict(department, "linear", {"X": X.tolist()})["predictions"]

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        r2 = r2_score(y_true, preds)
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))

        return {"department": department, "model_type": "linear",
                "metrics": {"R2": r2, "MAE": mae, "RMSE": rmse},
                "y_true": y_true.tolist(), "y_pred": preds}

    elif model_type == "arima":
        cleaned = clean_training_data(test_data, "arima")
        validate_arima_data(cleaned)
        values = np.array(cleaned["values"], dtype=float)
        preds = predict(department, "arima", {"steps_ahead": len(values)})["predictions"]
        return {"department": department, "model_type": "arima", "y_true": values.tolist(), "y_pred": preds}

    elif model_type == "prophet":
        cleaned = clean_training_data(test_data, "prophet")
        validate_prophet_data(cleaned)
        values = np.array(cleaned["values"], dtype=float)
        preds = predict(department, "prophet", {"steps_ahead": len(values)})["predictions"]
        return {"department": department, "model_type": "prophet", "y_true": values.tolist(), "y_pred": preds}

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

# ============================================================
# === MODEL INFO METADATA ====================================
# ============================================================

MODEL_INFO = {
    "linear": {
        "description": (
            "Linear Regression models the relationship between one or more independent variables (features X) "
            "and a continuous dependent variable (target y) by fitting a linear equation."
        ),
        "use_cases": [
            "Predict numeric values based on multiple input features",
            "Cost estimation or performance forecasting",
            "Trend analysis in business or finance"
        ],
        "inputs": {
            "training": {
                "X": "List of feature vectors (List[List[float]]) representing multiple training samples",
                "y": "List of target values (List[float]) corresponding to each sample"
            },
            "prediction": {
                "X": "List of feature vectors (List[List[float]]) for which predictions are required"
            }
        },
        "output": "List of predicted numeric values corresponding to the input feature vectors",
        "notes": (
            "All input features must be numeric. Missing values are not allowed. "
            "Inputs are normalized internally before fitting. Suitable for modeling linear relationships only."
        ),
    },
    "arima": {
        "description": (
            "ARIMA (AutoRegressive Integrated Moving Average) models univariate time series data to forecast "
            "future values based on its own past observations and trends."
        ),
        "use_cases": [
            "Forecasting future demand, revenue, or sales",
            "Predicting inventory or stock prices",
            "Short-term and medium-term time series prediction"
        ],
        "inputs": {
            "training": {
                "values": "List of numeric values (List[float]) representing the time series data in chronological order"
            },
            "prediction": {
                "steps_ahead": "Number of future time steps to forecast (int)"
            }
        },
        "output": "List of forecasted numeric values for the requested future steps",
        "notes": (
            "Univariate series only. The target series must be numeric and contain no missing values. "
            "Does not handle multiple features or exogenous variables in this simplified version."
        ),
    },
    "prophet": {
        "description": (
            "Prophet is a time series forecasting tool that captures trend, seasonality, and holiday effects "
            "to generate robust forecasts, even with missing or irregular data."
        ),
        "use_cases": [
            "Revenue, ridership, or web traffic prediction",
            "Forecasting seasonal sales or demand",
            "Planning resource allocation based on future trends"
        ],
        "inputs": {
            "training": {
                "dates": "List of date strings (List[str]) representing the time points of the observed series",
                "values": "List of numeric values (List[float]) corresponding to each date"
            },
            "prediction": {
                "steps_ahead": "Number of future time points to forecast (int)"
            }
        },
        "output": "List of forecasted values aligned with future dates",
        "notes": (
            "Automatically handles missing dates, trends, and seasonality. "
            "Requires at least one date/time column and one numeric target column. "
            "Supports daily, weekly, or monthly data."
        ),
    },
}
