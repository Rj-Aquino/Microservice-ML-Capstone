from fastapi import FastAPI, HTTPException, Query, Path
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any

from sklearn.metrics import r2_score

from preprocess import validate_and_normalize

from database import save_training_input, get_all_training_inputs
from models import train_model, predict  # your existing model functions
from scaler import delete_department_assets

from models import MODEL_INFO
from database import TrainingInput, SessionLocal

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Dynamic Forecasting Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] if you want to restrict it
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Request Models
# -----------------------
class TrainingRequest(BaseModel):
    department: str
    model_type: str
    data: Dict[str, Any]

    model_config = {"protected_namespaces": ()}

class PredictRequest(BaseModel):
    department: str
    model_type: str
    input: Dict[str, Any]

    model_config = {"protected_namespaces": ()}

class AccuracyRequest(BaseModel):
    department: str
    model_type: str
    test_data: Dict[str, Any]  # For linear: {"X": [...], "y": [...]}, ARIMA/Prophet: {"values": [...]} or {"dates": [...], "values": [...]}

    model_config = {"protected_namespaces": ()}

# -----------------------
# Endpoints
# -----------------------
@app.get("/model-info")
async def get_all_model_info():
    return {"models": MODEL_INFO}

@app.get("/model-info/{model_type}")
async def get_model_info(model_type: str):
    info = MODEL_INFO.get(model_type.lower())
    if not info:
        return {"error": f"Model type '{model_type}' not found."}
    return {model_type: info}

@app.post("/train")
def train(payload: TrainingRequest):
    try:
        raw_data = payload.data  # Keep raw data

        # 1️⃣ Validate and normalize raw input for training
        validated_data = validate_and_normalize(
            payload.model_type,
            raw_data,
            is_training=True,
            department=payload.department
        )

        # 2️⃣ Save the raw data (not scaled) to the DB
        save_training_input(payload.department, payload.model_type, raw_data)

        # 3️⃣ Train the model using normalized/scaled data
        result = train_model(payload.department, payload.model_type, validated_data)

        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/predict")
async def predict_model(payload: dict):
    department = payload.get("department")
    model_type = payload.get("model_type")
    input_data = payload.get("input")

    if not department:
        raise HTTPException(status_code=400, detail="Department is required for prediction.")
    if not model_type:
        raise HTTPException(status_code=400, detail="Model type is required.")

    # ✅ Pass department into validation
    validated = validate_and_normalize(model_type, input_data, is_training=False, department=department)
    
    preds = predict(department, model_type, validated)
    return {"department": department, "model_type": model_type, "predictions": preds}

@app.post("/validate_accuracy")
def validate_accuracy(payload: AccuracyRequest):
    """
    Validate model accuracy using test data.
    Returns R² for Linear, predicted vs actual for ARIMA/Prophet.
    Supports multi-feature X for Linear Regression.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score
    from fastapi import HTTPException

    dept = payload.department
    model_type = payload.model_type.lower()
    data = payload.test_data

    try:
        # === LINEAR REGRESSION ===
        if model_type == "linear":
            X_test = np.array(data.get("X", []), dtype=float)
            y_true = np.array(data.get("y", []), dtype=float)

            # Ensure proper shapes
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            if y_true.ndim > 1:
                y_true = y_true.ravel()

            # Predict using same department model
            prediction_response = predict(dept, model_type, {"X": X_test.tolist()})
            if isinstance(prediction_response, dict) and "predictions" in prediction_response:
                y_pred = np.array(prediction_response["predictions"], dtype=float)
            else:
                y_pred = np.array(prediction_response, dtype=float)

            score = r2_score(y_true, y_pred)
            return {
                "R2_score": float(score),
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist()
            }

        # === ARIMA MODEL ===
        elif model_type == "arima":
            series = np.array(data.get("values", []), dtype=float)
            if len(series) < 3:
                raise HTTPException(status_code=400, detail="ARIMA requires at least 3 values for testing")

            steps = len(series)
            prediction_response = predict(dept, model_type, {"steps_ahead": steps})
            y_pred = np.array(prediction_response.get("predictions", []), dtype=float)

            return {"y_true": series.tolist(), "y_pred": y_pred.tolist()}

        # === PROPHET MODEL ===
        elif model_type == "prophet":
            dates = pd.to_datetime(data.get("dates", []))
            values = np.array(data.get("values", []), dtype=float)
            if len(dates) != len(values):
                raise HTTPException(status_code=400, detail="'dates' and 'values' must have same length")

            steps = len(values)
            prediction_response = predict(dept, model_type, {"steps_ahead": steps})
            y_pred = np.array(prediction_response.get("predictions", []), dtype=float)

            return {"y_true": values.tolist(), "y_pred": y_pred.tolist()}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model_type: {model_type}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/data")
def read_all_training_data(
    department: str | None = Query(default=None, description="Filter by department"),
    model_type: str | None = Query(default=None, description="Filter by model type")
):
    try:
        return get_all_training_inputs(department=department, model_type=model_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/train/{train_id}")
async def delete_trained_model(train_id: int):
    """Delete a trained model entry and its related files."""
    db = SessionLocal()
    try:
        # 🔹 Fetch the entry first
        entry = db.query(TrainingInput).filter(TrainingInput.id == train_id).first()
        if not entry:
            raise HTTPException(status_code=404, detail=f"Training input with id {train_id} not found.")

        department = entry.department

        # 🔹 Delete from database
        db.delete(entry)
        db.commit()

        # 🔹 Delete related assets (models + scalers)
        if department:
            delete_department_assets(department)

        return {"detail": f"Training input {train_id} and assets for '{department}' deleted successfully."}

    finally:
        db.close()