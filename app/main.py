from fastapi import FastAPI, HTTPException, Query, Path
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from preprocess import inverse_transform_y, preprocess_input_data, preprocess_training_data
from database import save_training_input, get_all_training_inputs, delete_training_input
from models import train_model, predict  # your existing model functions

from validation import validate_and_normalize
from models import MODEL_INFO


app = FastAPI(title="Dynamic Forecasting Microservice")

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
        # 1️⃣ Validate and normalize raw input
        validated_data = validate_and_normalize(payload.model_type, payload.data, is_training=True)

        # 2️⃣ Preprocess for linear regression (multi-feature support)
        if payload.model_type.lower() == "linear":
            X_scaled, y_scaled = preprocess_training_data(validated_data, payload.department)
            # Flatten y for sklearn, X can be multi-dimensional
            validated_data = {"X": X_scaled.tolist(), "y": y_scaled.ravel().tolist()}

        # 3️⃣ Save the processed training input
        save_training_input(payload.department, payload.model_type, validated_data)

        # 4️⃣ Train the model
        result = train_model(payload.department, payload.model_type, validated_data)

        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def forecast(payload: PredictRequest):
    """
    Predict using a specific department's trained model.
    Supports multi-feature X for Linear Regression.
    """
    try:
        # ✅ Validate input
        validated_input = validate_and_normalize(payload.model_type, payload.input, is_training=False)

        # For linear, preprocess with scaler
        if payload.model_type.lower() == "linear":
            X_scaled = preprocess_input_data(validated_input, payload.department)
            validated_input = {"X": X_scaled.tolist()}

        predictions = predict(payload.department, payload.model_type, validated_input)
        return {"predictions": predictions}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/validate_accuracy")
def validate_accuracy(payload: AccuracyRequest):
    """
    Validate model accuracy using test data.
    Returns R² for Linear, predicted vs actual for ARIMA/Prophet.
    Supports multi-feature X for Linear Regression.
    """
    dept = payload.department
    model_type = payload.model_type.lower()
    data = payload.test_data

    try:
        if model_type == "linear":
            X_test = np.array(data.get("X", []), dtype=float)
            y_true = np.array(data.get("y", []), dtype=float)

            # Reshape if single feature
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            if y_true.ndim > 1:
                y_true = y_true.ravel()

            # Normalize using department scaler
            X_scaled = preprocess_input_data({"X": X_test.tolist()}, dept)
            y_pred = np.array(predict(dept, model_type, {"X": X_scaled.tolist()}), dtype=float)

            score = r2_score(y_true, y_pred)
            return {"R2_score": score, "y_true": y_true.tolist(), "y_pred": y_pred.tolist()}

        elif model_type == "arima":
            series = np.array(data.get("values", []), dtype=float)
            if len(series) < 3:
                raise HTTPException(status_code=400, detail="ARIMA requires at least 3 values for testing")

            steps = len(series)
            y_pred = np.array(predict(dept, model_type, {"steps_ahead": steps}), dtype=float)
            return {"y_true": series.tolist(), "y_pred": y_pred.tolist()}

        elif model_type == "prophet":
            dates = pd.to_datetime(data.get("dates", []))
            values = np.array(data.get("values", []), dtype=float)
            if len(dates) != len(values):
                raise HTTPException(status_code=400, detail="'dates' and 'values' must have same length")

            steps = len(values)
            y_pred = np.array(predict(dept, model_type, {"steps_ahead": steps}), dtype=float)
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

@app.delete("/train/{input_id}")
def delete_training_input_endpoint(input_id: int = Path(..., description="ID of the training input to delete")):
    """
    Delete a specific training input by its ID.
    """
    try:
        result = delete_training_input(input_id)
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
