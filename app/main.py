from fastapi import FastAPI, HTTPException, Query, Path
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any

from database import save_training_input, get_all_training_inputs
from models import train_model, predict, validate  # your existing model functions
from scaler import delete_and_retrain_department

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
async def train_model_endpoint(payload: dict):
    department = payload.get("department")
    model_type = payload.get("model_type")
    data = payload.get("data")

    if not department or not model_type or not data:
        raise HTTPException(status_code=400, detail="Missing required fields: department, model_type, or data.")

    # ✅ Save training input to database
    save_training_input(department, model_type, data)

    # ✅ Train the model (includes normalization)
    result = train_model(department, model_type, data)

    return result
    
@app.post("/predict")
async def predict_endpoint(payload: dict):
    department = payload.get("department")
    model_type = payload.get("model_type")
    input_data = payload.get("input_data")

    if not department or not model_type or not input_data:
        raise HTTPException(status_code=400, detail="Missing required fields: department, model_type, or input_data.")

    # ✅ Do NOT normalize here; the predict() function handles it.
    result = predict(department, model_type, input_data)
    return result

@app.post("/validate_accuracy")
def validate_accuracy(payload: AccuracyRequest):
    """
    Validate model accuracy using test data.
    Returns R² for Linear, predicted vs actual for ARIMA/Prophet.
    Supports multi-feature X for Linear Regression.
    """
    return validate(payload.department, payload.model_type, payload.test_data)

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
    """Delete a trained model entry and its related files, then retrain from remaining data."""
    db = SessionLocal()
    try:
        # 🔹 Fetch the entry first
        entry = db.query(TrainingInput).filter(TrainingInput.id == train_id).first()
        if not entry:
            raise HTTPException(status_code=404, detail=f"Training input with id {train_id} not found.")

        department = entry.department
        model_type = entry.model_type

        # 🔹 Delete from database
        db.delete(entry)
        db.commit()

        # 🔹 Delete assets and retrain if remaining data exists
        if department and model_type:
            result = delete_and_retrain_department(department, model_type)
        else:
            result = {"message": "No department/model_type info; skipped retrain."}

        return {
            "detail": f"Training input {train_id} deleted successfully.",
            "retrain_result": result
        }

    finally:
        db.close()
