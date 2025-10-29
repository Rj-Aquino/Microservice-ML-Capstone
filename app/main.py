from fastapi import FastAPI, HTTPException, Query
import numpy as np
from pydantic import BaseModel
from typing import Dict, Any

from database import save_training_input, get_all_training_inputs, TrainingInput, SessionLocal
from models import train_model, predict, validate, MODEL_INFO
from preprocess import clean_training_data  
from scaler import delete_and_retrain_department
from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# ⚙️ APP INITIALIZATION
# =====================================================
app = FastAPI(title="Dynamic Forecasting Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 🧩 Pydantic Request Models
# =====================================================
class TrainingRequest(BaseModel):
    department: str
    model_type: str
    data: Dict[str, Any]

class PredictRequest(BaseModel):
    department: str
    model_type: str
    data: Dict[str, Any]

class AccuracyRequest(BaseModel):
    department: str
    model_type: str
    data: Dict[str, Any]

# =====================================================
# 📘 Model Info Endpoints
# =====================================================
@app.get("/model-info")
async def get_all_model_info():
    return {"models": MODEL_INFO}

@app.get("/model-info/{model_type}")
async def get_model_info(model_type: str):
    info = MODEL_INFO.get(model_type.lower())
    if not info:
        raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not found.")
    return {model_type: info}

# =====================================================
# 🧠 TRAINING ENDPOINT
# =====================================================
@app.post("/train")
async def train_model_endpoint(payload: TrainingRequest):
    department = payload.department
    model_type = payload.model_type.lower()
    data = payload.data

    if department not in ["Operations", "Finance", "Inventory", "HR"]:
        raise HTTPException(status_code=400, detail="Department not supported.")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid 'data' format — expected dict.")

    # 🧹 Step 1: Clean the data (NO normalization here)
    try:
        cleaned_data = clean_training_data(data, model_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data cleaning failed: {str(e)}")

    # 💾 Step 2: Save cleaned (not normalized) data to DB
    save_training_input(department, model_type, cleaned_data)

    # 🧠 Step 3: Train model using cleaned data
    result = train_model(department, model_type, cleaned_data)

    return {"detail": "Model trained successfully."}

# =====================================================
# 🔮 PREDICTION ENDPOINT
# =====================================================
@app.post("/predict")
async def predict_endpoint(payload: PredictRequest):
    department = payload.department
    model_type = payload.model_type.lower()
    input_data = payload.data

    if not isinstance(input_data, dict):
        raise HTTPException(status_code=400, detail="Invalid 'input_data' format — expected dict.")

    result = predict(department, model_type, input_data)
    return {"detail": "Prediction successful.", "result": result}

# =====================================================
# 📈 VALIDATION / ACCURACY CHECK ENDPOINT
# =====================================================
@app.post("/validate_accuracy")
def validate_accuracy(payload: AccuracyRequest):
    department = payload.department
    model_type = payload.model_type.lower()
    test_data = payload.data

    result = validate(department, model_type, test_data)
    return {"detail": "Validation complete.", "result": result}

# =====================================================
# 🧾 TRAINING DATA RETRIEVAL ENDPOINT
# =====================================================
@app.get("/data")
def read_all_training_data(
    department: str | None = Query(default=None, description="Filter by department"),
    model_type: str | None = Query(default=None, description="Filter by model type")
):
    return get_all_training_inputs(department=department, model_type=model_type)

# =====================================================
# 🗑️ DELETE + RETRAIN ENDPOINT
# =====================================================
@app.delete("/train/{train_id}")
async def delete_trained_model(train_id: int):
    """
    Delete a training input, then retrain remaining data for that department/model.
    """
    db = SessionLocal()
    try:
        entry = db.query(TrainingInput).filter(TrainingInput.id == train_id).first()
        if not entry:
            raise HTTPException(status_code=404, detail=f"Training input with ID {train_id} not found.")

        department = entry.department
        model_type = entry.model_type

        db.delete(entry)
        db.commit()

        retrain_result = delete_and_retrain_department(department, model_type)
        return {
            "detail": f"Training input {train_id} deleted successfully.",
            "retrain_result": retrain_result,
        }

    finally:
        db.close()
