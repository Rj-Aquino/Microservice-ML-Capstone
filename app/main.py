import os
from fastapi import FastAPI, HTTPException, Query
import numpy as np
from pydantic import BaseModel
from typing import Dict, Any, Optional

from database import save_training_input, get_all_training_inputs, TrainingInput, SessionLocal, UploadedDataset
from models import train_model, predict, validate, MODEL_INFO, load_uploaded_dataset
from preprocess import clean_training_data  
from scaler import delete_and_retrain_department
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, UploadFile, Form, HTTPException
from datetime import datetime
import pandas as pd
import os
import json

from database import save_uploaded_dataset, get_uploaded_datasets, delete_uploaded_dataset

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
    dataset_id: Optional[str] = None
    department: Optional[str] = None
    model_type: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class PredictRequest(BaseModel):
    department: str
    model_type: str
    data: Optional[Dict[str, Any]] = None
    steps_ahead: Optional[int] = 1

class AccuracyRequest(BaseModel):
    department: str
    model_type: str
    data: Optional[Dict[str, Any]] = None

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
    dataset_id = getattr(payload, "dataset_id", None)
    data = payload.data  # Optional if using dataset_id
    department = None
    model_type = None

    if not dataset_id and not data:
        raise HTTPException(status_code=400, detail="Must provide either 'dataset_id' or raw 'data' for training.")

    # ============================
    # 1️⃣ Load prepared dataset if dataset_id is provided
    # ============================
    if dataset_id:
        print(f"📂 Loading prepared dataset: {dataset_id}")

        # Load prepared X/y
        prepared_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_prepared.json")
        if not os.path.exists(prepared_path):
            raise HTTPException(status_code=404, detail=f"Prepared JSON for dataset_id '{dataset_id}' not found.")

        with open(prepared_path, "r") as f:
            prepared = json.load(f)

        X = prepared.get("X")
        y = prepared.get("y")
        if not X or not y:
            raise HTTPException(status_code=400, detail="Prepared dataset missing 'X' or 'y'.")
        data = {"X": X, "y": y}

        # Load department/model_type from original uploaded dataset JSON
        original_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.json")
        if os.path.exists(original_path):
            with open(original_path, "r") as f:
                meta = json.load(f)
                department = meta.get("department")
                model_type = meta.get("model_type", "").lower()

    # ============================
    # 2️⃣ If data is raw JSON, use provided payload metadata
    # ============================
    if data and not dataset_id:
        department = getattr(payload, "department", None)
        model_type = getattr(payload, "model_type", None).lower() if getattr(payload, "model_type", None) else None

    # ============================
    # 3️⃣ Validate department and model_type
    # ============================
    valid_departments = ["Operations", "Finance", "Inventory", "HR"]
    valid_model_types = ["linear", "arima", "prophet"]

    if department not in valid_departments:
        raise HTTPException(status_code=400, detail=f"Department '{department}' not supported.")
    if model_type not in valid_model_types:
        raise HTTPException(status_code=400, detail=f"Model type '{model_type}' not supported.")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid 'data' format — expected dict or dataset_id.")

    print(f"🚀 Starting training for {department} - {model_type}")

    # ============================
    # 4️⃣ Clean the data
    # ============================
    try:
        cleaned_data = clean_training_data(data, model_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data cleaning failed: {str(e)}")

    # ============================
    # 5️⃣ Save cleaned data
    # ============================
    save_training_input(department, model_type, cleaned_data)

    # ============================
    # 6️⃣ Train the model
    # ============================
    result = train_model(department, model_type, cleaned_data)

    return {
        "detail": f"✅ Model trained successfully for {department} ({model_type}).",
        "department": department,
        "model_type": model_type,
        "dataset_id": dataset_id or "N/A",
        "result": result
    }

# =====================================================
# 🔮 PREDICTION ENDPOINT
# =====================================================
@app.post("/predict")
async def predict_endpoint(payload: PredictRequest):
    department = payload.department
    model_type = payload.model_type.lower()
    input_data = payload.data
    steps_ahead = getattr(payload, "steps_ahead", 1)  # only relevant for ARIMA/Prophet

    if not isinstance(input_data, dict):
        raise HTTPException(status_code=400, detail="Invalid 'data' format — expected dict.")

    # Linear models ignore steps_ahead
    if model_type == "linear":
        result = predict(department, model_type, input_data)
    else:  # ARIMA/Prophet
        result = predict(department, model_type, input_data, steps_ahead)

    return {"detail": "Prediction successful.", "result": result}

# =====================================================
# 📈 VALIDATION / ACCURACY CHECK ENDPOINT
# =====================================================
@app.post("/validate_accuracy")
def validate_accuracy(payload: AccuracyRequest):
    department = payload.department
    model_type = payload.model_type.lower()
    test_data = payload.data

    if not isinstance(test_data, dict):
        raise HTTPException(status_code=400, detail="Invalid 'data' format — expected dict.")

    # Call validate function
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


UPLOAD_DIR = "uploaded_datasets"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload-data")
async def upload_data(
    department: str = Form(...),
    model_type: str = Form(...),
    dataset_name: str = Form(...),
    file: UploadFile = None
):
    # 🔹 Validate file input
    if not file or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a valid CSV file.")

    # 🔹 Validate department
    valid_departments = ["Operations", "Finance", "Inventory", "HR"]
    if department not in valid_departments:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department '{department}'. Must be one of: {', '.join(valid_departments)}"
        )

    # 🔹 Validate model type
    valid_model_types = ["linear", "arima", "prophet"]
    if model_type.lower() not in valid_model_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type '{model_type}'. Must be one of: {', '.join(valid_model_types)}"
        )

    # 🔹 Generate dataset ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_id = f"{department}_{dataset_name}_{timestamp}"

    # 🔹 Save raw CSV
    csv_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    with open(csv_path, "wb") as buffer:
        buffer.write(await file.read())

    # 🔹 Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    # 🔹 Clean and convert
    df = df.dropna().drop_duplicates()
    data_json = df.to_dict(orient="list")

    # 🔹 Save JSON
    json_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.json")
    with open(json_path, "w") as f:
        json.dump({
            "department": department,
            "model_type": model_type.lower(),
            "dataset_name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "data": data_json
        }, f, indent=2)

    # 🔹 Store metadata in DB
    metadata = {
        "dataset_id": dataset_id,
        "department": department,
        "model_type": model_type.lower(),
        "dataset_name": dataset_name,
        "file_path": csv_path,
        "json_path": json_path,
        "records": len(df),
        "columns": list(df.columns),
    }
    save_uploaded_dataset(metadata)

    return {
        "message": "✅ Dataset uploaded and saved successfully.",
        "dataset_id": dataset_id,
        "records": len(df),
        "columns": list(df.columns)
    }

@app.post("/prepare-dataset")
async def prepare_dataset(
    dataset_id: str = Form(...),
    target_column: str = Form(...),
):
    db = SessionLocal()
    dataset = db.query(UploadedDataset).filter(UploadedDataset.dataset_id == dataset_id).first()
    if not dataset:
        db.close()
        return {"message": "❌ Dataset not found.", "dataset_id": dataset_id, "success": False}

    # Check if JSON file exists
    if not os.path.exists(dataset.json_path):
        db.close()
        return {
            "message": f"❌ Dataset file not found at '{dataset.json_path}'",
            "dataset_id": dataset_id,
            "success": False
        }

    # Load dataset JSON
    with open(dataset.json_path, "r") as f:
        data = json.load(f)["data"]

    df = pd.DataFrame(data)

    if target_column not in df.columns:
        db.close()
        return {"message": f"❌ Target column '{target_column}' not found.", "dataset_id": dataset_id, "success": False}

    # Convert and prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.factorize(X[col])[0]

    X_rows = X.values.tolist()
    y_rows = y.tolist()

    prepared_data = {
        "department": dataset.department,
        "model_type": dataset.model_type,
        "X": X_rows,
        "y": y_rows
    }

    prepared_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_prepared.json")
    with open(prepared_path, "w") as f:
        json.dump(prepared_data, f, indent=2)

    db.close()
    return {
        "message": "✅ Dataset prepared successfully.",
        "dataset_id": dataset_id,
        "target_column": target_column,
        "columns": list(X.columns),
        "prepared_path": prepared_path,
        "success": True
    }

@app.get("/datasets")
def list_datasets(
    department: Optional[str] = None,
    model_type: Optional[str] = None
):
    return get_uploaded_datasets(department, model_type)

@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    return delete_uploaded_dataset(dataset_id)