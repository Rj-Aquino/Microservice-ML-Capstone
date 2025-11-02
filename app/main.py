import os
from fastapi import FastAPI, HTTPException, Query
import numpy as np
from pydantic import BaseModel
from typing import Dict, Any, Optional

from database import save_training_input, get_all_training_inputs, TrainingInput, SessionLocal, UploadedDataset
from models import train_model, predict, validate, MODEL_INFO
from preprocess import clean_training_data  
from scaler import delete_and_retrain_department
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, UploadFile, Form, HTTPException
from datetime import datetime
import pandas as pd
import os
import json

from database import save_uploaded_dataset, get_uploaded_datasets, delete_uploaded_dataset, get_prepared_datasets, delete_prepared_dataset, PreparedDataset

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
    prepared_id: Optional[str] = None
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
    prepared_id = getattr(payload, "prepared_id", None)
    data = getattr(payload, "data", None)
    department = None
    model_type = None

    if not prepared_id and not data:
        raise HTTPException(status_code=400, detail="Must provide either 'prepared_id' or raw 'data' for training.")

    # ============================
    # 1️⃣ Load prepared dataset if prepared_id is provided
    # ============================
    if prepared_id:
        print(f"📂 Loading prepared dataset: {prepared_id}")

        # Locate prepared dataset in DB
        db = SessionLocal()
        prepared_entry = db.query(PreparedDataset).filter(PreparedDataset.prepared_id == prepared_id).first()
        if not prepared_entry:
            db.close()
            raise HTTPException(status_code=404, detail=f"Prepared dataset '{prepared_id}' not found in database.")

        prepared_path = prepared_entry.prepared_json_path
        if not os.path.exists(prepared_path):
            db.close()
            raise HTTPException(status_code=404, detail=f"Prepared JSON file not found at '{prepared_path}'")

        # Load prepared JSON
        with open(prepared_path, "r") as f:
            prepared_data = json.load(f)

        model_type = prepared_entry.model_type.lower()
        department = prepared_entry.department

        # Extract training data depending on model type
        if model_type == "linear":
            X = prepared_data.get("X")
            y = prepared_data.get("y")
            if not X or not y:
                db.close()
                raise HTTPException(status_code=400, detail="Prepared dataset missing 'X' or 'y' for linear model.")
            data = {"X": X, "y": y}

        elif model_type == "arima":
            series = prepared_data.get("data", {}).get("values")
            if not series:
                db.close()
                raise HTTPException(status_code=400, detail="Prepared ARIMA dataset missing 'values'.")
            data = {"values": series}

        elif model_type == "prophet":
            prophet_data = prepared_data.get("data")
            if not prophet_data:
                db.close()
                raise HTTPException(status_code=400, detail="Prepared Prophet dataset missing 'data'.")
            data = prophet_data

        else:
            db.close()
            raise HTTPException(status_code=400, detail=f"Unsupported model type '{model_type}'")

        db.close()

    # ============================
    # 2️⃣ If raw data provided (no prepared_id)
    # ============================
    elif data:
        department = getattr(payload, "department", None)
        model_type = getattr(payload, "model_type", None).lower() if getattr(payload, "model_type", None) else None

    # ============================
    # 3️⃣ Validate metadata
    # ============================
    valid_departments = ["Operations", "Finance", "Inventory", "HR"]
    valid_model_types = ["linear", "arima", "prophet"]

    if department not in valid_departments:
        raise HTTPException(status_code=400, detail=f"Department '{department}' not supported.")
    if model_type not in valid_model_types:
        raise HTTPException(status_code=400, detail=f"Model type '{model_type}' not supported.")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid 'data' format — expected dict or prepared_id.")

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
        "prepared_id": prepared_id or "N/A",
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


# =====================================
# Dataset Endpoints
# =====================================
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

    # 🔹 Ensure upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

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

    # 🔹 Save JSON representation
    json_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.json")
    with open(json_path, "w") as f:
        json.dump({
            "dataset_id": dataset_id,
            "department": department,
            "model_type": model_type.lower(),
            "dataset_name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "columns": list(df.columns),
            "records": len(df),
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

    saved_entry = save_uploaded_dataset(metadata)

    return {
        "message": "✅ Dataset uploaded and saved successfully.",
        "dataset_id": dataset_id,
        "records": len(df),
        "columns": list(df.columns),
        "db_id": saved_entry.get("id")
    }

@app.get("/datasets")
def list_datasets(
    department: Optional[str] = None,
    model_type: Optional[str] = None
):
    """List uploaded datasets."""
    return get_uploaded_datasets(department, model_type)

@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    """Delete uploaded dataset."""
    return delete_uploaded_dataset(dataset_id)

# =====================================
# Prepared Dataset Endpoints
# =====================================
@app.post("/prepare-dataset")
async def prepare_dataset(
    dataset_id: str = Form(...),
    target_column: str = Form(...),
    selected_columns: Optional[str] = Form(None),  # e.g. "date,feature1,feature2"
):
    db = SessionLocal()

    # 🔹 Fetch uploaded dataset
    dataset = db.query(UploadedDataset).filter(UploadedDataset.dataset_id == dataset_id).first()
    if not dataset:
        db.close()
        return {"message": "❌ Dataset not found.", "dataset_id": dataset_id, "success": False}

    if not os.path.exists(dataset.json_path):
        db.close()
        return {
            "message": f"❌ Dataset file not found at '{dataset.json_path}'",
            "dataset_id": dataset_id,
            "success": False
        }

    # 🔹 Load dataset JSON
    with open(dataset.json_path, "r") as f:
        raw_data = json.load(f)["data"]

    df = pd.DataFrame(raw_data)

    # 🔹 Apply column filtering (if specified)
    if selected_columns:
        selected_cols = [c.strip() for c in selected_columns.split(",") if c.strip()]

        # 🚫 Prevent target column from being reselected
        if target_column in selected_cols:
            db.close()
            return {
                "message": f"❌ Target column '{target_column}' should not be included in 'selected_columns'.",
                "success": False
            }

        # Check for missing selected columns
        missing = [c for c in selected_cols if c not in df.columns]
        if missing:
            db.close()
            return {"message": f"❌ Missing selected columns: {missing}", "success": False}

        # Keep only selected + target column
        df = df[selected_cols + [target_column]]

    # 🔹 Validate target column
    if target_column not in df.columns:
        db.close()
        return {"message": f"❌ Target column '{target_column}' not found.", "dataset_id": dataset_id, "success": False}

    model_type = dataset.model_type.lower()

    # 🔹 Generate a unique prepared ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prepared_id = f"{dataset_id}_prep_{timestamp}"

    # 🔹 Build prepared data structure
    if model_type == "linear":
        X = df.drop(columns=[target_column])
        y = df[target_column]

        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.factorize(X[col])[0]

        prepared_data = {
            "prepared_id": prepared_id,
            "dataset_id": dataset_id,
            "department": dataset.department,
            "model_type": model_type,
            "X_columns": list(X.columns),
            "target_column": target_column,
            "X": X.values.tolist(),
            "y": y.tolist(),
        }

    elif model_type == "arima":
        # Ensure target is numeric
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            db.close()
            return {"message": "❌ Target column must be numeric for ARIMA.", "success": False}

        # Ensure no extra columns (ARIMA is univariate)
        if len(df.columns) > 1:
            db.close()
            return {"message": "❌ ARIMA only supports one target column (no extra features).", "success": False}

        prepared_data = {
            "prepared_id": prepared_id,
            "dataset_id": dataset_id,
            "department": dataset.department,
            "model_type": model_type,
            "data": {"values": df[target_column].tolist()},
            "target_column": target_column,
        }

    elif model_type == "prophet":
        # Detect date/time column
        date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower() or col.lower() == "ds"]
        if not date_cols:
            db.close()
            return {"message": "❌ No date/time column found for Prophet.", "success": False}

        # Ensure only date + target (2 columns max)
        if len(df.columns) > 2:
            db.close()
            return {"message": "❌ Prophet currently supports only one date and one target column.", "success": False}

        date_col = date_cols[0]

        prepared_data = {
            "prepared_id": prepared_id,
            "dataset_id": dataset_id,
            "department": dataset.department,
            "model_type": model_type,
            "data": {
                "dates": df[date_col].astype(str).tolist(),
                "values": df[target_column].tolist(),
            },
            "target_column": target_column,
            "date_column": date_col,
        }

    else:
        db.close()
        return {"message": f"❌ Unsupported model type '{model_type}'.", "success": False}

    # 🔹 Save prepared JSON
    prepared_path = os.path.join(UPLOAD_DIR, f"{prepared_id}.json")
    with open(prepared_path, "w") as f:
        json.dump(prepared_data, f, indent=2)

    # 🔹 Store in DB
    prepared_entry = PreparedDataset(
        prepared_id=prepared_id,
        dataset_id=dataset.dataset_id,
        uploaded_dataset_id=dataset.id,
        department=dataset.department,
        model_type=model_type,
        target_column=target_column,
        prepared_json_path=prepared_path,
        columns_used=list(df.columns),
        rename_map=None,
        preprocessing_flags=None,
        created_at=datetime.now(),
    )

    db.add(prepared_entry)
    db.commit()
    db.close()

    return {
        "message": "✅ Dataset prepared successfully.",
        "prepared_id": prepared_id,
        "dataset_id": dataset_id,
        "model_type": model_type,
        "target_column": target_column,
        "columns": list(df.columns),
        "prepared_path": prepared_path,
        "success": True,
    }

@app.get("/prepared-datasets")
def list_prepared_datasets(
    department: Optional[str] = None,
    dataset_id: Optional[str] = None
):
    """List prepared datasets (optionally filtered by department or original dataset_id)."""
    return get_prepared_datasets(department, dataset_id)

@app.delete("/prepared-datasets/{prepared_id}")
def delete_prepared(prepared_id: str):
    """Delete a prepared dataset and its file."""
    return delete_prepared_dataset(prepared_id)