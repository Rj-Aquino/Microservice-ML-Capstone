import gzip
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from requests import Session

from database import (
    save_training_input,
    get_all_training_inputs,
    TrainingInput,
    SessionLocal,
    UploadedDataset,
    get_uploaded_datasets,
    get_prepared_datasets,
    delete_prepared_dataset,
    PreparedDataset,
    compress_json,
   decompress_json,
   delete_uploaded_dataset
)
from models import train_model, predict, validate, MODEL_INFO
from preprocess import clean_training_data
from scaler import delete_and_retrain_department

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

UPLOAD_DIR = "uploaded_datasets"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

    db: Session = SessionLocal()
    try:
        # ============================
        # 1️⃣ Load prepared dataset if prepared_id is provided
        # ============================
        if prepared_id:
            print(f"📂 Loading prepared dataset from DB: {prepared_id}")

            prepared_entry = db.query(PreparedDataset).filter(
                PreparedDataset.prepared_id == prepared_id
            ).first()

            if not prepared_entry:
                raise HTTPException(status_code=404, detail=f"Prepared dataset '{prepared_id}' not found.")

            department = prepared_entry.department
            model_type = prepared_entry.model_type
            data = prepared_entry.prepared_json  # ✅ Directly use the new column

        # ============================
        # 2️⃣ If raw data provided (no prepared_id)
        # ============================
        elif data:
            department = getattr(payload, "department", None)
            model_type = getattr(payload, "model_type", None)
            if not department or not model_type:
                raise HTTPException(status_code=400, detail="Missing 'department' or 'model_type' in payload.")
            department = department.lower()
            model_type = model_type.lower()

        else:
            raise HTTPException(status_code=400, detail="Must provide either 'prepared_id' or raw 'data' for training.")

        # ============================
        # 3️⃣ Validate metadata
        # ============================
        valid_departments = ["operations", "finance", "inventory", "hr"]
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")
    finally:
        db.close()

# =====================================================
# 🔮 PREDICTION ENDPOINT
# =====================================================
@app.post("/predict")
async def predict_endpoint(payload: PredictRequest):
    department = payload.department
    model_type = payload.model_type.lower()
    input_data = payload.data

    if not input_data:
        raise HTTPException(status_code=400, detail="Missing 'data' for prediction.")

    if not isinstance(input_data, dict):
        raise HTTPException(status_code=400, detail="Invalid 'data' format — expected dict")

    # The predict function internally handles steps_ahead for ARIMA/Prophet
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

    if not test_data:
        raise HTTPException(status_code=400, detail="Missing 'data' for prediction.")

    if not isinstance(test_data, dict):
        raise HTTPException(status_code=400, detail="Invalid 'data' format — expected dict.")

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

# =====================================================
# 🧾 DATASET UPLOAD / MANAGEMENT ENDPOINTS
# =====================================================
@app.post("/upload-data")
async def upload_data(
    department: str = Form(...),
    model_type: str = Form(...),
    dataset_name: str = Form(...),
    file: UploadFile = None
):
    if not file or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a valid CSV file.")

    valid_departments = ["operations", "finance", "inventory", "hr"]
    if department.lower() not in valid_departments:
        raise HTTPException(status_code=400, detail=f"Invalid department '{department}'.")

    valid_model_types = ["linear", "arima", "prophet"]
    if model_type.lower() not in valid_model_types:
        raise HTTPException(status_code=400, detail=f"Invalid model type '{model_type}'.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_id = f"{department.lower()}_{dataset_name}_{timestamp}"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    csv_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")

    with open(csv_path, "wb") as buffer:
        buffer.write(await file.read())

    df = pd.read_csv(csv_path)
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
    df = df.dropna().drop_duplicates()

    # ✅ Compress the data before saving
    compressed_data = compress_json(df.to_dict(orient="records"))

    metadata = {
        "dataset_id": dataset_id,
        "department": department.lower(),
        "model_type": model_type.lower(),
        "dataset_name": dataset_name,
        "records": len(df),
        "columns": list(df.columns),
        "data_compressed": compressed_data,  # ✅ compressed data
    }

    db: Session = SessionLocal()
    try:
        uploaded = UploadedDataset(**metadata)
        db.add(uploaded)
        db.commit()
        db.refresh(uploaded)
    finally:
        db.close()

    return {
        "message": "✅ Dataset uploaded and saved successfully (compressed).",
        "dataset_id": dataset_id,
        "records": len(df),
        "columns": list(df.columns),
        "compression": "gzip + base64",
        "success": True,
    }

@app.get("/datasets")
def list_datasets(department: Optional[str] = None, model_type: Optional[str] = None):
    return get_uploaded_datasets(department, model_type)


@app.delete("/datasets/{dataset_id}")
def delete_uploaded(dataset_id: str):
    return delete_uploaded_dataset(dataset_id)

@app.post("/prepare-dataset")
async def prepare_dataset(
    dataset_id: str = Form(...),
    target_column: str = Form(...),
    selected_columns: str | None = Form(None),
):
    """
    Prepare dataset with strict validation:
    - ARIMA: numeric target only, no selected_columns allowed
    - Prophet: requires date column; selected_columns must be date if provided
    - Linear: selected_columns define X features; all numeric; target must be numeric
    """
    db: Session = SessionLocal()
    try:
        dataset = db.query(UploadedDataset).filter(UploadedDataset.dataset_id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

        department = dataset.department.lower()
        model_type = dataset.model_type.lower()

        if not dataset.data_compressed:
            raise HTTPException(status_code=400, detail=f"No data found in uploaded dataset '{dataset_id}'.")

        try:
            raw_data = decompress_json(dataset.data_compressed)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to decompress dataset: {str(e)}")

        df = pd.DataFrame(raw_data)
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset payload is empty or invalid.")

        # --------------------
        # Ensure target exists and numeric
        # --------------------
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset.")

        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' must be numeric.")

        # --------------------
        # Parse selected columns
        # --------------------
        selected_cols: list[str] = []
        if selected_columns:
            selected_cols = [c.strip() for c in selected_columns.split(",") if c.strip()]
            missing = [c for c in selected_cols if c not in df.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing selected columns: {missing}")

        # --------------------
        # Model-specific validations
        # --------------------
        if model_type == "arima":
            if selected_cols:
                raise HTTPException(status_code=400, detail="ARIMA does not allow selected columns. Only numeric target is required.")
            prepared_json = {"values": df[target_column].tolist(), "target_column": target_column}

        elif model_type == "prophet":
            date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or c.lower() == "ds"]
            if not date_cols:
                raise HTTPException(status_code=400, detail="Prophet requires a date/time column.")
            date_col = date_cols[0]

            if selected_cols:
                if len(selected_cols) != 1 or selected_cols[0] != date_col:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prophet selected_columns must contain only the date column '{date_col}'."
                    )
            else:
                extra_cols = [c for c in df.columns if c not in [target_column, date_col]]
                if extra_cols:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prophet dataset can only have the date column '{date_col}' besides the target. Found extra: {extra_cols}"
                    )

            prepared_json = {
                "dates": df[date_col].astype(str).tolist(),
                "values": df[target_column].tolist(),
                "target_column": target_column,
                "date_column": date_col,
            }

        elif model_type == "linear":
            # Use selected_cols as X if provided, else all columns except target
            X_cols = selected_cols if selected_cols else [c for c in df.columns if c != target_column]

            # Validate all selected X columns are numeric
            non_numeric = [c for c in X_cols if not pd.api.types.is_numeric_dtype(df[c])]
            if non_numeric:
                raise HTTPException(status_code=400, detail=f"Selected feature columns must be numeric. Non-numeric: {non_numeric}")

            X = df[X_cols]
            y = df[target_column]

            prepared_json = {
                "X": X.values.tolist(),
                "y": y.tolist(),
                "target_column": target_column,
                "X_columns": X_cols,
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

        # --------------------
        # Save prepared dataset
        # --------------------
        prepared_id = f"{dataset_id}_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        prepared_entry = PreparedDataset(
            prepared_id=prepared_id,
            dataset_id=dataset.dataset_id,
            uploaded_dataset_id=dataset.id,
            department=department,
            model_type=model_type,
            target_column=target_column,
            columns_used=[target_column] if model_type == "arima" else list(df.columns),
            prepared_json=prepared_json,
            created_at=datetime.now(),
        )
        db.add(prepared_entry)
        db.commit()
        db.refresh(prepared_entry)

        return {
            "message": "✅ Dataset prepared successfully.",
            "prepared_id": prepared_id,
            "department": department,
            "model_type": model_type,
            "target_column": target_column,
            "columns": [target_column] if model_type == "arima" else list(df.columns),
            "samples": len(df),
            "train_ready_json": prepared_json,
            "success": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing dataset: {str(e)}")
    finally:
        db.close()

@app.get("/prepared-datasets")
def list_prepared_datasets(department: Optional[str] = None, dataset_id: Optional[str] = None):
    return get_prepared_datasets(department, dataset_id)

@app.delete("/prepared-datasets/{prepared_id}")
def delete_prepared(prepared_id: str):
    return delete_prepared_dataset(prepared_id)
