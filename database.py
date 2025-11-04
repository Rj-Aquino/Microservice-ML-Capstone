import base64
import gzip
import os
import json
import uuid
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, LargeBinary, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from dotenv import load_dotenv
from sqlalchemy.dialects.mysql import LONGBLOB

import pymysql
pymysql.install_as_MySQLdb()

# ==========================================================
# Database Setup (MySQL)
# ==========================================================
# Example: mysql+pymysql://user:password@localhost/db_name
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL environment variable is not set. Please define it before running the app.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

def compress_json(data):
    """Compress a JSON-serializable object using gzip + base64."""
    json_str = json.dumps(data)
    compressed = gzip.compress(json_str.encode("utf-8"))
    return base64.b64encode(compressed).decode("utf-8")

def decompress_json(encoded_str):
    """Decompress a base64-encoded gzip string back into JSON."""
    compressed = base64.b64decode(encoded_str)
    json_str = gzip.decompress(compressed).decode("utf-8")
    return json.loads(json_str)

# =====================================
# Helper: Unique ID Generator
# =====================================
def generate_prepared_id(department: str) -> str:
    """Generate a unique prepared dataset ID."""
    short_dept = department.upper().replace(" ", "")[:4]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_part = uuid.uuid4().hex[:4].upper()
    return f"PREP-{short_dept}-{timestamp}-{unique_part}"

# =====================================
# Database Models
# =====================================
class TrainingInput(Base):
    __tablename__ = "training_input"

    id = Column(Integer, primary_key=True, index=True)
    department = Column(String(100), index=True)
    model_type = Column(String(100), index=True)
    payload = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

class UploadedDataset(Base):
    __tablename__ = "uploaded_datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(String(100), unique=True, index=True)
    department = Column(String(100), nullable=False)
    model_type = Column(String(100), nullable=False)
    dataset_name = Column(String(255), nullable=False)
    columns = Column(JSON, nullable=False)
    records = Column(Integer, nullable=False)
    data_compressed = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    prepared_versions = relationship("PreparedDataset", back_populates="uploaded_dataset", cascade="all, delete")

class PreparedDataset(Base):
    __tablename__ = "prepared_datasets"

    id = Column(Integer, primary_key=True, index=True)
    prepared_id = Column(String(100), unique=True, index=True)
    dataset_id = Column(String(100), index=True)
    uploaded_dataset_id = Column(Integer, ForeignKey("uploaded_datasets.id", ondelete="CASCADE"))

    department = Column(String(100), nullable=False)
    model_type = Column(String(100), nullable=False)
    target_column = Column(String(100), nullable=False)
    columns_used = Column(JSON, nullable=False)
    rename_map = Column(JSON, nullable=True)
    preprocessing_flags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    prepared_json = Column(JSON, nullable=False)

    uploaded_dataset = relationship("UploadedDataset", back_populates="prepared_versions")

class StoredFile(Base):
    __tablename__ = "stored_files"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String(64), unique=True, index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    department = Column(String(100), nullable=True)
    content = Column(LONGBLOB, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(64), unique=True, index=True)
    model_type = Column(String(100), nullable=False)
    department = Column(String(100), nullable=False)
    model_blob = Column(LONGBLOB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    model_metadata = relationship("ModelMetadata", back_populates="model", cascade="all, delete")

class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(64), ForeignKey("trained_models.model_id", ondelete="CASCADE"))
    metrics = Column(JSON, nullable=True)
    features = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("TrainedModel", back_populates="model_metadata")

# Create all tables
Base.metadata.create_all(bind=engine)

# =====================================
# Helpers
# =====================================
def ensure_json(data: Any) -> Any:
    """Normalize raw JSON strings or objects into valid Python objects."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format in 'data'.")
    try:
        json.dumps(data)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="'data' must be JSON-serializable.")
    return data

def convert(obj: Any) -> Any:
    """Convert numpy/pandas/non-serializable objects to base Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(x) for x in obj]
    if pd.isna(obj):
        return None
    return obj

# =====================================
# Training Input Functions
# =====================================
def save_training_input(department: str, model_type: str, data: Any):
    """Save new training input into the database."""
    if not department or not model_type:
        raise HTTPException(status_code=400, detail="Missing 'department' or 'model_type'.")

    db = SessionLocal()
    try:
        data = ensure_json(data)
        clean_data = convert(data)
        entry = TrainingInput(
            department=department.strip(),
            model_type=model_type.lower().strip(),
            payload=clean_data,
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        print(f"[DB] ✅ Saved training input for {department} ({model_type}) id={entry.id}")
        return {"message": "Training input saved successfully.", "id": entry.id}
    except Exception as e:
        db.rollback()
        print(f"[DB] ❌ Error saving training input: {e}")
        raise HTTPException(status_code=500, detail=f"Database save failed: {str(e)}")
    finally:
        db.close()

def get_all_training_inputs(department: Optional[str] = None, model_type: Optional[str] = None):
    """Retrieve all stored training inputs, optionally filtered by department/model_type."""
    db: Session = SessionLocal()
    try:
        query = db.query(TrainingInput)
        if department:
            query = query.filter(TrainingInput.department == department)
        if model_type:
            query = query.filter(TrainingInput.model_type == model_type.lower())

        entries = query.order_by(TrainingInput.timestamp.desc()).all()
        return [
            {
                "id": e.id,
                "department": e.department,
                "model_type": e.model_type,
                "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "payload": e.payload,
            }
            for e in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve training inputs: {str(e)}")
    finally:
        db.close()

# =====================================
# Uploaded Dataset Functions
# =====================================
def get_uploaded_datasets(department: Optional[str] = None, model_type: Optional[str] = None):
    """Retrieve uploaded datasets with optional filters."""
    db: Session = SessionLocal()
    try:
        query = db.query(UploadedDataset)
        if department:
            query = query.filter(UploadedDataset.department == department)
        if model_type:
            query = query.filter(UploadedDataset.model_type == model_type.lower())
        entries = query.order_by(UploadedDataset.created_at.desc()).all()

        return [
            {
                "id": e.id,
                "dataset_id": e.dataset_id,
                "department": e.department,
                "model_type": e.model_type,
                "dataset_name": e.dataset_name,
                "records": e.records,
                "columns": e.columns,
                "created_at": e.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for e in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve datasets: {str(e)}")
    finally:
        db.close()

def delete_uploaded_dataset(dataset_id: str):
    """Delete uploaded dataset and all its prepared versions from the database (no file system)."""
    db: Session = SessionLocal()
    try:
        entry = db.query(UploadedDataset).filter(UploadedDataset.dataset_id == dataset_id).first()
        if not entry:
            raise HTTPException(status_code=404, detail=f"Dataset with ID '{dataset_id}' not found.")

        # 🔹 Delete all prepared datasets linked to this uploaded dataset
        prepared_versions = entry.prepared_versions
        for prep in prepared_versions:
            db.delete(prep)
        print(f"[DB] 🗑️ Deleted {len(prepared_versions)} prepared versions for dataset '{dataset_id}'")

        # 🔹 Delete the uploaded dataset entry
        db.delete(entry)
        db.commit()

        print(f"[DB] 🗑️ Deleted uploaded dataset '{dataset_id}' from database")

        return {"message": f"✅ Dataset '{dataset_id}' and its prepared versions deleted successfully from the database."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")
    finally:
        db.close()

# =====================================
# Prepared Dataset Functions
# =====================================
def get_prepared_datasets(department: Optional[str] = None, dataset_id: Optional[str] = None):
    """Retrieve prepared datasets, optionally filtered by department or dataset_id."""
    db: Session = SessionLocal()
    try:
        query = db.query(PreparedDataset)
        if department:
            query = query.filter(PreparedDataset.department == department)
        if dataset_id:
            query = query.filter(PreparedDataset.dataset_id == dataset_id)
        entries = query.order_by(PreparedDataset.created_at.desc()).all()

        return [
            {
                "id": e.id,
                "prepared_id": e.prepared_id,
                "dataset_id": e.dataset_id,
                "department": e.department,
                "target_column": e.target_column,
                "model_type": e.model_type,
                "columns_used": e.columns_used,
                "prepared_json": e.prepared_json,
                "created_at": e.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for e in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prepared datasets: {str(e)}")
    finally:
        db.close()

def delete_prepared_dataset(prepared_id: str):
    """Delete a prepared dataset from the database (no file system)."""
    db: Session = SessionLocal()
    try:
        entry = db.query(PreparedDataset).filter(PreparedDataset.prepared_id == prepared_id).first()
        if not entry:
            raise HTTPException(status_code=404, detail=f"Prepared dataset with ID '{prepared_id}' not found.")

        # 🔹 Delete the prepared dataset record
        db.delete(entry)
        db.commit()

        print(f"[DB] 🗑️ Deleted prepared dataset '{prepared_id}' from database")
        return {"message": f"✅ Prepared dataset '{prepared_id}' deleted successfully from the database."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete prepared dataset: {str(e)}")
    finally:
        db.close()

import pickle

def generate_file_id(prefix="FILE") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

# ==========================================================
# File Operations
# ==========================================================
def save_file_to_db(filename: str, content: bytes, file_type: str, department: Optional[str] = None):
    db = SessionLocal()
    try:
        file_id = generate_file_id()
        # Compress content before saving
        compressed_content = gzip.compress(content)
        new_file = StoredFile(
            file_id=file_id,
            filename=filename,
            file_type=file_type.lower(),
            content=compressed_content,
            department=department
        )
        db.add(new_file)
        db.commit()
        print(f"[DB] ✅ Stored file '{filename}' ({file_id})")
        return {"message": "File saved successfully.", "file_id": file_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    finally:
        db.close()

def read_file_from_db(filename: str, department: Optional[str] = None) -> bytes:
    """Read a stored file by its filename (and optional department)."""
    db = SessionLocal()
    try:
        query = db.query(StoredFile).filter(StoredFile.filename == filename)
        if department:
            query = query.filter(StoredFile.department == department)
        entry = query.first()
        if not entry:
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        print(f"[DB] 📄 Retrieved file {entry.filename} ({entry.file_id})")
        # Decompress content after reading
        return gzip.decompress(entry.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    finally:
        db.close()

def delete_file_by_type(department: str, file_type: str):
    """Delete a file by department + file_type."""
    db = SessionLocal()
    try:
        entry = db.query(StoredFile).filter(
            StoredFile.department == department,
            StoredFile.file_type == file_type.lower()
        ).first()
        if not entry:
            print(f"[DB] ⚠️ No file found for {department} ({file_type}), skipping.")
            return None
        db.delete(entry)
        db.commit()
        print(f"[DB] 🗑️ Deleted file {entry.filename} ({entry.file_id})")
        return entry.filename
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
    finally:
        db.close()

# ==========================================================
# Models Operations
# ==========================================================
def save_model_to_db(model_obj, model_type: str, department: str, metrics=None, features=None, notes=None):
    db = SessionLocal()
    try:
        model_id = generate_file_id(prefix="MODEL")
        model_blob = gzip.compress(pickle.dumps(model_obj))

        model_entry = TrainedModel(
            model_id=model_id,
            model_type=model_type.lower(),
            department=department,
            model_blob=model_blob
        )
        db.add(model_entry)
        db.commit()

        meta_entry = ModelMetadata(
            model_id=model_id,
            metrics=metrics or {},
            features=features or [],
            notes=notes or ""
        )
        db.add(meta_entry)
        db.commit()

        print(f"[DB] ✅ Saved model '{model_type}' ({model_id})")
        return {"message": "Model saved successfully.", "model_id": model_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")
    finally:
        db.close()

def load_model_from_db(department: str, model_type: str):
    """
    Load the trained model for a specific department and model_type.
    Raises 404 if no such model exists.
    """
    db = SessionLocal()
    try:
        entry = (
            db.query(TrainedModel)
            .filter(TrainedModel.department == department, TrainedModel.model_type == model_type.lower())
            .first()
        )
        if not entry:
            raise HTTPException(status_code=404, detail=f"No trained model found for {department} ({model_type})")
        
        model = pickle.loads(gzip.decompress(entry.model_blob))
        print(f"[DB] 🤖 Loaded model {entry.model_type} for {entry.department}")
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    finally:
        db.close()


