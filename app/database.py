import json
import os
import uuid
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

# =====================================
# Database Setup
# =====================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ML-Capstone.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

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
    department = Column(String, index=True)
    model_type = Column(String, index=True)
    payload = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


class UploadedDataset(Base):
    __tablename__ = "uploaded_datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(String, unique=True, index=True)
    department = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    dataset_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    json_path = Column(String, nullable=False)
    columns = Column(JSON, nullable=False)
    records = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    prepared_versions = relationship("PreparedDataset", back_populates="uploaded_dataset", cascade="all, delete")


class PreparedDataset(Base):
    __tablename__ = "prepared_datasets"

    id = Column(Integer, primary_key=True, index=True)
    prepared_id = Column(String, unique=True, index=True)  # ✅ Unique prepared dataset ID
    dataset_id = Column(String, index=True)  # same as UploadedDataset.dataset_id
    uploaded_dataset_id = Column(Integer, ForeignKey("uploaded_datasets.id", ondelete="CASCADE"))

    department = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    prepared_json_path = Column(String, nullable=False)
    columns_used = Column(JSON, nullable=False)
    rename_map = Column(JSON, nullable=True)
    preprocessing_flags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    uploaded_dataset = relationship("UploadedDataset", back_populates="prepared_versions")


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
def save_uploaded_dataset(metadata: dict):
    """Save dataset upload metadata to the database."""
    required = ["dataset_id", "department", "model_type", "dataset_name", "file_path", "json_path", "columns", "records"]
    for key in required:
        if key not in metadata:
            raise HTTPException(status_code=400, detail=f"Missing field in metadata: {key}")

    db = SessionLocal()
    try:
        new_entry = UploadedDataset(**metadata)
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)
        print(f"[DB] ✅ Saved uploaded dataset {metadata['dataset_name']} ({metadata['dataset_id']})")
        return {"message": "Dataset saved successfully.", "id": new_entry.id}
    except Exception as e:
        db.rollback()
        print(f"[DB] ❌ Error saving uploaded dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded dataset: {str(e)}")
    finally:
        db.close()


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
    """Delete uploaded dataset and all its prepared versions from DB and disk."""
    db: Session = SessionLocal()
    try:
        entry = db.query(UploadedDataset).filter(UploadedDataset.dataset_id == dataset_id).first()
        if not entry:
            raise HTTPException(status_code=404, detail=f"Dataset with ID '{dataset_id}' not found.")

        # 🔹 Delete prepared dataset files (if any)
        prepared_versions = entry.prepared_versions
        for prep in prepared_versions:
            if os.path.exists(prep.prepared_json_path):
                os.remove(prep.prepared_json_path)
                print(f"[FS] 🗑️ Deleted prepared file {prep.prepared_json_path}")

        # 🔹 Delete uploaded dataset files
        for path in [entry.file_path, entry.json_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"[FS] 🗑️ Deleted {path}")

        # 🔹 Delete from DB
        db.delete(entry)
        db.commit()
        print(f"[DB] 🗑️ Deleted dataset {dataset_id} and its prepared versions")

        return {"message": f"Dataset '{dataset_id}' and its prepared versions deleted successfully."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")
    finally:
        db.close()

# =====================================
# Prepared Dataset Functions
# =====================================
def save_prepared_dataset(metadata: dict):
    """Save prepared dataset metadata."""
    required = [
        "dataset_id", "uploaded_dataset_id", "prepared_json_path",
        "model_type", "target_column", "columns_used", "department"
    ]
    for key in required:
        if key not in metadata:
            raise HTTPException(status_code=400, detail=f"Missing field in prepared dataset metadata: {key}")

    db = SessionLocal()
    try:
        prepared_id = generate_prepared_id(metadata["department"])
        metadata["prepared_id"] = prepared_id

        new_entry = PreparedDataset(**metadata)
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)
        print(f"[DB] ✅ Saved prepared dataset {prepared_id} for {metadata['dataset_id']}")
        return {"message": "Prepared dataset saved successfully.", "prepared_id": prepared_id}
    except Exception as e:
        db.rollback()
        print(f"[DB] ❌ Error saving prepared dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save prepared dataset: {str(e)}")
    finally:
        db.close()


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
                "prepared_json_path": e.prepared_json_path,
                "created_at": e.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for e in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prepared datasets: {str(e)}")
    finally:
        db.close()


def delete_prepared_dataset(prepared_id: str):
    """Delete a prepared dataset and its file."""
    db: Session = SessionLocal()
    try:
        entry = db.query(PreparedDataset).filter(PreparedDataset.prepared_id == prepared_id).first()
        if not entry:
            raise HTTPException(status_code=404, detail=f"Prepared dataset with ID '{prepared_id}' not found.")

        if os.path.exists(entry.prepared_json_path):
            os.remove(entry.prepared_json_path)
            print(f"[FS] 🗑️ Deleted {entry.prepared_json_path}")

        db.delete(entry)
        db.commit()
        print(f"[DB] 🗑️ Deleted prepared dataset {prepared_id}")
        return {"message": f"Prepared dataset '{prepared_id}' deleted successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete prepared dataset: {str(e)}")
    finally:
        db.close()
