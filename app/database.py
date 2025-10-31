import json
import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# =====================================
# Database Setup
# =====================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ML-Capstone.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# =====================================
# Database Model
# =====================================
class TrainingInput(Base):
    __tablename__ = "training_input"

    id = Column(Integer, primary_key=True, index=True)
    department = Column(String, index=True)
    model_type = Column(String, index=True)
    payload = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# =====================================
# Helpers
# =====================================
def ensure_json(data: Any) -> Any:
    """
    Normalize raw JSON strings or objects into a valid Python object.
    Raises HTTPException if not JSON-serializable.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format in 'data'.")

    try:
        json.dumps(data)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="'data' must be JSON-serializable (dict, list, etc.).")

    return data


def convert(obj: Any) -> Any:
    """
    Convert numpy, pandas, and non-serializable objects into basic Python types.
    """
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
# Core Functions
# =====================================
def save_training_input(department: str, model_type: str, data: Any):
    """
    Save new training input into the database.
    Accepts JSON strings or Python objects (dict, list, etc.).
    """
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


def get_all_training_inputs(
    department: Optional[str] = None,
    model_type: Optional[str] = None,
):
    """
    Retrieve all stored training inputs, optionally filtered by department or model_type.
    Returns full payloads (no truncation).
    """
    db: Session = SessionLocal()
    try:
        query = db.query(TrainingInput)
        if department:
            query = query.filter(TrainingInput.department == department)
        if model_type:
            query = query.filter(TrainingInput.model_type == model_type.lower())

        entries = query.order_by(TrainingInput.timestamp.desc()).all()
        results = []

        for entry in entries:
            payload = entry.payload

            # ✅ Show full payload
            if isinstance(payload, (dict, list)):
                preview = payload
            else:
                preview = str(payload)

            results.append({
                "id": entry.id,
                "department": entry.department,
                "model_type": entry.model_type,
                "timestamp": entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "payload": preview,  # full data shown here
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve training inputs: {str(e)}")

    finally:
        db.close()

