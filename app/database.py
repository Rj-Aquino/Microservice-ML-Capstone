import ast
import json
from fastapi import HTTPException
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import numpy as np
from sqlalchemy.orm import Session

# Database setup
DATABASE_URL = "sqlite:///./ML-Capstone.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class TrainingInput(Base):
    __tablename__ = "training_input"
    id = Column(Integer, primary_key=True, index=True)
    department = Column(String, index=True)
    model_type = Column(String, index=True)
    payload = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Save training input (payload stored as dict, not JSON string)
def save_training_input(department: str, model_type: str, data: dict):
    db = SessionLocal()
    try:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            return obj

        clean_data = convert(data)

        db_entry = TrainingInput(
            department=department,
            model_type=model_type,
            payload=clean_data
        )

        db.add(db_entry)
        db.flush()  # Ensures the INSERT is actually sent to DB
        db.commit()
        print(f"[DB] ✅ Saved training input for {department} ({model_type}) id={db_entry.id}")
    except Exception as e:
        db.rollback()
        print(f"[DB] ❌ Error saving training input: {e}")
        raise
    finally:
        db.close()

def parse_payload_loose(payload_str: str):
    """Parse JSON or Python-style dicts/lists safely."""
    if not payload_str:
        return {}
    if isinstance(payload_str, dict):
        return payload_str
    try:
        return json.loads(payload_str)
    except Exception:
        try:
            return ast.literal_eval(payload_str)
        except Exception:
            return {"error": "Cannot parse payload"}

def get_all_training_inputs(department: str = None, model_type: str = None):
    db: Session = SessionLocal()
    try:
        query = db.query(TrainingInput)
        if department:
            query = query.filter(TrainingInput.department == department)
        if model_type:
            query = query.filter(TrainingInput.model_type == model_type)
        all_entries = query.order_by(TrainingInput.timestamp.desc()).all()

        results = []
        for entry in all_entries:
            payload_data = parse_payload_loose(entry.payload)

            # Build summary
            summary = {"X shape": None, "y count": None}
            if isinstance(payload_data, dict):
                X = payload_data.get("X")
                if isinstance(X, list) and X:
                    if all(isinstance(row, list) for row in X):
                        summary["X shape"] = f"{len(X)} samples × {len(X[0])} features"
                    elif all(not isinstance(row, list) for row in X):
                        summary["X shape"] = f"{len(X)} samples × 1 feature"

                y = payload_data.get("y")
                summary["y count"] = len(y) if isinstance(y, list) else None

            # Preview first 5 items per key
            preview = {}
            if isinstance(payload_data, dict):
                for k, v in payload_data.items():
                    if isinstance(v, list):
                        preview[k] = v[:5]
                    else:
                        preview[k] = v
            elif isinstance(payload_data, str):
                preview = {"raw": payload_data}

            results.append({
                "id": entry.id,
                "department": entry.department,
                "model_type": entry.model_type,
                "timestamp": entry.timestamp.strftime("%Y-%m-%d %H:%M:%S") if entry.timestamp else None,
                "payload_summary": summary,
                "payload_preview": preview
            })
        return results
    finally:
        db.close()