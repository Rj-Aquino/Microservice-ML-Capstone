import json
from fastapi import HTTPException
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import numpy as np

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
        # ✅ Convert NumPy arrays to lists before saving
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
            payload=json.dumps(clean_data)
        )
        db.add(db_entry)
        db.commit()
    finally:
        db.close()

# Get all training inputs, optionally filtered by department/model_type
def get_all_training_inputs(department: str = None, model_type: str = None):
    db = SessionLocal()
    try:
        query = db.query(TrainingInput)
        if department:
            query = query.filter(TrainingInput.department == department)
        if model_type:
            query = query.filter(TrainingInput.model_type == model_type)
        all_entries = query.order_by(TrainingInput.timestamp.desc()).all()

        results = []
        for entry in all_entries:
            try:
                payload_data = json.loads(entry.payload)
            except Exception:
                payload_data = {"error": "Invalid JSON format"}

            results.append({
                "id": entry.id,
                "department": entry.department,
                "model_type": entry.model_type,
                "timestamp": entry.timestamp.strftime("%Y-%m-%d %H:%M:%S") if entry.timestamp else None,
                "payload_summary": (
                    {
                        "X shape": f"{len(payload_data.get('X', []))} samples × "
                                   f"{len(payload_data['X'][0]) if payload_data.get('X') else 0} features"
                        if "X" in payload_data else None,
                        "y count": len(payload_data.get("y", [])) if "y" in payload_data else None,
                        "values count": len(payload_data.get("values", [])) if "values" in payload_data else None,
                        "dates count": len(payload_data.get("dates", [])) if "dates" in payload_data else None
                    }
                ),
                "payload_preview": (
                    {k: payload_data[k][:5] for k in payload_data if isinstance(payload_data[k], list)}
                    if isinstance(payload_data, dict) else None
                )
            })
        return results
    finally:
        db.close()