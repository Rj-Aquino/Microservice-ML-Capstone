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
        db_entry = TrainingInput(
            department=department,
            model_type=model_type,
            payload=data
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
        all_entries = query.all()

        results = []
        for entry in all_entries:
            results.append({
                "id": entry.id,
                "department": entry.department,
                "model_type": entry.model_type,
                "payload": entry.payload,
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else None
            })
        return results
    finally:
        db.close()

def delete_training_input(input_id: int):
    db = SessionLocal()
    try:
        entry = db.query(TrainingInput).filter(TrainingInput.id == input_id).first()
        if not entry:
            raise HTTPException(status_code=404, detail=f"Training input with id {input_id} not found.")
        
        db.delete(entry)
        db.commit()
        return {"message": f"Training input {input_id} deleted successfully."}
    finally:
        db.close()
