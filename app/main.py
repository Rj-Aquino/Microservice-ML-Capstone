from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

app = FastAPI(title="Simple Forecasting Microservice")

# initialize blank model
model = LinearRegression()

# Define input format
class DataInput(BaseModel):
    data: Dict[str, Any]

@app.get("/")
def root():
    return {"message": "Forecasting Microservice is running!"}

@app.post("/train")
def train_model(payload: DataInput):
    try:
        save_training_input(payload.data)  # store raw input
        data = payload.data
        X = np.array(data["X"]).reshape(-1, 1)
        y = np.array(data["y"])
        model.fit(X, y)
        joblib.dump(model, "model.pkl")
        return {"message": "Model trained successfully!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(payload: DataInput):
    try:
        model = joblib.load("model.pkl")
        input_data = np.array(payload.data["X"]).reshape(-1, 1)
        predictions = model.predict(input_data).tolist()
        return {"predictions": predictions}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not trained yet")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Database Related Code 

from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# Database setup
DATABASE_URL = "sqlite:///./ML-Capstone.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model for training data only
class TrainingInput(Base):
    __tablename__ = "training_input"
    id = Column(Integer, primary_key=True, index=True)
    payload = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Save only training inputs
def save_training_input(data: dict):
    db = SessionLocal()
    try:
        db_entry = TrainingInput(payload=data)  # store dict directly
        db.add(db_entry)
        db.commit()
    finally:
        db.close()