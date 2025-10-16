from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel
from typing import Dict, Any

from database import save_training_input, get_all_training_inputs, delete_training_input
from models import train_model, predict  # your existing model functions

app = FastAPI(title="Dynamic Forecasting Microservice")

# -----------------------
# Request Models
# -----------------------
class TrainingRequest(BaseModel):
    department: str
    model_type: str
    data: Dict[str, Any]

    model_config = {"protected_namespaces": ()}

class PredictRequest(BaseModel):
    department: str
    model_type: str
    input: Dict[str, Any]

    model_config = {"protected_namespaces": ()}

# -----------------------
# Endpoints
# -----------------------
@app.post("/train")
def train(payload: TrainingRequest):
    try:
        save_training_input(payload.department, payload.model_type, payload.data)
        result = train_model(payload.department, payload.model_type, payload.data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def forecast(payload: PredictRequest):
    try:
        predictions = predict(payload.department, payload.model_type, payload.input)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/data")
def read_all_training_data(
    department: str | None = Query(default=None, description="Filter by department"),
    model_type: str | None = Query(default=None, description="Filter by model type")
):
    try:
        return get_all_training_inputs(department=department, model_type=model_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/train/{input_id}")
def delete_training_input_endpoint(input_id: int = Path(..., description="ID of the training input to delete")):
    """
    Delete a specific training input by its ID.
    """
    try:
        result = delete_training_input(input_id)
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))