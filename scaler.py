import gzip
import pickle
import json
from fastapi import HTTPException
import numpy as np
from sqlalchemy.orm import Session

from database import (
    SessionLocal,
    TrainingInput,
    save_file_to_db,
    delete_file_by_type,
    StoredFile,
    TrainedModel,
)

# =======================================
# In-Memory Caches
# =======================================
scalers_X = {}
scalers_y = {}
feature_dims = {}

# =======================================
# Scaler DB Helpers
# =======================================

def save_scaler_to_db(department: str, scaler_type: str, scaler):
    """Save a fitted scaler to the DB as a StoredFile, compressed."""
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise HTTPException(status_code=400, detail="Scaler is not fitted yet.")
    content = gzip.compress(pickle.dumps(scaler))  # compress before saving
    filename = f"{department}_scaler_{scaler_type}.pkl"
    result = save_file_to_db(filename, content, file_type=f"scaler_{scaler_type}", department=department)
    return result

def load_scaler_from_db(department: str, scaler_type: str):
    """Load a scaler from the DB based on department and type, decompressing it."""
    db: Session = SessionLocal()
    try:
        entry = db.query(StoredFile).filter(
            StoredFile.department == department,
            StoredFile.file_type == f"scaler_{scaler_type}"
        ).order_by(StoredFile.uploaded_at.desc()).first()
        if not entry:
            raise HTTPException(status_code=400, detail=f"No {scaler_type} scaler found for {department}.")
        scaler = pickle.loads(gzip.decompress(entry.content))  # decompress before loading
        # Validate
        if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_") or np.isnan(scaler.mean_).any():
            raise HTTPException(status_code=400, detail=f"Scaler for {department} is corrupted or contains NaN values.")
        # Update in-memory cache
        if scaler_type == "X":
            scalers_X[department] = scaler
        else:
            scalers_y[department] = scaler
        feature_dims[department] = len(scaler.mean_)
        return scaler
    finally:
        db.close()

# =======================================
# Delete + Retrain Workflow
# =======================================
def delete_and_retrain_department(department: str, model_type: str):
    """
    Deletes the model (and scalers if linear) from DB for a department,
    then retrains using remaining TrainingInput entries.
    """
    from models import train_model  # avoid circular import
    deleted_files = []
    model_type_lower = model_type.lower()
    db: Session = SessionLocal()

    try:
        # ----------------------------
        # DELETE MODEL
        # ----------------------------
        if model_type_lower in ["linear", "arima"]:
            # TrainedModel table
            entry = db.query(TrainedModel).filter(
                TrainedModel.department == department,
                TrainedModel.model_type == model_type_lower
            ).first()
            if entry:
                db.delete(entry)
                db.commit()
                deleted_files.append(entry.model_id)
                print(f"[Model] 🗑️ Deleted {model_type} model for {department}")
            else:
                print(f"[Model] ⚠️ No {model_type} model found for {department}, skipping.")
        elif model_type_lower == "prophet":
            # StoredFile table
            deleted_file = delete_file_by_type(department, "prophet_model")
            if deleted_file:
                deleted_files.append(deleted_file)

        # ----------------------------
        # DELETE SCALERS (Linear Only)
        # ----------------------------
        if model_type_lower == "linear":
            for scaler_type in ["scaler_X", "scaler_y"]:
                deleted_scaler = delete_file_by_type(department, scaler_type)
                if deleted_scaler:
                    deleted_files.append(deleted_scaler)
            # Remove in-memory copies
            scalers_X.pop(department, None)
            scalers_y.pop(department, None)
            feature_dims.pop(department, None)
            print(f"[Clean-up] ✅ Cleared scalers and model for {department} ({model_type})")
        else:
            print(f"[Clean-up] ✅ Cleared only model for {department} ({model_type}) — no scalers used")

        # ----------------------------
        # RETRAIN USING DB DATA
        # ----------------------------
        remaining = db.query(TrainingInput).filter(
            TrainingInput.department == department,
            TrainingInput.model_type == model_type_lower
        ).all()

        if not remaining:
            return {
                "deleted_files": deleted_files,
                "retrained": False,
                "message": f"No remaining data found for {department}/{model_type}.",
            }

        combined_data = {"X": [], "y": [], "values": [], "dates": []}
        for entry in remaining:
            payload = entry.payload
            if not isinstance(payload, dict):
                continue
            if "X" in payload and "y" in payload:
                combined_data["X"].extend(payload["X"])
                combined_data["y"].extend(payload["y"])
            elif "values" in payload:
                combined_data["values"].extend(payload["values"])
                if "dates" in payload:
                    combined_data["dates"].extend(payload["dates"])

        # Prepare training data
        if model_type_lower == "linear":
            if not combined_data["X"] or not combined_data["y"]:
                return {"deleted_files": deleted_files, "retrained": False, "message": "No valid Linear data remaining."}
            train_data = {"X": combined_data["X"], "y": combined_data["y"]}
        elif model_type_lower == "arima":
            if not combined_data["values"]:
                return {"deleted_files": deleted_files, "retrained": False, "message": "No valid ARIMA data remaining."}
            train_data = {"values": combined_data["values"]}
        elif model_type_lower == "prophet":
            if not combined_data["dates"] or not combined_data["values"]:
                return {"deleted_files": deleted_files, "retrained": False, "message": "No valid Prophet data remaining."}
            train_data = {"dates": combined_data["dates"], "values": combined_data["values"]}
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model_type: {model_type}")

        # ----------------------------
        # RETRAIN MODEL
        # ----------------------------
        train_model(department, model_type_lower, train_data)
        print(f"[Retrain] ✅ Retrained {model_type} model for {department} using remaining data.")

        return {
            "deleted_files": deleted_files,
            "retrained": True,
            "message": f"{model_type.upper()} retrained successfully for {department}.",
        }

    finally:
        db.close()