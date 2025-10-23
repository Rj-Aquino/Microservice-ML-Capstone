import os
import json
from fastapi import HTTPException
import joblib
import numpy as np
from pytest import Session

from database import SessionLocal, TrainingInput, parse_payload_loose


SCALERS_DIR = "./scalers"  # ✅ use only this folder
BASE_MODEL_DIR = "saved_models"
os.makedirs(SCALERS_DIR, exist_ok=True)

scalers_X = {}
scalers_y = {}
feature_dims = {}

# ----------------------
# Helper: get scaler file path
# ----------------------
def get_scaler_path(department: str, scaler_type: str):
    """Return the file path for a department's scaler."""
    return os.path.join(SCALERS_DIR, f"{department}_scaler_{scaler_type}.pkl")

# ----------------------
# Save scalers
# ----------------------
def save_department_scalers(department: str):
    """Persist both X and y scalers and feature dimension info."""
    if department in scalers_X and department in scalers_y:
        x_scaler, y_scaler = scalers_X[department], scalers_y[department]
        if not hasattr(x_scaler, "mean_") or not hasattr(y_scaler, "mean_"):
            print(f"[Warning] Skipping save — scalers for {department} are not fitted yet.")
            return

        joblib.dump(x_scaler, get_scaler_path(department, "X"))
        joblib.dump(y_scaler, get_scaler_path(department, "y"))
        meta_path = os.path.join(SCALERS_DIR, f"{department}_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"feature_dim": feature_dims.get(department)}, f)

# ----------------------
# Load scalers
# ----------------------
def load_department_scalers(department: str):
    """Load department scalers and metadata if available, with validation."""
    x_path = get_scaler_path(department, "X")
    y_path = get_scaler_path(department, "y")
    meta_path = os.path.join(SCALERS_DIR, f"{department}_meta.json")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise HTTPException(status_code=400, detail=f"No saved scalers found for department '{department}'. Train first.")

    # Load both scalers
    scalers_X[department] = joblib.load(x_path)
    scalers_y[department] = joblib.load(y_path)

    # Validate that the scalers are fitted (mean_ and scale_ exist)
    x_scaler = scalers_X[department]
    if not hasattr(x_scaler, "mean_") or not hasattr(x_scaler, "scale_"):
        raise HTTPException(status_code=400, detail=f"Scaler for '{department}' is not properly fitted or corrupted.")

    # Load meta info (feature dimensions)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            feature_dims[department] = meta.get("feature_dim")

    # If meta missing, recover from scaler directly
    if department not in feature_dims or feature_dims[department] is None:
        feature_dims[department] = len(x_scaler.mean_)

    # Final safeguard: check mean_ for NaN
    if np.isnan(x_scaler.mean_).any() or np.isnan(x_scaler.scale_).any():
        raise HTTPException(status_code=400, detail=f"Scaler for '{department}' contains NaN values.")

# ----------------------
# Delete scalers and models for a department
# ----------------------
def delete_and_retrain_department(department: str, model_type: str):
    """
    Deletes all model/scaler files for a department + model_type,
    then retrains the model from remaining data in the database.
    """

    from models import train_model

    deleted_files = []

    # --- Delete model files only ---
    dept_folder = os.path.join(BASE_MODEL_DIR, department)
    if os.path.exists(dept_folder):
        for file in os.listdir(dept_folder):
            path = os.path.join(dept_folder, file)
            os.remove(path)
            deleted_files.append(path)

    # --- Delete scalers ---
    scaler_files = [
        f"{department}_scaler_X.pkl",
        f"{department}_scaler_y.pkl",
        f"{department}_meta.json"
    ]

    for file_name in scaler_files:
        path = os.path.join(SCALERS_DIR, file_name)
        if os.path.exists(path):
            os.remove(path)
            deleted_files.append(path)

    # Remove in-memory references
    if department in scalers_X: del scalers_X[department]
    if department in scalers_y: del scalers_y[department]
    if department in feature_dims: del feature_dims[department]

    print(f"[DEBUG] Deleted all model and scaler assets for {department}")

    # --- Fetch remaining training data from DB ---
    db: Session = SessionLocal()
    try:
        query = db.query(TrainingInput).filter(
            TrainingInput.department == department,
            TrainingInput.model_type == model_type
        )
        remaining_entries = query.all()
        if not remaining_entries:
            print(f"[DEBUG] No remaining data for {department} / {model_type}. Skipping retrain.")
            return {"deleted_files": deleted_files, "retrained": False, "message": "No remaining data to retrain."}

        # Combine all remaining data
        combined_X, combined_y = [], []
        for entry in remaining_entries:
            payload = parse_payload_loose(entry.payload)
            if "X" in payload and "y" in payload:
                combined_X.extend(payload["X"])
                combined_y.extend(payload["y"])

        if not combined_X or not combined_y:
            return {"deleted_files": deleted_files, "retrained": False, "message": "Remaining entries have no valid X/y."}

        retrain_data = {"X": combined_X, "y": combined_y}

        # --- Retrain model ---
        train_model(department, model_type, retrain_data)
        print(f"[DEBUG] Retrained {model_type} model for {department} from remaining data.")

        return {"deleted_files": deleted_files, "retrained": True, "message": "Model retrained from remaining data."}
    finally:
        db.close()
