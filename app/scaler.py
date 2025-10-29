import os
import json
from fastapi import HTTPException
import joblib
import numpy as np
from sqlalchemy.orm import Session

from database import SessionLocal, TrainingInput  # ✅ updated import

# =======================================
# CONSTANTS AND REGISTRIES
# =======================================
SCALERS_DIR = "./scalers"
BASE_MODEL_DIR = "saved_models"
os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

scalers_X = {}
scalers_y = {}
feature_dims = {}

# =======================================
# PATH HELPERS
# =======================================
def get_scaler_path(department: str, scaler_type: str):
    """Return the file path for a department's scaler."""
    return os.path.join(SCALERS_DIR, f"{department}_scaler_{scaler_type}.pkl")


# =======================================
# SAVE SCALERS
# =======================================
def save_department_scalers(department: str):
    """Persist both X and y scalers and feature dimension info."""
    if department in scalers_X and department in scalers_y:
        x_scaler, y_scaler = scalers_X[department], scalers_y[department]
        if not hasattr(x_scaler, "mean_") or not hasattr(y_scaler, "mean_"):
            print(f"[Warning] Skipping save — scalers for {department} not fitted yet.")
            return

        joblib.dump(x_scaler, get_scaler_path(department, "X"))
        joblib.dump(y_scaler, get_scaler_path(department, "y"))

        meta_path = os.path.join(SCALERS_DIR, f"{department}_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"feature_dim": feature_dims.get(department)}, f)
        print(f"[Scalers] ✅ Saved scalers for {department}")


# =======================================
# LOAD SCALERS
# =======================================
def load_department_scalers(department: str):
    """Load department scalers and metadata if available."""
    x_path = get_scaler_path(department, "X")
    y_path = get_scaler_path(department, "y")
    meta_path = os.path.join(SCALERS_DIR, f"{department}_meta.json")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise HTTPException(status_code=400, detail=f"No saved scalers found for '{department}'. Train first.")

    # Load scalers
    scalers_X[department] = joblib.load(x_path)
    scalers_y[department] = joblib.load(y_path)
    x_scaler = scalers_X[department]

    # Validate integrity
    if not hasattr(x_scaler, "mean_") or not hasattr(x_scaler, "scale_"):
        raise HTTPException(status_code=400, detail=f"Scaler for '{department}' is corrupted or not fitted.")
    if np.isnan(x_scaler.mean_).any() or np.isnan(x_scaler.scale_).any():
        raise HTTPException(status_code=400, detail=f"Scaler for '{department}' contains NaN values.")

    # Load meta info
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            feature_dims[department] = meta.get("feature_dim")

    # Recover missing dimension if needed
    if department not in feature_dims or feature_dims[department] is None:
        feature_dims[department] = len(x_scaler.mean_)

    print(f"[Scalers] ✅ Loaded scalers for {department} (features={feature_dims[department]})")


# =======================================
# DELETE + RETRAIN WORKFLOW
# =======================================
def delete_and_retrain_department(department: str, model_type: str):
    """
    Deletes the model and scaler files for a specific department + model_type,
    then retrains the model using remaining entries in the database.
    """
    from models import train_model  # local import to avoid circular dependency
    deleted_files = []

    # ----------------------------
    # DELETE MODEL FILES (specific model only)
    # ----------------------------
    dept_folder = os.path.join(BASE_MODEL_DIR, department)
    model_filename = f"{department}_{model_type.lower()}.pkl"
    model_path = os.path.join(dept_folder, model_filename)

    if os.path.exists(model_path):
        os.remove(model_path)
        deleted_files.append(model_path)
        print(f"[Model] 🗑️ Deleted {model_type} model for {department}")
    else:
        print(f"[Model] ⚠️ No {model_type} model found for {department} to delete")

    # ----------------------------
    # DELETE SCALER FILES (specific department only)
    # ----------------------------
    scaler_files = [
        f"{department}_scaler_X.pkl",
        f"{department}_scaler_y.pkl",
        f"{department}_meta.json",
    ]
    for fname in scaler_files:
        path = os.path.join(SCALERS_DIR, fname)
        if os.path.exists(path):
            os.remove(path)
            deleted_files.append(path)
            print(f"[Scaler] 🗑️ Deleted {fname}")

    # Remove in-memory copies
    scalers_X.pop(department, None)
    scalers_y.pop(department, None)
    feature_dims.pop(department, None)

    print(f"[Clean-up] ✅ Cleared scalers and model for '{department}' ({model_type})")

    # ----------------------------
    # RETRAIN USING REMAINING DB DATA
    # ----------------------------
    db: Session = SessionLocal()
    try:
        remaining = (
            db.query(TrainingInput)
            .filter(
                TrainingInput.department == department,
                TrainingInput.model_type == model_type.lower()
            )
            .all()
        )

        if not remaining:
            return {
                "deleted_files": deleted_files,
                "retrained": False,
                "message": f"No remaining data found for {department}/{model_type}.",
            }

        # Combine all valid payloads
        combined_data = {"X": [], "y": [], "values": [], "dates": []}

        for entry in remaining:
            payload = entry.payload
            if not isinstance(payload, dict):
                continue

            # Merge data based on structure
            if "X" in payload and "y" in payload:
                combined_data["X"].extend(payload["X"])
                combined_data["y"].extend(payload["y"])
            elif "values" in payload:
                combined_data["values"].extend(payload["values"])
                if "dates" in payload:
                    combined_data["dates"].extend(payload["dates"])

        # Prepare training data by model type
        if model_type.lower() == "linear":
            if not combined_data["X"] or not combined_data["y"]:
                return {
                    "deleted_files": deleted_files,
                    "retrained": False,
                    "message": "No valid Linear data remaining.",
                }
            train_data = {"X": combined_data["X"], "y": combined_data["y"]}

        elif model_type.lower() == "arima":
            if not combined_data["values"]:
                return {
                    "deleted_files": deleted_files,
                    "retrained": False,
                    "message": "No valid ARIMA data remaining.",
                }
            train_data = {"values": combined_data["values"]}

        elif model_type.lower() == "prophet":
            if not combined_data["dates"] or not combined_data["values"]:
                return {
                    "deleted_files": deleted_files,
                    "retrained": False,
                    "message": "No valid Prophet data remaining.",
                }
            train_data = {"dates": combined_data["dates"], "values": combined_data["values"]}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model_type: {model_type}")

        # ----------------------------
        # RETRAIN MODEL
        # ----------------------------
        train_model(department, model_type, train_data)
        print(f"[Retrain] ✅ Retrained {model_type} model for {department} using remaining data.")

        return {
            "deleted_files": deleted_files,
            "retrained": True,
            "message": f"{model_type.upper()} retrained successfully for {department}.",
        }

    finally:
        db.close()
