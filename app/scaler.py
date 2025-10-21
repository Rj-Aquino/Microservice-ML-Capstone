import os
import json
import joblib

SCALER_DIR = "saved_scalers"
BASE_MODEL_DIR = "saved_models"
os.makedirs(SCALER_DIR, exist_ok=True)

scalers_X = {}
scalers_y = {}
feature_dims = {}

def get_scaler_path(department: str, scaler_type: str):
    """Return the file path for a department's scaler."""
    return os.path.join(SCALER_DIR, f"{department}_{scaler_type}.pkl")

def save_department_scalers(department: str):
    """Persist both X and y scalers and feature dimension info."""
    if department in scalers_X and department in scalers_y:
        joblib.dump(scalers_X[department], get_scaler_path(department, "X"))
        joblib.dump(scalers_y[department], get_scaler_path(department, "y"))
        meta_path = os.path.join(SCALER_DIR, f"{department}_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"feature_dim": feature_dims.get(department)}, f)

def load_department_scalers(department: str):
    """Load department scalers and metadata if available."""
    x_path = get_scaler_path(department, "X")
    y_path = get_scaler_path(department, "y")
    meta_path = os.path.join(SCALER_DIR, f"{department}_meta.json")

    if os.path.exists(x_path) and os.path.exists(y_path):
        scalers_X[department] = joblib.load(x_path)
        scalers_y[department] = joblib.load(y_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                feature_dims[department] = meta.get("feature_dim")

# === CLEANUP FUNCTION ===
def delete_department_assets(department: str):
    """
    Deletes all model and scaler files associated with a department.
    Called automatically when a training entry is deleted.
    """
    deleted_files = []

    # Model cleanup (pkl/json)
    dept_folder = os.path.join(BASE_MODEL_DIR, department)
    if os.path.exists(dept_folder):
        for file in os.listdir(dept_folder):
            path = os.path.join(dept_folder, file)
            os.remove(path)
            deleted_files.append(path)

    # Scaler cleanup
    for suffix in ["X.pkl", "y.pkl", "meta.json"]:
        path = os.path.join(SCALER_DIR, f"{department}_{suffix}")
        if os.path.exists(path):
            os.remove(path)
            deleted_files.append(path)

    # Remove empty folder if applicable
    if os.path.isdir(dept_folder) and not os.listdir(dept_folder):
        os.rmdir(dept_folder)

    return {"deleted_files": deleted_files}