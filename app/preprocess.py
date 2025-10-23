import numpy as np
from fastapi import HTTPException
from sklearn.preprocessing import StandardScaler
from scaler import (
    load_department_scalers,
    save_department_scalers,
    scalers_X,
    scalers_y,
    feature_dims,
)

# ==============================
# AUTO-CLEANING FUNCTION
# ==============================
def clean_training_data(X: np.ndarray, y: np.ndarray, zscore_threshold: float = 3.0):
    """Remove NaNs, duplicates, and extreme outliers automatically."""
    initial_len = len(X)

    # Remove rows with NaN
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    X, y = X[mask], y[mask]

    # Remove duplicates
    Xy = np.hstack([X, y])
    Xy_unique = np.unique(Xy, axis=0)
    X, y = Xy_unique[:, :-1], Xy_unique[:, -1].reshape(-1, 1)

    # Remove outliers using z-score
    z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    mask = (z_scores < zscore_threshold).all(axis=1)
    X, y = X[mask], y[mask]

    cleaned_len = len(X)
    removed = initial_len - cleaned_len
    print(f"[Auto-Clean] Removed {removed} invalid or outlier rows (kept {cleaned_len}).")
    return X, y

# ==============================
# LINEAR MODEL VALIDATION
# ==============================
def validate_linear_data(data: dict, department: str, is_training=True):
    try:
        if is_training:
            # --- Extract and validate raw data ---
            X = data.get("X")
            y = data.get("y")
            if X is None or y is None:
                raise HTTPException(status_code=400, detail="Linear model requires 'X' and 'y'.")
            X = np.array(X, dtype=float)
            y = np.array(y, dtype=float).reshape(-1, 1)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[0] != y.shape[0]:
                raise HTTPException(status_code=400, detail="Length of 'X' and 'y' must match.")

            # --- Clean and validate numeric content ---
            X, y = clean_training_data(X, y)
            if np.isnan(X).any() or np.isnan(y).any():
                raise HTTPException(status_code=400, detail="Training data contains NaN values.")

            # --- Initialize and fit scalers ---
            scalers_X[department] = StandardScaler()
            scalers_y[department] = StandardScaler()
            X_scaled = scalers_X[department].fit_transform(X)
            y_scaled = scalers_y[department].fit_transform(y)

            # --- Save feature dimensions and scalers to `scalers/` folder ---
            feature_dims[department] = X.shape[1]
            save_department_scalers(department)

            return {"X": X_scaled.tolist(), "y": y_scaled.tolist()}

        else:
            # --- Prediction phase ---
            X = data.get("X")
            if X is None:
                raise HTTPException(status_code=400, detail="Linear prediction requires 'X'.")
            X = np.array(X, dtype=float)
            if np.isnan(X).any():
                raise HTTPException(status_code=400, detail="Input contains NaN values.")
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # --- Load scalers from `scalers/` folder ---
            load_department_scalers(department)
            if department not in scalers_X:
                raise HTTPException(status_code=400, detail=f"No scaler found for '{department}'. Train first.")

            expected_dim = feature_dims.get(department)
            if expected_dim and X.shape[1] != expected_dim:
                raise HTTPException(status_code=400, detail=f"Expected {expected_dim} features, got {X.shape[1]}.")

            X_scaled = scalers_X[department].transform(X)
            if np.isnan(X_scaled).any():
                raise HTTPException(status_code=400, detail="Scaled input produced NaNs — check scaler integrity.")

            return {"X": X_scaled.tolist(), "y": None}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid numeric values in X or y.")

# ==============================
# ARIMA VALIDATION
# ==============================
def validate_arima_data(data: dict, is_training=True):
    values = data.get("values")
    if is_training:
        if not values or len(values) < 3:
            raise HTTPException(status_code=400, detail="ARIMA requires at least 3 'values'.")
    else:
        if "steps_ahead" not in data:
            raise HTTPException(status_code=400, detail="ARIMA prediction requires 'steps_ahead'.")
    return data

# ==============================
# PROPHET VALIDATION
# ==============================
def validate_prophet_data(data: dict, is_training=True):
    if is_training:
        dates = data.get("dates")
        values = data.get("values")
        if not dates or not values:
            raise HTTPException(status_code=400, detail="Prophet requires 'dates' and 'values'.")
        if len(dates) != len(values):
            raise HTTPException(status_code=400, detail="'dates' and 'values' must have equal length.")
    else:
        if "steps_ahead" not in data:
            raise HTTPException(status_code=400, detail="Prophet prediction requires 'steps_ahead'.")
    return data

# ==============================
# DISPATCHER
# ==============================
def validate_and_normalize(model_type: str, data, is_training=True, department: str = None) -> dict:
    model_type = model_type.lower()
    print(f"\n[DEBUG] === validate_and_normalize() called ===")
    print(f"[DEBUG] model_type={model_type}, is_training={is_training}, department={department}")

    if model_type == "linear":
        if department is None:
            raise HTTPException(status_code=400, detail="Department is required for Linear model validation.")
        return validate_linear_data(data, department, is_training)

    elif model_type == "arima":
        return validate_arima_data(data, is_training)

    elif model_type == "prophet":
        return validate_prophet_data(data, is_training)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {model_type}")

# ==============================
# Y-INVERSE TRANSFORM
# ==============================
def inverse_transform_y(y_scaled, department: str):
    if department not in scalers_y:
        raise HTTPException(status_code=400, detail=f"No y scaler found for '{department}'. Train first.")
    return scalers_y[department].inverse_transform(y_scaled)
