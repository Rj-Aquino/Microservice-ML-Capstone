# validation.py
import numpy as np
from fastapi import HTTPException

def validate_linear_data(data: dict, is_training=True):
    """Validate and normalize input for linear regression with multiple features."""
    try:
        if is_training:
            X, y = data.get("X"), data.get("y")
            if X is None or y is None:
                raise HTTPException(status_code=400, detail="Linear model requires 'X' and 'y'")
            X = np.array(X, dtype=float)
            y = np.array(y, dtype=float).reshape(-1, 1)
            if X.shape[0] != y.shape[0]:
                raise HTTPException(status_code=400, detail="Length of 'X' and 'y' must match")
        else:
            X = data.get("X")
            if X is None:
                raise HTTPException(status_code=400, detail="Linear prediction requires 'X'")
            X = np.array(X, dtype=float)
            y = None

        # Normalize features individually to range [0,1]
        X = (X - np.min(X, axis=0)) / (np.ptp(X, axis=0, keepdims=True) + 1e-8)
        return {"X": X, "y": y}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid numeric values in X or y")

def validate_arima_data(data: dict, is_training=True):
    """Validate input for ARIMA model."""
    values = data.get("values")
    if is_training:
        if not values or len(values) < 3:
            raise HTTPException(status_code=400, detail="ARIMA requires at least 3 'values'")
    else:
        if "steps_ahead" not in data:
            raise HTTPException(status_code=400, detail="ARIMA prediction requires 'steps_ahead'")
    return data

def validate_prophet_data(data: dict, is_training=True):
    """Validate input for Prophet model."""
    if is_training:
        dates = data.get("dates")
        values = data.get("values")
        if not dates or not values:
            raise HTTPException(status_code=400, detail="Prophet requires 'dates' and 'values'")
        if len(dates) != len(values):
            raise HTTPException(status_code=400, detail="'dates' and 'values' must have equal length")
    else:
        if "steps_ahead" not in data:
            raise HTTPException(status_code=400, detail="Prophet prediction requires 'steps_ahead'")
    return data

def validate_and_normalize(model_type: str, data: dict, is_training=True) -> dict:
    """Dispatcher for all model validations."""
    model_type = model_type.lower()

    if model_type == "linear":
        return validate_linear_data(data, is_training)
    elif model_type == "arima":
        return validate_arima_data(data, is_training)
    elif model_type == "prophet":
        return validate_prophet_data(data, is_training)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {model_type}")
