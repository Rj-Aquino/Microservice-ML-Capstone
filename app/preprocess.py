import numpy as np
from scipy import stats
from fastapi import HTTPException

# ============================================================
# Helper: Z-score mask for outlier detection
# ============================================================
def _zscore_mask(values: np.ndarray, threshold: float = 2.0):
    """
    Boolean mask to remove outliers using z-score.
    - Uses sample std (ddof=1) for better sensitivity on small datasets.
    - If std is 0 or dataset < 2 points, no points are removed.
    """
    values = np.array(values, dtype=float)
    n = len(values)

    if n < 2:  # Too few points to reliably detect outliers
        return np.ones(n, dtype=bool)

    mean = values.mean()
    std = values.std(ddof=1)  # sample std
    if std == 0:
        return np.ones(n, dtype=bool)

    z_scores = np.abs((values - mean) / std)
    return z_scores < threshold

# ============================================================
# Unified Cleaning Function (with logs)
# ============================================================
# ============================================================
# Training-only Cleaning Function (with logs)
# ============================================================
def clean_training_data(data: dict, model_type: str):
    """
    Cleans raw training data across all model types with detailed logs.

    Performs:
      - NaN removal
      - Duplicate removal (if applicable)
      - Outlier removal (via z-score)
    """
    model_type = model_type.lower()
    print(f"\n🧹 Starting cleaning for model_type='{model_type}'")

    # -----------------------------
    # LINEAR MODEL CLEANING
    # -----------------------------
    if model_type == "linear":
        if not all(k in data for k in ("X", "y")):
            raise HTTPException(status_code=400, detail="Linear model data must include 'X' and 'y'.")

        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)
        print(f"Initial Linear data: X={X.shape}, y={y.shape}")

        # Remove NaNs
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        removed_nans = np.count_nonzero(~valid_mask)
        X, y = X[valid_mask], y[valid_mask]
        print(f"Removed {removed_nans} NaN rows. Remaining: {len(X)} samples")

        # Remove duplicates
        if len(X) > 1:
            unique_X, unique_idx = np.unique(X, axis=0, return_index=True)
            removed_dupes = len(X) - len(unique_X)
            X, y = X[unique_idx], y[unique_idx]
            print(f"Removed {removed_dupes} duplicate rows. Remaining: {len(X)} samples")

        # Outlier removal
        mask = _zscore_mask(y)
        removed_outliers = np.count_nonzero(~mask)
        X, y = X[mask], y[mask]
        print(f"Removed {removed_outliers} outliers. Final count: {len(X)} samples")

        print("🧾 Summary (LINEAR):", {"X_shape": X.shape, "y_shape": y.shape, "returned_samples": len(X)})
        return {"X": X.tolist(), "y": y.tolist()}

    # -----------------------------
    # ARIMA MODEL CLEANING
    # -----------------------------
    elif model_type == "arima":
        if "values" not in data:
            raise HTTPException(status_code=400, detail="ARIMA model data must include 'values'.")

        values = np.array(data["values"], dtype=float)
        print(f"Initial ARIMA data: {len(values)} samples")

        # 1️⃣ Remove NaNs
        before = len(values)
        values = values[~np.isnan(values)]
        removed_nans = before - len(values)
        print(f"Removed {removed_nans} NaN values. Remaining: {len(values)} samples")

        # 3️⃣ Remove outliers using z-score
        mask = _zscore_mask(values)
        removed_outliers = np.count_nonzero(~mask)
        values = values[mask]
        print(f"Removed {removed_outliers} outliers. Final count: {len(values)} samples")

        print("🧾 Summary (ARIMA):", {
            "returned_samples": len(values),
            "first_values": values[:5].tolist()
        })

        return {"values": values.tolist()}

    # -----------------------------
    # PROPHET MODEL CLEANING
    # -----------------------------
    elif model_type == "prophet":
        if not all(k in data for k in ("dates", "values")):
            raise HTTPException(status_code=400, detail="Prophet model data must include 'dates' and 'values'.")

        dates = np.array(data["dates"], dtype=str)
        values = np.array(data["values"], dtype=float)
        print(f"Initial Prophet data: {len(values)} samples")

        # 1️⃣ Remove NaNs
        valid_mask = ~np.isnan(values)
        removed_nans = np.count_nonzero(~valid_mask)
        dates, values = dates[valid_mask], values[valid_mask]
        print(f"Removed {removed_nans} NaN values. Remaining: {len(values)} samples")

        # 2️⃣ Remove duplicates based on dates (keep first occurrence)
        _, unique_idx = np.unique(dates, return_index=True)
        removed_dupes = len(dates) - len(unique_idx)
        dates, values = dates[unique_idx], values[unique_idx]
        print(f"Removed {removed_dupes} duplicate dates. Remaining: {len(values)} samples")

        # 3️⃣ Remove outliers using z-score
        mask = _zscore_mask(values)
        removed_outliers = np.count_nonzero(~mask)
        dates, values = dates[mask], values[mask]
        print(f"Removed {removed_outliers} outliers. Final count: {len(values)} samples")

        print("🧾 Summary (PROPHET):", {
            "returned_samples": len(values),
            "first_dates": dates[:3].tolist(),
            "first_values": values[:3].tolist()
        })

        return {"dates": dates.tolist(), "values": values.tolist()}

    # -----------------------------
    # UNSUPPORTED MODEL TYPE
    # -----------------------------
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {model_type}")

def clean_prediction_data(data: dict, model_type: str):
    model_type = model_type.lower()
    print(f"\n🧹 Starting prediction cleaning for model_type='{model_type}'")

    if model_type == "linear":
        if "X" not in data:
            raise HTTPException(status_code=400, detail="Prediction input must include 'X'.")
        X = np.array(data["X"], dtype=float)

        # Remove NaNs
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]

        # Remove duplicates
        if len(X) > 1:
            X = np.unique(X, axis=0)

        print("🧾 Summary (LINEAR PREDICT):", {"X_shape": X.shape, "returned_samples": len(X)})
        return {"X": X.tolist()}

    elif model_type == "arima":
        if "values" not in data:
            raise HTTPException(status_code=400, detail="ARIMA prediction must include 'values'.")

        values = np.array(data["values"], dtype=float)
        print(f"Initial ARIMA prediction data: {len(values)} samples")

        # Remove NaNs only
        before = len(values)
        values = values[~np.isnan(values)]
        removed_nans = before - len(values)
        print(f"Removed {removed_nans} NaN values. Remaining: {len(values)} samples")

        # ✅ No duplicate or outlier removal in prediction
        print("🧾 Summary (ARIMA PREDICT):", {
            "returned_samples": len(values),
            "first_values": values[:5].tolist()
        })

        return {"values": values.tolist()}

    elif model_type == "prophet":
        if not all(k in data for k in ("dates", "values")):
            raise HTTPException(status_code=400, detail="Prophet prediction must include 'dates' and 'values'.")

        dates = np.array(data["dates"], dtype=str)
        values = np.array(data["values"], dtype=float)
        print(f"Initial Prophet prediction data: {len(values)} samples")

        # Remove NaNs
        valid_mask = ~np.isnan(values)
        removed_nans = np.count_nonzero(~valid_mask)
        dates, values = dates[valid_mask], values[valid_mask]
        print(f"Removed {removed_nans} NaN values. Remaining: {len(values)} samples")

        # Remove duplicate dates (keep first)
        unique_dates, unique_idx = np.unique(dates, return_index=True)
        removed_dupes = len(dates) - len(unique_dates)
        dates, values = dates[unique_idx], values[unique_idx]
        print(f"Removed {removed_dupes} duplicate dates. Remaining: {len(values)} samples")

        print("🧾 Summary (PROPHET PREDICT):", {
            "returned_samples": len(values),
            "first_dates": dates[:3].tolist(),
            "first_values": values[:3].tolist()
        })

        return {"dates": dates.tolist(), "values": values.tolist()}

# ============================================================
# Validation Functions (Optional Layer)
# ============================================================
def validate_linear_data(data: dict):
    """Ensure Linear data is non-empty and properly shaped."""
    X, y = np.array(data["X"]), np.array(data["y"])
    if len(X) == 0 or len(y) == 0:
        raise HTTPException(status_code=400, detail="Linear data cannot be empty after cleaning.")
    if len(X) != len(y):
        raise HTTPException(status_code=400, detail="X and y must have the same number of samples.")
    return True


def validate_arima_data(data: dict):
    """Ensure ARIMA data is non-empty."""
    values = np.array(data["values"])
    if len(values) < 5:
        raise HTTPException(status_code=400, detail="ARIMA requires at least 5 data points.")
    return True


def validate_prophet_data(data: dict):
    """Ensure Prophet data is valid."""
    dates, values = np.array(data["dates"]), np.array(data["values"])
    if len(dates) != len(values) or len(dates) < 5:
        raise HTTPException(status_code=400, detail="Prophet requires at least 5 data points.")
    return True
