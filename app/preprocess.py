# preprocess.py
import numpy as np
from sklearn.preprocessing import StandardScaler

# Department-specific scalers
scalers_X = {}
scalers_y = {}

def validate_data(raw_data):
    """Ensures X and y exist and have compatible lengths."""
    if "X" not in raw_data or "y" not in raw_data:
        raise ValueError("Both 'X' and 'y' keys are required for training data.")

    X = np.array(raw_data["X"], dtype=float)
    y = np.array(raw_data["y"], dtype=float)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Training data cannot be empty.")
    if len(X) != len(y):
        raise ValueError("Length mismatch: X and y must have the same number of elements.")

    return X, y

def preprocess_training_data(raw_data, department):
    """
    Validates, cleans, removes duplicates, and normalizes training data
    for a specific department, now supporting multiple features.
    """
    X, y = validate_data(raw_data)  # X can be 2D now

    # 2️⃣ Remove NaNs row-wise
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    X, y = X[mask], y[mask]

    # 3️⃣ Remove duplicate rows
    Xy = np.hstack([X, y.reshape(-1, 1)])
    Xy_unique = np.unique(Xy, axis=0)
    X, y = Xy_unique[:, :-1], Xy_unique[:, -1].reshape(-1, 1)

    # 4️⃣ Initialize or get department-specific scalers
    if department not in scalers_X:
        scalers_X[department] = StandardScaler()
        scalers_y[department] = StandardScaler()

    # 5️⃣ Normalize
    X_scaled = scalers_X[department].fit_transform(X)
    y_scaled = scalers_y[department].fit_transform(y)

    return X_scaled, y_scaled

def preprocess_input_data(raw_data, department):
    """
    Prepare input data for prediction using department-specific scalers.
    """
    if "X" not in raw_data:
        raise ValueError("Key 'X' is required for prediction.")

    X = np.array(raw_data["X"], dtype=float)

    if department not in scalers_X:
        raise ValueError(f"No scaler found for department '{department}'. Train first.")

    # Use the fitted scaler from training
    X_scaled = scalers_X[department].transform(X)
    return X_scaled

def inverse_transform_y(y_scaled, department):
    """Convert normalized y back to original scale."""
    if department not in scalers_y:
        raise ValueError(f"No y scaler found for department '{department}'. Train first.")
    return scalers_y[department].inverse_transform(y_scaled)
