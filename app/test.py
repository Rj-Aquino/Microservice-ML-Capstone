import joblib
import numpy as np

department = "Bus_Finance"

x_scaler_path = f"saved_models/{department}/x_scaler.pkl"
y_scaler_path = f"saved_models/{department}/y_scaler.pkl"

x_scaler = joblib.load(x_scaler_path)
y_scaler = joblib.load(y_scaler_path)

print("X scaler mean:", x_scaler.mean_)
print("X scaler scale:", x_scaler.scale_)
print("y scaler mean:", y_scaler.mean_)
print("y scaler scale:", y_scaler.scale_)
