import requests

BASE_URL = "http://127.0.0.1:8000"

DEPARTMENTS = ["Bus_Operations", "Bus_Finance", "Bus_Inventory", "Bus_HR"]
MODELS = ["linear", "arima", "prophet"]

# Sample training data for each model
TRAINING_DATA = {
    "linear": {"X": [1, 2, 3], "y": [10, 20, 30]},
    "arima": {"values": [100, 110, 120, 130, 140]},
    "prophet": {"dates": ["2025-10-01", "2025-10-02", "2025-10-03"], "values": [50, 55, 60]}
}

# Sample prediction inputs
PREDICTION_INPUTS = {
    "linear": {"X": [4, 5]},
    "arima": {"steps_ahead": 3},
    "prophet": {"steps_ahead": 2}
}

def test_train_predict():
    for dept in DEPARTMENTS:
        for model in MODELS:
            train_payload = {
                "department": dept,
                "model_type": model,
                "data": TRAINING_DATA[model]
            }
            r = requests.post(f"{BASE_URL}/train", json=train_payload)
            print(f"Trained {dept} / {model}: {r.json()}")

            predict_payload = {
                "department": dept,
                "model_type": model,
                "input": PREDICTION_INPUTS[model]
            }
            r = requests.post(f"{BASE_URL}/predict", json=predict_payload)
            print(f"Predicted {dept} / {model}: {r.json()}")

def test_data_endpoint():
    print("\n--- /data endpoint testing ---")
    for dept in DEPARTMENTS + [None]:
        for model in MODELS + [None]:
            params = {}
            if dept: params["department"] = dept
            if model: params["model_type"] = model
            r = requests.get(f"{BASE_URL}/data", params=params)
            print(f"Data filter dept={dept}, model={model}: {r.json()}")

if __name__ == "__main__":
    test_train_predict()
    test_data_endpoint()
