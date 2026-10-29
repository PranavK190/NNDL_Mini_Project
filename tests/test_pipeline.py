import os
import json
import numpy as np
import pytest
import tensorflow as tf
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

def test_dataset_preprocessing_sequences():
    X_train_path = os.path.join(DATASET_DIR, "X_train.npy")
    y_train_path = os.path.join(DATASET_DIR, "y_train.npy")
    X_test_path = os.path.join(DATASET_DIR, "X_test.npy")
    y_test_path = os.path.join(DATASET_DIR, "y_test.npy")

    assert os.path.exists(X_train_path), "X_train.npy does not exist."
    assert os.path.exists(y_train_path), "y_train.npy does not exist."
    assert os.path.exists(X_test_path), "X_test.npy does not exist."
    assert os.path.exists(y_test_path), "y_test.npy does not exist."

    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)

    # Input sequence length requirement: 168 observations (1 week)
    assert X_train.shape[1] == 168, f"Expected input sequence length 168, got {X_train.shape[1]}"
    assert X_test.shape[1] == 168, f"Expected input sequence length 168, got {X_test.shape[1]}"

def test_model_loading_and_inference():
    model_path = os.path.join(MODELS_DIR, "mlp_model.keras")
    assert os.path.exists(model_path), "mlp_model.keras does not exist."

    model = tf.keras.models.load_model(model_path)
    sample_input = np.random.rand(1, 168)
    prediction = model.predict(sample_input, verbose=0)

    assert prediction.shape == (1, 1), f"Expected prediction shape (1, 1), got {prediction.shape}"
    assert not np.isnan(prediction[0, 0]), "Prediction contains NaN."
    assert not np.isinf(prediction[0, 0]), "Prediction is Infinite."

def test_24h_iterative_forecast_shape_and_finite_metrics():
    model_path = os.path.join(MODELS_DIR, "mlp_model.keras")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    X_test_path = os.path.join(DATASET_DIR, "X_test.npy")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    X_test = np.load(X_test_path)

    initial_window = X_test[-25].copy()
    preds = []
    current_win = initial_window.copy()

    for _ in range(24):
        pred = model.predict(current_win.reshape(1, -1), verbose=0)[0, 0]
        preds.append(pred)
        current_win = np.append(current_win[1:], pred)

    assert len(preds) == 24, f"Expected 24 forecast steps, got {len(preds)}"

    preds_orig = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    assert len(preds_orig) == 24
    assert np.all(np.isfinite(preds_orig)), "24h forecast contains non-finite values."

def test_metrics_json_structure_and_units():
    metrics_path = os.path.join(OUTPUTS_DIR, "metrics.json")
    assert os.path.exists(metrics_path), "metrics.json does not exist."

    with open(metrics_path) as f:
        metrics = json.load(f)

    required_keys = [
        "mlp_test_mae", "mlp_test_rmse",
        "naive_baseline_mae", "naive_baseline_rmse",
        "forecast_24h_mae", "forecast_24h_rmse", "units"
    ]

    for key in required_keys:
        assert key in metrics, f"Missing key '{key}' in metrics.json"

    assert metrics["units"] == "kW"
    assert np.isfinite(metrics["mlp_test_mae"])
    assert np.isfinite(metrics["mlp_test_rmse"])
    assert metrics["mlp_test_mae"] > 0
    assert metrics["mlp_test_rmse"] > 0
