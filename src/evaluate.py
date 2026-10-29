import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

MODEL_PATH = os.path.join(MODELS_DIR, "mlp_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

def evaluate_pipeline():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("[Evaluate] Loading dataset, model, and scaler artifacts...")
    X_test = np.load(os.path.join(DATASET_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATASET_DIR, "y_test.npy"))
    scaler = joblib.load(SCALER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    # 1. Evaluate single-step predictions across held-out test set
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Invert scaling to convert to original consumption units (kW)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Naive Baseline (same hour previous day: t - 24)
    # The input sequence length is 168 hours (7 days).
    # The value 24 hours ago corresponds to index -24 in the input window X_test
    naive_pred_scaled = X_test[:, -24]
    naive_pred_orig = scaler.inverse_transform(naive_pred_scaled.reshape(-1, 1)).flatten()

    # Calculate metrics for single-step prediction
    mlp_mae = float(mean_absolute_error(y_test_orig, y_pred_orig))
    mlp_rmse = float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)))

    naive_mae = float(mean_absolute_error(y_test_orig, naive_pred_orig))
    naive_rmse = float(np.sqrt(mean_squared_error(y_test_orig, naive_pred_orig)))

    print("\n--- Held-Out Test Set Evaluation (1-Hour Ahead) ---")
    print(f"MLP Model     -> MAE: {mlp_mae:.2f} kW, RMSE: {mlp_rmse:.2f} kW")
    print(f"Naive Baseline -> MAE: {naive_mae:.2f} kW, RMSE: {naive_rmse:.2f} kW")

    # 2. Perform 24-Hour Ahead Iterative Multi-step Forecasting
    # Use the window prior to the last 24 test points (X_test[-25]) to forecast the last 24 hours iteratively
    initial_window = X_test[-25].copy() # shape (168,)
    actual_next_24_scaled = y_test[-24:] # actual values for last 24 test points
    actual_next_24_orig = y_test_orig[-24:]

    iterative_preds_scaled = []
    current_window = initial_window.copy()

    for step in range(24):
        # Predict 1 step ahead
        pred = model.predict(current_window.reshape(1, -1), verbose=0)[0, 0]
        iterative_preds_scaled.append(pred)
        # Slide window forward: drop first element, append prediction
        current_window = np.append(current_window[1:], pred)

    iterative_preds_scaled = np.array(iterative_preds_scaled)
    forecast_24_orig = scaler.inverse_transform(iterative_preds_scaled.reshape(-1, 1)).flatten()

    # Calculate 24-hour forecast metrics
    forecast_mae = float(mean_absolute_error(actual_next_24_orig, forecast_24_orig))
    forecast_rmse = float(np.sqrt(mean_squared_error(actual_next_24_orig, forecast_24_orig)))

    print(f"\n--- 24-Hour Iterative Forecast Metrics ---")
    print(f"24h Forecast -> MAE: {forecast_mae:.2f} kW, RMSE: {forecast_rmse:.2f} kW")

    # Save metrics JSON
    metrics = {
        "mlp_test_mae": mlp_mae,
        "mlp_test_rmse": mlp_rmse,
        "naive_baseline_mae": naive_mae,
        "naive_baseline_rmse": naive_rmse,
        "forecast_24h_mae": forecast_mae,
        "forecast_24h_rmse": forecast_rmse,
        "units": "kW"
    }

    with open(os.path.join(OUTPUTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save 24h forecast results CSV
    forecast_df = pd.DataFrame({
        "hour_ahead": list(range(1, 25)),
        "actual_consumption": actual_next_24_orig,
        "predicted_consumption": forecast_24_orig
    })
    forecast_df.to_csv(os.path.join(OUTPUTS_DIR, "forecast_24h.csv"), index=False)

    # Save Plots
    # Plot 1: Test Set Actual vs Predicted (Sample 200 hours)
    plt.figure(figsize=(12, 6))
    sample_len = min(200, len(y_test_orig))
    plt.plot(y_test_orig[:sample_len], label="Actual Consumption", color="black", alpha=0.8)
    plt.plot(y_pred_orig[:sample_len], label="MLP Model Prediction", color="blue", linestyle="--")
    plt.plot(naive_pred_orig[:sample_len], label="Naive Baseline (t-24)", color="orange", linestyle=":", alpha=0.7)
    plt.title("Actual vs Predicted Electricity Consumption (Held-Out Test Sample)")
    plt.xlabel("Hours")
    plt.ylabel("Consumption (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "actual_vs_predicted.png"))
    plt.close()

    # Plot 2: 24-Hour Iterative Forecast
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 25), actual_next_24_orig, marker='o', label="Actual 24h Consumption", color="black")
    plt.plot(range(1, 25), forecast_24_orig, marker='s', label="Iterative 24h MLP Forecast", color="red", linestyle="--")
    plt.title("24-Hour Ahead Iterative Electricity Forecast")
    plt.xlabel("Forecast Horizon (Hours Ahead)")
    plt.ylabel("Consumption (kW)")
    plt.xticks(range(1, 25))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "forecast_24h.png"))
    plt.close()

    print("[Evaluate] Saved evaluation plots and metrics successfully.")

if __name__ == "__main__":
    evaluate_pipeline()
