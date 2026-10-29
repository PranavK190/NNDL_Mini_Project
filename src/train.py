import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_mlp_model

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

MODEL_PATH = os.path.join(MODELS_DIR, "mlp_model.keras")
LOSS_PLOT_PATH = os.path.join(OUTPUTS_DIR, "loss_plot.png")

def train(force_retrain=False, epochs=30, batch_size=32):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH) and not force_retrain:
        print(f"[Train] Model already exists at {MODEL_PATH}. Skipping training (use --retrain to force).")
        return

    print("[Train] Loading preprocessed training and validation sequences...")
    X_train_path = os.path.join(DATASET_DIR, "X_train.npy")
    y_train_path = os.path.join(DATASET_DIR, "y_train.npy")
    X_val_path = os.path.join(DATASET_DIR, "X_val.npy")
    y_val_path = os.path.join(DATASET_DIR, "y_val.npy")

    if not (os.path.exists(X_train_path) and os.path.exists(y_train_path)):
        raise FileNotFoundError("Preprocessed training data not found. Please run data_preprocessing.py first.")

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)

    print(f"[Train] Training data shape: {X_train.shape}, Val data shape: {X_val.shape}")

    input_dim = X_train.shape[1]
    model = build_mlp_model(input_dim=input_dim)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print("[Train] Starting neural network training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    model.save(MODEL_PATH)
    print(f"[Train] Model saved successfully to {MODEL_PATH}")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
    plt.title('MLP Neural Network Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()
    print(f"[Train] Saved training loss plot to {LOSS_PLOT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP Electricity Forecaster Model")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if saved model exists")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    args = parser.parse_args()

    train(force_retrain=args.retrain, epochs=args.epochs)
