import os
import sys
import zipfile
import urllib.request
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

ZIP_PATH = os.path.join(DATASET_DIR, "electricity_hourly_dataset.zip")
TSF_PATH = os.path.join(DATASET_DIR, "electricity_hourly_dataset.tsf")
ZENODO_URL = "https://zenodo.org/records/4656140/files/electricity_hourly_dataset.zip?download=1"

def ensure_directories():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

def download_dataset():
    ensure_directories()
    if os.path.exists(TSF_PATH):
        print(f"[Data Preprocessing] Found existing dataset at {TSF_PATH}")
        return True

    if os.path.exists(ZIP_PATH):
        print(f"[Data Preprocessing] Found existing zip archive at {ZIP_PATH}. Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(DATASET_DIR)
        return True

    print(f"[Data Preprocessing] Downloading dataset from Zenodo ({ZENODO_URL})...")
    try:
        req = urllib.request.Request(
            ZENODO_URL,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(ZIP_PATH, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("[Data Preprocessing] Download complete. Extracting archive...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(DATASET_DIR)
        print("[Data Preprocessing] Dataset extracted successfully.")
        return True
    except Exception as e:
        print(f"[Data Preprocessing Error] Automatic download failed: {e}")
        print("Fallback Instructions:")
        print("1. Manually download 'electricity_hourly_dataset.zip' from https://zenodo.org/records/4656140")
        print(f"2. Place 'electricity_hourly_dataset.zip' or 'electricity_hourly_dataset.tsf' in: {DATASET_DIR}")
        return False

def parse_tsf_file(filepath, series_id="T1"):
    print(f"[Data Preprocessing] Parsing series '{series_id}' from {filepath}...")
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    data_start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("@data"):
            data_start_idx = i + 1
            break

    if data_start_idx is None:
        raise ValueError("Invalid .tsf format: '@data' header not found.")

    selected_series = None
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if parts[0] == series_id:
            start_timestamp_str = parts[1]
            values = [float(v) for v in parts[2].split(",")]
            selected_series = (start_timestamp_str, values)
            break

    if selected_series is None:
        first_line = lines[data_start_idx].strip()
        parts = first_line.split(":")
        series_id = parts[0]
        start_timestamp_str = parts[1]
        values = [float(v) for v in parts[2].split(",")]
        print(f"[Data Preprocessing] Series not found. Falling back to first series '{series_id}'.")

    start_ts = pd.to_datetime(start_timestamp_str.replace("-00-01", " 00:00:00"))
    date_range = pd.date_range(start=start_ts, periods=len(values), freq='h')
    df = pd.DataFrame({"timestamp": date_range, "consumption": values})
    df.set_index("timestamp", inplace=True)
    return df, series_id

def create_sequences(data, input_seq_len=168):
    X, y = [], []
    for i in range(len(data) - input_seq_len):
        X.append(data[i : i + input_seq_len])
        y.append(data[i + input_seq_len])
    return np.array(X), np.array(y)

def process_data(series_id="T1", input_seq_len=168):
    if not download_dataset():
        sys.exit(1)

    df, actual_series_id = parse_tsf_file(TSF_PATH, series_id)

    # Data cleaning
    missing_count = df["consumption"].isnull().sum()
    if missing_count > 0:
        print(f"[Data Preprocessing] Found {missing_count} missing values. Forward-filling...")
        df["consumption"] = df["consumption"].ffill().bfill()

    df["consumption"] = df["consumption"].clip(lower=0)

    # Save raw time series CSV for reference/dashboard
    df.to_csv(os.path.join(DATASET_DIR, "selected_series.csv"))

    # Split chronologically (70% train, 15% validation, 15% test)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"[Data Preprocessing] Total observations: {n}")
    print(f"[Data Preprocessing] Split counts -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Fit MinMaxScaler ONLY on training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[["consumption"]])

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    train_scaled = scaler.transform(train_df[["consumption"]]).flatten()
    val_scaled = scaler.transform(val_df[["consumption"]]).flatten()
    test_scaled = scaler.transform(test_df[["consumption"]]).flatten()

    X_train, y_train = create_sequences(train_scaled, input_seq_len)

    val_full = np.concatenate([train_scaled[-input_seq_len:], val_scaled])
    X_val, y_val = create_sequences(val_full, input_seq_len)

    test_full = np.concatenate([val_scaled[-input_seq_len:], test_scaled])
    X_test, y_test = create_sequences(test_full, input_seq_len)

    print(f"[Data Preprocessing] Sequence shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[Data Preprocessing] Sequence shapes -> X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"[Data Preprocessing] Sequence shapes -> X_test: {X_test.shape}, y_test: {y_test.shape}")

    np.save(os.path.join(DATASET_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATASET_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATASET_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(DATASET_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(DATASET_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATASET_DIR, "y_test.npy"), y_test)

    dataset_info = {
        "series_id": actual_series_id,
        "total_observations": n,
        "train_observations": len(train_df),
        "val_observations": len(val_df),
        "test_observations": len(test_df),
        "min_val": float(df["consumption"].min()),
        "max_val": float(df["consumption"].max()),
        "mean_val": float(df["consumption"].mean()),
        "input_seq_len": input_seq_len,
        "start_time": str(df.index[0]),
        "end_time": str(df.index[-1])
    }

    with open(os.path.join(OUTPUTS_DIR, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=4)

    print("[Data Preprocessing] Preprocessing completed successfully.")

if __name__ == "__main__":
    process_data()
