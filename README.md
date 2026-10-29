# Electricity Consumption Forecasting Using Neural Networks

## Project Overview
This project is an end-to-end Neural Network & Deep Learning (NNDL) project titled **“Electricity Consumption Forecasting Using Neural Networks.”** It uses historical hourly electricity consumption data from the Monash Electricity Hourly Dataset (UCI Machine Learning Repository) to train a Keras Multilayer Perceptron (MLP) neural network.

The project provides an automated Windows launcher (`run_proj.bat`), modular Python scripts for preprocessing, training, and evaluation, zero-leakage scaling, a 24-hour iterative forecaster, an interactive Streamlit dashboard (`app.py`), pytest unit tests, and a report in `report/report.md`.

---

## 🚀 Quick Start (Windows Workflow)

### Primary Launch Method:
1. Double-click **`run_proj.bat`** in the project folder, or run it from Command Prompt / PowerShell:
   ```cmd
   run_proj.bat
   ```
2. The launcher will automatically:
   - Check system Python availability.
   - Create local virtual environment `.venv` (if not existing).
   - Upgrade pip and install `requirements.txt`.
   - Download and extract the Monash Electricity dataset from Zenodo if missing.
   - Preprocess data and generate 168-hour sliding window sequences.
   - Train the MLP neural network model (or reuse existing `models/mlp_model.keras`).
   - Run evaluation against Naive Baseline ($t-24$) and calculate MAE / RMSE.
   - Launch the interactive Streamlit dashboard automatically in your browser (`http://localhost:8501`).

### Force Retraining:
To force retraining the neural network from scratch:
```cmd
run_proj.bat --retrain
```

---

## 📁 Suggested Project Structure

```text
electricity-forecasting/
├── run_proj.bat             # Single-command Windows batch launcher
├── requirements.txt         # Python package dependencies
├── .gitignore               # Git ignore file for environment and caches
├── README.md                # Main documentation
├── app.py                   # Streamlit interactive dashboard
├── src/
│   ├── data_preprocessing.py# Data download, tsf parsing, scaling, sliding sequences
│   ├── model.py             # Keras MLP model architecture definition
│   ├── train.py             # Model training orchestrator & loss plotting
│   └── evaluate.py          # Metric computation (MAE, RMSE) & 24h forecast generator
├── datasets/                # Downloaded raw dataset & processed numpy arrays
├── models/                  # Saved model (.keras) and scaler (.pkl)
├── outputs/                 # Evaluation plots, dataset info, metrics JSON/CSV
├── tests/
│   └── test_pipeline.py     # Pytest unit tests
└── report/
    └── report.md            # Markdown report ready for conversion to PDF
```

---

## 📊 Quantitative Results Summary

| Model / Benchmark | Evaluation Task | MAE (kW) | RMSE (kW) |
| :--- | :--- | :---: | :---: |
| **MLP Neural Network** | 1-Hour Ahead Test Set | **4.26** | **8.30** |
| **Naive Baseline ($t-24$)** | 1-Hour Ahead Test Set | 5.80 | 13.86 |
| **24-Hour Iterative Forecast** | Multi-Step Day Ahead | **1.44** | **1.96** |

---

## 🛠️ Data Dataset Attribution & Fallback Instructions

### Dataset Source
- **Dataset Title:** Electricity Hourly Dataset (Monash Time Series Forecasting Archive)
- **Zenodo URL:** [https://zenodo.org/records/4656140](https://zenodo.org/records/4656140)
- **Original UCI Source:** UCI Machine Learning Repository (ElectricityLoadDiagrams20112014)

### Automatic Download & Fallback Procedure
`run_proj.bat` attempts automatic dataset downloading via Zenodo API. If automatic download fails or is blocked by network firewalls:
1. Download `electricity_hourly_dataset.zip` directly from [https://zenodo.org/records/4656140](https://zenodo.org/records/4656140).
2. Place `electricity_hourly_dataset.zip` directly into the `datasets/` directory inside this project folder.
3. Re-run `run_proj.bat`. The launcher will detect the ZIP archive, extract `electricity_hourly_dataset.tsf`, and proceed automatically.

---

## 🧪 Testing & Verification

Run tests using pytest inside the virtual environment:
```cmd
.venv\Scripts\pytest
```
All unit tests verify sequence length (168 observations), 24-hour iterative forecast length, model loading, and finite metric computation in original units ($\text{kW}$).

---

## 📄 Submission Report
A report is available in `report/report.md` detailing architecture, methodology, baseline comparison, limitations, Viva Q&A, and screenshot guide.
