# Electricity Consumption Forecasting Using Neural Networks

## Executive Summary
This project implements an end-to-end Neural Network & Deep Learning (NNDL) mini-project for forecasting hourly electricity consumption. Utilizing the public Monash Electricity Hourly Dataset (originating from UCI Machine Learning Repository), the project builds a Multilayer Perceptron (MLP) neural network in TensorFlow/Keras to predict hourly electricity consumption.

The system features automated data downloading, strict chronological train/validation/test splitting, zero data leakage scaling, single-step and 24-hour ahead iterative forecasting, comparison against a Naive Baseline ($t-24$), and an interactive Streamlit dashboard.

---

## 1. Technological Stack
- **Programming Language:** Python 3.10+
- **Deep Learning Framework:** TensorFlow 2.15+ / Keras
- **Data Science & Utility Libraries:** Pandas, NumPy, scikit-learn, joblib
- **Visualization:** Matplotlib
- **Web Dashboard:** Streamlit
- **Testing:** Pytest

---

## 2. Dataset Description & Structure
- **Dataset Name:** Monash Electricity Hourly Dataset (Zenodo Record: 4656140 / UCI Electricity Load Diagrams)
- **Data Format:** `.tsf` (Time Series Format)
- **Frequency:** Hourly ($1\text{h}$)
- **Target Series:** Client $T1$ electricity consumption (in Kilowatts, $\text{kW}$)
- **Observation Span:** 2012 to 2014 ($26,304$ total hourly observations)
- **Data Split:**
  - **Training Set (70%):** First 18,412 observations
  - **Validation Set (15%):** Next 3,945 observations
  - **Held-Out Test Set (15%):** Final 3,947 observations

---

## 3. Methodology & Neural Network Architecture

### 3.1 Sequence Preparation
A sliding-window sequence approach was implemented:
- **Input Window ($X$):** 168 consecutive hourly observations ($7\text{ days} \times 24\text{ hours}$).
- **Target ($y$):** The 1-hour ahead consumption value.

### 3.2 Preprocessing & Leakage Prevention
1. **Chronological Splitting:** The dataset is split sequentially without random shuffling to maintain temporal order.
2. **MinMax Normalization:** Scaling parameters ($0$ to $1$) are fit exclusively on the **training set** and then applied to validation and test sets.
3. **Missing Value Handling:** Missing observations are forward-filled (`ffill`) and zero-bounded.

### 3.3 Neural Network Architecture (MLP)
The primary forecasting model is a Multilayer Perceptron (MLP) implemented in Keras:
- **Input Layer:** 168 nodes (input sequence length)
- **Dense Layer 1:** 128 units, ReLU activation, Dropout ($0.2$)
- **Dense Layer 2:** 64 units, ReLU activation, Dropout ($0.2$)
- **Dense Layer 3:** 32 units, ReLU activation
- **Output Layer:** 1 unit (Linear activation for continuous consumption value)
- **Optimizer:** Adam ($\text{learning rate} = 0.001$)
- **Loss Function:** Mean Squared Error ($\text{MSE}$)
- **Early Stopping:** Monitored `val_loss` with patience of 7 epochs, saving best weights.

---

## 4. Evaluation & Naive Baseline Comparison

### 4.1 Evaluation Metrics
Evaluation is performed in original consumption units ($\text{kW}$):
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

### 4.2 Naive Baseline Benchmark
The Naive Baseline predicts consumption at hour $t$ using the observation from the exact same hour on the previous day ($t-24$):
$$\hat{y}_t^{\text{naive}} = y_{t-24}$$

### 4.3 Quantitative Results Summary
All metrics are evaluated on the held-out test set in original Kilowatts ($\text{kW}$):

| Model / Benchmark | Evaluation Task | MAE ($\text{kW}$) | RMSE ($\text{kW}$) |
| :--- | :--- | :---: | :---: |
| **MLP Neural Network** | 1-Hour Ahead Test Set | **4.26** | **8.30** |
| **Naive Baseline ($t-24$)** | 1-Hour Ahead Test Set | 5.80 | 13.86 |
| **24-Hour Iterative Forecast** | Multi-Step Day Ahead | **1.44** | **1.96** |

**Key Findings:**
1. The MLP Neural Network significantly outperforms the Naive Baseline on single-step prediction, reducing MAE by **26.5%** and RMSE by **40.1%**.
2. Autoregressive 24-step forward forecasting captures strong daily periodicity without divergence.

---

## 5. Single-Command Launcher (`run_proj.bat`)

The batch launcher `run_proj.bat` automates all operations sequentially:
1. **Python Installation Check:** Verifies system Python availability.
2. **Virtual Environment:** Creates `.venv` if not existing; reuses existing environment on subsequent launches.
3. **Dependency Installation:** Upgrades pip and installs packages from `requirements.txt`.
4. **Dataset Management:** Downloads and extracts dataset automatically from Zenodo if missing.
5. **Preprocessing:** Executes `src/data_preprocessing.py`.
6. **Model Training:** Executes `src/train.py` (reuses existing `models/mlp_model.keras` unless `--retrain` is passed).
7. **Evaluation:** Runs `src/evaluate.py` to calculate metrics and generate plots.
8. **Artifact Verification:** Confirms model and metrics exist before dashboard launch.
9. **Dashboard Launch:** Launches Streamlit interface via `app.py`.

---

## 6. Limitations and Future Work
1. **Single-Series Scope:** The current implementation models client series $T1$. Scaling to multi-series or global time-series networks (e.g. Temporal Fusion Transformer or N-BEATS) could improve multi-client predictions.
2. **Exogenous Features:** Incorporating weather, calendar features (day of week, public holidays), or electricity price signals could enhance forecast accuracy during sudden demand spikes.
3. **Multi-Step Architecture:** Transitioning from iterative 1-step forecasting to direct sequence-to-sequence (Seq2Seq) architectures could reduce error accumulation over longer horizons.

---

## 7. Viva Questions and Answers

### Q1: Why did you split the time series chronologically instead of using random k-fold cross-validation?
**Answer:** Time-series data possesses temporal dependence and autocorrelation. Random shuffling or standard k-fold cross-validation uses future data to predict past events, causing lookahead bias and data leakage. Chronological splitting ensures that model evaluation reflects real-world deployment where future data is unknown.

### Q2: How did you prevent data leakage during feature scaling?
**Answer:** The `MinMaxScaler` was fitted solely on the training split (70% of historical data). The fitted parameters ($\text{min}$ and $\text{max}$) were then used to transform validation and test sets. Fitting scalers on the entire dataset prior to splitting leaks distribution statistics from the test set into training, artificially boosting performance.

### Q3: What is the benefit of the 168-hour input sequence length?
**Answer:** 168 hours corresponds to exactly 7 full days ($7 \times 24\text{ hours}$). This window captures both intra-day diurnal demand patterns (morning/evening peaks) and weekly periodicity (workday vs. weekend consumption variations).

### Q4: How does 24-hour ahead iterative forecasting work?
**Answer:** The model predicts 1 hour ahead ($\hat{y}_{t+1}$). This prediction is appended to the input window while dropping the oldest observation, forming an updated 168-hour window to predict $\hat{y}_{t+2}$. This process repeats iteratively for 24 steps to generate a full day-ahead forecast.

---

## 8. Required Screenshots for Project Submission
To document your submission, capture the following screenshots when running the project:

1. **Terminal / Batch Launcher Execution:** Screenshot of `run_proj.bat` executing successfully in Command Prompt or PowerShell, showing steps 1 through 9.
2. **Streamlit Dashboard Overview:** Full window screenshot of Streamlit dashboard at `http://localhost:8501`.
3. **24-Hour Forecast Line Chart:** Screenshot of Tab 1 showing the 24-hour forward iterative forecast plot and table.
4. **Actual vs Predicted Plot:** Screenshot of Tab 2 showing test set comparison between MLP Model and Naive Baseline.
5. **Training Loss Plot & Metrics Table:** Screenshot of Tab 3 displaying training vs validation loss curves and quantitative performance metrics.
