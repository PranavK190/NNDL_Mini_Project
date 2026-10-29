@echo off
setlocal enabledelayedexpansion

:: =========================================================================
:: Single-Command Project Launcher: Electricity Consumption Forecasting
:: =========================================================================

echo.
echo =========================================================================
echo  Electricity Consumption Forecasting Using Neural Networks
echo  Project Setup, Pipeline Orchestrator, and Dashboard Launcher
echo =========================================================================
echo.

:: Ensure working directory is the folder where the batch file resides
cd /d "%~dp0"
set "PROJ_DIR=%~dp0"
set "VENV_DIR=%PROJ_DIR%.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"

:: -------------------------------------------------------------------------
:: 1. Check Python Availability
:: -------------------------------------------------------------------------
echo [Step 1/9] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on system PATH.
    echo Please install Python 3.10, 3.11, or 3.12 and ensure "Add Python to PATH" is checked.
    goto ERROR
)
for /f "tokens=*" %%i in ('python --version') do set SYSTEM_PYTHON_VER=%%i
echo Found %SYSTEM_PYTHON_VER%

:: -------------------------------------------------------------------------
:: 2. Create Virtual Environment (.venv) if missing
:: -------------------------------------------------------------------------
echo.
echo [Step 2/9] Setting up local virtual environment (.venv)...
if not exist "%PYTHON_EXE%" (
    echo Creating virtual environment at %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        goto ERROR
    )
    echo Virtual environment created successfully.
) else (
    echo Existing virtual environment found at %VENV_DIR%.
)

:: -------------------------------------------------------------------------
:: 3. Upgrade Pip & Install Dependencies
:: -------------------------------------------------------------------------
echo.
echo [Step 3/9] Checking and installing requirements...
"%PYTHON_EXE%" -m pip install --upgrade pip >nul 2>&1
"%PIP_EXE%" install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install required dependencies from requirements.txt.
    goto ERROR
)

:: -------------------------------------------------------------------------
:: 4. Check Dataset & Download if Needed
:: -------------------------------------------------------------------------
echo.
echo [Step 4/9] Verifying dataset...
if not exist "%PROJ_DIR%datasets\electricity_hourly_dataset.tsf" (
    if not exist "%PROJ_DIR%datasets\electricity_hourly_dataset.zip" (
        echo Dataset not found. Downloading Monash Electricity Hourly dataset...
    )
)

:: -------------------------------------------------------------------------
:: 5. Data Preprocessing & Sequence Generation
:: -------------------------------------------------------------------------
echo.
echo [Step 5/9] Running data preprocessing and sliding-window sequence generation...
"%PYTHON_EXE%" "%PROJ_DIR%src\data_preprocessing.py"
if %errorlevel% neq 0 (
    echo [ERROR] Data preprocessing failed.
    goto ERROR
)

:: -------------------------------------------------------------------------
:: 6. Model Training (Support --retrain)
:: -------------------------------------------------------------------------
echo.
echo [Step 6/9] Checking neural network model artifacts...
set RETRAIN_FLAG=
if "%1"=="--retrain" set RETRAIN_FLAG=--retrain
if "%1"=="-r" set RETRAIN_FLAG=--retrain

if "%RETRAIN_FLAG%"=="--retrain" (
    echo Force retraining requested via --retrain flag.
)

"%PYTHON_EXE%" "%PROJ_DIR%src\train.py" %RETRAIN_FLAG%
if %errorlevel% neq 0 (
    echo [ERROR] Model training failed.
    goto ERROR
)

:: -------------------------------------------------------------------------
:: 7. Evaluation & Naive Baseline Benchmark
:: -------------------------------------------------------------------------
echo.
echo [Step 7/9] Running evaluation pipeline and iterative 24h forecasting...
"%PYTHON_EXE%" "%PROJ_DIR%src\evaluate.py"
if %errorlevel% neq 0 (
    echo [ERROR] Evaluation stage failed.
    goto ERROR
)

:: -------------------------------------------------------------------------
:: 8. Verify Required Artifacts Before Launch
:: -------------------------------------------------------------------------
echo.
echo [Step 8/9] Verifying generated model and evaluation artifacts...
if not exist "%PROJ_DIR%models\mlp_model.keras" (
    echo [ERROR] Missing model artifact: models\mlp_model.keras
    goto ERROR
)
if not exist "%PROJ_DIR%outputs\metrics.json" (
    echo [ERROR] Missing evaluation metrics: outputs\metrics.json
    goto ERROR
)
echo All required artifacts are present.

:: -------------------------------------------------------------------------
:: 9. Launch Streamlit Dashboard
:: -------------------------------------------------------------------------
echo.
echo =========================================================================
echo  Launching Interactive Streamlit Dashboard...
echo  (Press Ctrl+C in this terminal window to stop the dashboard)
echo =========================================================================
echo.
"%PYTHON_EXE%" -m streamlit run "%PROJ_DIR%app.py"
if %errorlevel% neq 0 (
    echo [ERROR] Streamlit dashboard failed to launch.
    goto ERROR
)

goto END

:ERROR
echo.
echo =========================================================================
echo  [FAILURE] Execution stopped due to an error in the previous step.
echo =========================================================================
echo.
pause
exit /b 1

:END
echo.
echo Launcher finished successfully.
exit /b 0
