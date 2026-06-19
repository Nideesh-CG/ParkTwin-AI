@echo off
echo ==========================================================
echo   PARKTWIN AI - Digital Twin Launcher Console (Windows)
echo ==========================================================
echo.
echo Step 1: Installing python dependencies from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Warning: Some dependencies failed to install. Continuing...
)
echo.
echo Step 2: Training XGBoost forecasting models and pre-generating SHAP explanations...
python main.py --train
if %ERRORLEVEL% neq 0 (
    echo Warning: Model training returned an error. Continuing...
)
echo.
echo Step 3: Starting FastAPI API server in background...
start /B python main.py --api
echo.
echo Step 4: Starting Streamlit Dashboard Command Center...
python main.py --dashboard
echo.
pause
