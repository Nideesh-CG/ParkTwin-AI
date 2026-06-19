#!/bin/bash
echo "=========================================================="
echo "  PARKTWIN AI - Digital Twin Launcher Console (Unix/macOS)"
echo "=========================================================="
echo ""
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Warning: Dependency installation encountered issues. Continuing..."
fi
echo ""
echo "Step 2: Training forecasting models and pre-generating SHAP explanations..."
python main.py --train
if [ $? -ne 0 ]; then
    echo "Warning: Model training encountered issues. Continuing..."
fi
echo ""
echo "Step 3: Starting FastAPI API server in background..."
python main.py --api &
API_PID=$!
echo "FastAPI running with PID $API_PID"
echo ""
echo "Step 4: Starting Streamlit Dashboard Command Center..."
python main.py --dashboard

# Cleanup background process on exit
kill $API_PID
