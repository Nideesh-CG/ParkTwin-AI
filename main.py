import os
import sys
import argparse
import subprocess
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkTwinAI.Launcher")

def run_tests():
    """Run pytest suite."""
    logger.info("Executing unit tests via pytest...")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    sys.exit(result.returncode)

def train_models():
    """Load dataset, train XGBoost model, and pre-generate SHAP explanations."""
    logger.info("Retraining machine learning forecast models & SHAP explainability...")
    from app.data_pipeline import DataPipeline
    from services.forecast_service import ForecastService
    from services.explainability_service import ExplainabilityService
    
    dp = DataPipeline()
    df = dp.load_data()
    
    fs = ForecastService(df)
    logger.info("Step 1: Training XGBoost forecasting model...")
    metrics = fs.train_model()
    logger.info(f"Model trained successfully. Evaluation metrics: {metrics}")
    
    es = ExplainabilityService(fs)
    logger.info("Step 2: Pre-generating SHAP explanation plots...")
    ex_results = es.generate_explanations(df)
    logger.info(f"Explainability completed: {ex_results}")
    
    logger.info("All model retraining tasks completed successfully!")

def start_api():
    """Start the FastAPI backend server."""
    logger.info("Starting FastAPI backend server on http://127.0.0.1:8000...")
    logger.info("Access API documentation at http://127.0.0.1:8000/docs")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)

def start_dashboard():
    """Start the Streamlit dashboard application."""
    logger.info("Launching Streamlit Command Center Dashboard...")
    dashboard_path = os.path.join("dashboard", "dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])

def main():
    parser = argparse.ArgumentParser(
        description="PARKTWIN AI: A Self-Learning Congestion-Aware Parking Intelligence Digital Twin Launcher"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dashboard", action="store_true", help="Launch the Streamlit Smart City Traffic Command Center Dashboard")
    group.add_argument("--api", action="store_true", help="Launch the FastAPI REST API Server")
    group.add_argument("--train", action="store_true", help="Train the XGBoost forecasting model and pre-generate SHAP explanations")
    group.add_argument("--test", action="store_true", help="Run the automated pytest suite")
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.train:
        train_models()
    elif args.api:
        start_api()
    elif args.dashboard:
        start_dashboard()

if __name__ == "__main__":
    main()
