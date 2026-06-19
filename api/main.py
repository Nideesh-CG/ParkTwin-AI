import os
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd

from app.config import MODEL_DIR
from app.data_pipeline import DataPipeline
from services.hotspot_service import HotspotService
from services.pei_service import PEIService
from services.forecast_service import ForecastService
from services.explainability_service import ExplainabilityService
from services.detection_service import DetectionService
from services.delay_service import DelayService
from services.simulation_service import SimulationService

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkTwinAI.API")

app = FastAPI(
    title="PARKTWIN AI API",
    description="REST API interface for ParkTwin AI: A Self-Learning Congestion-Aware Parking Intelligence Digital Twin",
    version="1.0.0"
)

# Global variables for caching backend services
data_df = None
hotspots_summary = None
leaderboard = None
forecast_service = None

@app.on_event("startup")
def startup_event():
    """Load resources once at startup to keep APIs responsive."""
    global data_df, hotspots_summary, leaderboard, forecast_service
    logger.info("Initializing API backend resources...")
    try:
        # Load dataset
        pipeline = DataPipeline()
        data_df = pipeline.load_data()
        
        # Spatial Clustering and PEI Calculation
        hs = HotspotService(data_df)
        _, hotspots_summary = hs.run_spatial_clustering()
        
        pei_serv = PEIService(hotspots_summary)
        leaderboard = pei_serv.calculate_pei()
        
        # Initialize Forecast Service
        forecast_service = ForecastService(data_df)
        if os.path.exists(MODEL_DIR / "xgb_hotspot_model.json"):
            import xgboost as xgb
            forecast_service.model = xgb.XGBClassifier()
            forecast_service.model.load_model(str(MODEL_DIR / "xgb_hotspot_model.json"))
            forecast_service.is_trained = True
        else:
            logger.info("XGBoost model not pre-trained. Auto-training now...")
            forecast_service.train_model()
            
        logger.info("Backend resources initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize resources at startup: {e}")

class StatusResponse(BaseModel):
    status: str
    dataset_records: int
    total_hotspots: int
    model_trained: bool
    yolo_available: bool

class Hotspot(BaseModel):
    cluster_id: int
    latitude: float
    longitude: float
    violation_count: int
    avg_duration_minutes: float
    primary_vehicle_type: str
    primary_police_station: str
    primary_junction: str

class PEIItem(BaseModel):
    rank: int
    cluster_id: int
    primary_police_station: str
    primary_junction: str
    pei_score: float
    severity_label: str
    violation_count: int

class ForecastItem(BaseModel):
    rank: int
    police_station: str
    vehicle_type: str
    hour_of_day: int
    risk_probability: float

class SimulationRequest(BaseModel):
    cluster_id: int
    intervention: str
    traffic_flow: Optional[int] = 600

class SimulationResponse(BaseModel):
    hotspot_name: str
    intervention: str
    base_delay_vhl: float
    projected_delay_vhl: float
    delay_prevented_vhl: float
    improvement_percentage: float
    dollars_saved: float
    co2_avoided_kg: float

@app.get("/status", response_model=StatusResponse)
def get_status():
    """Get the current system status and counts."""
    det_service = DetectionService()
    return {
        "status": "online",
        "dataset_records": len(data_df) if data_df is not None else 0,
        "total_hotspots": len(hotspots_summary) if hotspots_summary is not None else 0,
        "model_trained": forecast_service.is_trained if forecast_service else False,
        "yolo_available": det_service.yolo_available
    }

@app.get("/hotspots", response_model=List[Hotspot])
def get_hotspots():
    """Retrieve all historical spatial hotspots discovered by DBSCAN."""
    if hotspots_summary is None or hotspots_summary.empty:
        raise HTTPException(status_code=503, detail="Hotspots service unavailable")
    return hotspots_summary.to_dict(orient="records")

@app.get("/pei", response_model=List[PEIItem])
def get_pei():
    """Get the Parking Externality Index (PEI) leaderboard."""
    if leaderboard is None or leaderboard.empty:
        raise HTTPException(status_code=503, detail="PEI service unavailable")
    return leaderboard.to_dict(orient="records")

@app.get("/forecast", response_model=List[ForecastItem])
def get_forecast():
    """Generate risk predictions for tomorrow."""
    if forecast_service is None:
        raise HTTPException(status_code=503, detail="Forecasting service unavailable")
    try:
        res = forecast_service.predict_tomorrow_hotspots()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect")
async def post_detect(file: UploadFile = File(...), threshold_sec: float = 5.0):
    """Upload a video and detect illegal parking violations."""
    temp_file = f"temp_upload_{file.filename}"
    try:
        with open(temp_file, "wb") as f:
            f.write(await file.read())
            
        detector = DetectionService()
        logs = detector.process_video(temp_file, stationary_threshold_sec=threshold_sec)
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.post("/simulate", response_model=SimulationResponse)
def post_simulate(req: SimulationRequest):
    """Simulate intervention outcomes on the target hotspot."""
    if leaderboard is None or leaderboard.empty:
        raise HTTPException(status_code=503, detail="PEI service unavailable")
        
    # Find hotspot row
    row = leaderboard[leaderboard['cluster_id'] == req.cluster_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Hotspot cluster_id {req.cluster_id} not found")
        
    row = row.iloc[0]
    is_junction = row['primary_junction'] != 'No Junction'
    
    # Run causal delay & simulation
    ds = DelayService()
    ds.TRAFFIC_FLOW_DEFAULT = req.traffic_flow
    delay_stats = ds.estimate_causal_delay(row['pei_score'], row['avg_duration_minutes'], is_junction)
    
    sim = SimulationService()
    sim_report = sim.run_simulation(delay_stats["vehicle_hours_lost"], req.intervention, row['primary_junction'])
    metrics = sim_report["metrics"]
    
    return {
        "hotspot_name": sim_report["hotspot_name"],
        "intervention": sim_report["intervention"],
        "base_delay_vhl": metrics["base_delay_vhl"],
        "projected_delay_vhl": metrics["projected_delay_vhl"],
        "delay_prevented_vhl": metrics["delay_prevented_vhl"],
        "improvement_percentage": metrics["improvement_percentage"],
        "dollars_saved": metrics["dollars_saved"],
        "co2_avoided_kg": metrics["fuel_saved_liters"] * sim.CO2_KG_PER_LITER
    }

@app.get("/explain")
def get_explain():
    """Retrieve model explanation summary metrics."""
    if forecast_service is None or not forecast_service.is_trained:
        raise HTTPException(status_code=503, detail="Forecasting model is not trained yet")
    return {
        "features": forecast_service.feature_names,
        "importances": forecast_service.get_feature_importance(),
        "explanations_saved_directory": "outputs/explanations/"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
