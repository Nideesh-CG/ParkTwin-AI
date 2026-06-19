import os
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from app.config import MODEL_DIR
from app.data_pipeline import DataPipeline
from services.hotspot_service import HotspotService
from services.pei_service import PEIService
from services.forecast_service import ForecastService
from services.delay_service import DelayService
from services.simulation_service import SimulationService
from services.detection_service import DetectionService
from services.traffic_memory_service import TrafficMemoryService
from services.immunity_service import ImmunityService
from api.main import app

client = TestClient(app)

# Fixture to load dataset once
@pytest.fixture(scope="module")
def dataset():
    dp = DataPipeline()
    try:
        df = dp.load_data()
        return df
    except Exception as e:
        pytest.skip(f"Failed to load dataset: {e}")

# 1. Dataset Loading Tests
def test_data_pipeline(dataset):
    assert dataset is not None
    assert len(dataset) > 0
    
    # Expected columns check
    expected_cols = ['latitude', 'longitude', 'created_datetime', 'closed_datetime', 
                     'police_station', 'junction', 'vehicle_type', 'vehicle_number', 'duration_minutes']
    for col in expected_cols:
        assert col in dataset.columns
        
    # Check datetimes are parsed
    assert pd.api.types.is_datetime64_any_dtype(dataset['created_datetime'])
    assert pd.api.types.is_datetime64_any_dtype(dataset['closed_datetime'])
    
    # Check null imputation
    assert dataset['police_station'].isna().sum() == 0
    assert dataset['vehicle_type'].isna().sum() == 0

# 2. Historical Hotspot DBSCAN Tests
def test_hotspot_dbscan(dataset):
    hs = HotspotService(dataset)
    clustered, summary = hs.run_spatial_clustering()
    
    assert clustered is not None
    assert summary is not None
    assert 'cluster_id' in clustered.columns
    
    if len(summary) > 0:
        assert 'latitude' in summary.columns
        assert 'longitude' in summary.columns
        assert 'violation_count' in summary.columns
        assert 'avg_duration_minutes' in summary.columns
        
    # Check temporal breakdowns
    _, tod, weekly = hs.get_temporal_breakdowns(clustered)
    assert isinstance(tod, dict)
    assert isinstance(weekly, dict)

# 3. PEI Calculation Tests
def test_pei_calculation(dataset):
    hs = HotspotService(dataset)
    _, summary = hs.run_spatial_clustering()
    
    pei_serv = PEIService(summary)
    leaderboard = pei_serv.calculate_pei()
    
    if len(leaderboard) > 0:
        assert 'pei_score' in leaderboard.columns
        assert 'severity_label' in leaderboard.columns
        
        # Verify bounds
        assert leaderboard['pei_score'].max() <= 100.0
        assert leaderboard['pei_score'].min() >= 0.0
        
        # Check severity labels mapping
        for _, row in leaderboard.iterrows():
            score = row['pei_score']
            label = row['severity_label']
            if score <= 40.0:
                assert label == 'Moderate'
            elif score <= 70.0:
                assert label == 'High'
            else:
                assert label == 'Critical'

# 4. Forecast Outputs (XGBoost) Tests
def test_forecast_xgboost(dataset):
    fs = ForecastService(dataset)
    # Train on small subset
    metrics = fs.train_model(dataset.head(500))
    
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert fs.is_trained
    
    # Verify predictions format
    forecasts = fs.predict_tomorrow_hotspots()
    assert len(forecasts) <= 10
    if len(forecasts) > 0:
        item = forecasts[0]
        assert 'police_station' in item
        assert 'risk_probability' in item
        assert 0.0 <= item['risk_probability'] <= 1.0

# 5. Causal Delay Estimation Tests
def test_delay_estimation():
    ds = DelayService()
    
    # 1 hour car violation, near a junction, high PEI
    res = ds.estimate_causal_delay(pei_score=80.0, duration_minutes=60.0, is_junction=True)
    assert isinstance(res, dict)
    assert res["vehicle_hours_lost"] > 0
    assert "assumptions" in res
    
    # Non-junction should have lower delay than junction
    res_no_junc = ds.estimate_causal_delay(pei_score=80.0, duration_minutes=60.0, is_junction=False)
    assert res_no_junc["vehicle_hours_lost"] < res["vehicle_hours_lost"]

# 6. Simulation Scenarios Tests
def test_simulation_intervention():
    sim = SimulationService()
    
    # Baseline VHL
    base_vhl = 100.0
    
    # Tow Vehicle (80% duration reduction)
    res_tow = sim.run_simulation(base_vhl, "Tow Vehicle", "Test St")
    assert res_tow["metrics"]["projected_delay_vhl"] == 20.0
    assert res_tow["metrics"]["improvement_percentage"] == 80.0
    assert res_tow["metrics"]["fuel_saved_liters"] > 0
    assert res_tow["metrics"]["co2_avoided_kg"] > 0
    assert res_tow["metrics"]["dollars_saved"] > 0

# 7. Detection Pipeline Tracker Tests
def test_detection_pipeline():
    detector = DetectionService()
    assert hasattr(detector, "yolo_available")
    
    # Run simulated tracking generator
    tracks = detector._generate_simulated_tracks(1, 640, 480)
    assert len(tracks) >= 2
    assert tracks[0]["track_id"] == 1
    assert tracks[0]["vehicle_type"] == "CAR"

# 8. API Endpoint Responses Tests
def test_api_endpoints():
    # Use context manager to trigger startup lifespan events in TestClient
    with TestClient(app) as test_client:
        # Test GET /status
        response = test_client.get("/status")
        assert response.status_code == 200
        res_json = response.json()
        assert "status" in res_json
        assert res_json["status"] == "online"
        
        # Test GET /hotspots
        response = test_client.get("/hotspots")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        
        # Test GET /pei
        response = test_client.get("/pei")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        
        # Test POST /simulate
        sim_payload = {"cluster_id": 0, "intervention": "Tow Vehicle", "traffic_flow": 600}
        response = test_client.post("/simulate", json=sim_payload)
        assert response.status_code in [200, 404]

# 9. Traffic Memory Service Tests
def test_traffic_memory_service(dataset):
    hs = HotspotService(dataset)
    clustered, summary = hs.run_spatial_clustering()
    pei_serv = PEIService(summary)
    leaderboard = pei_serv.calculate_pei()
    
    tms = TrafficMemoryService(leaderboard, clustered)
    profiles = tms.generate_profiles()
    
    assert isinstance(profiles, dict)
    if len(profiles) > 0:
        first_key = list(profiles.keys())[0]
        p = profiles[first_key]
        assert "recurrence_count" in p
        assert "avg_duration" in p
        assert "best_intervention" in p
        assert "avg_recovery_time" in p
        assert "escalated_events" in p

# 10. Traffic Immunity Service Tests
def test_traffic_immunity_service(dataset):
    hs = HotspotService(dataset)
    clustered, summary = hs.run_spatial_clustering()
    pei_serv = PEIService(summary)
    leaderboard = pei_serv.calculate_pei()
    
    tms = TrafficMemoryService(leaderboard, clustered)
    profiles = tms.generate_profiles()
    
    ims = ImmunityService(leaderboard, profiles)
    scores = ims.calculate_immunity_scores()
    
    assert isinstance(scores, dict)
    if len(scores) > 0:
        first_key = list(scores.keys())[0]
        s = scores[first_key]
        assert "tis_score" in s
        assert "classification" in s
        assert s["tis_score"] >= 0.0 and s["tis_score"] <= 100.0
        assert s["classification"] in ["Fragile", "Adaptive", "Resilient"]
        
        # Test simulated immunity projections
        sim_res = ims.estimate_simulated_immunity(first_key, "Tow Vehicle")
        assert sim_res is not None
        assert "tis_before" in sim_res
        assert "tis_after" in sim_res
        assert "improvement" in sim_res
