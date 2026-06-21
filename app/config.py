import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# Output Subdirectories
REPORTS_DIR = OUTPUT_DIR / "reports"
EXPLANATIONS_DIR = OUTPUT_DIR / "explanations"
DETECTIONS_DIR = OUTPUT_DIR / "detections"
HEATMAPS_DIR = OUTPUT_DIR / "heatmaps"
SIMULATIONS_DIR = OUTPUT_DIR / "simulations"

# Create all output directories
for d in [REPORTS_DIR, EXPLANATIONS_DIR, DETECTIONS_DIR, HEATMAPS_DIR, SIMULATIONS_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Default City Settings (Bangalore Center)
DEFAULT_LATITUDE = 12.9716
DEFAULT_LONGITUDE = 77.5946

# PEI Weights
PEI_WEIGHTS = {
    "frequency": 0.30,
    "duration": 0.25,
    "peak_hour": 0.20,
    "junction_criticality": 0.15,
    "density": 0.10
}

# Detection settings
DEFAULT_STATIONARY_THRESHOLD_SEC = 5.0 # Seconds for demo
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    # Mapping for auto-rickshaws if standard model tracks them as motorcycle/car or specific classes
}

# Clustering Settings (DBSCAN)
DBSCAN_EPS_KM = 0.1  # 100 meters
DBSCAN_MIN_SAMPLES = 5
EARTH_RADIUS_KM = 6371.0088

# Performance Settings
MAX_SAMPLE_SIZE = 25000  # Cap data loaded in memory for heavy ML ops
