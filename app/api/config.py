import os
from pathlib import Path

# Model Selection Preference: "auto" | "automl" | "fallback"
SERVED_MODEL_PREFERENCE = os.environ.get("SERVED_MODEL_PREFERENCE", "auto").lower()

# Paths
API_DIR = Path(__file__).resolve().parent
APP_DIR = API_DIR.parent
PROJECT_ROOT = APP_DIR.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
XAI_DIR = ARTIFACTS_DIR / "xai"
GLOBAL_XAI_DIR = XAI_DIR / "global"
LOCAL_XAI_DIR = XAI_DIR / "local"
SAMPLES_DIR = XAI_DIR / "samples"
TEXT_DEMO_DIR = XAI_DIR / "text_demo"
IMAGE_DEMO_DIR = XAI_DIR / "image_demo"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_TEXT_DIR = PROJECT_ROOT / "data" / "text"

FRONTEND_DIST_DIR = APP_DIR / "frontend" / "dist"
STATIC_FALLBACK_DIR = APP_DIR / "static_fallback"

# Risk Thresholds
PROBABILITY_THRESHOLD = 0.50
LOW_RISK_THRESHOLD = 0.30
MEDIUM_RISK_THRESHOLD = 0.60

def get_risk_level(probability: float) -> str:
    """Classify failure probability into Low, Medium, or High risk."""
    if probability < LOW_RISK_THRESHOLD:
        return "Low"
    elif probability <= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    else:
        return "High"
