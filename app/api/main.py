"""FastAPI Application for Predictive Maintenance Explainable AI Dashboard."""
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.config import (
    PROJECT_ROOT,
    ARTIFACTS_DIR,
    FRONTEND_DIST_DIR,
    STATIC_FALLBACK_DIR,
    RESULTS_DIR,
)
from app.api.schemas import (
    SensorInput,
    PredictionResponse,
    ExplanationResponse,
    WhatIfRequest,
    WhatIfResponse,
    TextExplainRequest,
    TextExplainResponse,
    ImageDemoResponse,
)
from app.api.model_service import model_service
from app.api.xai_service import explain_sensor_reading, get_global_explanations, get_sample_cases
from app.api.text_demo_service import explain_maintenance_text
from app.api.image_demo_service import get_image_demo_items
from src.xai.what_if import analyze_what_if
from prometheus_fastapi_instrumentator import Instrumentator
from app.api.metrics import xai_predictions_total

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pm_xai_api")

app = FastAPI(
    title="Predictive Maintenance XAI Dashboard",
    description="Explainable AI API for Machine Failure Prediction and Root-Cause Analysis in Industry 4.0",
    version="1.0.0",
)

# Enable CORS for local Vite and external client calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics Instrumentation (Phase 5 DevOps)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)

# Mount artifacts folder
if ARTIFACTS_DIR.exists():
    app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")

# Mount frontend assets if dist exists
dist_assets = FRONTEND_DIST_DIR / "assets"
if dist_assets.exists():
    app.mount("/assets", StaticFiles(directory=str(dist_assets)), name="assets")


def get_frontend_mode() -> str:
    """Returns 'react' if compiled React dist exists, else 'static_fallback'."""
    index_path = FRONTEND_DIST_DIR / "index.html"
    return "react" if index_path.exists() else "static_fallback"


@app.get("/")
def read_root():
    """Serves the built React SPA if present, otherwise serves the static fallback page."""
    if get_frontend_mode() == "react":
        return FileResponse(str(FRONTEND_DIST_DIR / "index.html"))
    fallback_index = STATIC_FALLBACK_DIR / "index.html"
    if fallback_index.exists():
        return FileResponse(str(fallback_index))
    return JSONResponse(
        content={
            "app": "Predictive Maintenance XAI Dashboard",
            "status": "ready",
            "message": "Frontend not compiled yet. Access /docs for interactive Swagger UI."
        }
    )


@app.get("/health")
def health_check():
    """Health status and model operational check."""
    model_loaded = model_service.served_model is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/version")
def version_info():
    """Version metadata and operational mode."""
    return {
        "app_name": "Predictive Maintenance XAI Dashboard",
        "api_version": "1.0.0",
        "phase": "phase4",
        "frontend_mode": get_frontend_mode(),
        "model_served": model_service.model_info.get("served_model_name", "Unknown"),
    }


@app.get("/model-card")
def get_model_card():
    """Returns the standardized Model Card JSON."""
    card_path = ARTIFACTS_DIR / "model_card.json"
    if not card_path.exists():
        raise HTTPException(status_code=404, detail="Model card artifact not found.")
    with open(card_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/model/info")
def get_model_details():
    """Returns technical architecture, training metrics, and explanation metadata."""
    feature_list = [
        "type (Machine quality variant: L, M, H)",
        "air_temp (Air temperature in Kelvin)",
        "process_temp (Process temperature in Kelvin)",
        "rotational_speed (Spindle rotational velocity in rpm)",
        "torque (Drive shaft mechanical torque in Nm)",
        "tool_wear (Cumulative tool operation duration in min)"
    ]
    return {
        "served_model_source": model_service.model_info.get("served_model_source"),
        "model_name": model_service.model_info.get("served_model_name"),
        "model_type": model_service.model_info.get("served_model_type"),
        "metrics": model_service.model_info.get("metrics"),
        "explanation_model_used": model_service.model_info.get("explanation_model_type"),
        "surrogate_used": model_service.model_info.get("surrogate_used", False),
        "reason_for_selection": model_service.model_info.get("reason_for_selection"),
        "feature_list": feature_list
    }


@app.get("/global-explanations")
def get_global_xai():
    """Returns precomputed global SHAP and PDP interpretability summaries."""
    try:
        return get_global_explanations()
    except Exception as e:
        logger.error("Error retrieving global explanations: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/samples")
def get_sample_records():
    """Returns precomputed representative high, medium, and low risk sample instances."""
    try:
        return get_sample_cases()
    except Exception as e:
        logger.error("Error retrieving samples: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image-demo", response_model=ImageDemoResponse)
def get_image_demo():
    """Returns multi-modal image explainability demonstration items."""
    try:
        return get_image_demo_items()
    except Exception as e:
        logger.error("Error retrieving image demo: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict_machine_failure(sensor_input: SensorInput):
    """Predicts machine failure probability and risk level from sensor telemetry."""
    try:
        res = model_service.predict(sensor_input)
        risk = getattr(res.prediction, "risk_level", "low").lower()
        xai_predictions_total.labels(risk_level=risk).inc()
        return res
    except Exception as e:
        logger.error("Error during failure prediction: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse)
def explain_machine_failure(sensor_input: SensorInput):
    """Computes dynamic local SHAP feature attributions and maintenance recommendations."""
    try:
        return explain_sensor_reading(sensor_input)
    except Exception as e:
        logger.error("Error during failure explanation: %s", e)
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.post("/what-if", response_model=WhatIfResponse)
def what_if_simulation(request: WhatIfRequest):
    """Performs counterfactual sensitivity analysis between baseline and modified telemetry."""
    try:
        return analyze_what_if(request.original_input, request.modified_input)
    except Exception as e:
        logger.error("Error during what-if analysis: %s", e)
        raise HTTPException(status_code=500, detail=f"What-if error: {str(e)}")


@app.post("/text-explain", response_model=TextExplainResponse)
def explain_notes(request: TextExplainRequest):
    """Analyzes maintenance notes with TF-IDF classifier and LIME token attribution."""
    try:
        return explain_maintenance_text(request)
    except Exception as e:
        logger.error("Error during text explanation: %s", e)
        raise HTTPException(status_code=500, detail=f"Text explain error: {str(e)}")
