"""XAI Service API wrapper for Tabular Explainability."""
import json
from pathlib import Path
from typing import List, Dict, Any
from app.api.config import GLOBAL_XAI_DIR, SAMPLES_DIR
from app.api.schemas import SensorInput, ExplanationResponse
from src.xai.tabular_explainer import TabularExplainerService, generate_all_global_artifacts, generate_local_sample_artifacts


def explain_sensor_reading(sensor_input: SensorInput) -> ExplanationResponse:
    """Computes dynamic SHAP and LIME explanations for an incoming telemetry reading."""
    explainer = TabularExplainerService.get_instance()
    return explainer.explain_instance(sensor_input)


def get_global_explanations() -> Dict[str, Any]:
    """Retrieves precomputed global model interpretability metrics and feature summaries."""
    json_path = GLOBAL_XAI_DIR / "global_explanation.json"
    if not json_path.exists():
        generate_all_global_artifacts()
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sample_cases() -> List[Dict[str, Any]]:
    """Retrieves precomputed representative local test sample cases."""
    samples = []
    if not SAMPLES_DIR.exists() or len(list(SAMPLES_DIR.glob("sample_*.json"))) == 0:
        generate_local_sample_artifacts()

    for sample_file in sorted(SAMPLES_DIR.glob("sample_*.json")):
        with open(sample_file, "r", encoding="utf-8") as f:
            samples.append(json.load(f))
    return samples
