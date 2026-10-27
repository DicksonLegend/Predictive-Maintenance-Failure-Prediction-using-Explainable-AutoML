"""Unit and Integration Tests for FastAPI Application Endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_1_health_endpoint():
    """1. GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert data["model_loaded"] is True
    assert "timestamp" in data


def test_2_version_endpoint():
    """2. GET /version returns 200."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "Predictive Maintenance XAI Dashboard"
    assert data["api_version"] == "1.0.0"
    assert data["phase"] == "phase4"
    assert "frontend_mode" in data
    assert "model_served" in data


def test_3_model_card_endpoint():
    """3. GET /model-card returns 200 and contains dataset field."""
    response = client.get("/model-card")
    assert response.status_code == 200
    data = response.json()
    assert "dataset" in data
    assert "model_name" in data
    assert "metrics" in data
    assert "intended_use" in data
    assert "explanation_methods" in data


def test_4_global_explanations_endpoint():
    """4. GET /global-explanations returns 200."""
    response = client.get("/global-explanations")
    assert response.status_code == 200
    data = response.json()
    assert "top_global_features" in data
    assert len(data["top_global_features"]) > 0
    assert "feature_interpretations" in data


def test_5_samples_endpoint():
    """5. GET /samples returns 200 and at least one sample."""
    response = client.get("/samples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    sample0 = data[0]
    assert "sample_id" in sample0
    assert "input_features" in sample0
    assert "predicted_probability" in sample0
    assert "risk_level" in sample0


def test_6_predict_endpoint():
    """6. POST /predict with valid input returns failure_probability between 0 and 1."""
    payload = {
        "type": "L",
        "air_temp": 300.0,
        "process_temp": 310.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 50.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    prob = data["prediction"]["failure_probability"]
    assert 0.0 <= prob <= 1.0
    assert data["prediction"]["predicted_class"] in [0, 1]
    assert data["prediction"]["risk_level"] in ["Low", "Medium", "High"]
    assert "model" in data


def test_7_explain_endpoint():
    """7. POST /explain with valid input returns top_reasons or explanation_method."""
    payload = {
        "type": "M",
        "air_temp": 302.5,
        "process_temp": 311.0,
        "rotational_speed": 1350.0,
        "torque": 65.0,
        "tool_wear": 210.0
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_reasons" in data
    assert len(data["top_reasons"]) > 0
    assert "explanation_method" in data
    assert "recommendation" in data
    assert "local_shap_plot" in data


def test_8_what_if_endpoint():
    """8. POST /what-if with valid input returns original_probability and modified_probability."""
    payload = {
        "original_input": {
            "type": "L",
            "air_temp": 303.0,
            "process_temp": 312.0,
            "rotational_speed": 1360.0,
            "torque": 68.0,
            "tool_wear": 200.0
        },
        "modified_input": {
            "type": "L",
            "air_temp": 300.0,
            "process_temp": 310.0,
            "rotational_speed": 1500.0,
            "torque": 40.0,
            "tool_wear": 20.0
        }
    }
    response = client.post("/what-if", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "original_probability" in data
    assert "modified_probability" in data
    assert "probability_change" in data
    assert "interpretation" in data
    assert "recommendation" in data
    assert 0.0 <= data["original_probability"] <= 1.0
    assert 0.0 <= data["modified_probability"] <= 1.0


def test_9_text_explain_endpoint():
    """9. POST /text-explain with sample text returns predicted_label."""
    payload = {
        "text": "Machine making loud noise and vibrating heavily near spindle."
    }
    response = client.post("/text-explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert data["predicted_label"] in ["urgent", "normal"]
    assert "probability" in data
    assert "important_words" in data
    assert len(data["important_words"]) > 0


def test_10_image_demo_endpoint():
    """10. GET /image-demo returns 200 and items list."""
    response = client.get("/image-demo")
    assert response.status_code == 200
    data = response.json()
    assert "image_demo_mode" in data
    assert "items" in data
    assert isinstance(data["items"], list)
    assert len(data["items"]) >= 1
    item0 = data["items"][0]
    assert "original" in item0
    assert "heatmap" in item0
    assert "overlay" in item0
    assert "predicted_class" in item0
    assert "explanation" in item0
