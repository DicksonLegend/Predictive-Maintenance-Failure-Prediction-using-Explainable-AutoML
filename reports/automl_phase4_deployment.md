# Phase 4 Deployment Report: Serving AutoML Predictive Maintenance Models

**Project Title:** Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0  
**Document:** AutoML Phase 4 Deployment Specifications  
**Date:** 2026-10-27  

---

## 1. Deployment Objective
The deployment objective of Phase 4 is to transition the winning AutoML failure prediction model from offline benchmarking scripts into a production-ready, low-latency RESTful API service paired with an interactive decision-support interface. This fulfills the AutoML rubric requirement to serve the best model through a modern web framework (FastAPI) and prepare the system for containerized Phase 5 DevOps deployment.

---

## 2. Model Export & Serialization
In Phase 3, competitive AutoML benchmarking across FLAML and AutoGluon established that **FLAML AutoML (LightGBM)** achieved the highest test $F_1$-score (**0.6281**, test precision = 0.7170, test ROC-AUC = 0.9171) under an identical 60-second time budget.

The model was serialized using `joblib` to:
```
models/flaml_model.pkl
```
The serialized asset packages both the internal preprocessing transformer (`_transformer`) and the optimized LightGBM gradient boosted decision tree estimator (`model.estimator`).

### Serving Priority Hierarchy
The serving layer (`app/api/model_service.py`) implements a resilient fallback hierarchy:
1. **Priority 1 (Phase 3 AutoML Winner):** Loads `models/flaml_model.pkl` from `results/phase3_best_model.json`.
2. **Priority 2 (Phase 2 Manual Winner):** Falls back to `models/baseline_rf.pkl` from `results/phase2_best_model.json` if AutoML fails to deserialize.
3. **Priority 3 (Safe Dynamic Fallback):** Trains an emergency balanced Random Forest on `data/processed/train_manual.csv` (`random_state=42`, `class_weight='balanced'`) and serializes to `models/fallback_rf.pkl`.

---

## 3. FastAPI Prediction Service Architecture
The inference service is hosted via FastAPI with an ASGI runtime (Uvicorn). Key engineering characteristics:
- **Stateless Design:** No server-side session state; enables horizontal pod autoscaling in Phase 5.
- **Strict Pydantic Validation:** Ingests `SensorInput` with type constraints, physical plausibility checks ($200K \le T \le 400K$), and non-negative constraints for rotational speed, torque, and tool wear.
- **Dual Endpoint Decoupling:**
  - High-throughput inference: `POST /predict` executes in $< 20\text{ms}$.
  - In-depth explainability: `POST /explain` computes local Shapley values and renders attribution waterfalls in $< 250\text{ms}$.

---

## 4. Health and Version Endpoints
To support future Kubernetes liveness/readiness probes and automated health checks in CI/CD pipelines, two dedicated monitoring endpoints are provided:

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2026-10-27T07:13:37.996123Z"
}
```

### `GET /version`
```json
{
  "app_name": "Predictive Maintenance XAI Dashboard",
  "api_version": "1.0.0",
  "phase": "phase4",
  "frontend_mode": "react",
  "model_served": "FLAML - lgbm"
}
```

---

## 5. Local Running Instructions

### Windows (PowerShell):
```powershell
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Run the FastAPI ASGI server with Uvicorn
.venv\Scripts\python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Linux / macOS:
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run the FastAPI ASGI server with Uvicorn
.venv/bin/python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running:
- **Interactive Web Dashboard:** Navigate to `http://localhost:8000/`
- **Swagger OpenAPI Documentation:** Navigate to `http://localhost:8000/docs`
- **ReDoc API Documentation:** Navigate to `http://localhost:8000/redoc`

---

## 6. Prediction Demo Example

### cURL Request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "L",
    "air_temp": 302.5,
    "process_temp": 311.2,
    "rotational_speed": 1360.0,
    "torque": 66.8,
    "tool_wear": 218.0
  }'
```

### Response Payload:
```json
{
  "input": {
    "type": "L",
    "air_temp": 302.5,
    "process_temp": 311.2,
    "rotational_speed": 1360.0,
    "torque": 66.8,
    "tool_wear": 218.0
  },
  "prediction": {
    "failure_probability": 0.8624,
    "predicted_class": 1,
    "risk_level": "High",
    "threshold": 0.5
  },
  "model": {
    "name": "FLAML - lgbm",
    "source": "Phase 3 AutoML (FLAML)"
  }
}
```

---

## 7. Model Card Summary
The deployment includes a standardized Model Card serialized to [`artifacts/model_card.json`](file:///d:/AutoML_XAI/artifacts/model_card.json) and exposed via `GET /model-card`. It establishes:
- **Data Lineage:** 8,000 training records, 3.39% baseline failure prevalence.
- **Audited Metrics:** Test F1 = 0.6281, Precision = 0.7170, Recall = 0.5588, ROC-AUC = 0.9171.
- **Fairness Guarantee:** Disparate impact is minimal ($FPR < 1.5\%$ across all quality tiers).
- **Leakage Prevention:** Controlled experiments demonstrated that post-hoc inspection flags artificially inflate F1 to 0.8244; production pipelines strictly filter downstream variables.

---

## 8. Deployment Limitations
1. **Cold Start Latency:** The initial TreeExplainer initialization requires $\sim 1.5\text{s}$ to cache the background sample distribution; subsequent inferences execute in milliseconds.
2. **Dynamic Plot Disk I/O:** Dynamic local SHAP waterfall plots are written to `artifacts/xai/local/shap_local_dynamic.png`; in high-concurrency production deployments, in-memory Base64 data URIs or Redis caching are recommended.

---

## 9. Preparation for DevOps Phase 5
The Phase 4 architecture is cleanly prepared for Phase 5 containerization and CI/CD automation:
- **Separation of Concerns:** Clear demarcation between backend service (`app/api/`), static frontend assets (`app/frontend/dist/`), models (`models/`), and artifacts (`artifacts/`).
- **Standardized Health Checks:** Ready for Kubernetes `livenessProbe` and `readinessProbe` targeting `/health`.
- **Pre-Compiled Frontend:** The React frontend compiles to pure HTML/CSS/JS in `app/frontend/dist/`, which FastAPI serves statically without requiring a Node.js runtime inside the production Docker container.
- **Automated Test Suite:** `tests/test_api.py` passes 10/10 automated tests using `TestClient`, enabling automated integration gates in GitHub Actions.
