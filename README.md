# Explainable AutoML for Predictive Maintenance

An industrial machine learning decision support system uniting **Automated Machine Learning (AutoML)** and **Explainable AI (XAI)** to predict equipment failure from sensor telemetry and deliver transparent, actionable root-cause explanations.

---

## 1. Problem Statement
In modern smart manufacturing, unexpected machinery breakdowns result in massive financial losses, supply chain bottlenecks, and severe safety hazards. 

This project tackles the **AI4I 2020 Predictive Maintenance** challenge: given telemetry measurements (air temperature, process temperature, spindle rotational speed, torque, and cumulative tool wear), predict whether the equipment is prone to imminent failure (`machine_failure` = 1) versus nominal operation (`machine_failure` = 0). Crucially, the system uncovers the underlying physical failure mechanisms (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure) via game-theoretic Shapley attributions (SHAP), local linear surrogates (LIME), partial dependence curves, and interactive what-if counterfactual analysis.

---

## 2. Quickstart & Setup Instructions

### Environment Setup
1. Open PowerShell / Command Prompt in the project root:
   ```powershell
   cd d:\AutoML_XAI
   ```
2. Activate the pre-configured Python 3.11 virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 3. Phase 4 Execution & Deployment Commands

Phase 4 delivers an interactive Explainable AI Dashboard featuring a **FastAPI** backend and a reactive **React (Vite)** single-page application.

### 1. Generate XAI Artifacts
Generates global SHAP summary, bar plots, feature importance, Partial Dependence Plots (PDP), local audited sample cases, NLP text explanations, and visual spatial saliency overlays:

**Windows (PowerShell):**
```powershell
.venv\Scripts\python scripts/generate_xai_artifacts.py --mode fast
```
**Linux / macOS:**
```bash
.venv/bin/python scripts/generate_xai_artifacts.py --mode fast
```
*(For final mode with 200 background samples and 5 local cases, replace `--mode fast` with `--mode final`)*.

### 2. Build React Frontend
Installs frontend dependencies and compiles the React application into `app/frontend/dist/` (falls back automatically to `app/static_fallback/index.html` if Node.js is unavailable):

**Windows (PowerShell):**
```powershell
.venv\Scripts\python scripts/build_frontend.py
```
**Linux / macOS:**
```bash
.venv/bin/python scripts/build_frontend.py
```

### 3. Run FastAPI Web Service
Starts the local ASGI server hosting both RESTful inference/explanation endpoints and the interactive React dashboard:

**Windows (PowerShell):**
```powershell
.venv\Scripts\python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```
**Linux / macOS:**
```bash
.venv/bin/python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```
- **Web Dashboard:** Access `http://localhost:8000/` in any browser.
- **Interactive OpenAPI Docs:** Access `http://localhost:8000/docs`.

### 4. Run Automated Test Suite
Executes the comprehensive integration test suite covering all 10 core API endpoints:

**Windows (PowerShell):**
```powershell
.venv\Scripts\python -m pytest tests/test_api.py -v
```
**Linux / macOS:**
```bash
.venv/bin/python -m pytest tests/test_api.py -v
```

### 5. Validate Phase 4 Deliverables
Runs the automated verification checklist ensuring all required code, artifacts, models, reports, and tests pass:

**Windows (PowerShell):**
```powershell
.venv\Scripts\python scripts/validate_phase4.py
```
**Linux / macOS:**
```bash
.venv/bin/python scripts/validate_phase4.py
```

---

## 4. Previous Phase Commands

### Phase 1: Data Ingestion & EDA
```powershell
.venv\Scripts\python scripts/run_phase1.py
.venv\Scripts\python scripts/validate_phase1.py
```

### Phase 2: Manual RF Baseline & HPO Benchmarking
```powershell
.venv\Scripts\python scripts/run_phase2.py --mode fast
.venv\Scripts\python scripts/validate_phase2.py
```

### Phase 3: AutoML Benchmarking (FLAML vs AutoGluon) & Auditing
```powershell
.venv\Scripts\python scripts/run_phase3.py --mode fast
.venv\Scripts\python scripts/validate_phase3.py
```

---

## 5. Directory Structure Overview

```
predictive_maintenance_xai_automl/
├── app/
│   ├── api/                     # FastAPI ASGI Service Layer
│   │   ├── main.py              # Application entrypoint & routes
│   │   ├── config.py            # Path constants and risk thresholds
│   │   ├── schemas.py           # Pydantic v2 schemas and validators
│   │   ├── model_service.py     # Inference engine with fallback hierarchy
│   │   ├── xai_service.py       # Tabular SHAP/LIME controller
│   │   ├── text_demo_service.py # NLP shift notes explanation controller
│   │   ├── image_demo_service.py# Visual saliency explanation controller
│   │   └── recommendation_service.py # Prescriptive rule mapping engine
│   ├── frontend/                # React (Vite) Single Page Application
│   │   ├── dist/                # Production-compiled static bundle
│   │   └── src/                 # Reactive component hierarchy
│   └── static_fallback/        # Standalone vanilla CSS/JS fallback UI
├── src/
│   ├── data/                    # Ingestion, preprocessing, synthetic generator
│   ├── models/                  # Manual RF, Grid, Random, Optuna HPO
│   ├── automl/                  # FLAML & AutoGluon benchmarking runners
│   └── xai/                     # Tabular SHAP/LIME, What-If, Text & Image XAI
├── artifacts/
│   ├── model_card.json          # Machine-readable governance Model Card
│   └── xai/                     # Precomputed & dynamic interpretability assets
├── data/
│   ├── raw/                     # Raw AI4I 2020 dataset
│   └── processed/               # Stratified 80/20 train/test splits
├── models/                      # Serialized models (.pkl)
├── reports/                     # Academic markdown reports & outlines
│   ├── model_card.md            # Standardized model card
│   ├── xai_phase4_results.md    # Phase 4 XAI results report
│   ├── automl_phase4_deployment.md # Phase 4 deployment specifications
│   └── screenshots/             # Embedded UI screenshots & instructions
├── results/                     # Run logs, metadata JSONs, benchmark tables
├── scripts/                     # Operational runners and automated validators
└── tests/                       # Automated API integration test suite
```

---

## 6. Model Governance & Operational Safeguards

- **Served Model:** FLAML AutoML LightGBM Classifier (Phase 3 Winner).
- **Test Metrics:** F1-score = 0.6281, ROC-AUC = 0.9171, Precision = 0.7170, Recall = 0.5588.
- **Risk Thresholds:** Low Risk ($P < 0.30$), Medium Risk ($0.30 \le P \le 0.60$), High Risk ($P > 0.60$).
- **Human-in-the-Loop Safeguard:** The system operates strictly as decision-support. Work orders derived from Shapley attributions require certified mechanical review prior to equipment intervention.
