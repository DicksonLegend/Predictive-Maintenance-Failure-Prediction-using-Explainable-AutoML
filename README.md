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

## 5. Phase 5 — DevOps, Containerization & Monitoring Layer

Phase 5 operationalizes the complete machine learning and explainable AI system with automated CI/CD pipelines, containerization, Kubernetes orchestration, Infrastructure-as-Code (Terraform), and a Prometheus/Grafana monitoring stack.

### DevOps Architecture Diagram

```text
                                      +---------------------------------------------+
                                      |                DEVELOPER                    |
                                      |   GitFlow: feature/* -> develop -> main     |
                                      +---------------------------------------------+
                                                             │ (git push)
                                                             ▼
                                      +---------------------------------------------+
                                      |          GITHUB ACTIONS CI/CD               |
                                      |  1. Quality: Ruff, Bandit, Gitleaks, Pytest |
                                      |  2. Build & Scan: Docker Multi-Stage, Trivy |
                                      |  3. Deploy Verify: KinD Ephemeral K8s Rollout|
                                      +---------------------------------------------+
                                                             │ (ghcr.io push)
                                                             ▼
                 ┌───────────────────────────────────────────┴──────────────────────────────────────────┐
                 ▼                                                                                      ▼
+----------------------------------+                                              +----------------------------------+
|      KUBERNETES ORCHESTRATION    |                                              |     MONITORING & ALERTING        |
|  - Namespace: xai-predictive-maint|                                              |  - Prometheus Server (:9090)     |
|  - Deployment: 2 replicas        | ────── Scraped via /metrics ───────────────► |  - Scrapes: http_requests_total, |
|  - RollingUpdate (Zero-Downtime) |                                              |    latency histograms,           |
|  - HPA: Auto-scales 1 to 4 pods  |                                              |    xai_predictions_total         |
|  - Service: ClusterIP (:80)      |                                              |  - Alertmanager: ApiDown, 5xx    |
|  - Non-Root Security (UID 10001) |                                              |  - Grafana Dashboard (:3000)     |
+----------------------------------+                                              +----------------------------------+
```

### DevOps Quickstart Commands

#### 1. Build Container Image (Lite Production Image)
```powershell
.\scripts\build_image.ps1
# Linux / macOS: ./scripts/build_image.sh
```

#### 2. Run Container Locally via Docker Compose
```powershell
.\scripts\run_app.ps1
# Linux / macOS: ./scripts/run_app.sh
```
- Web Application: [http://localhost:8000](http://localhost:8000)
- Metrics Endpoint: [http://localhost:8000/metrics](http://localhost:8000/metrics)

#### 3. Deploy to Kubernetes (Kustomize)
```powershell
.\scripts\deploy_k8s.ps1
# Linux / macOS: ./scripts/deploy_k8s.sh
```

#### 4. Launch Prometheus & Grafana Monitoring Stack
```powershell
.\scripts\start_monitoring.ps1
# Linux / macOS: ./scripts/start_monitoring.sh
```
- Grafana Dashboard: [http://localhost:3000](http://localhost:3000) (Credentials: `admin` / `admin`)
- Prometheus Web UI: [http://localhost:9090](http://localhost:9090)

#### 5. Generate Synthetic Telemetry Load
```powershell
.venv\Scripts\python scripts/generate_load.py --requests 300 --concurrency 5
```

#### 6. Validate Complete Phase 5 DevOps Suite
```powershell
.venv\Scripts\python scripts/validate_phase5.py
```

### Documentation & Runbooks
- **Operator Runbook:** [DEVOPS_SETUP_GUIDE.md](DEVOPS_SETUP_GUIDE.md) — Step-by-step manual execution guide covering GitHub setup, branch protection, Kubernetes scaling, and Terraform.
- **Technical Report:** [reports/devops_phase5_results.md](reports/devops_phase5_results.md) — Comprehensive 16-section academic write-up detailing security gates, container architecture, and the Rubric Coverage Matrix.
- **Architectural Assumptions:** [ASSUMPTIONS_PHASE5.md](ASSUMPTIONS_PHASE5.md) — Engineering decisions, verified metric schemas, and fallback logic.

---

## 6. Directory Structure Overview

```
predictive_maintenance_xai_automl/
├── app/
│   ├── api/                     # FastAPI ASGI Service Layer
│   │   ├── main.py              # Application entrypoint & Prometheus instrumentator
│   │   ├── config.py            # Path constants, model preference, risk thresholds
│   │   ├── metrics.py           # Custom Prometheus Counter (xai_predictions_total)
│   │   ├── schemas.py           # Pydantic v2 schemas and validators
│   │   ├── model_service.py     # Inference engine with directory & fallback loading
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
├── k8s/                         # Declarative Kubernetes Manifests (Kustomize)
│   ├── namespace.yaml           # Isolated xai-predictive-maintenance namespace
│   ├── configmap.yaml           # Environment configurations
│   ├── secret.yaml              # Secret key placeholders
│   ├── deployment.yaml          # RollingUpdate deployment (2 replicas, non-root)
│   ├── service.yaml             # ClusterIP port 80 -> 8000
│   ├── hpa.yaml                 # Horizontal Pod Autoscaler (1-4 replicas)
│   └── kustomization.yaml       # Kustomize aggregation manifest
├── terraform/                   # Infrastructure as Code (Docker Provider)
│   ├── versions.tf              # Terraform & provider requirements
│   ├── variables.tf             # Input variables with validation constraints
│   ├── main.tf                  # Bridge network, volume, Prometheus & Grafana
│   ├── outputs.tf               # Connection URLs and container IDs
│   └── terraform.tfvars.example # Example variable definitions
├── monitoring/                  # Observability Configuration
│   ├── prometheus/              # Scrape configs & alert rules (ApiDown, HighErrorRate)
│   ├── grafana/                 # Provisioned datasources & 8-panel operations dashboard
│   └── alertmanager/            # Webhook routing configuration
├── .github/                     # Collaboration & Automation
│   ├── workflows/ci-cd.yml      # Three-stage GitHub Actions CI/CD pipeline
│   ├── PULL_REQUEST_TEMPLATE.md # Quality-checked PR template
│   └── ISSUE_TEMPLATE/          # Standardized bug report & feature request forms
├── Dockerfile                   # Three-stage production containerfile
├── docker-compose.yml           # Application container compose configuration
├── docker-compose.monitoring.yml# Monitoring stack compose configuration
├── requirements.txt             # Complete project dependencies
├── requirements-lite.txt        # Lightweight runtime dependencies (<1.5 GB image)
├── requirements-dev.txt         # Dev, linting, security, and test tools
├── ruff.toml                    # Ruff linter configuration
├── .bandit                      # Bandit SAST security configuration
├── .dockerignore                # Build context exclusion rules
├── CONTRIBUTING.md              # GitFlow branching & contribution governance
├── DEVOPS_SETUP_GUIDE.md        # Comprehensive operations runbook
├── ASSUMPTIONS_PHASE5.md        # Architectural decisions & assumptions
├── artifacts/                   # Model Card JSON & XAI static assets
├── models/                      # AutoGluon directory predictor & Scikit-Learn models
├── reports/                     # Academic reports across Phases 2, 3, 4, and 5
└── tests/                       # Automated API integration tests (11/11 passing)
```

---

## 7. Model Governance & Operational Safeguards

- **Ultimate Served Model:** AutoGluon Tabular (`WeightedEnsemble_L2` Stacked Ensemble — Phase 3 Final Mode Winner).
- **Test Metrics:** F1-score = **0.7797**, ROC-AUC = **0.9735**, PR-AUC = **0.8490**, Precision = **0.9200**, Recall = **0.6765**, Accuracy = **0.9870**.
- **Fallback Hierarchy:** Gracefully falls back to Phase 2 Random Forest (`models/baseline_rf.pkl`, Test F1 = 0.7132) when running inside lightweight container environments without AutoGluon.
- **Risk Thresholds:** Low Risk ($P < 0.30$), Medium Risk ($0.30 \le P \le 0.60$), High Risk ($P > 0.60$).
- **Human-in-the-Loop Safeguard:** The system operates strictly as decision-support. Work orders derived from Shapley attributions require certified mechanical review prior to equipment intervention.

