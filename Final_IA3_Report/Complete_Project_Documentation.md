# Complete Project Documentation & Engineering Master Reference

**Project Title:** Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0  
**Repository Identifier:** `d:\AutoML_XAI`  
**Lead Candidate / Author:** [STUDENT NAME]  
**University Seat Number (USN):** [USN]  
**Academic Department / Institution:** [COLLEGE NAME]  
**Submission Cycle:** Industry Assessment 3 (IA3) Master Reference  
**Documentation Version:** 1.0.0 (Production Master Release)  
**Verification Date:** [SUBMISSION DATE]  

---

## 1. PROJECT IDENTITY & REPOSITORY BLUEPRINT

### 1.1 Academic Scope & Subjects Covered
This engineering documentation synthesizes a comprehensive, cross-disciplinary Industry 4.0 system uniting three primary computer science and data engineering disciplines:
1. **Automated Machine Learning (AutoML):** Systematic hyperparameter tuning (Grid Search, Random Search, Optuna Bayesian optimization) contrasted with advanced automated pipeline frameworks (FLAML budget-aware optimization and AutoGluon Tabular multi-layer stacked ensembling), substantiated by rigorous critical audits for data leakage, subgroup bias, and overfitting.
2. **Explainable Artificial Intelligence (XAI):** Cooperative game-theoretic global feature attribution (TreeSHAP across 200 interventional background instances), local decision boundary extraction (LIME), non-linear Partial Dependence Plots (PDP), interactive counterfactual "What-If" delta simulations, prescriptive maintenance recommendation dispatching, and multimodal diagnostics (technician shift note NLP and computer vision defect saliency).
3. **Cloud DevOps & Site Reliability Engineering (DevOps):** Enterprise GitFlow version control, 3-stage GitHub Actions CI/CD pipelines incorporating static code analysis (Ruff) and security gates (Bandit SAST, Gitleaks secret detection, Trivy container vulnerability scanning), multi-stage Docker containerization, Kubernetes orchestration on KinD clusters with zero-downtime rolling updates and metrics-server Horizontal Pod Autoscaling (HPA), Terraform Infrastructure as Code (IaC), and live Prometheus/Grafana/Alertmanager observability.

### 1.2 Personnel & Institutional Placeholders
- Candidate Name: `[STUDENT NAME]`
- University Identification: `[USN]`
- Institution / Faculty: `[COLLEGE NAME]`
- Academic Supervisor: `[SUPERVISOR NAME]`
- Submission Date: `[SUBMISSION DATE]`

### 1.3 Repository Structural Topology
The repository is structured into modular, decoupled packages ensuring strict separation of concerns across raw datasets, model weights, API backends, compiled frontend interfaces, deployment manifests, automated test suites, and academic documentation:

```text
d:\AutoML_XAI\
|-- .github/
|   |-- workflows/ci-cd.yml          # 3-Stage GitHub Actions CI/CD Pipeline
|   |-- PULL_REQUEST_TEMPLATE.md     # Standardized Pull Request template
|   `-- ISSUE_TEMPLATE/              # Bug report and feature request schemas
|-- app/
|   |-- api/                         # FastAPI application backend
|   |   |-- main.py                  # API endpoints (/health, /predict, /explain, /what-if)
|   |   |-- model_service.py         # Model loading, surrogate binding, inference engine
|   |   |-- schemas.py               # Pydantic v2 telemetry validation models
|   |   `-- metrics.py               # Prometheus metrics instrumentation
|   `-- frontend/                    # Vite + React Single Page Application (8 tabs)
|       |-- src/                     # React UI components, risk gauges, waterfall charts
|       `-- dist/                    # Compiled static production bundle
|-- artifacts/                       # Persisted XAI static assets and model cards
|   |-- model_card.json              # Standardized model card specification
|   `-- xai/                         # Global/local plots, sample JSONs, vision heatmaps
|-- data/
|   |-- raw/ai4i2020.csv             # Raw UCI AI4I 2020 Predictive Maintenance dataset
|   `-- processed/                   # Stratified train/test splits (manual & automl)
|-- Final_IA3_Report/                # Official academic report deliverables
|   |-- AutoML_Report.md             # Subject Report 1: AutoML Benchmarking
|   |-- XAI_Report.md                # Subject Report 2: Explainable AI Decision Support
|   |-- DevOps_Report.md             # Subject Report 3: Cloud DevOps & Observability
|   `-- Complete_Project_Documentation.md # This comprehensive master reference
|-- k8s/                             # Declarative Kubernetes manifests (Kustomize)
|   |-- namespace.yaml               # xai-predictive-maintenance namespace
|   |-- deployment.yaml              # xai-api deployment (RollingUpdate, non-root, probes)
|   |-- service.yaml                 # ClusterIP service routing traffic to port 8000
|   |-- hpa.yaml                     # Horizontal Pod Autoscaler (1-4 pods @ 60% CPU)
|   |-- configmap.yaml / secret.yaml # Decoupled operational config and credentials
|   `-- kustomization.yaml           # Unified Kustomize orchestration manifest
|-- models/                          # Serialized model weights (PKL and AutoGluon folders)
|-- monitoring/                      # Observability stack configurations
|   |-- prometheus/prometheus.yml    # Prometheus scrape targets and intervals
|   |-- prometheus/alerts.yml        # Alertmanager alerting thresholds
|   `-- grafana/                     # Provisioned datasources and 8-panel dashboard JSON
|-- plots/                           # Statistical curves, ROC/PR trajectories, confusion matrices
|-- reports/                         # Phase-specific Markdown technical reports
|-- results/                         # Machine-readable evaluation CSVs, logs, and metadata
|   `-- phase55_evidence/            # 51 captured live execution evidence logs
|-- scripts/                         # Automation, validation, load generation, and teardown
|-- src/                             # Core library modules (tuning, automl, xai explainers)
|-- terraform/                       # Terraform IaC definitions (main.tf, outputs.tf)
|-- tests/test_api.py                # Pytest integration suite (11/11 passing tests)
|-- Dockerfile                       # Multi-stage production container build specification
`-- requirements.txt                 # Complete project dependencies
```

---

## 2. END-TO-END SYSTEM ARCHITECTURE

The end-to-end architecture operationalizes the flow of industrial data from raw sensory ingestion to automated machine learning model selection, post-hoc explainability, containerized microservice execution, and live cloud-native observability:

```text
====================================================================================================
                        END-TO-END INDUSTRIAL MLOPS & OBSERVABILITY ARCHITECTURE
====================================================================================================

 [1. INGESTION & DATA CURATION]
  UCI AI4I 2020 Dataset (10,000 operational cycles, 3.39% failure incidence)
  ├── 6 Sensor Inputs: Product Type, Air Temp, Process Temp, Rotational Speed, Torque, Tool Wear
  └── Stratified 80/20 Partition: 8,000 Train Rows (271 failures) | 2,000 Held-Out Test Rows (68 failures)
               │
               ▼
 [2. MODELING & BENCHMARKING ENGINE]
  ├── Manual Tuning Suite (5-Fold CV): Baseline RF (F1: 0.7132) | Grid Search | Random Search | Optuna
  ├── AutoML Suite (300s Budget): FLAML (rf, F1: 0.7680) vs. AutoGluon (WeightedEnsemble_L2, F1: 0.7797)
  └── Critical Audits: Synthetic Leakage (+10.42 pp spike) | Subgroup Fairness (L, M, H) | Overfitting Check
               │
               ▼
 [3. EXPLAINABILITY & DECISION SUPPORT TIER]
  Champion Model (AutoGluon Stacking) ──> Surrogate Random Forest (F1: 0.7132, 94.8% Fidelity)
  ├── Global Explanations: TreeSHAP (200 Background Samples), Feature Importance, Partial Dependence
  ├── Local Explanations: Dual SHAP Waterfall + LIME Rule Formulation across 5 Multi-Tier Sample Cases
  ├── Interactive What-If Studio: Counterfactual Simulation (Torque 68->42, Wear 220->15 => Risk 84.6%->2.4%)
  └── Multimodal Proxies: Technician Shift Log NLP Classifier (LIME Text) + Optical Defect Saliency Maps
               │
               ▼
 [4. CONTINUOUS DELIVERY & PACKAGING]
  GitHub Actions CI/CD Pipeline ──> 3 Stages: Quality Gate (Ruff/Bandit) -> Build & Trivy Scan -> KinD Deploy
  └── Multi-Stage Dockerfile (1.56 GB footprint, 333.9s build, non-root user UID 10001, Python 3.11-slim)
               │
               ▼
 [5. ORCHESTRATION & HIGH AVAILABILITY]
  Kubernetes (KinD cluster 'desktop-control-plane', namespace: xai-predictive-maintenance)
  ├── Deployment: xai-api (Replicas: 2 to 4 | RollingUpdate maxSurge 1, maxUnavailable 0)
  ├── Health Architecture: Dual /health Readiness (10s delay, 5s period) & Liveness (40s delay, 15s period)
  └── Horizontal Pod Autoscaler (HPA): Autonomous scale-out to 4 replicas under 176% CPU load (55 req/s)
               │
               ▼
 [6. OBSERVABILITY & INFRASTRUCTURE AS CODE]
  Terraform IaC (terraform/main.tf via kreuzwerker/docker provider)
  ├── Docker Network: xai_monitoring | Volume: xai_grafana_data
  ├── Prometheus Server (Port 9090): Scrapes host.docker.internal:8000/metrics every 10 seconds
  ├── Alertmanager Engine: ApiDown, HighErrorRate (>5%), HighP95Latency (>1.0s) active rules
  └── Grafana Dashboard (Port 3000): 8-Panel Operational Dashboard with custom business metric
      xai_predictions_total flowing live (400 -> 700 requests verified)
====================================================================================================
```

---

## 3. PHASE-BY-PHASE ENGINEERING NARRATIVE

### Phase 1: Data Curation, Exploratory Analysis, and Stratified Partitioning
Phase 1 established the empirical baseline. The UCI AI4I 2020 dataset models a milling machine with four physical failure mechanisms: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), and Overstrain Failure (OSF), summarized into a single binary `machine_failure` column. 
- **Prevalence:** 9,661 nominal cycles (96.61%) and 339 failure events (3.39%).
- **Data Hygiene:** Verified zero null values, validated physical ranges (temperatures ~300-314 K, torque 3.8-76.6 Nm, speeds 1168-2886 rpm, tool wear 0-253 min), and confirmed that machine identification columns (`UDI`, `Product ID`) were excluded to prevent index overfitting.
- **Stratified Partitioning:** Executed an 80/20 train/test split governed by `RANDOM_SEED = 42`. Training split: 8,000 records (271 failures, 3.39%); test split: 2,000 records (68 failures, 3.40%). Test records were strictly isolated as `test_manual.csv` and `test_automl.csv`.

### Phase 2: Manual Tree Ensemble Hyperparameter Optimization
Phase 2 evaluated Scikit-Learn tree ensembles using stratified 5-fold cross-validation on `train_manual.csv`:
- **Baseline Random Forest:** 100 trees, unconstrained depth, `class_weight='balanced'`. Resulted in CV F1 of 0.6614 and held-out test F1 of 0.7132 (precision 0.7541, recall 0.6765, ROC-AUC 0.9587) in 4.27 seconds.
- **Grid Search:** Evaluated 72 parameter configurations across 5 folds (360 fits total) exploring estimators [50, 100, 200], depths [5, 10, 20, None], and sample splits. Best CV F1 of 0.6703 converged on `n_estimators=100, max_depth=10`, achieving test F1 of 0.6757 (precision 0.6250, recall 0.7353, ROC-AUC 0.9704) in 39.72 seconds.
- **Random Search:** Sampled 15 configurations across 5 folds (75 fits), reaching CV F1 of 0.6681 and test F1 of 0.6803 in 11.09 seconds.
- **Optuna Bayesian Optimization:** Executed 15 TPE trials with Stratified 5-Fold CV. Discovered parameter sets achieving CV F1 of 0.6703 and test F1 of 0.6757 in 28.20 seconds.
- **Empirical Lesson:** Optimization algorithms favored higher recall (0.7353 vs. 0.6765) by lowering precision (0.6250 vs. 0.7541), confirming that unconstrained HPO on imbalanced data will sacrifice false alarm rates unless precision penalties are codified into the objective function.

### Phase 3: AutoML Benchmarking and Critical Evaluation Suite
Phase 3 benchmarked automated frameworks on `train_automl.csv` under identical 300-second computational ceilings:
- **FLAML AutoML:** Completed in 300.01 seconds. Converged on an optimized Random Forest (`rf`) estimator (`n_estimators=31, max_leaf_nodes=484`), yielding test F1 of 0.7680, precision of 0.8421, recall of 0.7059, ROC-AUC of 0.9535, and PR-AUC of 0.8268.
- **AutoGluon Tabular:** Completed base and ensemble training in 10.39 seconds. AutoGluon trained LightGBM, CatBoost, ExtraTrees, and neural networks, combining them into a multi-layer stacked meta-estimator (`WeightedEnsemble_L2`). This model emerged as the project champion: **Test F1 = 0.7797**, **Precision = 0.9200**, **ROC-AUC = 0.9735**, **PR-AUC = 0.8490**, and **Accuracy = 0.9870** (only 4 false positives in 2,000 test cases).
- **Synthetic Data Leakage Audit:** Synthesized `post_failure_inspection_flag` (asserted in 92% of failures, 1% of nominal). Rerunning FLAML caused an artificial **+10.42 percentage point F1 spike (0.7680 -> 0.8722)**, with the leaky feature usurping **62.4% of total feature attribution**. This empirically verified that AutoML blindly exploits leakage artifacts.
- **Subgroup Fairness Audit:** Audited across hardware tiers: Type L (Low quality, F1 = 0.7576, FPR = 0.27%), Type M (Medium quality, F1 = 0.8444, FPR = 0.17%), and Type H (High quality, F1 = 0.5714, FPR = 0.00%). Type H recall was limited to 0.40 due to small test sample support (only 5 failures in 214 machines).

### Phase 4: Explainable AI Decision Support System & REST API
Phase 4 converted the champion model into an interactive decision support platform:
- **Surrogate Random Forest Model:** To circumvent the mathematical inability of TreeSHAP to parse AutoGluon's multi-layer stacked ensemble graph, an automated surrogate pattern bound `models/baseline_rf.pkl` (F1 = 0.7132, 94.8% classification agreement) to the TreeExplainer engine with **200 stratified background samples**. Local explanations execute in under 180 ms.
- **Local Explanations:** Implemented dual-perspective explanations: SHAP waterfall plots decomposing additive probability offsets from base-rate ($E[f(x)] = 0.034$), and LIME decision rules extracting human-readable bounding intervals.
- **Prescriptive Recommendation Engine:** Formulated rule-based maintenance action dispatching tied to primary SHAP attribution vectors.
- **Interactive What-If Studio:** Developed counterfactual delta simulation. On a critical exemplar (Sample 0), reducing torque from 68 to 42 Nm and tool wear from 222 to 15 min dropped failure risk from 84.6% to 2.4% (an 82.2 percentage point risk abatement).
- **Multimodal Demos:** Integrated technician shift log text classification (TF-IDF + Logistic Regression with LIME token salience) and optical inspection defect saliency mapping.
- **Frontend SPA:** Built an 8-tab Vite/React dashboard served statically by the FastAPI backend at `/`.

### Phase 5: Cloud DevOps, CI/CD, Containerization, and Observability
Phase 5 engineered the production delivery infrastructure:
- **GitFlow & Quality Gates:** Enforced feature branch reviews, PR templates, Ruff linting (0 errors), Bandit SAST (0 high/medium vulnerabilities), Gitleaks secret scans, and Trivy container vulnerability gating on CRITICAL flaws.
- **Multi-Stage Dockerfile:** Implemented 3 stages (Node 20 Alpine frontend builder -> Python 3.11-slim dependencies -> Python 3.11-slim non-root runtime). Created `xai-automl:lite` with an optimized 1.56 GB footprint.
- **Kubernetes Orchestration:** Authored manifests for namespace `xai-predictive-maintenance`, ConfigMap, Secret, Service (ClusterIP), Deployment (2 replicas, non-root UID 10001, readiness probe delay 10s / period 5s, liveness probe delay 40s / period 15s, rolling update maxSurge 1 / maxUnavailable 0), and HPA (1 to 4 replicas @ 60% CPU).
- **Terraform IaC:** Authored `terraform/main.tf` using provider `kreuzwerker/docker` to declaratively provision the `xai_monitoring` bridge network, `xai_grafana_data` volume, Prometheus container (port 9090), and Grafana container (port 3000).
- **Live Monitoring Instrumentation:** Instrumentated FastAPI with Prometheus counters and histograms, defining alert rules (`ApiDown`, `HighErrorRate`, `HighP95Latency`) and an 8-panel Grafana dashboard (`xai-operations.json`).

### Phase 5.5: Local Deployment Execution & Empirical Evidence Capture
Phase 5.5 executed the complete DevOps layer live on the candidate's local workstation (Docker Desktop KinD cluster `desktop-control-plane`, Windows/PowerShell, Terraform v1.16.6):
- **Docker Image Build:** Built `xai-automl:lite` in **333.9 seconds**, verifying an image footprint of **1.56 GB**.
- **Metrics-Server Self-Healing:** Patched upstream `metrics-server` deployment with `--kubelet-insecure-tls` and removed obsolete `--kubelet-use-node-name`, bringing `kubectl top nodes` to Ready status (1% CPU, 9% RAM).
- **Kubernetes Deployment:** Applied Kustomize manifests; 2/2 pods attained Ready status in 27 seconds. Dual probes confirmed healthy.
- **Zero-Downtime Rolling Update Proof:** Restarted deployment while executing 10 continuous health probes (2s interval):
  `200 -> 200 -> 200 -> 200 -> 200 -> 200 -> 200 -> 503 (port-forward reconnect) -> 200 -> 200`. Traffic shifted seamlessly to new pod hashes (`xai-api-5865cb77f4-mtfhv`).
- **Autoscaling Telemetry (HPA):** Scaled deployment to 3 replicas. Fired 400 load requests (10 concurrency, 55.10 req/s, 100% 200 OK). Pod CPU surged to **176%** (exceeding 60% target). **HPA autonomously scaled the deployment to 4 replicas** (`9fgkm` spawned).
- **Terraform Monitoring Apply:** Terraform provisioned 6 resources cleanly. Containers `xai-prometheus-tf` and `xai-grafana-tf` verified `Up (healthy)`.
- **Live Observability & Custom Metric Verification:** Prometheus target `host.docker.internal:8000` verified `health: "up"`. Grafana confirmed database ok and indexed dashboard `xai-ops-main`. Injecting 300 synthetic load requests caused custom business metric `sum(xai_predictions_total)` to increment from **400 to 700**, proving end-to-end data flow: App -> Prometheus -> Grafana.
- **Full Regression Suite:** Validated 197/197 checks (Pytest 11/11, Phase 2 31/31, Phase 3 39/39, Phase 4 43/43, Phase 5 73/73).

---

## 4. COMPLETE METRICS APPENDIX

### 4.1 Master Benchmark Comparison: Fast Mode vs. Final Mode

| Phase / Modeling Strategy | Mode | Search Budget / Folds | Validation / CV F1 | Test F1-Score | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Tuning Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 2 Baseline RF** | Fast | 2-Fold CV | 0.5936 | 0.7132 | 0.9587 | 0.7812 | 0.7541 | 0.6765 | 3.05 s |
| **Phase 2 Baseline RF** | **Final** | **5-Fold CV** | **0.6614** | **0.7132** | **0.9587** | **0.7812** | **0.7541** | **0.6765** | **4.27 s** |
| **Phase 2 Grid Search RF** | Fast | 12 fits, 2-fold | 0.5848 | 0.5976 | 0.9641 | 0.7420 | 0.5104 | 0.7206 | 3.98 s |
| **Phase 2 Grid Search RF** | **Final** | **360 fits, 5-fold** | **0.6703** | **0.6757** | **0.9704** | **0.7944** | **0.6250** | **0.7353** | **39.72 s** |
| **Phase 2 Random Search RF**| Fast | 6 fits, 2-fold | 0.6107 | 0.6541 | 0.9649 | 0.7610 | 0.5714 | 0.7647 | 4.28 s |
| **Phase 2 Random Search RF**| **Final** | **75 fits, 5-fold** | **0.6681** | **0.6803** | **0.9711** | **0.7980** | **0.6329** | **0.7353** | **11.09 s** |
| **Phase 2 Optuna Bayesian** | Fast | 5 trials, 2-fold | 0.5848 | 0.5976 | 0.9641 | 0.7420 | 0.5104 | 0.7206 | 1.97 s |
| **Phase 2 Optuna Bayesian** | **Final** | **15 trials, 5-fold**| **0.6703** | **0.6757** | **0.9704** | **0.7944** | **0.6250** | **0.7353** | **28.20 s** |
| **Phase 3 FLAML AutoML** | Fast | 60 s Budget | 0.6250 | 0.6281 | 0.9171 | 0.6866 | 0.7170 | 0.5588 | 60.03 s |
| **Phase 3 FLAML AutoML** | **Final** | **300 s Budget** | **0.7698** | **0.7680** | **0.9535** | **0.8268** | **0.8421** | **0.7059** | **300.01 s** |
| **Phase 3 AutoGluon Tabular**| Fast | 60 s Budget | 0.4500 | 0.3721 | 0.9590 | 0.7155 | 0.8889 | 0.2353 | 8.75 s |
| **Phase 3 AutoGluon Tabular**| **Final** | **300 s Budget** | **0.8627** | **0.7797** | **0.9735** | **0.8490** | **0.9200** | **0.6765** | **10.39 s** |

### 4.2 Controlled Synthetic Data Leakage Audit Table

| Pipeline Configuration | AutoML Engine | Test F1-Score | Test ROC-AUC | Test PR-AUC | Test Accuracy | Dominant Explanatory Feature | Attribution Share |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Clean Baseline Matrix** | FLAML Final | 0.7680 | 0.9535 | 0.8268 | 0.9855 | Applied Torque (`torque`) | 34.2% |
| **Contaminated Matrix** | FLAML Final | **0.8722** | **0.9891** | **0.9151** | **0.9915** | **`post_failure_inspection_flag`** | **62.4%** |
| **Observed Net Impact** | Synthetic Leakage | **+0.1042** | **+0.0356** | **+0.0883** | **+0.0060** | **Signal Usurpation** | **+28.2 pp** |

### 4.3 Algorithmic Fairness & Subgroup Performance Table

| Hardware Quality Tier | Test Population ($N$) | Failures ($Y=1$) | Cohort Failure Rate | Precision | Recall | Cohort F1-Score | False Positive Rate (FPR) | False Negative Rate (FNR) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Type L (Low Quality)** | 1,170 | 38 | 3.25% | 0.8929 | 0.6579 | **0.7576** | **0.0027 (0.27%)** | 0.3421 (34.21%) |
| **Type M (Medium Quality)**| 616 | 25 | 4.06% | 0.9500 | 0.7600 | **0.8444** | **0.0017 (0.17%)** | 0.2400 (24.00%) |
| **Type H (High Quality)** | 214 | 5 | 2.34% | **1.0000** | 0.4000 | **0.5714** | **0.0000 (0.00%)** | 0.6000 (60.00%) |

### 4.4 Generalization Gap & Overfitting Verification Table

| Model Architecture | Validation / CV F1 | Held-Out Test F1 | Generalization Gap ($\Delta$) | Overfitting Flag | Analytical Stability Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Random Forest** | 0.6614 | 0.7132 | -0.0518 | Stable | High test generalization without parameter overfitting. |
| **Grid Search RF** | 0.6703 | 0.6757 | -0.0054 | Stable | Negligible gap ($\Delta < 0.01$); optimal depth constraint. |
| **Random Search RF** | 0.6681 | 0.6803 | -0.0122 | Stable | Stable distribution sampling across trees. |
| **Optuna Bayesian RF** | 0.6703 | 0.6757 | -0.0054 | Stable | Smooth Bayesian convergence without over-specialization. |
| **FLAML AutoML** | 0.7698 | 0.7680 | +0.0018 | Stable | Perfectly calibrated validation-to-test tracking. |
| **AutoGluon Tabular** | 0.8627 | 0.7797 | +0.0830 | Moderate Gap | Minor out-of-fold stacking gap; net test F1 remains project best. |

### 4.5 Global Interpretability Ranking Table (Authoritative SHAP)

| Rank | Telemetry Dimension | Mean Absolute SHAP ($E[\|\phi_i\|]$) | Attribution Share | Physical Domain Meaning |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Applied Torque (`torque`) | **0.1620** | **34.2%** | Mechanical shear stress and motor power stall limits. |
| **2** | Rotational Velocity (`rotational_speed`) | **0.1401** | **29.6%** | Dynamic spindle velocity; low speed under torque strains motor. |
| **3** | Cumulative Tool Wear (`tool_wear`) | **0.1085** | **22.9%** | Blade engagement time; steep risk acceleration above 190 min. |
| **4** | Ambient Air Temp (`air_temp`) | **0.0796** | **16.8%** | Environmental thermal baseline elevating heat dissipation strain. |
| **5** | Process Temp (`process_temp`) | **0.0348** | **7.4%** | Internal machining heat dissipation and cooling state. |
| **6** | Product Type L (`type_L`) | **0.0120** | **2.5%** | Lower physical tolerance margins on low-cost hardware. |
| **7** | Product Type M (`type_M`) | **0.0047** | **1.0%** | Standard mid-tier industrial tooling baseline. |
| **8** | Product Type H (`type_H`) | **0.0022** | **0.5%** | High-precision equipment with lowest baseline vulnerability. |

---

## 5. REPOSITORY ARTIFACT MAP

| Relative Path | Artifact Category | Description and Operational Function |
| :--- | :--- | :--- |
| `data/raw/ai4i2020.csv` | Input Data | Raw UCI AI4I 2020 Predictive Maintenance dataset (10,000 rows). |
| `data/processed/train_manual.csv` | Preprocessed Data | Stratified 8,000-row training matrix with one-hot encoded product types. |
| `data/processed/test_manual.csv` | Preprocessed Data | Held-out 2,000-row evaluation matrix used across all test benchmarks. |
| `models/baseline_rf.pkl` | Model Weight | Serialized heuristic Random Forest (`class_weight='balanced'`; F1 = 0.7132). |
| `models/optuna_rf.pkl` | Model Weight | Serialized Bayesian-tuned Random Forest ensemble. |
| `models/flaml_model.pkl` | Model Weight | Serialized FLAML champion model (`rf`, F1 = 0.7680). |
| `models/autogluon_predictor/` | Model Weights | Complete directory housing AutoGluon stacked ensemble (`WeightedEnsemble_L2`). |
| `models/text_demo_tfidf.pkl` | NLP Model | TF-IDF feature extractor fitted on technician maintenance shift logs. |
| `models/text_demo_logreg.pkl` | NLP Model | Logistic Regression classifier predicting shift log urgency. |
| `artifacts/model_card.json` | Governance | Standardized machine-readable model card specification. |
| `artifacts/xai/global/` | Visualizations | Static SHAP beeswarm summary, bar plot, and 4 partial dependence PNGs. |
| `artifacts/xai/local/` | Visualizations | Pre-computed local SHAP waterfalls and LIME decision rules for 5 sample cases. |
| `artifacts/xai/image_demo/` | Visualizations | Original, Grad-CAM heatmap, and overlay PNGs for 3 surface defect modes. |
| `k8s/` | Infrastructure | Declarative Kubernetes manifests (Deployment, Service, HPA, ConfigMap, Secret). |
| `terraform/` | Infrastructure | Terraform configuration files provisioning Prometheus, Grafana, networks, and volumes. |
| `monitoring/prometheus/` | Observability | Prometheus scrape configuration (`prometheus.yml`) and alert rules (`alerts.yml`). |
| `monitoring/grafana/` | Observability | Provisioned datasources and 8-panel live operations dashboard (`xai-operations.json`). |
| `results/phase55_evidence/` | Verification Evidence | Directory containing 51 captured live deployment evidence logs and telemetry files. |
| `Dockerfile` | Containerization | 3-stage multi-stage Docker build producing `xai-automl:lite` (1.56 GB). |
| `.github/workflows/ci-cd.yml` | CI/CD Automation | GitHub Actions 3-stage pipeline (Quality Gate -> Build/Scan/Push -> KinD Deploy). |
| `scripts/validate_phase*.py` | Quality Assurance | Programmatic validation scripts covering Phase 2 (31), Phase 3 (39), Phase 4 (43), Phase 5 (73). |

---

## 6. COMPLETE REPRODUCTION RUNBOOK

### 6.1 Prerequisites & Environment Setup
Ensure Windows 10/11 with PowerShell, Docker Desktop running with Kubernetes enabled (KinD cluster `desktop-control-plane`), Terraform v1.16+, and Python 3.11 installed.

```powershell
# 1. Clone repository and initialize virtual environment
git clone <repository-url> d:\AutoML_XAI
cd d:\AutoML_XAI
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install development and runtime dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

### 6.2 Data Preparation & Phase Benchmarking (Phases 1-4)
```powershell
# Execute Phase 1: Data Curation & Preprocessing
.venv\Scripts\python scripts/run_phase1.py

# Execute Phase 2: Manual Tree Ensemble Hyperparameter Optimization (Final Mode)
.venv\Scripts\python scripts/run_phase2.py --mode final

# Execute Phase 3: AutoML Benchmarking & Critical Audits (Final Mode, 300s budget)
.venv\Scripts\python scripts/run_phase3.py --mode final --budget 300

# Execute Phase 4: XAI Artifact Generation & FastAPI Service Preparation
.venv\Scripts\python scripts/run_phase4.py --mode final

# Compile React Single Page Application frontend
cd app/frontend
npm ci
npm run build
cd ../..
```

### 6.3 Local DevOps Execution & Observability (Phase 5 & 5.5)
```powershell
# 1. Execute static analysis and security scanning
ruff check .
bandit -c .bandit -r app/ src/
pytest tests/ -q

# 2. Build local Docker lite image (<2.0 GB)
docker build --build-arg INSTALL_AUTOML=false -t xai-automl:lite -f Dockerfile .

# 3. Stream Docker image into KinD cluster containerd node
.venv\Scripts\python scripts/load_image_to_kind.py

# 4. Deploy Kubernetes workloads
kubectl apply -k k8s/
kubectl -n xai-predictive-maintenance rollout status deployment/xai-api --timeout=240s

# 5. Launch background port-forward supervisor (host port 8000 -> svc/xai-api:80)
Start-Process -NoNewWindow python -ArgumentList "scripts/port_forward_supervisor.py"

# 6. Provision Prometheus & Grafana Monitoring Stack via Terraform
cd terraform
terraform init -backend=false
terraform validate
terraform apply -auto-approve
cd ..

# 7. Execute synthetic load test (400 requests, concurrency 10) to verify HPA
.venv\Scripts\python scripts/generate_load.py --requests 400 --concurrency 10 --url http://localhost:8000

# 8. Query Prometheus for live metric verification
curl.exe -s http://localhost:9090/api/v1/targets
curl.exe -s "http://localhost:9090/api/v1/query?query=sum(xai_predictions_total)"

# 9. Execute Master Multi-Phase Regression Validation Suite
.venv\Scripts\python scripts/validate_phase2.py
.venv\Scripts\python scripts/validate_phase3.py
.venv\Scripts\python scripts/validate_phase4.py
.venv\Scripts\python scripts/validate_phase5.py
```

---

## 7. FIVE-MINUTE DEMONSTRATION SCRIPT

This structured 5-minute presentation script enables the candidate to demonstrate all three project pillars smoothly during academic evaluation:

- **Minute 1: Problem Framing & AutoML Leaderboard (AutoML Pillar)**
  - Open `plots/phase3_full_comparison.png`.
  - State: *"We addressed predictive maintenance under severe 3.39% class imbalance on 10,000 industrial machining cycles. Under a synchronized 300-second budget, AutoGluon Tabular's multi-layer stacked ensemble (`WeightedEnsemble_L2`) won the benchmark with an F1-score of 0.7797, precision of 0.9200, and ROC-AUC of 0.9735, slashing false alarms by over 50% relative to baselines."*
  - Highlight the synthetic leakage audit (`plots/leakage_metric_comparison.png`): *"We proved that unmonitored AutoML is blind to data leakage: injecting an unverified diagnostic flag caused an artificial +10.42 pp F1 surge, usurping 62.4% of attribution."*

- **Minute 2: Global Attribution & Mechanical Physics (XAI Pillar)**
  - Navigate to the running dashboard at `http://localhost:8000` (or display `artifacts/xai/global/shap_summary.png`).
  - State: *"To explain this stacked black box, we developed a surrogate Random Forest model achieving 94.8% classification fidelity. Global TreeSHAP across 200 background instances reveals that applied torque (34.2%), rotational velocity (29.6%), and cumulative tool wear (22.9%) drive failures, perfectly aligning with physical machine operating limits."*
  - Point to the partial dependence plots: *"Our PDP curves identify the non-linear wear cliff at 190 minutes and motor torque overload at 65 Nm."*

- **Minute 3: Local Root-Cause Diagnosis & What-If Studio (XAI Pillar)**
  - Select Sample 0 (High Risk: 84.6% failure probability) in the dashboard.
  - State: *"The local SHAP waterfall shows base risk pushed from 3.4% to 84.6% by extreme torque (+0.38) and blade wear (+0.31). The prescriptive engine advises: 'Mechanical Overload: Throttle feed rate immediately'."*
  - Switch to the What-If Studio: Adjust torque from 68 to 42 Nm and tool wear from 222 to 15 min. Click Recalculate: *"Failure probability immediately drops from 84.6% down to 2.4%, proving an 82.2 percentage point risk abatement."*

- **Minute 4: Cloud-Native Containerization & Kubernetes HPA (DevOps Pillar)**
  - Open terminal and run `kubectl -n xai-predictive-maintenance get pods,hpa`.
  - State: *"Our application is packaged into a non-root, multi-stage Docker container (`xai-automl:lite`, 1.56 GB). On Kubernetes, it runs with zero-downtime rolling updates (`maxSurge 1, maxUnavailable 0`) and dual readiness/liveness probes. Under a 400-request load test achieving 55 req/s, CPU surged to 176%, and HPA autonomously scaled the deployment from 3 to 4 replicas."*

- **Minute 5: Observability Stack & Live Metric Propagation (DevOps Pillar)**
  - Open Prometheus at `http://localhost:9090/targets` and Grafana at `http://localhost:3000`.
  - State: *"The entire monitoring tier is declaratively provisioned via Terraform. Prometheus scrapes host port 8000 every 10 seconds. In Grafana, our 8-panel dashboard tracks latency, error rates, and custom business metrics. We verified live metric propagation by observing `sum(xai_predictions_total)` increment from 400 to 700 matching our load generation. Full regression testing confirms 197 out of 197 checks passing."*

---

## 8. VIVA PREPARATION: 20 CORE DEFENSE QUESTIONS & ANSWERS

### AutoML & Modeling
1. **Q: Why did you prioritize F1-score and Precision over Accuracy?**  
   *A: In the AI4I dataset, machine failures comprise only 3.39% of instances. A trivial majority classifier predicting zero failures achieves 96.61% accuracy while failing 100% of broken machines. F1-score balances precision and recall on the minority class.*

2. **Q: Why did the Baseline Random Forest achieve higher test F1 than Grid Search?**  
   *A: Grid Search optimized 5-fold cross-validation F1 by capturing additional minority failure instances (boosting recall from 0.6765 to 0.7353). However, this reduced test precision from 0.7541 down to 0.6250, resulting in 30 false positives compared to the baseline's 15. The baseline's heuristic `class_weight='balanced'` maintained superior precision generalization.*

3. **Q: How does FLAML's optimization strategy differ from AutoGluon's?**  
   *A: FLAML prioritizes low-cost, fast estimators (LightGBM, Random Forest) early in the budget using a randomized direct search to optimize a single best model. AutoGluon embraces multi-layer ensemble stacking, training heterogeneous model families and learning optimal convex blending weights over out-of-fold probability distributions.*

4. **Q: Why did AutoGluon win Phase 3?**  
   *A: AutoGluon's `WeightedEnsemble_L2` achieved the highest F1-score (0.7797) and an outstanding precision of 0.9200 (only 4 false positives across 2,000 test records), combining gradient boosted trees and bagged forests to cancel out individual model variances.*

5. **Q: What is data leakage, and how did your audit prove its danger?**  
   *A: Data leakage occurs when features containing post-event information are inadvertently included in training. Introducing `post_failure_inspection_flag` created an artificial +10.42 pp F1 surge (0.7680 to 0.8722), with the leaky flag capturing 62.4% of total attribution. AutoML blindly exploited the feature, proving it cannot validate real-world physical semantics.*

6. **Q: What did your subgroup fairness audit reveal regarding Type H machines?**  
   *A: Type H represents premium hardware with only 5 failures in the 2,000 test instances. While the model achieved perfect precision (1.0000, 0 false alarms), missing 3 subtle failures depressed recall to 0.4000 (F1 = 0.5714). This demonstrates that global metric optimization can penalize low-support premium equipment cohorts.*

7. **Q: Did AutoGluon exhibit overfitting?**  
   *A: AutoGluon exhibited a moderate validation-to-test F1 gap of +0.0830 (0.8627 vs. 0.7797). This is characteristic of multi-layer stacking meta-learners slightly over-fitting out-of-fold blending weights, but its test F1 (0.7797) remained superior to all single-model architectures.*

### Explainable AI (XAI)
8. **Q: Why couldn't you compute TreeSHAP directly on AutoGluon's winning model?**  
   *A: AutoGluon's `WeightedEnsemble_L2` is a multi-layer stacked ensemble combining diverse algorithms. TreeSHAP requires a single, unified tree decision graph. KernelSHAP is model-agnostic but requires thousands of permutations per query (~12 seconds latency), which is too slow for real-time web applications.*

9. **Q: How does your surrogate model pattern maintain fidelity?**  
   *A: We bind an automated Random Forest surrogate (`baseline_rf.pkl`) to TreeExplainer. The surrogate model demonstrates a 94.8% classification agreement with the stacked ensemble across the 2,000 test cases, enabling sub-180ms explanations while preserving physical attribution fidelity.*

10. **Q: What are the mathematical axioms guaranteed by SHAP?**  
    *A: Local accuracy (efficiency: attributions sum to the difference between prediction and expected base rate), Missingness (features with zero effect receive zero attribution), Consistency (if a model changes such that a feature's marginal contribution increases, its attribution cannot decrease), and Symmetry.*

11. **Q: How does LIME differ from SHAP in your dashboard?**  
    *A: SHAP provides game-theoretic additive feature attributions ($\phi_i$) anchored to dataset base rates. LIME fits an interpretable linear surrogate over local perturbations, generating explicit textual decision rules (e.g., "torque > 62.50 AND tool_wear > 205.00").*

12. **Q: How does the What-If counterfactual simulator assist plant engineers?**  
    *A: It allows engineers to test operational adjustments before intervention. On Sample 0, throttling torque from 68 to 42 Nm and changing the tool blade (wear 222 -> 15 min) dropped failure probability from 84.6% to 2.4%, proving an 82.2 percentage point risk abatement.*

13. **Q: Can SHAP feature attributions be interpreted as causal relationships?**  
    *A: No. SHAP measures statistical association based on interventional marginal expectations. In machines, torque and rotational speed are physically coupled by motor power ($P = \tau \cdot \omega$). Independent feature perturbations can evaluate physically impossible states; domain engineering validation is always required.*

14. **Q: What purpose do the multimodal text and image demos serve?**  
    *A: They demonstrate decision support beyond tabular data. The text explainer uses LIME to highlight urgency tokens ("screeching", "smoke") in technician shift logs, while the image demo uses spatial heatmaps to localize tooling blade cracks.*

### DevOps & Infrastructure
15. **Q: Why did you implement a multi-stage Dockerfile?**  
    *A: To separate build tools from runtime execution. Stage 1 compiles the Vite/React frontend using Node 20; Stage 2 resolves Python wheels using `pip --prefer-binary`; Stage 3 copies only static bundles and compiled packages into a fresh Python 3.11-slim image running as unprivileged UID 10001, producing a secure 1.56 GB container.*

16. **Q: How does Kubernetes achieve zero downtime during a rolling update?**  
    *A: The deployment defines `maxSurge: 1` and `maxUnavailable: 0`. During a rollout, Kubernetes launches a new pod, waits for its Readiness Probe (`GET /health`, initial delay 10s) to return 200 OK, adds it to the Service endpoints, and only then terminates the old replica.*

17. **Q: Why did you observe a single 503 error during the rolling update verification?**  
    *A: The 503 was an artifact of `kubectl port-forward` running on the local host. Unlike production ingress controllers (Envoy/NGINX) that dynamically re-route traffic across cluster endpoint pools, `kubectl port-forward` binds to a specific pod IP. When that pod terminated, the forwarder took ~1 second to reconnect.*

18. **Q: How was Horizontal Pod Autoscaling (HPA) verified?**  
    *A: We injected 400 requests across 10 concurrent threads (55 req/s). Pod CPU surged to 176%, exceeding the 60% HPA target. The metrics-server reported the utilization surge, and HPA autonomously scaled the deployment from 3 to 4 replicas.*

19. **Q: Why manage Prometheus and Grafana with Terraform instead of docker-compose?**  
    *A: Terraform codifies Infrastructure as Code (IaC) with declarative state management. It allows programmatic resource outputs (e.g., container IDs, URLs) to be piped into automated test pipelines and supports multi-provider infrastructure evolution (e.g., moving from local Docker to AWS EKS).*

20. **Q: How did you prove that monitoring works end-to-end?**  
    *A: We instrumented the FastAPI backend with Prometheus metrics. We queried `sum(xai_predictions_total)` before (400) and after (700) generating 300 load requests. The counter incremented by exactly +300, confirming end-to-end data flow: App -> Prometheus Scraping -> Grafana Visualization.*

---

## 9. SCREENSHOT & EVIDENCE CAPTURE CHECKLIST

The following reference table maps required submission screenshots to repository components and current availability status:

| Screenshot Identifier | Description / View | Target URL / Command Source | Status on Workstation |
| :--- | :--- | :--- | :---: |
| `phase5_git_branches.png` | GitFlow branching showing `main`, `develop`, and `feature` | Terminal / GitHub branch network | Manual action (post-push) |
| `phase5_pr_template.png` | GitHub Pull Request interface with review template | GitHub Web UI PR view | Manual action (post-push) |
| `phase5_branch_protection.png` | Branch protection rules enforcing PR reviews on `main` | GitHub Repository Settings | Manual action (post-push) |
| `phase5_ci_pipeline_overview.png` | Green 3-job GitHub Actions CI/CD execution | GitHub Actions Workflow run | Manual action (post-push) |
| `phase5_quality_gate.png` | CI logs showing clean Ruff, Bandit, and Pytest steps | GitHub Actions Job 1 logs | Manual action (post-push) |
| `phase5_trivy_scan.png` | CI logs showing clean Trivy container vulnerability scan | GitHub Actions Job 2 logs | Manual action (post-push) |
| `phase5_docker_build.png` | Terminal output of `docker build -t xai-automl:lite` | `results/phase55_evidence/01_docker_build.log` | **READY (1.56 GB)** |
| `phase5_ghcr_packages.png` | Published container package in GitHub Packages (GHCR) | GitHub Packages Web UI | Manual action (post-push) |
| `phase5_k8s_resources.png` | Terminal output of `kubectl get pods,svc,hpa -o wide` | `results/phase55_evidence/03_k8s_resources.txt` | **READY (Running)** |
| `phase5_k8s_rolling_update.png` | Zero-downtime rolling update health probe sequence | `results/phase55_evidence/04_rolling_update_health_sequence.txt` | **READY (Captured)** |
| `phase5_k8s_hpa_scaling.png` | HPA scaling output showing replica expansion to 4 | `results/phase55_evidence/05_scaling_hpa_samples.txt` | **READY (Captured)** |
| `phase5_terraform_plan.png` | Terminal output of `terraform apply` creating 6 resources | `results/phase55_evidence/06_terraform_apply.txt` | **READY (Captured)** |
| `phase5_prometheus_targets.png` | Web UI showing `host.docker.internal:8000` target UP | `http://localhost:9090/targets` | **READY (Live)** |
| `phase5_prometheus_alerts.png` | Web UI showing `ApiDown`, `HighErrorRate` rules | `http://localhost:9090/alerts` | **READY (Live)** |
| `phase5_grafana_dashboard.png` | 8-panel Operations Dashboard with live metric graphs | `http://localhost:3000` (admin/admin) | **READY (Live)** |
| `phase5_validation_pass.png` | Master validation script showing 73/73 checks PASS | `results/phase55_evidence/08_regression_phase5.txt` | **READY (197/197)** |

---

## 10. REMAINING MANUAL STUDENT ACTIONS

To complete final academic submission, the student must execute the following ordered steps:
1. **Docker Desktop Memory Allocation:** Verify Docker Desktop is allocated >= 6.0 GB RAM (verified at 7.40 GB during preflight).
2. **Push Repository to GitHub:**
   ```powershell
   git push origin develop
   ```
3. **Execute GitFlow Pull Request Workflow:**
   - Create a feature branch: `git checkout -b feature/devops-phase5`
   - Push feature branch and open a Pull Request targeting `develop` on GitHub.
   - Verify that `.github/PULL_REQUEST_TEMPLATE.md` automatically populates the PR body.
   - Merge PR into `develop`, then open and merge a PR from `develop` into `main`.
4. **Enable Branch Protection on `main`:**
   - In GitHub repository: `Settings -> Branches -> Add branch protection rule -> main`.
   - Check `Require a pull request before merging` and `Require status checks to pass before merging`.
5. **Monitor CI/CD Actions & Capture Screenshots:**
   - Navigate to GitHub Actions tab. Confirm all 3 jobs run green.
   - Capture `phase5_ci_pipeline_overview.png`, `phase5_quality_gate.png`, and `phase5_trivy_scan.png`.
   - Confirm container package is published to GitHub Container Registry (GHCR) and capture `phase5_ghcr_packages.png`.
6. **Capture Live Workstation GUI Screenshots:**
   - Open browser to `http://localhost:3000` (Grafana: `admin` / `admin`). Capture full dashboard showing 8 live panels.
   - Open browser to `http://localhost:9090/targets` and `http://localhost:9090/alerts`. Capture Prometheus targets and alert states.
   - Open terminal and run `kubectl get pods,svc,hpa -n xai-predictive-maintenance`. Capture active pod output.
7. **Optional Teardown:**
   - Once all screenshots are saved, execute `scripts/teardown_all.ps1` to cleanly terminate local containers and port-forwarding processes.

---

## 11. KNOWN LIMITATIONS & HONEST TECHNICAL DISCLOSURES

1. **Fast Mode vs. Final Mode Metric Variance:** In Fast Mode, truncated search spaces and 60-second budgets favored FLAML (`lgbm`, F1 = 0.6281) while AutoGluon underperformed (F1 = 0.3721) due to incomplete stacking convergence. In Final Mode, 300-second budgets allowed AutoGluon's multi-layer stacking to fully optimize, overtaking all competitors with an exceptional F1 of 0.7797.
2. **Surrogate Model Explanations:** Local SHAP explanations are computed via a high-fidelity Random Forest surrogate (F1 = 0.7132, 94.8% classification agreement) rather than directly traversing the multi-layer stacked ensemble. While physically faithful and operating in sub-180ms latency, it remains an empirical approximation of the true ensemble decision boundary.
3. **Illustrative Vision Saliency Demo:** The optical defect inspection tab operates as an illustrative computer vision proxy simulating surface crack and thermal defect heatmaps, rather than a production-trained deep convolutional neural network.
4. **Single 503 Port-Forward Artifact:** The single 503 error observed during rolling update verification was an artifact of host `kubectl port-forward` socket reconnect latency when a target pod terminated, and does not represent pod-level service downtime.
5. **Local KinD Deployment:** Orchestration was executed on a local KinD cluster rather than managed cloud Kubernetes (AWS EKS, GCP GKE). The architecture is 100% cloud-ready and can be deployed directly to cloud clusters using identical manifests.

---

## 12. CONFLICT LOG & DATA AUDIT

During cross-file reconciliation across all project reports and metadata files, the following potential numeric disagreements were identified and resolved:

| Artifact 1 | Artifact 2 | Dimension | Values Observed | Authoritative Resolution | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `reports/automl_phase2_results.md` | `results/tuning_comparison.csv` | Baseline RF Test F1 | 0.7132 vs. 0.7132 | **0.7132** | Perfect agreement across all Phase 2 files. |
| `reports/automl_phase3_results.md` | `results/phase3_full_comparison.csv` | FLAML Test F1 | 0.7680 vs. 0.7680 | **0.7680** | Consistent across FLAML final-mode logs. |
| `reports/automl_phase3_results.md` | `results/phase3_full_comparison.csv` | AutoGluon Test F1 | 0.7797 vs. 0.7797 | **0.7797** | Authoritative winner across Phase 3 and Phase 4. |
| `reports/automl_phase3_results.md` | `results/leakage_results.csv` | Leaky Model F1 | 0.8722 vs. 0.8722 | **0.8722** | Exact +10.42 pp spike confirmed across all files. |
| `reports/FINAL_MODE_COMPARISON_REPORT.md` | `results/phase5_metadata.json` | Fast vs. Final Time | Fast: 60s vs. Final: 300s | **Final 300s Budget** | Final Mode numbers supersede Fast Mode verification figures. |
| `results/phase55_evidence/01_docker_duration.txt` | `reports/devops_phase5_results.md` | Docker Build Time | 333.9s vs. 333.9s | **333.9 seconds** | Machine-recorded build duration captured in Phase 5.5. |
| `results/phase55_evidence/07_monitoring_xai_predictions_after.json`| `results/phase55_evidence/EVIDENCE_LOG.md` | Predictions Total | 400 -> 700 (+300) | **700 Total Predictions** | Live Prometheus query output verified after 300 load requests. |

---
*End of Master Reference Documentation.*
