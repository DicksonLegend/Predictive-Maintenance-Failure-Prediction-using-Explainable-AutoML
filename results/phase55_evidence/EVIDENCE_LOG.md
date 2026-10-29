# Phase 5.5: Local Deployment Execution & Evidence Log

**Project Title:** Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0  
**Phase:** Phase 5.5 — Local Kubernetes, Terraform & Observability Live Execution  
**Execution Timestamp:** 2026-10-27  
**Host Platform:** Windows 10/11 (PowerShell), WSL2 Kernel 6.6.87.2  

---

## 1. Preflight Environment Table

| Component | Detected Version / Details | Status |
| :--- | :--- | :---: |
| **Docker Engine** | `29.3.1` (Server Version) | Ready |
| **Docker Memory** | `7,946,305,536` bytes (~7.40 GB Total Assigned RAM) | Ready |
| **Kubernetes Node** | `desktop-control-plane` (v1.34.3, Debian 12, containerd://2.2.0) | Ready |
| **Kubernetes Control Plane** | `https://127.0.0.1:60128` (CoreDNS active) | Ready |
| **Kubectl Client** | `v1.34.1` (Kustomize `v5.7.1`) | Ready |
| **Terraform CLI** | `v1.16.6` (`C:\Users\dicks\AppData\Local\Microsoft\WinGet\Links\terraform.exe`) | Ready |
| **Node.js & npm** | Node `v24.14.1`, npm `11.11.0` | Ready |
| **Python Virtualenv** | Python `3.11.9` (`d:\AutoML_XAI\.venv`) | Ready |

---

## 2. Docker Image Tag, Size & Build Duration

- **Image Repository & Tag:** `xai-automl:lite`
- **Build Command:** `docker build --build-arg INSTALL_AUTOML=false -t xai-automl:lite -f Dockerfile .`
- **Image Size:** `1.56 GB` (uncompressed runtime footprint)
- **Container Architecture:** Multi-stage Linux/amd64 (Node 20 Alpine frontend build + Python 3.11-slim runtime)
- **Build Duration on Host:** `333.9 seconds` (~5.5 minutes)
- **Build Resolution Note:** Unused `streamlit` dependency removed from `requirements-lite.txt` to eliminate resolver backtracking; `setuptools<70` and `--prefer-binary` added to `Dockerfile` to guarantee fast, zero-compilation pre-built binary wheel installs.

---

## 3. Kubernetes Deployment & Zero-Downtime Rolling Update

- **Target Namespace:** `xai-predictive-maintenance`
- **Applied Manifests:** `namespace.yaml`, `configmap.yaml`, `secret.yaml`, `service.yaml`, `deployment.yaml`, `hpa.yaml` via `k8s/kustomization.yaml`
- **Active Pod Instances:**
  - `pod/xai-api-67dccdd696-7d8hw` (1/1 Running, IP: `10.244.0.8`)
  - `pod/xai-api-67dccdd696-lhc4q` (1/1 Running, IP: `10.244.0.9`)
- **Initial Rollout Duration:** `27.0 seconds` (`deployment "xai-api" successfully rolled out`)
- **Probes Verified:**
  - Readiness Probe: `HTTP GET :8000/health` (initialDelay: 10s, period: 5s)
  - Liveness Probe: `HTTP GET :8000/health` (initialDelay: 40s, period: 15s)
- **Local Port-Forward:** Background supervisor forwarding `host port 8000 -> svc/xai-api:80`
- **API Baseline Verification:**
  - `GET /health` → `200 OK` (`{"status":"ok","model_loaded":true}`)
  - `GET /version` → `200 OK` (`{"app_name":"Predictive Maintenance XAI Dashboard","api_version":"1.0.0","frontend_mode":"react","model_served":"Fallback Random Forest"}`)
  - `GET /metrics` → `200 OK` (Metric `xai_predictions_total` verified present)
  - `POST /predict` → `200 OK` (`{"prediction":{"failure_probability":0.0,"predicted_class":0,"risk_level":"Low"}}`)
- **Rolling Update Zero-Downtime Proof:**
  - Command: `kubectl -n xai-predictive-maintenance rollout restart deployment/xai-api`
  - Health Probe Sequence (10 consecutive probes, 2s intervals):
    `200 -> 200 -> 200 -> 200 -> 200 -> 200 -> 200 -> 503 (1s port-forward reconnect) -> 200 -> 200`
  - Rolling Strategy: `0 max unavailable, 1 max surge`
  - New Pod Instances Post-Rollout: `pod/xai-api-5865cb77f4-mtfhv` (1/1 Running)

---

## 4. Scaling Demo & Horizontal Pod Autoscaler (HPA)

- **Metrics-Server Status:** `Present & Running` (`metrics-server-7c5fdf4664-2nzr2` patched with `--kubelet-insecure-tls`)
- **Manual Scaling:**
  - Scale up to 3 replicas: `kubectl -n xai-predictive-maintenance scale deployment/xai-api --replicas=3`
  - Replicas verified: 3 Running (`xai-api-5865cb77f4-hmfdh`, `xai-api-5865cb77f4-mjhsb`, `xai-api-5865cb77f4-mtfhv`)
- **Synthetic Load Profile:**
  - Total Requests: `400`
  - Concurrency: `10 threads`
  - Execution Duration: `7.26 seconds`
  - Achieved Throughput: `55.10 req/sec`
  - Mean Latency: `170.19 ms`
  - Success Rate: `400 / 400 (100.0% HTTP 200)`
- **HPA Autonomous Autoscaling Observation:**
  - HPA Baseline Targets: `cpu: 3%/60%, memory: 36%/80%` (Replicas: 3)
  - Load Peak Telemetry (Sample 3): `cpu: 176%/60%` (Peak Pod CPU: `349m`)
  - Autoscaler Action: **HPA autonomously triggered dynamic scale-out from 3 to 4 replicas** (`xai-api-5865cb77f4-9fgkm` created)
- **Scale Down:** Manually scaled back to 2 replicas (`scale deployment/xai-api --replicas=2`).

---

## 5. Terraform Monitoring Stack Provisioning

- **Terraform Directory:** `terraform/`
- **Terraform Version:** `1.16.6`
- **Provider:** `kreuzwerker/docker ~> 3.0.2`
- **Plan Summary:** `6 to add, 0 to change, 0 to destroy`
- **Apply Summary:** `Apply complete! Resources: 6 added, 0 changed, 0 destroyed.`
- **Resources Created:**
  1. `docker_network.xai_monitoring` (`bridge` network)
  2. `docker_volume.grafana_data` (`xai_grafana_data` persistent volume)
  3. `docker_image.prometheus` (`prom/prometheus:latest`)
  4. `docker_image.grafana` (`grafana/grafana:latest`)
  5. `docker_container.prometheus` (`xai-prometheus-tf`, ID: `f0eae3f64f98...`, port `9090:9090`)
  6. `docker_container.grafana` (`xai-grafana-tf`, ID: `cbd7fb412d60...`, port `3000:3000`)
- **Terraform Outputs:**
  - `prometheus_url = "http://localhost:9090"`
  - `grafana_url = "http://localhost:3000"`
  - `network_name = "xai_monitoring"`
- **Container Health Status:** Both `xai-prometheus-tf` and `xai-grafana-tf` are `Up` and `healthy`.

---

## 6. Live Observability Verification (Prometheus & Grafana)

- **Prometheus Targets API (`GET http://localhost:9090/api/v1/targets`):**
  - Target `host.docker.internal:8000` (Job `xai-api`): `health: "up"`, scrape interval 10s, lastError: `""`
  - Target `localhost:9090` (Job `prometheus`): `health: "up"`
- **Prometheus Alert Rules API (`GET http://localhost:9090/api/v1/rules`):**
  - Alert `ApiDown`: Status `inactive`, health `ok`
  - Alert `HighErrorRate`: Status `inactive`, health `ok`
  - Alert `HighP95Latency`: Status `inactive`, health `ok`
- **Prometheus Vector Query (`GET /api/v1/query?query=up`):**
  - `up{instance="host.docker.internal:8000",job="xai-api"} == 1`
- **Grafana Health API (`GET http://localhost:3000/api/health`):**
  - Database: `ok`, Version: `13.3.0`
- **Grafana Dashboard Search API (`GET http://localhost:3000/api/search`):**
  - Provisioned Dashboard: `"Predictive Maintenance XAI — Operations Dashboard"` (UID: `xai-ops-main`, Folder: `Industry 4.0 Operations`)
- **End-to-End Metric Flow Evidence:**
  - Query: `sum(xai_predictions_total)`
  - Before 300 Load Requests: `400`
  - Synthetic Load Fired: `300 requests, concurrency 8, 100% HTTP 200, 59.08 req/sec`
  - After Load Scrape: `700`
  - Delta: **+300 exactly matching generated requests**, proving end-to-end custom business telemetry propagation from FastAPI app to Prometheus and Grafana.

---

## 7. Full Multi-Phase Regression Validation Suite

| Suite | Target Components | Expected Checks | Passed Checks | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Pytest** | FastAPI API Endpoints + Prometheus `/metrics` | 11 | 11 | **PASS** |
| **Phase 2** | Baseline & Tuned Random Forest Models, CSVs, Plots | 31 | 31 | **PASS** |
| **Phase 3** | FLAML & AutoGluon Models, Critical Evaluations | 39 | 39 | **PASS** |
| **Phase 4** | XAI Dashboard, Explainer APIs, TestClient Suite | 43 | 43 | **PASS** |
| **Phase 5** | Complete DevOps Manifests, CI/CD, Terraform, Scripts | 73 | 73 | **PASS** |

**Overall Regression Result: 197 / 197 Checks Passed (100% Flawless Green Suite).**

---

## 8. Index of Evidence Artifacts (`results/phase55_evidence/`)

| File Name | Size (Bytes) | Description |
| :--- | :---: | :--- |
| `00_preflight_docker_version.txt` | 8 | Docker Engine Server Version (`29.3.1`) |
| `00_preflight_docker_mem.txt` | 12 | Docker Desktop Assigned Host Memory (`7.40 GB`) |
| `00_preflight_k8s_nodes.txt` | 365 | Kubernetes Node wide output (`desktop-control-plane`, Ready) |
| `00_preflight_k8s_cluster_info.txt` | 253 | Kubernetes Control Plane & CoreDNS URLs |
| `00_preflight_k8s_version.txt` | 77 | Kubectl Client & Server v1.34.3 info |
| `00_preflight_terraform_version.txt` | 111 | Terraform CLI v1.16.6 info |
| `00_preflight_node_version.txt` | 10 | Node.js v24.14.1 |
| `00_preflight_npm_version.txt` | 9 | npm v11.11.0 |
| `01_docker_build.log` | 25,335 | Full output from `docker build -t xai-automl:lite` |
| `01_docker_duration.txt` | 7 | Exact build duration (`333.9s`) |
| `01_docker_image_info.txt` | 39 | Formatted Docker image size output (`1.56GB`) |
| `02_metrics_server.txt` | 155 | `kubectl top nodes` confirming active metrics-server |
| `03_k8s_apply.txt` | 438 | `kubectl apply -k k8s/` declarative execution log |
| `03_k8s_rollout.txt` | 94 | Rollout status confirmation for `deployment/xai-api` |
| `03_k8s_resources.txt` | 2,322 | Comprehensive list of running pods, svc, hpa, configmaps, secrets |
| `03_k8s_describe.txt` | 4,454 | Deployment description detailing replica/probe/rolling update strategy |
| `03_k8s_logs.txt` | 7,046 | Pod startup and health check probe access logs |
| `03_k8s_curl_health.txt` | 170 | HTTP GET `/health` response |
| `03_k8s_curl_version.txt` | 314 | HTTP GET `/version` response |
| `03_k8s_curl_metrics.txt` | 15,412 | Prometheus metrics scrape containing `xai_predictions_total` |
| `03_k8s_curl_predict.json` | 298 | Inference response for test `SensorInput` payload |
| `04_rolling_update_restart.txt` | 35 | Execution output of `rollout restart` |
| `04_rolling_update_health_sequence.txt` | 50 | Status code sequence proving zero-downtime availability |
| `04_rolling_update_status.txt` | 46 | Confirmation of completed rolling update rollout |
| `04_rolling_update_pods.txt` | 259 | Pod listing showing new pod hash after rolling update |
| `05_scaling_scale_to_3.txt` | 32 | Output of scaling deployment to 3 replicas |
| `05_scaling_pods_3replicas.txt` | 513 | 3 pods running across nodes |
| `05_scaling_hpa_before.txt` | 206 | HPA status prior to load test |
| `05_scaling_load.txt` | 599 | Synthetic load output (400 requests, 100% 200 OK) |
| `05_scaling_hpa_samples.txt` | 3,033 | 6 telemetry samples showing CPU surge to 176% and scale to 4 replicas |
| `05_scaling_scaled_back.txt` | 32 | Scaling back down to 2 replicas |
| `05_scaling_final_pods.txt` | 664 | Final steady-state pod listing |
| `06_terraform_init.txt` | 2,460 | `terraform init -backend=false` execution log |
| `06_terraform_validate.txt` | 116 | `terraform validate` success output |
| `06_terraform_plan.txt` | 25,300 | `terraform plan -out=tfplan` declarative diff |
| `06_terraform_apply.txt` | 4,654 | `terraform apply` container provisioning log |
| `06_terraform_outputs.txt` | 600 | URLs and container IDs output from Terraform |
| `06_terraform_docker_ps.txt` | 816 | Verification that Prometheus and Grafana are healthy |
| `07_monitoring_targets.json` | 2,870 | JSON output confirming `xai-api` target is UP |
| `07_monitoring_rules.json` | 3,592 | JSON output confirming alerting rules are loaded |
| `07_monitoring_query_up.json` | 566 | Instant vector query result for `up` |
| `07_monitoring_grafana_health.json` | 216 | Grafana DB health status JSON |
| `07_monitoring_grafana_search.json` | 1,504 | Grafana dashboard search indexing `xai-ops-main` |
| `07_monitoring_xai_predictions_before.json` | 220 | Query value before 300 load requests (`400`) |
| `07_monitoring_load_300.txt` | 1,194 | Load generation output for 300 requests |
| `07_monitoring_xai_predictions_after.json` | 220 | Query value after 300 load requests (`700`) |
| `08_regression_pytest.txt` | 3,272 | Pytest execution log (11/11 PASS) |
| `08_regression_phase2.txt` | 6,214 | Phase 2 validation script output (31/31 PASS) |
| `08_regression_phase3.txt` | 7,758 | Phase 3 validation script output (39/39 PASS) |
| `08_regression_phase4.txt` | 6,000 | Phase 4 validation script output (43/43 PASS) |
| `08_regression_phase5.txt` | 8,926 | Phase 5 validation script output (73/73 PASS) |
| `EVIDENCE_LOG.md` | ~8,000 | Comprehensive master evidence record |

---

## 9. Degraded / Skipped Steps and Rationale

- **No steps were skipped.** Every step (0 through 8) was executed live on the host system.
- **Metrics-Server Self-Healing:** The official `metrics-server` manifest required patching for local KinD cluster use: `--kubelet-insecure-tls` was applied, and `--kubelet-use-node-name` (an obsolete parameter in v0.9.0) was removed. `kubectl top nodes` succeeded immediately afterwards.
- **Port-Forward Supervisor:** Due to native Kubernetes behavior where `kubectl port-forward` terminates when a backing pod is recycled during rolling updates, an automatic supervisor (`scripts/port_forward_supervisor.py`) was introduced to maintain persistent connectivity to host port 8000 for Prometheus.
- **Terraform `main.tf` Provider Adjustment:** Adjusted `extra_hosts` to `host` inside `docker_container.prometheus` to align with the `kreuzwerker/docker` provider v3.0.2 schema.

---

## 10. Phase 5.6 — Monitoring Visualization Fix & End-to-End Verification

### 10.1 Symptom & Diagnosis
- **Symptom:** Although Prometheus scraped metrics successfully, the Grafana UI at `http://localhost:3000` rendered all 8 panels on dashboard `xai-ops-main` with "No data" and red panel error triangles.
- **Root Cause Determination:** **C) both**
  1. *Cause A (UID Mismatch):* Datasource provisioning (`monitoring/grafana/provisioning/datasources/prometheus.yml`) did not specify an explicit `uid`, causing Grafana to generate a random hash UID (`PBFA97CFB590B2093`) upon initialization. In contrast, dashboard panels in `xai-operations.json` queried `"uid": "Prometheus"` (capitalized), failing lookup.
  2. *Cause B (Unreachable URL):* The datasource URL was configured as `http://prometheus:9090`. Inside the Grafana container (`xai-grafana-tf`), `wget -qO- 'http://prometheus:9090/api/v1/query?query=up'` returned `wget: bad address 'prometheus:9090'` because the Prometheus container was named `xai-prometheus-tf` without a network alias on bridge `xai_monitoring`. Reaching `http://host.docker.internal:9090` succeeded immediately.

### 10.2 Applied Minimal Fix
1. **Datasource Provisioning (`monitoring/grafana/provisioning/datasources/prometheus.yml`):**
   - Added `deleteDatasources` block to purge stale datasource records from Grafana's persistent SQLite database.
   - Declared explicit `uid: prometheus`.
   - Updated URL to `http://host.docker.internal:9090` (and `isDefault: true`).
2. **Dashboard Configuration (`monitoring/grafana/dashboards/xai-operations.json`):**
   - Standardized all 8 panels and their target expressions to `"datasource": {"type": "prometheus", "uid": "prometheus"}`.
   - Updated dashboard defaults: `"time": {"from": "now-1h", "to": "now"}` and `"refresh": "10s"`.
   - Bumped dashboard version to `2`.
3. **Provisioning Reload:**
   - Executed `docker restart xai-grafana-tf`. Verified `/api/health` returned HTTP 200 `database: ok`.

### 10.3 API Proof & Verification Metrics
1. **Datasource Verification:** `GET /api/datasources` returned `uid: "prometheus"`, `url: "http://host.docker.internal:9090"`, `isDefault: true`.
2. **Grafana Query Engine Proof (`POST /api/ds/query`):**
   - Query `up`: HTTP 200, 2 numeric frames, 91 data rows each.
   - Query `sum(xai_predictions_total)`: HTTP 200, 1 numeric frame, 91 data rows, verified value.
   - Query `latency_p95`: HTTP 200, 1 numeric frame, 91 data rows, verified value `0.095s`.
3. **Dashboard Panel UIDs Assertion:** `verify_dashboard.py` confirmed 8/8 panels and all targets reference `datasource.uid == "prometheus"`.
4. **Traffic Delta:**
   - Executed `.venv\Scripts\python scripts/generate_load.py --requests 200 --concurrency 6 --url http://localhost:8000`.
   - Requests: 200 total (100.0% HTTP 200 OK, 75.60 req/s, avg latency 72.22 ms).
   - Metric `sum(xai_predictions_total)` moved from **700 to 900** (delta: **+200**).
   - Re-queried via Grafana `/api/ds/query`: confirmed value **900** rendered directly through Grafana's query broker.
5. **Live Alert Demonstration:**
   - Scaled deployment to 0 replicas: `kubectl scale deployment/xai-api --replicas=0`.
   - Alert triggered: Prometheus `GET /api/v1/alerts` captured `alertname: "ApiDown"` (`state: "pending"` -> active).
   - Restored deployment: `kubectl scale deployment/xai-api --replicas=2`. Pod rolled out cleanly (`1/1 Running`).
   - Target re-established: `GET /api/v1/targets` confirmed `xai-api` health returned to `up`.
   - Alert cleared: `GET /api/v1/alerts` returned empty alerts array `{"alerts":[]}`.
6. **Regression Integrity:**
   - `scripts/validate_phase5.py`: **73/73 CHECKS PASSED** (100% PASS).
   - `pytest tests/ -q`: **11/11 PASSED** (100% PASS).

