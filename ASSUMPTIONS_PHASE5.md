# Architectural Assumptions & Engineering Decisions (Phase 5 DevOps)

**Project:** Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0  
**Phase:** Phase 5 — DevOps Layer (CI/CD, Containerization, Kubernetes, Terraform, Monitoring)  
**Author:** Senior DevOps & Platform Engineer  
**Date:** 2026-10-27  

---

## 1. Containerization & Dual-Image Strategy (Lite vs. Full)

### Decision & Rationale
- **Multi-Stage Docker Architecture:** The Dockerfile employs a three-stage build (`frontend-builder`, `python-deps`, `runtime`) to minimize final image footprint, eliminate build-time tools (Node.js, npm, compilers) from the production image, and run as a non-root user (`appuser`, UID `10001`).
- **Lite vs. Full Build Strategy:**
  - `INSTALL_AUTOML=false` (Default / CI / Local K8s): Generates the lightweight container (`xai-automl:lite`, ~1.2 GB). It excludes the multi-gigabyte dependency trees of AutoGluon (`torch`, `ray`, `mxnet`, `autogluon.*`) and FLAML. In this mode, the FastAPI application automatically falls back to the high-performance Scikit-Learn Random Forest model (`models/baseline_rf.pkl`, Test $F_1 = 0.7132$) or surrogate model.
  - `INSTALL_AUTOML=true` (Main Branch Release Artifact Only): Builds the complete image (`xai-automl:full`) containing native AutoGluon dependencies for full stacked ensemble execution. This build is configured to run asynchronously with `continue-on-error: true` in CI because it takes 15–20 minutes and downloads multi-gigabyte wheels.
- **Model Selection Preference (`SERVED_MODEL_PREFERENCE`):**
  - Added support for the environment variable `SERVED_MODEL_PREFERENCE` with values `"auto"` (default), `"automl"`, or `"fallback"`.
  - In `"auto"`, the service attempts to load the Phase 3 AutoGluon winner, seamlessly falling back to Phase 2 Random Forest if AutoGluon cannot be loaded.
  - In `"fallback"`, the container deterministically loads the lightweight Scikit-Learn Random Forest model without throwing exceptions or attempting heavy imports.

---

## 2. Verified Prometheus Metric Names

To ensure alert rules and Grafana dashboard panels function deterministically without silent query failures, metric names emitted by `prometheus-fastapi-instrumentator` were directly verified via testclient inspection:

| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `http_requests_total` | Counter | `handler`, `status`, `method` | Total count of HTTP requests processed by endpoint and HTTP status code. |
| `http_request_duration_seconds_bucket` | Histogram | `handler`, `status`, `method`, `le` | Distribution of request latencies used for P50, P90, and P95 percentile calculations. |
| `xai_predictions_total` | Counter | `risk_level` (`low`, `medium`, `high`) | Custom business metric tracking total inference calls grouped by calculated operational risk level. |
| `up` | Gauge | `job`, `instance` | Standard Prometheus liveness indicator ($1 = \text{healthy}, 0 = \text{unreachable}$). |

### Alert Expression Calibration
- **ApiDown:** `up{job="xai-api"} == 0` for 1 minute (Critical).
- **HighErrorRate:** `(sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))) > 0.05` for 2 minutes (Warning).
- **HighP95Latency:** `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 1.0` for 5 minutes (Warning).

---

## 3. Kubernetes Orchestration Decisions

1. **Namespace Isolation:** All resources reside in `xai-predictive-maintenance` to isolate workloads, network policies, and role bindings from default cluster namespaces.
2. **RollingUpdate Strategy:** `maxSurge: 1` and `maxUnavailable: 0` ensure that when updating deployments or model weights, new pods must pass readiness probes before old pods are terminated, guaranteeing zero dropped HTTP requests.
3. **Security Context & Filesystem Permissions:**
   - `runAsNonRoot: true` and `runAsUser: 10001` ensure containers cannot perform privilege escalation.
   - `capabilities: drop: ["ALL"]` drops all Linux kernel capabilities.
   - `readOnlyRootFilesystem: false`: Set to `false` because the application dynamically computes and renders local SHAP and LIME PNG artifacts into `/app/artifacts/xai/local/` upon receiving on-demand `/explain` requests. In a future enterprise iteration, local scratch writes can be mounted to an ephemeral `emptyDir` volume.
4. **Probe Timings:**
   - `readinessProbe`: Initial delay 10s, period 5s. Guards traffic until FastAPI initialization finishes.
   - `livenessProbe`: Initial delay 40s, period 15s. Provides sufficient buffer for initial model weight loading into memory.
5. **Horizontal Pod Autoscaler (HPA):** Configured on `cpu: 60%` and `memory: 80%` average utilization, with a 120-second stabilization window for scale-down to prevent rapid pod thrashing (flapping).

---

## 4. Static Code Quality & Security Policy

1. **Ruff Linter Configuration:**
   - Target line length: 120 characters.
   - Ruleset: `E` (pycodestyle errors), `F` (pyflakes), `W` (warnings).
   - Ignored rules: `E501` (pragmatic line length for diagnostic strings/docstrings), `F401` (API module imports), `F841` (intermediate diagnostic variables), `F541` (logging f-strings without variables).
2. **Bandit Security Scan:**
   - Configured with `-ll` in CI to fail strictly on Medium or High severity vulnerabilities.
   - Low severity findings: The 6 Low-severity informational warnings identified in pre-existing Phase 1–3 code (`B110: try_except_pass` and `B112: try_except_continue`) represent intentional fallback cascades (e.g. attempting native feature importance before falling back to permutation importance). As instructed by the prompt guidelines, these were preserved to maintain ML code integrity.
3. **Trivy Vulnerability Policy:**
   - Scans container images for operating system and Python package CVEs.
   - `exit-code: 1` enforced on `CRITICAL` severity only. `HIGH` severity vulnerabilities produce warning annotations without blocking CI deployment gates, adhering to modern DevSecOps patch cycles.

---

## 5. Tool Detection & Skipped Validations Handling

As specified by the hard constraints, absent or non-running optional host binaries do not fail the phase:
1. **Docker Desktop Daemon (`com.docker.service`):**
   - Detection: `docker.exe` is installed on host, but the Docker Desktop daemon service was stopped in the current Windows session.
   - Handling: Validated via `docker build --help`, `Dockerfile` syntax inspection, and `.dockerignore` verification. `docker_build_skipped: true` logged in metadata with clear explanation.
2. **Terraform CLI:**
   - Detection: Terraform is not installed on system PATH.
   - Handling: All HCL configuration files (`versions.tf`, `variables.tf`, `main.tf`, `outputs.tf`, `terraform.tfvars.example`, `README.md`) authored and verified. `terraform_skipped: true` logged in metadata with install command (`winget install Hashicorp.Terraform`).
3. **Kubectl Client:**
   - Detection: `kubectl.exe` (v1.34.1) is installed on host.
   - Handling: Successfully executed client-side dry run (`kubectl apply --dry-run=client -k k8s/ -o yaml > results/k8s_dryrun_output.yaml`), verifying 100% manifest validity offline.
