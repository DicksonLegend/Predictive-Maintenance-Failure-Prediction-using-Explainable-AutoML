MASTER PROMPT — PHASE 5: DEVOPS LAYER (CI/CD, CONTAINERIZATION, KUBERNETES, TERRAFORM, MONITORING)

You are a senior DevOps / Platform engineer. You are continuing the locked project:
"Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0"

Phases 1–4 are COMPLETE and validated in Final Mode (113/113 checks passing):
- Phase 1: data pipeline (AI4I 2020, 10,000 rows, target=machine_failure)
- Phase 2: manual baseline + Grid/Random/Optuna tuning
- Phase 3: FLAML vs AutoGluon (300s budget). WINNER: AutoGluon WeightedEnsemble_L2, Test F1=0.7797, stored at models/autogluon_predictor/ (a DIRECTORY). Fallback hierarchy in app/api/model_service.py falls back to scikit-learn Random Forest (models/baseline_rf.pkl, F1=0.7132) when AutoGluon is unavailable.
- Phase 4: FastAPI backend (app/api/) + compiled React SPA (app/frontend/dist/) + XAI artifacts (artifacts/) + 10/10 API tests passing (tests/test_api.py)

PHASE 5 OBJECTIVE:
Wrap the existing application in a production-grade DevOps layer satisfying the DevOps rubric:
version control & collaboration, CI/CD automation, containerization & orchestration,
Infrastructure-as-Code, monitoring/logging/alerting, secrets management, and security scanning.

Do not ask questions. Make reasonable engineering decisions and document every one in ASSUMPTIONS_PHASE5.md.
ALL CHANGES MUST BE ADDITIVE. You must NOT break any existing functionality.
After every modification to app/ or src/, re-run:
  .venv\Scripts\python scripts/validate_phase4.py
and ensure it still reports 43/43 PASS. If it fails, fix your change immediately.

The user's machine is WINDOWS with PowerShell. Docker Desktop is installed.
kubectl MAY be available (Docker Desktop Kubernetes). Terraform MAY be installed.
Node.js/npm ARE installed (used in Phase 4). GitHub CLI may or may not be present.
Design every script so it degrades gracefully when an optional tool is absent.

==================================================
HARD CONSTRAINTS — READ FIRST
==================================================
1. NEVER run: git push, gh repo create, kubectl apply (real), terraform apply, or docker push.
   These require the user's credentials/cluster and are documented as MANUAL STEPS instead.
   You MAY run: git init/add/commit locally, docker build (LITE variant only),
   docker run (ephemeral, then remove), terraform init/validate/plan,
   kubectl apply --dry-run=client, pytest, and all validation scripts.
2. DO NOT build the Docker image with INSTALL_AUTOML=true — the AutoGluon dependency tree
   is multi-GB and will exceed any sandbox timeout. Build ONLY with INSTALL_AUTOML=false.
3. DO NOT modify data/, results/, plots/, models/, artifacts/ from previous phases
   (except appending new files where explicitly instructed).
4. DO NOT rewrite app/api/main.py. Make SURGICAL additive edits only (metrics instrumentation).
5. Keep the total new-file footprint reasonable. Prefer quality over quantity.
6. If an optional binary (terraform/kubectl/docker) is missing, SKIP that validation step,
   log it in results/phase5_metadata.json under "skipped_validations", and CONTINUE.
   Never fail the phase because an optional tool is absent.

==================================================
DELIVERABLE 1 — CODE QUALITY & SECURITY TOOLING
==================================================
Create:
- requirements-dev.txt containing (at minimum): ruff, bandit, pytest, httpx,
  prometheus-fastapi-instrumentator
- ruff.toml with a pragmatic ruleset (line-length=120, ignore E501 in generated files,
  exclude .venv, artifacts, models, data)
- .bandit config or inline flags excluding tests/ and .venv/
- Add these packages to requirements.txt ONLY if absent (do not duplicate, do not remove anything).

Fix any ruff/bandit violations your NEW code introduces. For pre-existing violations in
Phase 1–4 code: fix trivial ones automatically; for anything non-trivial, add a targeted
`# noqa` with a one-line justification comment. NEVER refactor working ML code for style.

Verify by running:
  .venv\Scripts\python -m ruff check app src scripts tests
  .venv\Scripts\python -m bandit -r app src -x tests,.venv
  .venv\Scripts\python -m pytest tests/ -q
All three must exit cleanly (bandit may report LOW-severity informational findings; that is acceptable —
document them, do not chase zero at the cost of breaking code).

==================================================
DELIVERABLE 2 — PROMETHEUS INSTRUMENTATION (surgical edit)
==================================================
Install prometheus-fastapi-instrumentator into .venv.

Edit app/api/main.py ADDITIVELY:
- Import Instrumentator from prometheus_fastapi_instrumentator
- After app creation and BEFORE static mounts, add:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)
- Mount order matters: /metrics must be registered before the catch-all StaticFiles mount at "/".
  Verify the existing static mount does not shadow it; if it does, re-order so API routes win.
- Add ONE custom business metric: a Prometheus Counter named
  `xai_predictions_total` with label `risk_level` (values: low|medium|high).
  Increment it inside the /predict handler based on the computed risk band.
  Put the Counter definition in app/api/config.py or a new app/api/metrics.py and import it —
  do NOT inline business logic changes into the handler beyond the single increment call.

Add a new test to tests/test_api.py:
  test_11_metrics_endpoint — asserts GET /metrics returns 200 and the body contains
  "xai_predictions_total" and standard http_request metrics.

Then re-run the FULL existing test suite (all 11 tests must pass) AND
scripts/validate_phase4.py (must still be 43/43).

==================================================
DELIVERABLE 3 — CONTAINERIZATION (multi-stage Dockerfile)
==================================================
Create .dockerignore excluding: .venv, .git, __pycache__, node_modules,
app/frontend/node_modules, *.pyc, results/fast_mode_backup, .pytest_cache, *.log

Create requirements-lite.txt = requirements.txt MINUS autogluon.tabular and flaml
(keep: fastapi, uvicorn, pydantic, pandas, numpy, scikit-learn, matplotlib, seaborn,
 shap, lime, joblib, prometheus-fastapi-instrumentator, and anything else app/api imports).
Generate it programmatically from requirements.txt so they cannot drift; add a comment header
explaining the lite/full split.

Create Dockerfile with THREE stages and an ARG:
  ARG INSTALL_AUTOML=false

  Stage 1 "frontend-builder": FROM node:20-alpine
    - WORKDIR /fe ; COPY app/frontend/package*.json ./
    - RUN npm ci (fall back to `npm install` if no lockfile exists — check first and log which was used)
    - COPY app/frontend/ ./ ; RUN npm run build   (produces /fe/dist)

  Stage 2 "python-deps": FROM python:3.11-slim
    - ARG INSTALL_AUTOML ; ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
    - COPY requirements-lite.txt requirements.txt ./
    - RUN pip install --upgrade pip && pip install -r requirements-lite.txt
    - RUN if [ "$INSTALL_AUTOML" = "true" ]; then pip install -r requirements.txt; fi

  Stage 3 "runtime": FROM python:3.11-slim
    - ENV PYTHONUNBUFFERED=1 PORT=8000
    - Create non-root user "appuser" (uid 10001) and switch to it
    - COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
    - COPY --from=python-deps /usr/local/bin /usr/local/bin
    - COPY app/ ./app/ ; COPY src/ ./src/
    - COPY models/ ./models/ ; COPY artifacts/ ./artifacts/
    - COPY data/processed/ ./data/processed/
    - COPY reports/model_card.md ./reports/model_card.md
    - COPY --from=frontend-builder /fe/dist ./app/frontend/dist
    - RUN mkdir -p /app/logs && chown -R appuser /app
    - EXPOSE 8000
    - HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
        CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"
    - CMD ["python","-m","uvicorn","app.api.main:app","--host","0.0.0.0","--port","8000"]
    - Label the image: org.opencontainers.image.title, .description, .version=1.0.0

IMPORTANT: app/api/model_service.py already falls back to the scikit-learn Random Forest when
AutoGluon cannot be imported/loaded. Verify this fallback path works inside the LITE container.
Additionally, add support for an environment variable SERVED_MODEL_PREFERENCE
(values: "auto" | "automl" | "fallback", default "auto") in app/api/config.py, honoured by
model_service.py, so the container can deterministically choose the lightweight surrogate.
"auto" preserves today's exact behaviour. This must be a small, safe, additive change —
re-run validate_phase4.py afterwards.

Build and verify (LITE only):
  docker build --build-arg INSTALL_AUTOML=false -t xai-automl:lite -f Dockerfile .
  docker run -d --name xai-smoke -p 8001:8000 xai-automl:lite
  (wait for healthy, then:) curl http://localhost:8001/health  and  /metrics
  POST one valid payload to /predict and confirm 200 + a probability
  docker rm -f xai-smoke
Record image size, build duration, and smoke-test results in results/phase5_metadata.json.
If `docker build` is unavailable in the sandbox, run `docker build --help`-level detection,
log "docker_build_skipped": true with the reason, and continue. NEVER fail the phase on this.

Create docker-compose.yml (single-service app profile) mapping 8000:8000 with the lite image,
a healthcheck, restart: unless-stopped, and env SERVED_MODEL_PREFERENCE.

==================================================
DELIVERABLE 4 — MONITORING STACK (Prometheus + Grafana + Alertmanager)
==================================================
Create monitoring/prometheus/prometheus.yml:
  - global scrape_interval: 10s, evaluation_interval: 10s
  - rule_files: ["/etc/prometheus/alerts.yml"]
  - alerting → alertmanager:9093
  - scrape job "xai-api": targets ["host.docker.internal:8000"]
    (add a comment explaining that on Linux the user must switch to 172.17.0.1 or the compose service name)
  - scrape job "prometheus": targets ["localhost:9090"]

Create monitoring/prometheus/alerts.yml with at least:
  - ApiDown            (up == 0 for 1m, severity critical)
  - HighErrorRate      (rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05 for 2m, severity warning)
  - HighP95Latency     (histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 1 for 5m, severity warning)
  Use metric names that prometheus-fastapi-instrumentator actually emits — VERIFY the real names
  by inspecting the /metrics output from your smoke-test container and adjust the rules to match exactly.
  Document the verified metric names in ASSUMPTIONS_PHASE5.md.

Create monitoring/grafana/provisioning/datasources/prometheus.yml (auto-provision the Prometheus DS)
Create monitoring/grafana/provisioning/dashboards/dashboards.yml
Create monitoring/grafana/dashboards/xai-operations.json — a valid Grafana dashboard JSON with panels:
  Request Rate, Error Rate, P95 Latency, Predictions by Risk Level (from xai_predictions_total),
  API Up/Down stat panel. Use the exact metric names you verified above.

Create monitoring/alertmanager/alertmanager.yml (minimal: a null/webhook receiver, no external deps).

Create docker-compose.monitoring.yml:
  services: prometheus (prom/prometheus:latest, 9090:9090, mounts the two prometheus files,
             extra_hosts: ["host.docker.internal:host-gateway"]),
            grafana (grafana/grafana:latest, 3000:3000, GF_SECURITY_ADMIN_PASSWORD=admin, provisioning mounts),
            alertmanager (prom/alertmanager:latest, 9093:9093, OPTIONAL — put it behind
             compose profile "alerting" so `docker compose -f docker-compose.monitoring.yml up -d`
             works without it and `--profile alerting` includes it)
  Use a named volume for grafana data. Set restart: unless-stopped on all.

==================================================
DELIVERABLE 5 — KUBERNETES MANIFESTS
==================================================
Create k8s/ with:
- namespace.yaml            (name: xai-predictive-maintenance)
- configmap.yaml            (SERVED_MODEL_PREFERENCE, RISK thresholds, LOG_LEVEL)
- secret.yaml               (stringData with a clearly-marked DEMO_API_KEY placeholder +
                             a prominent comment: "DEMO ONLY — in production use SealedSecrets /
                             External Secrets Operator / cloud KMS. Never commit real credentials.")
- deployment.yaml:
    name xai-api, namespace as above, replicas: 2
    strategy: RollingUpdate, maxSurge 1, maxUnavailable 0
    image: xai-automl:lite, imagePullPolicy: IfNotPresent
    ports: containerPort 8000 named http
    envFrom: configMapRef + secretRef
    resources: requests cpu 100m / memory 512Mi ; limits cpu 1000m / memory 2Gi
    readinessProbe: httpGet /health port 8000, initialDelaySeconds 10, periodSeconds 5, failureThreshold 3
    livenessProbe:   httpGet /health port 8000, initialDelaySeconds 40, periodSeconds 15, failureThreshold 3
    securityContext: runAsNonRoot true, runAsUser 10001, allowPrivilegeEscalation false,
                     readOnlyRootFilesystem false (document why: the app writes dynamic SHAP plots),
                     capabilities drop ALL
    labels: app=xai-api, tier=backend, version=v1
- service.yaml              (ClusterIP, port 80 → targetPort 8000, selector app=xai-api)
- hpa.yaml                  (autoscaling/v2, minReplicas 1, maxReplicas 4,
                             CPU averageUtilization 60, memory averageUtilization 80,
                             behavior block with sensible scaleDown stabilizationWindowSeconds 120)
- kustomization.yaml        (namespace + resources list so `kubectl apply -k k8s/` works)

Validate syntax if kubectl is present:
  kubectl apply --dry-run=client -k k8s/ -o yaml > results/k8s_dryrun_output.yaml
If kubectl is absent, log "kubectl_skipped": true and continue. Do NOT attempt a real apply.

==================================================
DELIVERABLE 6 — INFRASTRUCTURE AS CODE (Terraform)
==================================================
Create terraform/ with:
- versions.tf   (terraform >= 1.5, kreuzwerker/docker provider ~> 3.0)
- variables.tf  (grafana_admin_password default "admin" + validation block,
                 prometheus_port default 9090, grafana_port default 3000,
                 api_host default "host.docker.internal")
- main.tf       (docker provider; docker_network "xai_monitoring";
                 docker_volume for grafana data;
                 docker_image + docker_container for prometheus and grafana mirroring the compose stack,
                 with correct bind mounts to ../monitoring/... , ports, restart policies,
                 healthchecks, and a depends_on chain)
- outputs.tf    (grafana_url, prometheus_url, network_name, container IDs)
- terraform.tfvars.example
- README.md     (init/plan/apply/destroy commands + what it provisions)

If terraform is installed, run (and capture output to results/):
  terraform -chdir=terraform init -backend=false
  terraform -chdir=terraform validate
  terraform -chdir=terraform plan -out=tfplan  (plan only — NEVER apply)
Save the plan output to results/terraform_plan.txt.
If terraform is absent, log "terraform_skipped": true with the install command and continue.

==================================================
DELIVERABLE 7 — CI/CD PIPELINE (GitHub Actions)
==================================================
Create .github/workflows/ci-cd.yml with three jobs.

Triggers: push to [main, develop], pull_request to [main], and workflow_dispatch.

Job 1 "quality":
  runs-on: ubuntu-latest
  - actions/checkout@v4
  - actions/setup-python@v5 with python-version 3.11 and pip cache
  - pip install -r requirements-dev.txt
  - ruff check (upload annotations on failure)
  - bandit -r app src -x tests (continue-on-error: false, but allow LOW findings — use `-ll` to fail only on medium+)
  - gitleaks/gitleaks-action@v2 (secret scanning)
  - pip install -r requirements-lite.txt
  - pytest tests/ -q --junitxml=results/pytest-report.xml ; upload the xml as an artifact

Job 2 "build-scan-push":
  needs: quality ; runs-on: ubuntu-latest
  permissions: contents: read, packages: write
  - checkout
  - docker/setup-buildx-action@v3
  - docker/metadata-action@v5 → images: ghcr.io/${{ github.repository }}, tags: type=sha + type=ref,event=branch + latest on main
  - docker/build-push-action@v6:
      build-args: INSTALL_AUTOML=false   (lite image for CI speed)
      push: ${{ github.event_name != 'pull_request' }}
      cache-from/to: type=gha
  - aquasecurity/trivy-action@master: scan the built image, format table,
    severity CRITICAL,HIGH, exit-code 1 on CRITICAL only (HIGH must not block — document why),
    ignore-unfixed: true
  - On main only: a second build with INSTALL_AUTOML=true tagged `:full-${{ github.sha }}`,
    pushed to GHCR, with `continue-on-error: true` and a comment explaining it is the
    release artifact and may take 15+ minutes.

Job 3 "deploy-verify":
  needs: build-scan-push ; runs-on: ubuntu-latest ; if: github.ref == 'refs/heads/main'
  - checkout
  - helm/kind-action@v1 → create a kind cluster
  - kind load docker-image (build the lite image locally in the runner first)
  - kubectl apply -k k8s/
  - kubectl -n xai-predictive-maintenance rollout status deployment/xai-api --timeout=180s
  - kubectl get pods,svc,hpa -A  (echo for logs)
  - port-forward + curl /health and /metrics, assert 200
  - kubectl delete -k k8s/ (cleanup)
  This job is the "multi-stage deployment verification" gate.

Keep the YAML valid, commented, and pinned to major versions. Validate YAML syntax locally with python yaml.safe_load.

==================================================
DELIVERABLE 8 — COLLABORATION ARTEFACTS
==================================================
Create .github/PULL_REQUEST_TEMPLATE.md with checkboxes:
  description, related issue, type of change, self-review, tests added/passing,
  lint passing, docs updated, screenshots, security considerations, deployment notes.
Create .github/ISSUE_TEMPLATE/bug_report.md and feature_request.md (simple YAML front-matter forms).
Create CONTRIBUTING.md documenting the GitFlow strategy actually used:
  main (protected, production) ← develop (integration) ← feature/* , bugfix/* , hotfix/*
  Include: how to branch, commit message convention (Conventional Commits),
  how to open a PR, required checks, code owners note, and the definition of done.

Initialize git locally if not already initialized:
  git init (if needed) ; create a sensible .gitignore addition for terraform (.terraform/, *.tfstate*, tfplan),
  grafana volume data, and results/k8s_dryrun_output.yaml
  git add -A ; git commit -m "chore(phase5): add DevOps layer — CI/CD, Docker, Kubernetes, Terraform, monitoring"
Do NOT create a remote. Do NOT push. Print the exact commands the user must run to push
(both the `gh` CLI one-liner and the plain git remote/push variant) into DEVOPS_SETUP_GUIDE.md.

==================================================
DELIVERABLE 9 — HELPER SCRIPTS (PowerShell primary, bash mirror)
==================================================
scripts/build_image.ps1        (param -Full switch → INSTALL_AUTOML=true; default lite; prints size & duration)
scripts/run_app.ps1            (docker compose up the app, wait for health, print URLs)
scripts/deploy_k8s.ps1         (kubectl apply -k k8s/ ; rollout status ; port-forward svc/xai-api 8000:80 in background ; print URLs)
scripts/teardown_k8s.ps1       (kubectl delete -k k8s/ --ignore-not-found)
scripts/start_monitoring.ps1   (docker compose -f docker-compose.monitoring.yml --profile alerting up -d ; print Grafana/Prometheus URLs + default creds)
scripts/stop_monitoring.ps1
scripts/generate_load.py       (argparse: --requests N default 300, --concurrency C default 5, --url default http://localhost:8000;
                                fires realistic /predict payloads across all risk bands using a thread pool;
                                prints achieved RPS and status-code histogram. stdlib + httpx only.)
Mirror each .ps1 with an equivalent .sh for Linux/macOS. All scripts must be idempotent and fail loudly with clear messages.

==================================================
DELIVERABLE 10 — DOCUMENTATION & REPORT
==================================================
Create DEVOPS_SETUP_GUIDE.md — an ordered, numbered, copy-pasteable runbook for the USER covering
every manual step the agent cannot perform:
  A. Prerequisites (enable Docker Desktop Kubernetes, winget install Hashicorp.Terraform, winget install GitHub.cli, gh auth login, verification commands)
  B. Push to GitHub (gh repo create ... --private --source=. --remote=origin --push, plus the manual git remote alternative)
  C. Demonstrate the branching/PR workflow (create develop, create feature/demo-branch, trivial commit, gh pr create, gh pr merge)
  D. Enable branch protection (exact GitHub UI click-path)
  E. Observe CI/CD (where to look, what to screenshot)
  F. Local container run + smoke test
  G. Kubernetes deploy, rolling update demo (kubectl set image ...), scaling demo (kubectl scale --replicas=4 then watch HPA), teardown
  H. Terraform apply for the monitoring stack, Grafana login (admin/admin), dashboard import path, Prometheus targets/alerts pages
  I. Generate load + capture Grafana screenshots
  J. Run validate_phase5.py
  K. A screenshot checklist table (16+ items) mapping each screenshot → the report section it belongs in
  Every command must be given for Windows PowerShell first, then bash.

Create reports/devops_phase5_results.md — report-ready academic write-up with these sections:
  1. Phase 5 Objective
  2. DevOps Ecosystem & Version Control Strategy (GitFlow, PR template, branch protection, collaboration)
  3. CI/CD Pipeline Design (three-job architecture diagram in ASCII, stage-by-stage explanation, quality gates)
  4. Static Code Analysis & Security Scanning (ruff, bandit, gitleaks, Trivy — what each catches, sample findings)
  5. Containerization (multi-stage Dockerfile explained stage-by-stage, lite vs full build-arg rationale,
     image size reduction, non-root user, HEALTHCHECK, .dockerignore)
  6. Container Registry Strategy (GHCR, SHA + branch + latest tagging, GHA cache)
  7. Kubernetes Orchestration (namespace, Deployment, probes, resource requests/limits,
     RollingUpdate strategy with maxSurge/maxUnavailable, Service, HPA scaling policy, security context)
  8. Infrastructure as Code (Terraform docker provider — what it provisions, plan output summary,
     why IaC over manual, how the same code targets cloud providers)
  9. Monitoring, Logging & Alerting (Prometheus scrape config, verified metric names,
     three alert rules with thresholds and rationale, Grafana dashboard panels, structured app logging,
     kubectl logs, CI log artifacts; ELK noted as future work)
  10. Secrets Management (K8s Secret, GitHub Actions secrets/GITHUB_TOKEN, gitleaks prevention,
      production recommendation: SealedSecrets/ESO/KMS)
  11. Deployment Verification in CI (the kind-cluster deploy-verify job — why this proves the manifests are valid)
  12. Cloud Readiness (explicit honest statement: validated on local Kubernetes via Docker Desktop;
      identical manifests + Terraform patterns apply to EKS/GKE/AKS; what would change for a real cloud deployment)
  13. DevOps Rubric Coverage Matrix (table: each rubric criterion → where it is satisfied → evidence file/screenshot)
  14. Limitations & Future Work
  15. Reproduction Instructions (pointer to DEVOPS_SETUP_GUIDE.md)
  16. Screenshots to Capture (checklist mirrored from the guide)

Update README.md with a "Phase 5 — DevOps" section: architecture diagram, quickstart
(build → run → deploy → monitor), links to the guide and the DevOps report.

==================================================
DELIVERABLE 11 — METADATA, LOGS & VALIDATION
==================================================
Create results/phase5_metadata.json containing:
  phase, timestamp, mode:"final", tool_versions (docker, kubectl, terraform, node, npm, python — null if absent),
  docker_build: {performed, image_tag, size_bytes, duration_seconds, install_automl},
  container_smoke_test: {performed, health_ok, metrics_ok, predict_ok},
  terraform: {performed, init_ok, validate_ok, plan_ok, plan_file},
  kubectl: {available, dry_run_ok, output_file},
  quality_gates: {ruff_ok, bandit_findings_count, pytest_passed, pytest_total},
  skipped_validations: [...], assumptions: [...], installation_issues: [...]

Create results/phase5_run.log (timestamped) and results/phase5_error_log.json (empty list if clean).

Create scripts/validate_phase5.py checking existence AND non-emptiness of every Phase 5 artefact
listed below, PLUS re-running:
  - pytest tests/ -q                      (must be 11/11)
  - scripts/validate_phase4.py            (must still be 43/43)
  - scripts/validate_phase3.py            (must still be 39/39)
  - scripts/validate_phase2.py            (must still be 31/31)
  - python -c "import yaml,glob; [yaml.safe_load(open(f,encoding='utf-8')) for f in glob.glob('.github/workflows/*.yml')+glob.glob('k8s/*.yaml')]"
  - optional (skip gracefully if tool missing): docker image inspect xai-automl:lite,
    terraform -chdir=terraform validate, kubectl apply --dry-run=client -k k8s/
Print a grouped PASS/FAIL checklist with sizes and a final OVERALL STATUS, in the same visual
style as validate_phase2/3/4.py. Exit non-zero on any FAIL.

Required artefact checklist for validate_phase5.py:
  requirements-dev.txt, ruff.toml, .dockerignore, requirements-lite.txt,
  Dockerfile, docker-compose.yml, docker-compose.monitoring.yml,
  monitoring/prometheus/prometheus.yml, monitoring/prometheus/alerts.yml,
  monitoring/grafana/provisioning/datasources/prometheus.yml,
  monitoring/grafana/provisioning/dashboards/dashboards.yml,
  monitoring/grafana/dashboards/xai-operations.json,
  monitoring/alertmanager/alertmanager.yml,
  k8s/namespace.yaml, k8s/configmap.yaml, k8s/secret.yaml, k8s/deployment.yaml,
  k8s/service.yaml, k8s/hpa.yaml, k8s/kustomization.yaml,
  terraform/versions.tf, terraform/variables.tf, terraform/main.tf, terraform/outputs.tf,
  terraform/terraform.tfvars.example, terraform/README.md,
  .github/workflows/ci-cd.yml, .github/PULL_REQUEST_TEMPLATE.md,
  .github/ISSUE_TEMPLATE/bug_report.md, .github/ISSUE_TEMPLATE/feature_request.md,
  CONTRIBUTING.md, DEVOPS_SETUP_GUIDE.md,
  scripts/build_image.ps1, scripts/build_image.sh, scripts/run_app.ps1, scripts/run_app.sh,
  scripts/deploy_k8s.ps1, scripts/deploy_k8s.sh, scripts/teardown_k8s.ps1, scripts/teardown_k8s.sh,
  scripts/start_monitoring.ps1, scripts/start_monitoring.sh,
  scripts/stop_monitoring.ps1, scripts/stop_monitoring.sh, scripts/generate_load.py,
  scripts/validate_phase5.py,
  reports/devops_phase5_results.md, ASSUMPTIONS_PHASE5.md,
  results/phase5_metadata.json, results/phase5_run.log, results/phase5_error_log.json
  Plus conditional: results/terraform_plan.txt (only if terraform ran),
                    results/k8s_dryrun_output.yaml (only if kubectl ran)

Update .gitignore to exclude: .terraform/, *.tfstate, *.tfstate.*, tfplan,
  monitoring/grafana/data/, results/k8s_dryrun_output.yaml, .ruff_cache/, .bandit

==================================================
EXECUTION ORDER (follow exactly)
==================================================
1.  Snapshot the current state: run validate_phase2/3/4.py and record 31/39/43 PASS baselines.
2.  Deliverable 1 (quality tooling) → get ruff/bandit/pytest green.
3.  Deliverable 2 (metrics instrumentation) → re-run pytest (11/11) + validate_phase4 (43/43).
4.  Deliverable 3 (Docker) → build LITE image, smoke-test in a container, tear it down.
5.  Deliverable 4 (monitoring configs) → validate YAML/JSON parse cleanly.
6.  Deliverable 5 (k8s manifests) → dry-run if kubectl present.
7.  Deliverable 6 (terraform) → init/validate/plan if terraform present.
8.  Deliverable 7 (CI/CD YAML) → validate YAML parses.
9.  Deliverable 8 (collaboration + local git commit).
10. Deliverable 9 (helper scripts) → syntax-check each (.ps1 via `powershell -NoProfile -Command "Get-Command -Syntax"` or a parse check; .sh via `bash -n`).
11. Deliverable 10 (docs/report).
12. Deliverable 11 (metadata + validate_phase5.py) → run it until OVERALL STATUS: PASS.
13. Final regression: re-run validate_phase2.py, validate_phase3.py, validate_phase4.py,
    and pytest. ALL must still pass. If anything regressed, fix it before finishing.

==================================================
FINAL OUTPUT REQUIRED FROM YOU
==================================================
1.  Full list of files created/modified (mark MODIFIED files explicitly, especially app/api/main.py,
    app/api/config.py, app/api/model_service.py, tests/test_api.py, requirements.txt, README.md, .gitignore).
2.  Full output of scripts/validate_phase5.py.
3.  Regression confirmation table: Phase 2 (31/31), Phase 3 (39/39), Phase 4 (43/43), pytest (11/11).
4.  Docker lite image: tag, size, build duration, smoke-test results (/health, /metrics, /predict).
5.  Quality gate results: ruff status, bandit finding count by severity, gitleaks config present.
6.  Terraform plan summary (or "skipped — not installed", with the install command).
7.  kubectl dry-run status (or "skipped — not available").
8.  Every manual step the user must perform, as a short ordered list (pointing to DEVOPS_SETUP_GUIDE.md for detail).
9.  Any assumptions made and any tools skipped.
10. A message saying exactly:
    "Phase 5 completed. DevOps layer implemented and validated. The AutoML + XAI + DevOps project is ready for final reporting. The separate Agentic AI project can now begin."

Do NOT push to any remote. Do NOT apply to any real cluster. Do NOT run terraform apply.
Do NOT start a new project or phase. Stop after Phase 5 validation.