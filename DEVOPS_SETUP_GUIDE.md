# DevOps Operations Runbook & Manual Execution Guide (Phase 5)

**Project:** Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0  
**Target Environment:** Windows 10/11 (PowerShell) with Linux/macOS Bash mirrors  
**Audience:** Platform Engineers, University Evaluators, and DevOps Practitioners  

---

## Overview
This runbook contains the step-by-step instructions for all operations requiring user credentials, cloud access, or cluster privileges that automated agents cannot perform autonomously. Follow each section in sequence.

---

## Section A: Prerequisites & Environment Setup

### 1. Enable Kubernetes in Docker Desktop
1. Open **Docker Desktop**.
2. Click the gear icon (**Settings**) in the top navigation bar.
3. Select **Kubernetes** from the left sidebar.
4. Check **Enable Kubernetes** and click **Apply & restart**.
5. Verify your Kubernetes cluster is online:
   - **PowerShell:**
     ```powershell
     kubectl cluster-info
     kubectl get nodes
     ```
   - **Bash:**
     ```bash
     kubectl cluster-info
     kubectl get nodes
     ```

### 2. Install Terraform & GitHub CLI
- **PowerShell (winget):**
  ```powershell
  winget install Hashicorp.Terraform
  winget install GitHub.cli
  ```
- **Bash (Homebrew / APT):**
  ```bash
  # macOS Homebrew:
  brew install terraform gh
  # Ubuntu / Debian:
  sudo apt-get update && sudo apt-get install -y terraform gh
  ```

### 3. Authenticate with GitHub
```powershell
gh auth login
```
*Select `GitHub.com` -> `HTTPS` -> Authenticate with your web browser.*

---

## Section B: Remote Repository Initialization & Git Push

### Option 1: One-Liner via GitHub CLI (Recommended)
```powershell
gh repo create Predictive-Maintenance-AutoML-XAI-DevOps --private --source=. --remote=origin --push
```

### Option 2: Plain Git Remote & Push
1. Create a repository named `Predictive-Maintenance-AutoML-XAI-DevOps` on [GitHub](https://github.com/new).
2. Execute the following in your local workspace:
   - **PowerShell:**
     ```powershell
     git remote add origin https://github.com/<YOUR-USERNAME>/Predictive-Maintenance-AutoML-XAI-DevOps.git
     git branch -M main
     git push -u origin main
     ```
   - **Bash:**
     ```bash
     git remote add origin https://github.com/<YOUR-USERNAME>/Predictive-Maintenance-AutoML-XAI-DevOps.git
     git branch -M main
     git push -u origin main
     ```

---

## Section C: GitFlow Branching & Pull Request Workflow

To demonstrate collaborative software engineering for the university evaluation:

1. **Create and push the `develop` integration branch:**
   - **PowerShell:**
     ```powershell
     git checkout -b develop
     git push -u origin develop
     ```
   - **Bash:**
     ```bash
     git checkout -b develop
     git push -u origin develop
     ```

2. **Create a demonstration feature branch:**
   - **PowerShell:**
     ```powershell
     git checkout -b feature/demo-monitoring-tune
     # Make a trivial documentation edit:
     Add-Content -Path "ASSUMPTIONS_PHASE5.md" -Value "`n<!-- Feature branch demonstration -->"
     git commit -am "feat(monitoring): demonstrate PR workflow with telemetry tuning"
     git push -u origin feature/demo-monitoring-tune
     ```
   - **Bash:**
     ```bash
     git checkout -b feature/demo-monitoring-tune
     echo "<!-- Feature branch demonstration -->" >> ASSUMPTIONS_PHASE5.md
     git commit -am "feat(monitoring): demonstrate PR workflow with telemetry tuning"
     git push -u origin feature/demo-monitoring-tune
     ```

3. **Open and merge a Pull Request via CLI or Web:**
   - **PowerShell / Bash:**
     ```powershell
     gh pr create --base develop --head feature/demo-monitoring-tune --title "feat(monitoring): demonstrate GitFlow PR" --body "Demonstrates automated PR template and CI integration."
     gh pr merge --merge --delete-branch
     ```

---

## Section D: Branch Protection Governance

Configure branch protection rules to enforce the Definition of Done:
1. Navigate to your GitHub repository: `https://github.com/<YOUR-USERNAME>/<REPO>/settings/branches`.
2. Click **Add branch protection rule**.
3. Set **Branch name pattern** to `main`.
4. Enable:
   - [x] **Require a pull request before merging** (Require approvals: 1).
   - [x] **Require status checks to pass before merging**.
   - Select checks: `Code Quality & Security Gate`, `Build, Scan & Push Container Images`.
   - [x] **Do not allow bypassing the above settings**.
5. Click **Create** or **Save changes**.

---

## Section E: GitHub Actions CI/CD Pipeline Monitoring

1. In your GitHub repository, click the **Actions** tab.
2. Select the latest workflow run: **CI/CD Pipeline — Code Quality, Container Security & Deployment Verification**.
3. Observe all three pipeline stages:
   - **Stage 1 (quality):** Ruff linting, Bandit SAST, Gitleaks secrets audit, Pytest suite (11/11 tests).
   - **Stage 2 (build-scan-push):** Docker multi-stage build, Trivy vulnerability scan, GHCR container push.
   - **Stage 3 (deploy-verify):** Ephemeral KinD Kubernetes cluster creation, manifest deployment, rollout verification, live health curl checks.
4. Capture screenshots for your report (see Section K).

---

## Section F: Local Container Execution & Smoke Testing

### 1. Build and Run Container Image Locally
- **PowerShell:**
  ```powershell
  # Build lightweight container image
  .\scripts\build_image.ps1

  # Launch container via Docker Compose
  .\scripts\run_app.ps1
  ```
- **Bash:**
  ```bash
  chmod +x scripts/*.sh
  ./scripts/build_image.sh
  ./scripts/run_app.sh
  ```

### 2. Smoke Test Endpoints
- **PowerShell:**
  ```powershell
  Invoke-RestMethod -Uri "http://localhost:8000/health"
  Invoke-RestMethod -Uri "http://localhost:8000/version"
  (Invoke-WebRequest -Uri "http://localhost:8000/metrics").Content.Substring(0, 500)
  ```
- **Bash:**
  ```bash
  curl -s http://localhost:8000/health | jq .
  curl -s http://localhost:8000/version | jq .
  curl -s http://localhost:8000/metrics | head -n 20
  ```

---

## Section G: Kubernetes Deployment, Rolling Update & Autoscaling

### 1. Deploy Application to Local Kubernetes
- **PowerShell:**
  ```powershell
  .\scripts\deploy_k8s.ps1
  ```
- **Bash:**
  ```bash
  ./scripts/deploy_k8s.sh
  ```

### 2. Verify Resources & Rolling Update Strategy
- **PowerShell / Bash:**
  ```powershell
  # Check deployed pods and service
  kubectl -n xai-predictive-maintenance get pods,svc,hpa

  # Demonstrate zero-downtime rolling update by updating container image
  kubectl -n xai-predictive-maintenance set image deployment/xai-api xai-api=xai-automl:lite

  # Watch the rolling deployment in real time (maxSurge 1, maxUnavailable 0)
  kubectl -n xai-predictive-maintenance rollout status deployment/xai-api
  ```

### 3. Demonstrate Horizontal Pod Autoscaling (HPA)
- **PowerShell / Bash:**
  ```powershell
  # In Terminal 1, watch the HPA status:
  kubectl -n xai-predictive-maintenance get hpa -w

  # In Terminal 2, simulate burst load:
  .venv\Scripts\python scripts/generate_load.py --requests 1000 --concurrency 20
  ```

### 4. Teardown Kubernetes Resources
- **PowerShell:**
  ```powershell
  .\scripts\teardown_k8s.ps1
  ```
- **Bash:**
  ```bash
  ./scripts/teardown_k8s.sh
  ```

---

## Section H: Terraform Monitoring Stack Deployment

### 1. Initialize and Plan via Terraform
- **PowerShell:**
  ```powershell
  cd terraform
  terraform init
  terraform validate
  terraform plan -out=tfplan
  ```
- **Bash:**
  ```bash
  cd terraform
  terraform init
  terraform validate
  terraform plan -out=tfplan
  ```

### 2. Apply Infrastructure (Manual Approval)
```powershell
terraform apply tfplan
```
*Prometheus and Grafana containers will launch on the `xai_monitoring` Docker bridge network.*

### 3. Alternative: Docker Compose Launch
If Terraform is not installed on your system, launch the identical stack using Docker Compose:
- **PowerShell:**
  ```powershell
  .\scripts\start_monitoring.ps1
  ```
- **Bash:**
  ```bash
  ./scripts/start_monitoring.sh
  ```

---

## Section I: Load Generation & Grafana Dashboard Inspection

1. **Generate Synthetic Telemetry Load (Run once before capturing Grafana screenshot):**
   ```powershell
   .venv\Scripts\python scripts/generate_load.py --requests 500 --concurrency 10
   ```
   > [!IMPORTANT]
   > **Prerequisite for Grafana Screenshot (`phase5_grafana_dashboard.png`):**
   > Always run `generate_load.py` once prior to capturing the dashboard screenshot. This generates live HTTP request volume, populates the inference counter (`sum(xai_predictions_total)`), and generates realistic latency histogram distributions for the visual panels.

2. **Access Grafana Dashboard:**
   - URL: [http://localhost:3000](http://localhost:3000)
   - Login: `admin` / `admin`
   - Navigate to: **Dashboards** -> **Industry 4.0 Operations** -> **Predictive Maintenance XAI — Operations Dashboard**.
   - Observe live panels:
     - API Status (`UP`)
     - Total Inferences
     - HTTP Request Volume by Handler
     - Latency Distribution (P50 vs P95)
     - Predictions by Risk Level (`Low`, `Medium`, `High`)
     - HTTP Error Rate (5xx %)

3. **Access Prometheus Target Health & Alerts:**
   - URL: [http://localhost:9090/targets](http://localhost:9090/targets) (Confirm `xai-api` endpoint is `UP`).
   - URL: [http://localhost:9090/alerts](http://localhost:9090/alerts) (Inspect `ApiDown`, `HighErrorRate`, `HighP95Latency`).

---

## Section J: Automated Regression Validation

Run the master Phase 5 validation script:
- **PowerShell:**
  ```powershell
  .venv\Scripts\python scripts/validate_phase5.py
  ```
- **Bash:**
  ```bash
  python scripts/validate_phase5.py
  ```
*Confirm that `OVERALL STATUS: PASS` is reported with 100% check satisfaction.*

---

## Section K: Screenshot Evidence Checklist Table

Capture the following 16 screenshots and place them in `reports/screenshots/` for inclusion in academic submissions:

| # | Screenshot Filename | Subject / View | Target Report Section |
| :---: | :--- | :--- | :--- |
| 1 | `phase5_git_branches.png` | Terminal showing `git branch -a` with `main` and `develop` | Sec 2: Version Control |
| 2 | `phase5_pr_template.png` | GitHub Web UI showing filled PR Template with quality checkboxes | Sec 2: Version Control |
| 3 | `phase5_branch_protection.png` | GitHub Settings showing protected `main` branch rules | Sec 2: Version Control |
| 4 | `phase5_ci_pipeline_overview.png` | GitHub Actions dashboard showing all 3 jobs green | Sec 3: CI/CD Pipeline |
| 5 | `phase5_quality_gate.png` | Expanded Job 1 logs showing Ruff, Bandit, and Pytest passing | Sec 4: Static Security |
| 6 | `phase5_trivy_scan.png` | Expanded Job 2 logs showing clean Trivy vulnerability table | Sec 4: Static Security |
| 7 | `phase5_docker_build.png` | Terminal output of `scripts/build_image.ps1` with size & time | Sec 5: Containerization |
| 8 | `phase5_ghcr_packages.png` | GitHub Packages UI showing published `xai-automl:lite` image | Sec 6: Container Registry |
| 9 | `phase5_k8s_resources.png` | Terminal showing `kubectl get pods,svc,hpa -n xai-predictive-maintenance` | Sec 7: Kubernetes |
| 10 | `phase5_k8s_rolling_update.png` | Terminal showing `kubectl rollout status` zero-downtime update | Sec 7: Kubernetes |
| 11 | `phase5_k8s_hpa_scaling.png` | Terminal showing HPA scaling pods under load from 2 to 4 | Sec 7: Kubernetes |
| 12 | `phase5_terraform_plan.png` | Terminal showing `terraform plan` clean execution output | Sec 8: Infrastructure as Code |
| 13 | `phase5_prometheus_targets.png` | Prometheus Web UI at `/targets` showing `xai-api` state UP | Sec 9: Monitoring |
| 14 | `phase5_prometheus_alerts.png` | Prometheus Web UI at `/alerts` showing configured alert rules | Sec 9: Monitoring |
| 15 | `phase5_grafana_dashboard.png` | Full Grafana dashboard showing live metrics under load (run `generate_load.py` once prior to capture) | Sec 9: Monitoring |
| 16 | `phase5_validation_pass.png` | Terminal showing `validate_phase5.py` with OVERALL STATUS: PASS | Sec 13: Rubric Matrix |

---

## Appendix: Post-Submission Disk Reclaim & VHDX Compaction

> [!WARNING]
> **DO NOT EXECUTE PRIOR TO SUBMISSION / GRADING:**
> Running these procedures will shut down the active deployment stack and WSL2. Perform these steps only after all evaluation screenshots have been captured and the academic project has been formally submitted.

### When to Run
- After completing all manual demonstrations, taking all required screenshots, and submitting reports.
- When you want to reclaim 10 to 18+ GB of disk space on Windows drive C:.

### Step-by-Step Execution

#### Option 1: Full Teardown & Maximum Space Reclaim (Recommended)
This script tears down all Kubernetes resources and Terraform containers, purges all BuildKit caches, and runs the disk compactor:
```powershell
# Standard Teardown & Compaction (preserves container images for fast re-launch):
.\scripts\full_teardown_and_reclaim.ps1

# Maximum Teardown & Compaction (additionally purges unused container images):
.\scripts\full_teardown_and_reclaim.ps1 -IncludeImages
```

#### Option 2: Standalone Docker VHDX Compaction
If you have already run `.\scripts\teardown_all.ps1` or stopped Docker Desktop, you can shrink the physical WSL2 virtual hard disk file (`docker_data.vhdx`) directly:
```powershell
.\scripts\compact_docker_disk.ps1
```
*How it works: The script shuts down WSL2 (`wsl --shutdown`), locates the Docker VHDX under `%LOCALAPPDATA%\Docker`, and invokes Windows `diskpart` in read-only mode to compact the sparse file back to its actual utilized footprint.*
