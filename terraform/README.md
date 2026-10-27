# Infrastructure as Code (IaC) — Monitoring Stack Provisioning

This directory defines the automated Terraform deployment of the **Industry 4.0 Predictive Maintenance XAI Monitoring Stack** using the `kreuzwerker/docker` provider.

---

## 1. Architecture Provisioned
- **Docker Bridge Network:** `xai_monitoring` for isolated inter-container communication.
- **Persistent Volume:** `xai_grafana_data` preserving user sessions, preferences, and custom dashboard state.
- **Prometheus Container:** `prom/prometheus:latest` running on port `9090`, configured to scrape `/metrics` from the FastAPI application via `host.docker.internal:8000`, with automated alerting evaluated from `alerts.yml`.
- **Grafana Container:** `grafana/grafana:latest` running on port `3000`, auto-provisioned with the Prometheus data source and the `xai-operations.json` operations dashboard.

---

## 2. Prerequisites
1. **Docker Desktop:** Ensure Docker Desktop is installed and running.
2. **Terraform CLI:** Install Terraform `>= 1.5.0`:
   ```powershell
   winget install Hashicorp.Terraform
   ```
   Or on Linux/macOS:
   ```bash
   brew tap hashicorp/tap && brew install hashicorp/tap/terraform
   ```

---

## 3. Workflow Commands

### Step A: Initialize Terraform
Downloads the Docker provider plugins and sets up the local state storage:
```powershell
cd terraform
terraform init
```

### Step B: Validate Syntax & Configuration
Verifies that all HCL blocks, dependencies, and variables conform to the Terraform specification:
```powershell
terraform validate
```

### Step C: Generate Execution Plan (Dry Run)
Inspects the target Docker daemon and computes the exact set of resources to be provisioned:
```powershell
terraform plan -out=tfplan
```

### Step D: Apply Infrastructure (Manual Step)
Provisions the network, volume, and containers:
```powershell
terraform apply tfplan
```
*Note: As per DevOps governance constraints, automated scripts only run plan; apply is executed manually by the authorized platform engineer.*

### Step E: Destroy Infrastructure (Teardown)
Cleanly removes containers and networks when operations cease:
```powershell
terraform destroy -auto-approve
```

---

## 4. Default Access Points
- **Prometheus Web UI:** [http://localhost:9090](http://localhost:9090)
- **Grafana Dashboard:** [http://localhost:3000](http://localhost:3000) (Credentials: `admin` / `admin`)
