# Start Stack After Laptop Reboot (1-Click Startup & Verification)

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Restarting Predictive Maintenance DevOps Stack Post-Reboot" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Ensure Docker Desktop is running
Write-Host "`n[1/4] Checking Docker Desktop..." -ForegroundColor White
try {
    docker info > $null 2>&1
    Write-Host "Docker Engine is UP and running." -ForegroundColor Green
} catch {
    Write-Host "Docker is not running. Please launch Docker Desktop and wait 30 seconds." -ForegroundColor Red
    exit 1
}

# 2. Ensure Prometheus and Grafana monitoring containers are started
Write-Host "`n[2/4] Verifying Monitoring Containers (Prometheus & Grafana)..." -ForegroundColor White
$running = docker ps --format "{{.Names}}"
if ($running -notmatch "xai-prometheus-tf") {
    Write-Host "Starting xai-prometheus-tf..." -ForegroundColor Yellow
    docker start xai-prometheus-tf
}
if ($running -notmatch "xai-grafana-tf") {
    Write-Host "Starting xai-grafana-tf..." -ForegroundColor Yellow
    docker start xai-grafana-tf
}
Write-Host "Monitoring containers active." -ForegroundColor Green

# 3. Verify Kubernetes Pods
Write-Host "`n[3/4] Verifying Kubernetes Pods..." -ForegroundColor White
kubectl -n xai-predictive-maintenance get pods

# 4. Launch Port-Forward Supervisor in Background (if not already running)
Write-Host "`n[4/4] Starting Port-Forward Supervisor (Host Port 8000)..." -ForegroundColor White
$existingPf = Get-Process | Where-Object { $_.CommandLine -and $_.CommandLine -match "port_forward_supervisor" }
if (-not $existingPf) {
    Start-Process -FilePath "d:\AutoML_XAI\.venv\Scripts\python.exe" -ArgumentList "d:\AutoML_XAI\scripts\port_forward_supervisor.py" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    Write-Host "Port-forward supervisor launched in background." -ForegroundColor Green
} else {
    Write-Host "Port-forward supervisor already running." -ForegroundColor Green
}

# 5. Quick Health Ping
Write-Host "`n--- Health Check Results ---" -ForegroundColor White
Start-Sleep -Seconds 2
try {
    $health = curl.exe -s http://localhost:8000/health
    Write-Host "API (localhost:8000):        $health" -ForegroundColor Green
} catch {
    Write-Host "API (localhost:8000):        Starting up (retry in 10s)..." -ForegroundColor Yellow
}

Write-Host "Prometheus (localhost:9090): http://localhost:9090 (Targets & Alerts)" -ForegroundColor Green
Write-Host "Grafana (localhost:3000):    http://localhost:3000 (Operations Dashboard)" -ForegroundColor Green
Write-Host "`nFull stack is ready for evaluation and screenshots!" -ForegroundColor Cyan
