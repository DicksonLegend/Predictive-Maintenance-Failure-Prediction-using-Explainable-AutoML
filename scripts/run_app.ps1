<#
.SYNOPSIS
    Starts the Predictive Maintenance XAI Application container via Docker Compose.
#>
$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting XAI Predictive Maintenance Application..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

docker compose up -d

Write-Host "Waiting for container healthcheck to report healthy..." -ForegroundColor Yellow
$healthy = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    $status = docker inspect --format="{{.State.Health.Status}}" xai-predictive-maintenance-api 2>$null
    if ($status -eq "healthy") {
        $healthy = $true
        break
    }
    Write-Host "Current health status: $status (waiting...)"
}

if (-not $healthy) {
    Write-Host "Warning: Application did not become healthy within 60s." -ForegroundColor Yellow
} else {
    Write-Host "Application is HEALTHY!" -ForegroundColor Green
}

Write-Host "------------------------------------------------------------" -ForegroundColor Green
Write-Host "Access Endpoints:" -ForegroundColor Green
Write-Host "  Dashboard & UI:  http://localhost:8000"
Write-Host "  Swagger Docs:    http://localhost:8000/docs"
Write-Host "  Prometheus:      http://localhost:8000/metrics"
Write-Host "  Health Check:    http://localhost:8000/health"
Write-Host "------------------------------------------------------------" -ForegroundColor Green
