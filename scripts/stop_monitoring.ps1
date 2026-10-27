<#
.SYNOPSIS
    Stops the Prometheus, Grafana, and Alertmanager monitoring stack.
#>
$ErrorActionPreference = "Stop"

Write-Host "Stopping Industry 4.0 Monitoring Stack..." -ForegroundColor Yellow
docker compose -f docker-compose.monitoring.yml --profile alerting down

Write-Host "Monitoring stack cleanly stopped." -ForegroundColor Green
