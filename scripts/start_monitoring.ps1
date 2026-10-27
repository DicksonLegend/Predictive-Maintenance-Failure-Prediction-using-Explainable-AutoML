<#
.SYNOPSIS
    Starts the Prometheus, Grafana, and Alertmanager monitoring stack.
#>
$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting Industry 4.0 Monitoring Stack (Prometheus + Grafana)..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

docker compose -f docker-compose.monitoring.yml --profile alerting up -d

Write-Host "------------------------------------------------------------" -ForegroundColor Green
Write-Host "MONITORING STACK ONLINE!" -ForegroundColor Green
Write-Host "Access Points:"
Write-Host "  Grafana Dashboard:  http://localhost:3000"
Write-Host "  Credentials:        admin / admin"
Write-Host "  Prometheus UI:      http://localhost:9090"
Write-Host "  Alertmanager UI:    http://localhost:9093"
Write-Host "------------------------------------------------------------" -ForegroundColor Green
