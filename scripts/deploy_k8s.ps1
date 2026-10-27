<#
.SYNOPSIS
    Deploys the XAI Predictive Maintenance manifests to Kubernetes via Kustomize.
#>
$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Deploying XAI Predictive Maintenance to Kubernetes..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

kubectl apply -k k8s/

Write-Host "Waiting for deployment rollout to complete..." -ForegroundColor Yellow
kubectl -n xai-predictive-maintenance rollout status deployment/xai-api --timeout=180s

Write-Host "Starting background port-forward (8000 -> 80)..." -ForegroundColor Yellow
$job = Start-Job -ScriptBlock {
    kubectl -n xai-predictive-maintenance port-forward svc/xai-api 8000:80
}

Write-Host "------------------------------------------------------------" -ForegroundColor Green
Write-Host "DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
Write-Host "Cluster Resources:"
kubectl -n xai-predictive-maintenance get pods,svc,hpa
Write-Host "------------------------------------------------------------" -ForegroundColor Green
Write-Host "Access Endpoints (via port-forward):"
Write-Host "  Dashboard & UI:  http://localhost:8000"
Write-Host "  Health Status:   http://localhost:8000/health"
Write-Host "  Prometheus:      http://localhost:8000/metrics"
Write-Host "To stop port-forward, run: Get-Job | Stop-Job"
Write-Host "------------------------------------------------------------" -ForegroundColor Green
