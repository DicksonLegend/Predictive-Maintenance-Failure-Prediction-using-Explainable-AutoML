<#
.SYNOPSIS
    Tears down the XAI Predictive Maintenance Kubernetes deployment.
#>
$ErrorActionPreference = "Stop"

Write-Host "Tearing down Kubernetes resources for XAI Predictive Maintenance..." -ForegroundColor Yellow
kubectl delete -k k8s/ --ignore-not-found

Write-Host "Cleaning up any running port-forward background jobs..." -ForegroundColor Yellow
Get-Job | Stop-Job -PassThru | Remove-Job

Write-Host "Kubernetes resources successfully removed." -ForegroundColor Green
