# Comprehensive Teardown Script for Phase 5.5 Local Stack (PowerShell)
# DO NOT RUN AUTOMATICALLY: Keep stack active until UI screenshots are captured.

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Teardown: Predictive Maintenance DevOps Stack" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Stop background port-forward and supervisors
Write-Host "[1/3] Terminating port-forward processes on host port 8000..." -ForegroundColor Yellow
Get-Process | Where-Object { $_.ProcessName -match "kubectl" -or ($_.CommandLine -and $_.CommandLine -match "port_forward") } | Stop-Process -Force -ErrorAction SilentlyContinue

# 2. Delete Kubernetes manifests
Write-Host "[2/3] Deleting Kubernetes namespace and resources..." -ForegroundColor Yellow
kubectl delete -k k8s/ --ignore-not-found

# 3. Destroy Terraform Monitoring Stack
Write-Host "[3/3] Destroying Terraform provisioned monitoring containers..." -ForegroundColor Yellow
$terraformBin = "C:\Users\dicks\AppData\Local\Microsoft\WinGet\Links\terraform.exe"
if (-not (Test-Path $terraformBin)) {
    $terraformBin = "terraform"
}
& $terraformBin -chdir=terraform destroy -auto-approve

Write-Host "All local Phase 5.5 resources have been safely torn down." -ForegroundColor Green
