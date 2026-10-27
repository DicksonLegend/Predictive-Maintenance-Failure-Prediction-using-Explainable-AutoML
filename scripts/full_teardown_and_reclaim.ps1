# Post-Submission Full Teardown and Maximum Disk Reclaim Automation
# WARNING: Run ONLY after project evaluation, final screenshots, and academic submission.

param(
    [switch]$IncludeImages = $false
)

Write-Host "==========================================================================" -ForegroundColor Red
Write-Host "POST-SUBMISSION FULL TEARDOWN & DISK RECLAIM" -ForegroundColor Yellow
Write-Host "This script performs an end-to-end teardown and maximizes disk space recovery:" -ForegroundColor White
Write-Host "  1. Destroys all running Kubernetes pods, services, and Terraform containers" -ForegroundColor White
Write-Host "  2. Prunes all Docker BuildKit cache layers" -ForegroundColor White
if ($IncludeImages) {
    Write-Host "  3. [-IncludeImages active] Prunes ALL unused Docker images (incl. xai-automl:lite)" -ForegroundColor Yellow
    Write-Host "     Note: Future demonstrations will require re-running scripts/build_image.ps1" -ForegroundColor Yellow
}
Write-Host "  4. Compacts the WSL2 / Docker VHDX disk using diskpart" -ForegroundColor White
Write-Host "==========================================================================" -ForegroundColor Red

$confirm = Read-Host "Proceed with full teardown and deep disk reclaim? (y/N)"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Host "Teardown aborted." -ForegroundColor Gray
    exit 0
}

# 1. Teardown active stack
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "`n[Step 1/4] Running teardown_all.ps1..." -ForegroundColor Cyan
& (Join-Path $scriptRoot "teardown_all.ps1")

# 2. Prune Build Cache
Write-Host "`n[Step 2/4] Pruning Docker build cache..." -ForegroundColor Cyan
docker builder prune -a -f

# 3. Optional image prune
if ($IncludeImages) {
    Write-Host "`n[Step 3/4] Pruning unused Docker images (-IncludeImages specified)..." -ForegroundColor Yellow
    docker image prune -a -f
} else {
    Write-Host "`n[Step 3/4] Skipping full image prune (images preserved for future demos; use -IncludeImages to remove)." -ForegroundColor Gray
}

# 4. Compact Docker VHDX disk
Write-Host "`n[Step 4/4] Executing Docker VHDX compaction..." -ForegroundColor Cyan
& (Join-Path $scriptRoot "compact_docker_disk.ps1")

Write-Host "`nFull teardown and deep disk recovery complete!" -ForegroundColor Green
