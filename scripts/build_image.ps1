<#
.SYNOPSIS
    Builds the Predictive Maintenance XAI Docker container image.
.DESCRIPTION
    Builds either the lightweight production image (default) or the full AutoML release image.
.PARAMETER Full
    Switch to build the full image with INSTALL_AUTOML=true (includes AutoGluon and FLAML).
#>
param (
    [switch]$Full
)

$ErrorActionPreference = "Stop"

$imageTag = if ($Full) { "xai-automl:full" } else { "xai-automl:lite" }
$installAutoml = if ($Full) { "true" } else { "false" }

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Building Docker Image: $imageTag (INSTALL_AUTOML=$installAutoml)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$startTime = Get-Date

try {
    & docker build --build-arg INSTALL_AUTOML=$installAutoml -t $imageTag -f Dockerfile .
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build exited with code $LASTEXITCODE"
    }

    $elapsed = (Get-Date) - $startTime
    $size = & docker image inspect $imageTag --format "{{.Size}}"

    Write-Host "------------------------------------------------------------" -ForegroundColor Green
    Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "Tag:      $imageTag"
    Write-Host "Duration: $($elapsed.TotalSeconds.ToString('F1')) seconds"
    Write-Host "Size:     $([math]::Round([int64]$size / 1MB, 1)) MB"
    Write-Host "------------------------------------------------------------" -ForegroundColor Green
}
catch {
    Write-Host "Error during Docker build: $_" -ForegroundColor Red
    exit 1
}
