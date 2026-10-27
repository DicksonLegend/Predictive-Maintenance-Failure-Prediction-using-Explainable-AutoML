# Post-Submission WSL / Docker VHDX Disk Compactor
# WARNING: Run ONLY after project evaluation, final screenshots, and stack teardown.

Write-Host "==========================================================================" -ForegroundColor Red
Write-Host "WARNING: DOCKER VHDX COMPACTION SCRIPT" -ForegroundColor Yellow
Write-Host "This script will shut down WSL2 to shrink the physical size of Docker's" -ForegroundColor Yellow
Write-Host "virtual hard disk (ext4.vhdx / docker_data.vhdx) on your Windows C: drive." -ForegroundColor Yellow
Write-Host "DO NOT RUN THIS WHILE THE DEPLOYMENT STACK OR DEMOS ARE ACTIVE!" -ForegroundColor Red
Write-Host "Run this ONLY after capturing all UI screenshots and submitting the project." -ForegroundColor Yellow
Write-Host "==========================================================================" -ForegroundColor Red

$confirm = Read-Host "Are you sure you want to shut down WSL and compact Docker virtual disks now? (y/N)"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Host "Operation cancelled by user." -ForegroundColor Gray
    exit 0
}

# 1. Shut down WSL
Write-Host "`n[1/4] Shutting down WSL2 and Docker VM..." -ForegroundColor Cyan
wsl --shutdown
Start-Sleep -Seconds 5

# 2. Locate Docker VHDX files under LOCALAPPDATA
Write-Host "[2/4] Auto-discovering Docker VHDX disks..." -ForegroundColor Cyan
$vhdxFiles = Get-ChildItem -Path "$env:LOCALAPPDATA\Docker" -Recurse -Filter "*.vhdx" -ErrorAction SilentlyContinue

if (-not $vhdxFiles -or $vhdxFiles.Count -eq 0) {
    Write-Host "No Docker VHDX files located under $env:LOCALAPPDATA\Docker." -ForegroundColor Yellow
    exit 1
}

# 3. Compact each discovered VHDX file using Diskpart
foreach ($file in $vhdxFiles) {
    $path = $file.FullName
    $sizeBeforeMB = [math]::Round($file.Length / 1MB, 2)
    $sizeBeforeGB = [math]::Round($file.Length / 1GB, 2)
    
    Write-Host "`nProcessing: $path" -ForegroundColor White
    Write-Host "Current Size: $sizeBeforeMB MB ($sizeBeforeGB GB)" -ForegroundColor White

    # Generate temporary diskpart script
    $tempScript = [System.IO.Path]::GetTempFileName()
    @"
select vdisk file="$path"
attach vdisk readonly
compact vdisk
detach vdisk
"@ | Set-Content -Path $tempScript -Encoding ASCII

    Write-Host "Compacting disk via Windows diskpart..." -ForegroundColor Cyan
    try {
        Start-Process -FilePath "diskpart.exe" -ArgumentList "/s `"$tempScript`"" -Wait -NoNewWindow
    } finally {
        if (Test-Path $tempScript) { Remove-Item -Path $tempScript -Force }
    }

    # Refresh file info
    $updatedFile = Get-Item -Path $path
    $sizeAfterMB = [math]::Round($updatedFile.Length / 1MB, 2)
    $sizeAfterGB = [math]::Round($updatedFile.Length / 1GB, 2)
    $reclaimedGB = [math]::Round(($file.Length - $updatedFile.Length) / 1GB, 2)

    Write-Host "New Size:     $sizeAfterMB MB ($sizeAfterGB GB)" -ForegroundColor Green
    Write-Host "Reclaimed:    $reclaimedGB GB on C: drive!" -ForegroundColor Green
}

Write-Host "`n[4/4] Docker disk compaction complete. You may restart Docker Desktop whenever needed." -ForegroundColor Green
