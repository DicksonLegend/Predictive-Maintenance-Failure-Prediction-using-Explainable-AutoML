# Step 4: Live Alert Demo Automation
Write-Host "=== Starting Live Alert Demo ==="

# 1. Scale deployment to 0
Write-Host "Scaling deployment/xai-api to 0 replicas..."
kubectl -n xai-predictive-maintenance scale deployment/xai-api --replicas=0
kubectl -n xai-predictive-maintenance get pods

# 2. Wait for alert to trigger (15s scrape interval + 60s for duration = ~75s)
Write-Host "Waiting 75 seconds for Prometheus scrape and alert evaluation (for: 1m)..."
Start-Sleep -Seconds 75

# 3. Capture alert state
Write-Host "Capturing Prometheus alerts and rules..."
curl.exe -s http://localhost:9090/api/v1/alerts | Out-File -FilePath results/phase56_evidence/04_alert_firing.json -Encoding utf8
curl.exe -s http://localhost:9090/api/v1/rules | Out-File -FilePath results/phase56_evidence/04_rules_firing.json -Encoding utf8

# 4. Immediate restoration
Write-Host "Restoring deployment/xai-api to 2 replicas..."
kubectl -n xai-predictive-maintenance scale deployment/xai-api --replicas=2
kubectl -n xai-predictive-maintenance rollout status deployment/xai-api --timeout=240s

# Verify health endpoint on host
Write-Host "Verifying host :8000 health..."
$retries = 0
$health_ok = $false
while ($retries -lt 15) {
    Start-Sleep -Seconds 3
    try {
        $res = curl.exe -s http://localhost:8000/health
        if ($res -like '*"status":"ok"*') {
            $health_ok = $true
            Write-Host "Host :8000 is healthy: $res"
            break
        }
    } catch {}
    $retries++
}

# 5. Wait 75 seconds for Prometheus to observe target recovery and clear alert
Write-Host "Waiting 75 seconds for target to re-establish and Prometheus up health..."
Start-Sleep -Seconds 75

curl.exe -s http://localhost:9090/api/v1/targets | Out-File -FilePath results/phase56_evidence/04_target_recovered.json -Encoding utf8
curl.exe -s http://localhost:9090/api/v1/alerts | Out-File -FilePath results/phase56_evidence/04_alert_cleared.json -Encoding utf8

Write-Host "=== Live Alert Demo Completed Successfully ==="
