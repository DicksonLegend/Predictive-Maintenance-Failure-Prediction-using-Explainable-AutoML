#!/usr/bin/env bash
# ==============================================================================
# Starts the Predictive Maintenance XAI Application via Docker Compose (Bash mirror)
# ==============================================================================
set -euo pipefail

echo "============================================================"
echo "Starting XAI Predictive Maintenance Application..."
echo "============================================================"

docker compose up -d

echo "Waiting for container healthcheck to report healthy..."
for i in {1..30}; do
    sleep 2
    STATUS=$(docker inspect --format="{{.State.Health.Status}}" xai-predictive-maintenance-api 2>/dev/null || echo "starting")
    if [[ "$STATUS" == "healthy" ]]; then
        echo "Application is HEALTHY!"
        break
    fi
    echo "Current health status: $STATUS (waiting...)"
done

echo "------------------------------------------------------------"
echo "Access Endpoints:"
echo "  Dashboard & UI:  http://localhost:8000"
echo "  Swagger Docs:    http://localhost:8000/docs"
echo "  Prometheus:      http://localhost:8000/metrics"
echo "  Health Check:    http://localhost:8000/health"
echo "------------------------------------------------------------"
