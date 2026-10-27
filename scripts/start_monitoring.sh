#!/usr/bin/env bash
# ==============================================================================
# Starts the Prometheus, Grafana, and Alertmanager monitoring stack (Bash mirror)
# ==============================================================================
set -euo pipefail

echo "============================================================"
echo "Starting Industry 4.0 Monitoring Stack (Prometheus + Grafana)..."
echo "============================================================"

docker compose -f docker-compose.monitoring.yml --profile alerting up -d

echo "------------------------------------------------------------"
echo "MONITORING STACK ONLINE!"
echo "Access Points:"
echo "  Grafana Dashboard:  http://localhost:3000"
echo "  Credentials:        admin / admin"
echo "  Prometheus UI:      http://localhost:9090"
echo "  Alertmanager UI:    http://localhost:9093"
echo "------------------------------------------------------------"
