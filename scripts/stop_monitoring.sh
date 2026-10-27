#!/usr/bin/env bash
# ==============================================================================
# Stops the Prometheus, Grafana, and Alertmanager monitoring stack (Bash mirror)
# ==============================================================================
set -euo pipefail

echo "Stopping Industry 4.0 Monitoring Stack..."
docker compose -f docker-compose.monitoring.yml --profile alerting down

echo "Monitoring stack cleanly stopped."
