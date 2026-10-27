#!/usr/bin/env bash
# ==============================================================================
# Deploys the XAI Predictive Maintenance manifests to Kubernetes (Bash mirror)
# ==============================================================================
set -euo pipefail

echo "============================================================"
echo "Deploying XAI Predictive Maintenance to Kubernetes..."
echo "============================================================"

kubectl apply -k k8s/

echo "Waiting for deployment rollout to complete..."
kubectl -n xai-predictive-maintenance rollout status deployment/xai-api --timeout=180s

echo "Starting background port-forward (8000 -> 80)..."
kubectl -n xai-predictive-maintenance port-forward svc/xai-api 8000:80 >/dev/null 2>&1 &
PF_PID=$!

echo "------------------------------------------------------------"
echo "DEPLOYMENT SUCCESSFUL!"
echo "Cluster Resources:"
kubectl -n xai-predictive-maintenance get pods,svc,hpa
echo "------------------------------------------------------------"
echo "Access Endpoints (via port-forward PID ${PF_PID}):"
echo "  Dashboard & UI:  http://localhost:8000"
echo "  Health Status:   http://localhost:8000/health"
echo "  Prometheus:      http://localhost:8000/metrics"
echo "To stop port-forward, run: kill ${PF_PID}"
echo "------------------------------------------------------------"
