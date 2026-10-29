#!/usr/bin/env bash
# Comprehensive Teardown Script for Phase 5.5 Local Stack (Bash)
# DO NOT RUN AUTOMATICALLY: Keep stack active until UI screenshots are captured.

set -euo pipefail

echo "========================================================"
echo "Teardown: Predictive Maintenance DevOps Stack"
echo "========================================================"

echo "[1/3] Terminating port-forward processes..."
pkill -f "kubectl.*port-forward" || true
pkill -f "port_forward_supervisor" || true

echo "[2/3] Deleting Kubernetes namespace and resources..."
kubectl delete -k k8s/ --ignore-not-found

echo "[3/3] Destroying Terraform provisioned monitoring containers..."
terraform -chdir=terraform destroy -auto-approve

echo "All local Phase 5.5 resources have been safely torn down."
