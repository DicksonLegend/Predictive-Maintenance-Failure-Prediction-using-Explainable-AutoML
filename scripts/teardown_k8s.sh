#!/usr/bin/env bash
# ==============================================================================
# Tears down the XAI Predictive Maintenance Kubernetes deployment (Bash mirror)
# ==============================================================================
set -euo pipefail

echo "Tearing down Kubernetes resources for XAI Predictive Maintenance..."
kubectl delete -k k8s/ --ignore-not-found

echo "Kubernetes resources successfully removed."
