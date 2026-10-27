#!/usr/bin/env bash
# ==============================================================================
# Builds the Predictive Maintenance XAI Docker container image (Bash mirror)
# Usage: ./scripts/build_image.sh [--full]
# ==============================================================================
set -euo pipefail

IMAGE_TAG="xai-automl:lite"
INSTALL_AUTOML="false"

if [[ "${1:-}" == "--full" ]]; then
    IMAGE_TAG="xai-automl:full"
    INSTALL_AUTOML="true"
fi

echo "============================================================"
echo "Building Docker Image: ${IMAGE_TAG} (INSTALL_AUTOML=${INSTALL_AUTOML})"
echo "============================================================"

START_TIME=$(date +%s)

docker build --build-arg INSTALL_AUTOML="${INSTALL_AUTOML}" -t "${IMAGE_TAG}" -f Dockerfile .

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

SIZE_BYTES=$(docker image inspect "${IMAGE_TAG}" --format "{{.Size}}")
SIZE_MB=$(awk "BEGIN {printf \"%.1f\", ${SIZE_BYTES}/1048576}")

echo "------------------------------------------------------------"
echo "BUILD SUCCESSFUL!"
echo "Tag:      ${IMAGE_TAG}"
echo "Duration: ${DURATION}s"
echo "Size:     ${SIZE_MB} MB"
echo "------------------------------------------------------------"
