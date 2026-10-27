# ==============================================================================
# Multi-Stage Dockerfile for Predictive Maintenance XAI Application (Phase 5)
# Stage 1: Build React Single Page Application (frontend-builder)
# Stage 2: Install Python dependencies (python-deps, with optional AutoML build arg)
# Stage 3: Minimal, secure, non-root runtime container (runtime)
# ==============================================================================

ARG INSTALL_AUTOML=false

# ------------------------------------------------------------------------------
# Stage 1: Frontend Builder
# ------------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder
WORKDIR /fe
COPY app/frontend/package*.json ./
RUN npm ci || npm install
COPY app/frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Python Dependencies Builder
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS python-deps
ARG INSTALL_AUTOML
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /install
COPY requirements-lite.txt requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-lite.txt
RUN if [ "$INSTALL_AUTOML" = "true" ]; then pip install -r requirements.txt; fi

# ------------------------------------------------------------------------------
# Stage 3: Minimal Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Predictive Maintenance XAI Dashboard" \
      org.opencontainers.image.description="Explainable AutoML-based Machine Failure Prediction API and Dashboard" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="Dickson / MLOps Team"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    SERVED_MODEL_PREFERENCE="auto"

WORKDIR /app

# Create non-root user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser

# Copy installed Python packages from deps stage
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy application source code and artifacts
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY artifacts/ ./artifacts/
COPY data/processed/ ./data/processed/
COPY reports/model_card.md ./reports/model_card.md

# Copy compiled React frontend from stage 1
COPY --from=frontend-builder /fe/dist ./app/frontend/dist

# Setup logs and permissions
RUN mkdir -p /app/logs && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
