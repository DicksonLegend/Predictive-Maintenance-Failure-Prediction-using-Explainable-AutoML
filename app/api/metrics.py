"""Prometheus Metrics definitions for Phase 5 DevOps monitoring."""
from prometheus_client import Counter

# Custom business metric: tracking failure predictions across operational risk bands
xai_predictions_total = Counter(
    "xai_predictions_total",
    "Total count of machine failure inference predictions grouped by risk level",
    ["risk_level"]
)
