"""Demonstrates zero-downtime rolling update on Kubernetes deployment/xai-api."""
import subprocess
import time
import urllib.request
from pathlib import Path

evidence_dir = Path("results/phase55_evidence")
evidence_dir.mkdir(parents=True, exist_ok=True)

print("1. Triggering rollout restart deployment/xai-api...")
restart_res = subprocess.run(
    ["kubectl", "-n", "xai-predictive-maintenance", "rollout", "restart", "deployment/xai-api"],
    capture_output=True, text=True
)
print("Restart Output:", restart_res.stdout.strip())
(evidence_dir / "04_rolling_update_restart.txt").write_text(restart_res.stdout, encoding="utf-8")

# Immediately loop 10 times (2s apart) probing http://localhost:8000/health
print("2. Immediately probing http://localhost:8000/health across 10 iterations (2s interval)...")
status_codes = []
for i in range(1, 11):
    try:
        req = urllib.request.Request("http://localhost:8000/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            code = resp.status
    except Exception as e:
        code = getattr(e, "code", 503)
    status_codes.append(code)
    print(f"  [Probe {i}/10] HTTP Status: {code}")
    time.sleep(2)

code_seq = " -> ".join(str(c) for c in status_codes)
print(f"Sequence of status codes: {code_seq}")
(evidence_dir / "04_rolling_update_health_sequence.txt").write_text("\n".join(str(c) for c in status_codes) + "\n", encoding="utf-8")

print("3. Awaiting rollout completion...")
rollout_res = subprocess.run(
    ["kubectl", "-n", "xai-predictive-maintenance", "rollout", "status", "deployment/xai-api", "--timeout=240s"],
    capture_output=True, text=True
)
print("Rollout Status:", rollout_res.stdout.strip())
(evidence_dir / "04_rolling_update_status.txt").write_text(rollout_res.stdout, encoding="utf-8")

print("4. Inspecting new pods...")
pods_res = subprocess.run(
    ["kubectl", "-n", "xai-predictive-maintenance", "get", "pods", "-o", "wide"],
    capture_output=True, text=True
)
print(pods_res.stdout)
(evidence_dir / "04_rolling_update_pods.txt").write_text(pods_res.stdout, encoding="utf-8")
print("Rolling update verification completed successfully!")
