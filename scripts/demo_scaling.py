"""Demonstrates manual scaling, load generation, HPA sampling, and scaling back."""
import subprocess
import time
import sys
from pathlib import Path

evidence_dir = Path("results/phase55_evidence")
evidence_dir.mkdir(parents=True, exist_ok=True)

print("1. Scaling deployment/xai-api to 3 replicas...")
scale3_res = subprocess.run(
    ["kubectl", "-n", "xai-predictive-maintenance", "scale", "deployment/xai-api", "--replicas=3"],
    capture_output=True, text=True
)
print(scale3_res.stdout)
(evidence_dir / "05_scaling_scale_to_3.txt").write_text(scale3_res.stdout, encoding="utf-8")

# Wait for 3 pods to be ready
subprocess.run(
    ["kubectl", "-n", "xai-predictive-maintenance", "rollout", "status", "deployment/xai-api", "--timeout=120s"],
    capture_output=True, text=True
)

print("2. Capturing 3 running pods...")
pods_res = subprocess.run(
    ["kubectl", "-n", "xai-predictive-maintenance", "get", "pods", "-o", "wide"],
    capture_output=True, text=True
)
print(pods_res.stdout)
(evidence_dir / "05_scaling_pods_3replicas.txt").write_text(pods_res.stdout, encoding="utf-8")

print("3. Inspecting initial HPA status...")
hpa_before = subprocess.run(
    ["kubectl", "-n", "xai-predictive-maintenance", "get", "hpa"],
    capture_output=True, text=True
)
print(hpa_before.stdout)
(evidence_dir / "05_scaling_hpa_before.txt").write_text(hpa_before.stdout, encoding="utf-8")

print("4. Launching load generation (400 requests, concurrency 10)...")
t0 = time.perf_counter()
load_res = subprocess.run(
    [sys.executable, "scripts/generate_load.py", "--requests", "400", "--concurrency", "10", "--url", "http://localhost:8000"],
    capture_output=True, text=True
)
load_duration = time.perf_counter() - t0
print(load_res.stdout)
(evidence_dir / "05_scaling_load.txt").write_text(load_res.stdout, encoding="utf-8")

print("5. Sampling HPA and pod resource usage 6 times (10s intervals)...")
sample_logs = []
for i in range(1, 7):
    ts = time.strftime("%H:%M:%S")
    hpa_sample = subprocess.run(["kubectl", "-n", "xai-predictive-maintenance", "get", "hpa"], capture_output=True, text=True)
    top_sample = subprocess.run(["kubectl", "top", "pods", "-n", "xai-predictive-maintenance"], capture_output=True, text=True)
    
    header = f"=== SAMPLE {i}/6 at {ts} ==="
    print(header)
    print("HPA:\n", hpa_sample.stdout.strip())
    print("Top Pods:\n", top_sample.stdout.strip())
    
    sample_logs.append(f"{header}\n[HPA]\n{hpa_sample.stdout}\n[TOP PODS]\n{top_sample.stdout}\n")
    if i < 6:
        time.sleep(10)

(evidence_dir / "05_scaling_hpa_samples.txt").write_text("\n".join(sample_logs), encoding="utf-8")

print("6. Scaling back deployment/xai-api to 2 replicas...")
scale2_res = subprocess.run(
    ["kubectl", "-n", "xai-predictive-maintenance", "scale", "deployment/xai-api", "--replicas=2"],
    capture_output=True, text=True
)
print(scale2_res.stdout)
(evidence_dir / "05_scaling_scaled_back.txt").write_text(scale2_res.stdout, encoding="utf-8")

# Capture final pods
final_pods = subprocess.run(
    ["kubectl", "-n", "xai-predictive-maintenance", "get", "pods", "-o", "wide"],
    capture_output=True, text=True
)
print("Final Pods:\n", final_pods.stdout)
(evidence_dir / "05_scaling_final_pods.txt").write_text(final_pods.stdout, encoding="utf-8")

print("Step 5 scaling verification complete!")
