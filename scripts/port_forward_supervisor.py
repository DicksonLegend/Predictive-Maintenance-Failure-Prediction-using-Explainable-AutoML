"""Supervises kubectl port-forward svc/xai-api 8000:80, auto-restarting if dropped."""
import subprocess
import time
import sys

print("Starting kubectl port-forward supervisor on host port 8000 -> svc/xai-api:80...")
while True:
    try:
        proc = subprocess.Popen(
            ["kubectl", "-n", "xai-predictive-maintenance", "port-forward", "svc/xai-api", "8000:80"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print("Port-forward child process started (PID:", proc.pid, ")")
        for line in proc.stdout:
            if "Forwarding from" in line:
                print(line.strip())
        proc.wait()
        print("Port-forward disconnected, reconnecting in 1s...")
        time.sleep(1)
    except Exception as e:
        print("Supervisor error:", e)
        time.sleep(2)
