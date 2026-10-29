"""Streams docker save of xai-automl:lite directly into KinD containerd."""
import subprocess
import time

print("Saving xai-automl:lite and importing into desktop-control-plane (containerd k8s.io namespace)...")
t0 = time.perf_counter()

save_proc = subprocess.Popen(["docker", "save", "xai-automl:lite"], stdout=subprocess.PIPE)
load_proc = subprocess.Popen(["docker", "exec", "-i", "desktop-control-plane", "ctr", "-n", "k8s.io", "images", "import", "-"], stdin=save_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
save_proc.stdout.close()

out, err = load_proc.communicate()
duration = time.perf_counter() - t0
print(f"Import completed in {duration:.1f}s. Returncode: {load_proc.returncode}")
if out:
    print(out.decode("utf-8", errors="replace"))
if err:
    print(err.decode("utf-8", errors="replace"))

# Verify image in KinD
chk = subprocess.run(["docker", "exec", "desktop-control-plane", "crictl", "images"], capture_output=True, text=True)
for line in chk.stdout.splitlines():
    if "xai-automl" in line:
        print("Found in KinD:", line)
