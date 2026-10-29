"""Builds Docker lite image, captures build duration and full logs cleanly."""
import subprocess
import time
import sys
from pathlib import Path

log_path = Path("results/phase55_evidence/01_docker_build.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

print("Starting Docker build for xai-automl:lite...")
start_time = time.perf_counter()

proc = subprocess.run(
    ["docker", "build", "--build-arg", "INSTALL_AUTOML=false", "-t", "xai-automl:lite", "-f", "Dockerfile", "."],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
)

duration = time.perf_counter() - start_time
full_output = proc.stdout.decode("utf-8", errors="replace")
log_path.write_text(full_output, encoding="utf-8")

print(f"Docker build finished in {duration:.1f} seconds. Exit code: {proc.returncode}")

if proc.returncode == 0:
    inspect_res = subprocess.run(
        ["docker", "images", "xai-automl:lite", "--format", "{{.Repository}}:{{.Tag}} {{.Size}} {{.CreatedSince}}"],
        capture_output=True,
        text=True
    )
    img_info = inspect_res.stdout.strip()
    print("Image Info:", img_info)
    Path("results/phase55_evidence/01_docker_image_info.txt").write_text(img_info + "\n", encoding="utf-8")
    with open("results/phase55_evidence/01_docker_duration.txt", "w", encoding="utf-8") as f:
        f.write(f"{duration:.1f}\n")
else:
    print("Build failed! Last 20 lines of log:")
    print("\n".join(full_output.splitlines()[-20:]))
    sys.exit(proc.returncode)
