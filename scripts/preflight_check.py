"""Pre-flight system environment check for Phase 5.5."""
import subprocess
from pathlib import Path

out_dir = Path("results/phase55_evidence")
out_dir.mkdir(parents=True, exist_ok=True)

commands = [
    ("docker version --format \"{{.Server.Version}}\"", "00_preflight_docker_version.txt"),
    ("docker info --format \"{{.MemTotal}}\"", "00_preflight_docker_mem.txt"),
    ("kubectl get nodes -o wide", "00_preflight_k8s_nodes.txt"),
    ("kubectl cluster-info", "00_preflight_k8s_cluster_info.txt"),
    ("kubectl version", "00_preflight_k8s_version.txt"),
    ("terraform version", "00_preflight_terraform_version.txt"),
    ("node --version", "00_preflight_node_version.txt"),
    ("npm --version", "00_preflight_npm_version.txt"),
]

print("=== STEP 0: PRE-FLIGHT SYSTEM CHECKS ===")
for cmd, filename in commands:
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        output = (res.stdout.strip() + "\n" + res.stderr.strip()).strip()
        (out_dir / filename).write_text(output + "\n", encoding="utf-8")
        print(f"[{cmd}] -> {filename}")
        print(output)
        print("-" * 50)
    except Exception as e:
        err_msg = f"ERROR executing {cmd}: {e}"
        (out_dir / filename).write_text(err_msg + "\n", encoding="utf-8")
        print(err_msg)
        print("-" * 50)
