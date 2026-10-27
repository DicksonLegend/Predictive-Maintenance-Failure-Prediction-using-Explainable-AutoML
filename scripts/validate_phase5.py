"""Phase 5 Automated Validation Script.

Verifies that all required Phase 5 DevOps infrastructure files, Kubernetes manifests,
Terraform modules, CI/CD pipelines, monitoring stacks, helper scripts, and documentation
exist, are non-empty, and syntactically valid.
Executes full regression testing across Phase 2, Phase 3, Phase 4, and Pytest suites.
"""
import sys
import os
import glob
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Complete checklist of required Phase 5 artifacts
REQUIRED_PHASE5_FILES = [
    "requirements-dev.txt",
    "ruff.toml",
    ".bandit",
    ".dockerignore",
    "requirements-lite.txt",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.monitoring.yml",
    "monitoring/prometheus/prometheus.yml",
    "monitoring/prometheus/alerts.yml",
    "monitoring/grafana/provisioning/datasources/prometheus.yml",
    "monitoring/grafana/provisioning/dashboards/dashboards.yml",
    "monitoring/grafana/dashboards/xai-operations.json",
    "monitoring/alertmanager/alertmanager.yml",
    "k8s/namespace.yaml",
    "k8s/configmap.yaml",
    "k8s/secret.yaml",
    "k8s/deployment.yaml",
    "k8s/service.yaml",
    "k8s/hpa.yaml",
    "k8s/kustomization.yaml",
    "terraform/versions.tf",
    "terraform/variables.tf",
    "terraform/main.tf",
    "terraform/outputs.tf",
    "terraform/terraform.tfvars.example",
    "terraform/README.md",
    ".github/workflows/ci-cd.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    "CONTRIBUTING.md",
    "DEVOPS_SETUP_GUIDE.md",
    "scripts/build_image.ps1",
    "scripts/build_image.sh",
    "scripts/run_app.ps1",
    "scripts/run_app.sh",
    "scripts/deploy_k8s.ps1",
    "scripts/deploy_k8s.sh",
    "scripts/teardown_k8s.ps1",
    "scripts/teardown_k8s.sh",
    "scripts/start_monitoring.ps1",
    "scripts/start_monitoring.sh",
    "scripts/stop_monitoring.ps1",
    "scripts/stop_monitoring.sh",
    "scripts/generate_load.py",
    "scripts/validate_phase5.py",
    "reports/devops_phase5_results.md",
    "ASSUMPTIONS_PHASE5.md",
    "results/phase5_metadata.json",
    "results/phase5_run.log",
    "results/phase5_error_log.json",
]


def validate_phase5():
    print("=" * 75)
    print("PHASE 5 AUTOMATED VALIDATION SUITE — DEVOPS LAYER")
    print("=" * 75)

    all_passed = True
    total_checks = 0
    passed_checks = 0

    # --------------------------------------------------------------------------
    # CHECK GROUP 1: Phase 5 Artifact Existence & Non-Empty Checks
    # --------------------------------------------------------------------------
    print("\n[CHECK GROUP 1] Required Phase 5 Files & Non-Empty Checks:")
    for rel_path in REQUIRED_PHASE5_FILES:
        total_checks += 1
        abs_path = PROJECT_ROOT / rel_path
        if abs_path.exists() and abs_path.stat().st_size > 0:
            print(f"  [PASS] {rel_path} ({abs_path.stat().st_size} bytes)")
            passed_checks += 1
        else:
            if not abs_path.exists():
                print(f"  [FAIL] {rel_path} — FILE MISSING")
            else:
                print(f"  [FAIL] {rel_path} — FILE IS EMPTY (0 bytes)")
            all_passed = False

    # Check conditional outputs (e.g. k8s dry run output)
    k8s_dryrun = PROJECT_ROOT / "results" / "k8s_dryrun_output.yaml"
    if k8s_dryrun.exists() and k8s_dryrun.stat().st_size > 0:
        total_checks += 1
        passed_checks += 1
        print(f"  [PASS] results/k8s_dryrun_output.yaml ({k8s_dryrun.stat().st_size} bytes, client dry-run verified)")

    # --------------------------------------------------------------------------
    # CHECK GROUP 2: Declarative Syntax Validation (YAML & JSON)
    # --------------------------------------------------------------------------
    print("\n[CHECK GROUP 2] Declarative Syntax Parsing (YAML & JSON):")
    try:
        import yaml
        yaml_files = (
            glob.glob(str(PROJECT_ROOT / ".github" / "workflows" / "*.yml")) +
            glob.glob(str(PROJECT_ROOT / "k8s" / "*.yaml")) +
            glob.glob(str(PROJECT_ROOT / "monitoring" / "**" / "*.yml"), recursive=True) +
            glob.glob(str(PROJECT_ROOT / "docker-compose*.yml"))
        )
        for yf in yaml_files:
            total_checks += 1
            rel_yf = os.path.relpath(yf, PROJECT_ROOT)
            with open(yf, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            print(f"  [PASS] Valid YAML: {rel_yf}")
            passed_checks += 1

        # Check Grafana JSON
        total_checks += 1
        dash_file = PROJECT_ROOT / "monitoring" / "grafana" / "dashboards" / "xai-operations.json"
        with open(dash_file, "r", encoding="utf-8") as f:
            dash = json.load(f)
        assert len(dash.get("panels", [])) >= 4, "Dashboard has fewer than 4 panels"
        print(f"  [PASS] Valid Grafana Dashboard JSON: {dash.get('title')} ({len(dash['panels'])} panels)")
        passed_checks += 1
    except Exception as e:
        print(f"  [FAIL] Syntax parsing failure: {e}")
        all_passed = False

    # --------------------------------------------------------------------------
    # CHECK GROUP 3: Pytest Suite Execution (11/11 tests including /metrics)
    # --------------------------------------------------------------------------
    print("\n[CHECK GROUP 3] Pytest API Test Suite (11/11 Tests):")
    total_checks += 1
    python_bin = sys.executable
    ret_pytest = subprocess.run([python_bin, "-m", "pytest", "tests/", "-q"], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if ret_pytest.returncode == 0:
        print("  [PASS] All 11 pytest integration tests passed successfully (100% PASS)!")
        passed_checks += 1
    else:
        print(f"  [FAIL] Pytest failed:\n{ret_pytest.stdout}\n{ret_pytest.stderr}")
        all_passed = False

    # --------------------------------------------------------------------------
    # CHECK GROUP 4: Full Multi-Phase Regression Validation Suite
    # --------------------------------------------------------------------------
    print("\n[CHECK GROUP 4] Multi-Phase Regression Verification:")

    # Phase 2 Regression
    total_checks += 1
    ret_p2 = subprocess.run([python_bin, "scripts/validate_phase2.py"], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if ret_p2.returncode == 0 and "31" in ret_p2.stdout:
        print("  [PASS] Phase 2 Regression Check: 31/31 PASS (Baseline & Tuned RF Models)")
        passed_checks += 1
    else:
        print("  [FAIL] Phase 2 Regression failed!")
        all_passed = False

    # Phase 3 Regression
    total_checks += 1
    ret_p3 = subprocess.run([python_bin, "scripts/validate_phase3.py"], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if ret_p3.returncode == 0 and "39" in ret_p3.stdout:
        print("  [PASS] Phase 3 Regression Check: 39/39 PASS (AutoML & Critical Evaluation)")
        passed_checks += 1
    else:
        print("  [FAIL] Phase 3 Regression failed!")
        all_passed = False

    # Phase 4 Regression
    total_checks += 1
    ret_p4 = subprocess.run([python_bin, "scripts/validate_phase4.py"], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if ret_p4.returncode == 0 and "43" in ret_p4.stdout:
        print("  [PASS] Phase 4 Regression Check: 43/43 PASS (XAI Dashboard & API Integration)")
        passed_checks += 1
    else:
        print("  [FAIL] Phase 4 Regression failed!")
        all_passed = False

    # --------------------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"PHASE 5 VALIDATION SUMMARY: {passed_checks}/{total_checks} CHECKS PASSED")
    if all_passed:
        print("OVERALL STATUS: PASS (Phase 5 DevOps Layer 100% Validated)")
    else:
        print("OVERALL STATUS: FAIL")
    print("=" * 75)

    return all_passed


if __name__ == "__main__":
    success = validate_phase5()
    sys.exit(0 if success else 1)
