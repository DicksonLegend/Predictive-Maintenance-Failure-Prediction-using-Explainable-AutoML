"""Phase 2 Validation Script.

Verifies the existence and non-emptiness of all 31 required Phase 2 artifacts.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent

# Exact checklist of files required by Phase 2 specifications (31 items)
REQUIRED_FILES = [
    "src/models/__init__.py",
    "src/models/baseline.py",
    "src/models/tuning.py",
    "src/models/evaluation.py",
    "scripts/run_phase2.py",
    "results/phase2_metadata.json",
    "results/baseline_metrics.csv",
    "results/grid_search_results.csv",
    "results/random_search_results.csv",
    "results/optuna_results.csv",
    "results/tuning_comparison.csv",
    "results/best_params.json",
    "results/phase2_best_model.json",
    "results/optuna_trials.csv",
    "results/phase2_summary.md",
    "results/phase2_run.log",
    "models/baseline_rf.pkl",
    "models/grid_rf.pkl",
    "models/random_rf.pkl",
    "models/optuna_rf.pkl",
    "plots/baseline_confusion_matrix.png",
    "plots/best_tuned_confusion_matrix.png",
    "plots/tuning_f1_comparison.png",
    "plots/tuning_time_comparison.png",
    "plots/roc_curve_phase2.png",
    "reports/automl_phase2_results.md",
    "reports/screenshots/phase2_baseline_confusion_matrix.png",
    "reports/screenshots/phase2_best_tuned_confusion_matrix.png",
    "reports/screenshots/phase2_tuning_f1_comparison.png",
    "reports/screenshots/phase2_tuning_time_comparison.png",
    "reports/screenshots/phase2_roc_curve_phase2.png",
]


def validate_phase2() -> bool:
    """Validates existence and non-empty status of all required Phase 2 artifacts."""
    print("=" * 75)
    print("PHASE 2 ARTIFACT VALIDATION CHECKLIST")
    print("=" * 75)

    all_passed = True
    passed_count = 0

    for relative_path in REQUIRED_FILES:
        target_path = ROOT_DIR / relative_path
        exists = target_path.exists()
        is_non_empty = target_path.is_file() and target_path.stat().st_size > 0

        if exists and is_non_empty:
            status = "PASS"
            passed_count += 1
            print(f"[{status}]  {relative_path:<52} (size: {target_path.stat().st_size} bytes)")
        else:
            status = "FAIL"
            all_passed = False
            reason = "missing" if not exists else "empty file"
            print(f"[{status}]  {relative_path:<52} ({reason})")

    print("-" * 75)
    print(f"Total Items Checked : {len(REQUIRED_FILES)}")
    print(f"Passed              : {passed_count}")
    print(f"Failed              : {len(REQUIRED_FILES) - passed_count}")
    print("-" * 75)

    if all_passed:
        print("OVERALL STATUS: PASS (Phase 2 validation fully satisfied)")
        print("=" * 75)
        return True
    else:
        print("OVERALL STATUS: FAIL (Some Phase 2 artifacts are missing or empty)")
        print("=" * 75)
        return False


if __name__ == "__main__":
    success = validate_phase2()
    sys.exit(0 if success else 1)
