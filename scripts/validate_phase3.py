"""Phase 3 Validation Script.

Verifies the existence and non-emptiness of all mandatory and dynamic Phase 3 artifacts.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent

# Core static Phase 3 files (36 items)
STATIC_REQUIRED_FILES = [
    "src/automl/__init__.py",
    "src/automl/common.py",
    "src/automl/flaml_runner.py",
    "src/automl/leakage.py",
    "src/automl/critical_evaluation.py",
    "src/automl/feature_importance.py",
    "scripts/run_phase3.py",
    "results/phase3_metadata.json",
    "results/phase3_run.log",
    "results/tool_error_log.json",
    "results/flaml_leaderboard.csv",
    "results/flaml_metrics.csv",
    "results/flaml_best_config.json",
    "models/flaml_model.pkl",
    "results/automl_comparison.csv",
    "results/phase3_full_comparison.csv",
    "results/overfitting_check.csv",
    "results/leakage_results.csv",
    "results/leakage_metadata.json",
    "results/class_imbalance_report.csv",
    "results/group_performance.csv",
    "results/interpretability_notes.csv",
    "results/phase3_best_model.json",
    "results/phase3_test_predictions.csv",
    "results/phase3_summary.md",
    "reports/automl_phase3_results.md",
    "plots/phase3_model_comparison_f1.png",
    "plots/phase3_time_comparison.png",
    "plots/phase3_full_comparison.png",
    "plots/overfitting_check.png",
    "plots/leakage_metric_comparison.png",
    "plots/leakage_feature_importance.png",
    "plots/phase3_class_distribution.png",
    "plots/phase3_confusion_matrix_best_model.png",
    "plots/phase3_precision_recall_curve.png",
    "plots/group_performance.png",
]


def validate_phase3() -> bool:
    """Validates presence and integrity of Phase 3 artifacts."""
    print("=" * 75)
    print("PHASE 3 ARTIFACT VALIDATION CHECKLIST")
    print("=" * 75)

    all_passed = True
    passed_count = 0
    total_checked = 0

    # 1. Check Static Artifacts
    for relative_path in STATIC_REQUIRED_FILES:
        total_checked += 1
        target_path = ROOT_DIR / relative_path
        exists = target_path.exists()
        is_non_empty = target_path.is_file() and target_path.stat().st_size > 0

        if exists and is_non_empty:
            passed_count += 1
            print(f"[PASS]  {relative_path:<52} (size: {target_path.stat().st_size} bytes)")
        else:
            all_passed = False
            reason = "missing" if not exists else "empty file"
            print(f"[FAIL]  {relative_path:<52} ({reason})")

    # 2. Check Dynamic Second Tool Artifacts
    meta_path = ROOT_DIR / "results" / "phase3_metadata.json"
    second_tool = "AutoGluon"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                second_tool = meta.get("second_tool_name", "AutoGluon")
        except Exception:
            pass

    print("-" * 75)
    print(f"Checking Dynamic Second Tool Artifacts (Second Tool: {second_tool}):")

    dynamic_items = []
    if second_tool == "AutoGluon":
        dynamic_items = [
            ("results/autogluon_leaderboard.csv", "file"),
            ("results/autogluon_metrics.csv", "file"),
            ("models/autogluon_predictor", "dir"),
        ]
    else:
        dynamic_items = [
            ("results/pycaret_leaderboard.csv", "file"),
            ("results/pycaret_metrics.csv", "file"),
            ("models/pycaret_model.pkl", "file"),
        ]

    for rel_path, item_type in dynamic_items:
        total_checked += 1
        target = ROOT_DIR / rel_path
        exists = target.exists()

        if item_type == "file":
            valid = exists and target.is_file() and target.stat().st_size > 0
            size_str = f"size: {target.stat().st_size} bytes" if exists else "missing"
        else:  # dir
            # Ensure directory exists and has at least one child
            valid = exists and target.is_dir() and any(target.iterdir())
            count = len(list(target.iterdir())) if exists and target.is_dir() else 0
            size_str = f"directory containing {count} files" if exists else "missing"

        if valid:
            passed_count += 1
            print(f"[PASS]  {rel_path:<52} ({size_str})")
        else:
            all_passed = False
            reason = "missing" if not exists else "empty"
            print(f"[FAIL]  {rel_path:<52} ({reason})")

    print("-" * 75)
    print(f"Total Items Checked : {total_checked}")
    print(f"Passed              : {passed_count}")
    print(f"Failed              : {total_checked - passed_count}")
    print("-" * 75)

    if all_passed:
        print("OVERALL STATUS: PASS (Phase 3 validation fully satisfied)")
        print("=" * 75)
        return True
    else:
        print("OVERALL STATUS: FAIL (Some Phase 3 artifacts are missing or empty)")
        print("=" * 75)
        return False


if __name__ == "__main__":
    success = validate_phase3()
    sys.exit(0 if success else 1)
