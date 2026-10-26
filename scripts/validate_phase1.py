"""Phase 1 Validation Script.

Verifies the presence and integrity of all required Phase 1 files and artifacts.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent

# Exact checklist of files required by Phase 1 specifications
REQUIRED_FILES = [
    "PROJECT_SPEC.md",
    "PLAN.md",
    "README.md",
    "ASSUMPTIONS.md",
    "requirements.txt",
    ".gitignore",
    "data/processed/train_manual.csv",
    "data/processed/test_manual.csv",
    "data/processed/train_automl.csv",
    "data/processed/test_automl.csv",
    "data/processed/metadata.json",
    "results/dataset_summary.csv",
    "results/missing_values.csv",
    "results/class_distribution.csv",
    "results/feature_summary.csv",
    "plots/class_distribution.png",
    "plots/correlation_heatmap.png",
    "plots/missing_values.png",
    "plots/feature_distributions.png",
    "reports/automl_report_outline.md",
    "reports/xai_report_outline.md",
]


def validate_phase1() -> bool:
    """Validates existence of each required Phase 1 artifact and prints status."""
    print("=" * 70)
    print("PHASE 1 ARTIFACT VALIDATION CHECKLIST")
    print("=" * 70)

    all_passed = True
    passed_count = 0

    for relative_path in REQUIRED_FILES:
        target_path = ROOT_DIR / relative_path
        exists = target_path.exists()
        is_non_empty = target_path.is_file() and target_path.stat().st_size > 0

        if exists and is_non_empty:
            status = "PASS"
            passed_count += 1
            print(f"[{status}]  {relative_path:<45} (size: {target_path.stat().st_size} bytes)")
        else:
            status = "FAIL"
            all_passed = False
            reason = "missing" if not exists else "empty file"
            print(f"[{status}]  {relative_path:<45} ({reason})")

    print("-" * 70)
    print(f"Total Items Checked : {len(REQUIRED_FILES)}")
    print(f"Passed              : {passed_count}")
    print(f"Failed              : {len(REQUIRED_FILES) - passed_count}")
    print("-" * 70)

    if all_passed:
        print("OVERALL STATUS: PASS (Phase 1 validation fully satisfied)")
        print("=" * 70)
        return True
    else:
        print("OVERALL STATUS: FAIL (Some Phase 1 artifacts are missing)")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = validate_phase1()
    sys.exit(0 if success else 1)
