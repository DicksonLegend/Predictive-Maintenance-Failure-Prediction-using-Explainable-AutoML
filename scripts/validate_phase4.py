"""Phase 4 Automated Validation Script.

Verifies that all required Phase 4 application code, XAI modules, static/React
frontend assets, artifacts, test suites, and academic reports exist and are non-empty.
Executes the API test suite and outputs granular PASS/FAIL status.
"""
import sys
import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_FILES = [
    "app/__init__.py",
    "app/api/main.py",
    "app/api/config.py",
    "app/api/schemas.py",
    "app/api/model_service.py",
    "app/api/xai_service.py",
    "app/api/text_demo_service.py",
    "app/api/image_demo_service.py",
    "app/api/recommendation_service.py",
    "src/xai/tabular_explainer.py",
    "src/xai/text_explainer.py",
    "src/xai/image_explainer.py",
    "src/xai/what_if.py",
    "scripts/generate_xai_artifacts.py",
    "scripts/build_frontend.py",
    "scripts/run_phase4.py",
    "scripts/validate_phase4.py",
    "tests/test_api.py",
    "results/phase4_metadata.json",
    "results/phase4_run.log",
    "results/phase4_error_log.json",
    "artifacts/model_card.json",
    "reports/model_card.md",
    "reports/xai_phase4_results.md",
    "reports/automl_phase4_deployment.md",
    "reports/screenshots/phase4_ui_screenshot_instructions.md",
    "artifacts/xai/global/shap_summary.png",
    "artifacts/xai/global/shap_bar.png",
    "artifacts/xai/global/feature_importance.png",
    "artifacts/xai/global/pdp_torque.png",
    "artifacts/xai/global/pdp_tool_wear.png",
    "artifacts/xai/global/pdp_rotational_speed.png",
    "artifacts/xai/global/pdp_process_temp.png",
    "artifacts/xai/global/global_explanation.json",
    "artifacts/xai/samples/sample_0.json",
    "artifacts/xai/local/shap_local_sample_0.png",
    "artifacts/xai/local/lime_local_sample_0.png",
    "artifacts/xai/text_demo/text_sample_0.json",
    "artifacts/xai/image_demo/image_0_metadata.json",
    "models/text_demo_tfidf.pkl",
    "models/text_demo_logreg.pkl",
]


def validate_phase4():
    print("=" * 70)
    print("PHASE 4 AUTOMATED VALIDATION SUITE")
    print("=" * 70)

    all_passed = True
    check_count = 0
    pass_count = 0

    # 1. Check all required files
    print("\n[CHECK GROUP 1] Required Files & Non-Empty Checks:")
    for rel_path in REQUIRED_FILES:
        check_count += 1
        abs_path = PROJECT_ROOT / rel_path
        if abs_path.exists() and abs_path.stat().st_size > 0:
            print(f"  [PASS] {rel_path} ({abs_path.stat().st_size} bytes)")
            pass_count += 1
        else:
            if not abs_path.exists():
                print(f"  [FAIL] {rel_path} — FILE MISSING")
            else:
                print(f"  [FAIL] {rel_path} — FILE IS EMPTY (0 bytes)")
            all_passed = False

    # 2. Dynamic Frontend Mode Check
    print("\n[CHECK GROUP 2] Dynamic Frontend Mode Verification:")
    check_count += 1
    meta_path = PROJECT_ROOT / "results" / "phase4_metadata.json"
    frontend_mode = "unknown"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            frontend_mode = meta.get("frontend_mode", "unknown")
        except Exception:
            pass

    if frontend_mode == "react":
        dist_index = PROJECT_ROOT / "app" / "frontend" / "dist" / "index.html"
        if dist_index.exists() and dist_index.stat().st_size > 0:
            print(f"  [PASS] Frontend mode is 'react': app/frontend/dist/index.html verified ({dist_index.stat().st_size} bytes)")
            pass_count += 1
        else:
            print("  [FAIL] Frontend mode is 'react' but app/frontend/dist/index.html is missing or empty!")
            all_passed = False
    elif frontend_mode == "static_fallback":
        static_index = PROJECT_ROOT / "app" / "static_fallback" / "index.html"
        if static_index.exists() and static_index.stat().st_size > 0:
            print(f"  [PASS] Frontend mode is 'static_fallback': app/static_fallback/index.html verified ({static_index.stat().st_size} bytes)")
            pass_count += 1
        else:
            print("  [FAIL] Frontend mode is 'static_fallback' but app/static_fallback/index.html is missing or empty!")
            all_passed = False
    else:
        # Check if either dist/index.html or static_fallback/index.html exists
        dist_index = PROJECT_ROOT / "app" / "frontend" / "dist" / "index.html"
        static_index = PROJECT_ROOT / "app" / "static_fallback" / "index.html"
        if dist_index.exists() and dist_index.stat().st_size > 0:
            print(f"  [PASS] Detected compiled React frontend at: app/frontend/dist/index.html ({dist_index.stat().st_size} bytes)")
            pass_count += 1
        elif static_index.exists() and static_index.stat().st_size > 0:
            print(f"  [PASS] Detected static fallback frontend at: app/static_fallback/index.html ({static_index.stat().st_size} bytes)")
            pass_count += 1
        else:
            print("  [FAIL] No valid frontend index.html detected in either dist/ or static_fallback/")
            all_passed = False

    # 3. Execute FastAPI TestClient API Tests
    print("\n[CHECK GROUP 3] Executing API Integration Tests (TestClient):")
    check_count += 1
    try:
        from fastapi.testclient import TestClient
        from app.api.main import app

        client = TestClient(app)

        # Run all 10 tests
        r1 = client.get("/health")
        assert r1.status_code == 200 and r1.json()["status"] == "ok"

        r2 = client.get("/version")
        assert r2.status_code == 200

        r3 = client.get("/model-card")
        assert r3.status_code == 200 and "dataset" in r3.json()

        r4 = client.get("/global-explanations")
        assert r4.status_code == 200 and "top_global_features" in r4.json()

        r5 = client.get("/samples")
        assert r5.status_code == 200 and len(r5.json()) >= 1

        r6 = client.post("/predict", json={
            "type": "L", "air_temp": 300.0, "process_temp": 310.0,
            "rotational_speed": 1500.0, "torque": 40.0, "tool_wear": 50.0
        })
        assert r6.status_code == 200 and 0.0 <= r6.json()["prediction"]["failure_probability"] <= 1.0

        r7 = client.post("/explain", json={
            "type": "M", "air_temp": 302.5, "process_temp": 311.0,
            "rotational_speed": 1350.0, "torque": 65.0, "tool_wear": 210.0
        })
        assert r7.status_code == 200 and "top_reasons" in r7.json()

        r8 = client.post("/what-if", json={
            "original_input": {"type": "L", "air_temp": 302.0, "process_temp": 311.0, "rotational_speed": 1350.0, "torque": 68.0, "tool_wear": 200.0},
            "modified_input": {"type": "L", "air_temp": 300.0, "process_temp": 310.0, "rotational_speed": 1500.0, "torque": 40.0, "tool_wear": 20.0}
        })
        assert r8.status_code == 200 and "probability_change" in r8.json()

        r9 = client.post("/text-explain", json={"text": "Machine vibrating heavily with screeching bearing noise."})
        assert r9.status_code == 200 and "predicted_label" in r9.json()

        r10 = client.get("/image-demo")
        assert r10.status_code == 200 and len(r10.json()["items"]) >= 1

        print("  [PASS] All 10 FastAPI TestClient API tests passed successfully!")
        pass_count += 1
    except Exception as e:
        print(f"  [FAIL] API integration tests failed: {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 70)
    print(f"PHASE 4 VALIDATION SUMMARY: {pass_count}/{check_count} CHECKS PASSED")
    if all_passed:
        print("OVERALL STATUS: PASS (Phase 4 100% Complete & Validated)")
    else:
        print("OVERALL STATUS: FAIL")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    passed = validate_phase4()
    sys.exit(0 if passed else 1)
