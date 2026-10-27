"""Script to generate all Explainable AI artifacts for Phase 4.

Supports:
  --mode fast   (50 background samples, 3 local sample test cases)
  --mode final  (200 background samples, 5 local sample test cases)
"""
import argparse
import sys
import os
import json
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai.tabular_explainer import generate_all_global_artifacts, generate_local_sample_artifacts
from src.xai.text_explainer import text_explainer_service
from src.xai.image_explainer import generate_image_demo_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_xai_artifacts")


def main():
    parser = argparse.ArgumentParser(description="Generate Explainable AI Artifacts")
    parser.add_argument("--mode", choices=["fast", "final"], default="fast", help="Execution mode (fast or final)")
    args = parser.parse_args()

    bg_size = 50 if args.mode == "fast" else 200
    num_samples = 3 if args.mode == "fast" else 5

    logger.info("==================================================")
    logger.info("GENERATING XAI ARTIFACTS — MODE: %s", args.mode.upper())
    logger.info("Background sample size: %d, Local sample cases: %d", bg_size, num_samples)
    logger.info("==================================================")

    # 1. Global Tabular Artifacts
    logger.info("[1/4] Generating global SHAP, Feature Importance, and PDP plots...")
    g_meta = generate_all_global_artifacts(sample_size=bg_size)
    logger.info("Global artifacts generated with top features: %s", g_meta.get("top_global_features"))

    # 2. Local Tabular Samples
    logger.info("[2/4] Generating local SHAP, LIME, and sample JSONs...")
    s_meta = generate_local_sample_artifacts(num_samples=num_samples)
    logger.info("Generated %d local sample cases.", len(s_meta))

    # 3. Text XAI Demo
    logger.info("[3/4] Generating NLP text maintenance log explanation artifacts...")
    text_explainer_service.generate_demo_artifacts()

    # 4. Image XAI Demo
    logger.info("[4/4] Generating multi-modal spatial saliency image artifacts...")
    generate_image_demo_artifacts()

    logger.info("==================================================")
    logger.info("ALL XAI ARTIFACTS SUCCESSFULLY GENERATED")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
