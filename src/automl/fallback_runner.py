"""Fallback AutoML runner using PyCaret in case AutoGluon is unavailable."""

import time
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd

from src.automl.common import compute_automl_metrics
from src.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR, TARGET_COLUMN
from src.utils import set_random_seed, setup_logger

logger = setup_logger("fallback_runner")


def run_pycaret_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mode: str = "fast",
    time_budget_seconds: int = 60,
    random_state: int = RANDOM_SEED,
    model_save_path: Path = MODELS_DIR / "pycaret_model.pkl",
    leaderboard_save_path: Path = RESULTS_DIR / "pycaret_leaderboard.csv",
    metrics_save_path: Path = RESULTS_DIR / "pycaret_metrics.csv",
) -> Tuple[Any, Dict[str, Any]]:
    """Executes PyCaret classification search as a fallback AutoML engine."""
    set_random_seed(random_state)
    logger.info(f"Starting PyCaret classification fallback search (mode={mode})...")

    from pycaret.classification import (
        ClassificationExperiment,
        compare_models,
        predict_model,
        pull,
        setup,
    )

    train_data = X_train.copy()
    train_data[TARGET_COLUMN] = y_train.values

    folds = 2 if mode == "fast" else 5
    n_select = 1

    start_time = time.perf_counter()
    exp = ClassificationExperiment()
    exp.setup(
        data=train_data,
        target=TARGET_COLUMN,
        session_id=random_state,
        fold=folds,
        verbose=False,
    )

    # Compare models optimizing for F1
    best_model = exp.compare_models(sort="F1", n_select=n_select, verbose=False)
    actual_training_time = time.perf_counter() - start_time

    # Pull leaderboard table
    leaderboard_df = exp.pull()
    leaderboard_save_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(leaderboard_df, pd.DataFrame):
        leaderboard_df.to_csv(leaderboard_save_path, index=False)
        logger.info(f"Saved PyCaret leaderboard to {leaderboard_save_path}")

    # Save model
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_save_path)
    logger.info(f"Saved PyCaret model to {model_save_path}")

    # Evaluate on held-out test data
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test.values

    predictions = exp.predict_model(best_model, data=test_data)
    y_pred = predictions["prediction_label"].to_numpy()
    y_prob = None
    if "prediction_score" in predictions.columns:
        # If prediction_label is 1, score is P(1); if 0, score is P(0)
        scores = predictions["prediction_score"].to_numpy()
        labels = predictions["prediction_label"].to_numpy()
        y_prob = np.where(labels == 1, scores, 1.0 - scores)

    metrics = compute_automl_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    val_f1 = np.nan
    if "F1" in leaderboard_df.columns:
        val_f1 = float(leaderboard_df["F1"].iloc[0])

    model_name = type(best_model).__name__

    metrics_summary = {
        "method": "PyCaret AutoML (Fallback)",
        "tool": "PyCaret",
        "best_model": model_name,
        "cv_or_validation_f1": round(val_f1, 4) if not np.isnan(val_f1) else np.nan,
        "test_f1": metrics["test_f1"],
        "test_roc_auc": metrics["test_roc_auc"],
        "test_pr_auc": metrics["test_pr_auc"],
        "test_precision": metrics["test_precision"],
        "test_recall": metrics["test_recall"],
        "test_accuracy": metrics["test_accuracy"],
        "training_time_seconds": round(actual_training_time, 2),
        "requested_time_budget_seconds": time_budget_seconds,
        "preprocessing_notes": "PyCaret automated classification pipeline (imputation, encoding, scaling)",
        "interpretability_notes": "Moderate to High depending on winning estimator",
    }

    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(metrics_save_path, index=False)
    logger.info(f"Saved PyCaret metrics to {metrics_save_path}")

    metrics_summary["confusion_matrix"] = metrics["confusion_matrix"]
    metrics_summary["y_pred"] = metrics["y_pred"]
    metrics_summary["y_prob"] = metrics["y_prob"]
    metrics_summary["best_params"] = {"model_type": model_name}

    return best_model, metrics_summary
