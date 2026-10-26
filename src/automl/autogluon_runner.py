"""AutoGluon Tabular runner for Predictive Maintenance benchmarking."""

import time
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from src.automl.common import compute_automl_metrics
from src.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR, TARGET_COLUMN
from src.utils import set_random_seed, setup_logger

logger = setup_logger("autogluon_runner")


def run_autogluon_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    time_budget_seconds: int = 60,
    random_state: int = RANDOM_SEED,
    model_save_dir: Path = MODELS_DIR / "autogluon_predictor",
    leaderboard_save_path: Path = RESULTS_DIR / "autogluon_leaderboard.csv",
    metrics_save_path: Path = RESULTS_DIR / "autogluon_metrics.csv",
    importance_save_path: Path = RESULTS_DIR / "autogluon_feature_importance.csv",
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    """Executes AutoGluon-Tabular multi-layer stacking search with fixed time budget."""
    set_random_seed(random_state)
    logger.info(f"Starting AutoGluon Tabular fit (time_limit={time_budget_seconds}s, eval_metric=f1)...")

    # Combine into single DataFrame for AutoGluon API
    train_data = X_train.copy()
    train_data[TARGET_COLUMN] = y_train.values

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test.values

    import shutil
    if model_save_dir.exists():
        try:
            shutil.rmtree(model_save_dir)
        except Exception as e:
            logger.warning(f"Could not clean existing AutoGluon directory: {e}")
    model_save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        eval_metric="f1",
        problem_type="binary",
        path=str(model_save_dir),
    )

    predictor.fit(
        train_data=train_data,
        time_limit=time_budget_seconds,
        presets="medium_quality",
        verbosity=1,
    )
    actual_training_time = time.perf_counter() - start_time
    logger.info(f"AutoGluon training finished in {actual_training_time:.2f}s")

    # 1. Leaderboard generation on held-out test set
    leaderboard_df = predictor.leaderboard(test_data, silent=True)
    leaderboard_save_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_df.to_csv(leaderboard_save_path, index=False)
    logger.info(f"Saved AutoGluon leaderboard to {leaderboard_save_path}")

    # 2. Extract best model and validation score
    best_model_name = getattr(predictor, "model_best", getattr(predictor, "_model_best", "WeightedEnsemble_L2"))
    if callable(best_model_name):
        best_model_name = best_model_name()
    val_f1 = np.nan
    if "score_val" in leaderboard_df.columns:
        best_row = leaderboard_df[leaderboard_df["model"] == best_model_name]
        if not best_row.empty:
            val_f1 = float(best_row["score_val"].values[0])

    # 3. Test Evaluation
    y_pred = predictor.predict(X_test).to_numpy()
    y_prob = None
    try:
        proba_df = predictor.predict_proba(X_test)
        if isinstance(proba_df, pd.DataFrame) and 1 in proba_df.columns:
            y_prob = proba_df[1].to_numpy()
        elif isinstance(proba_df, pd.DataFrame) and "1" in proba_df.columns:
            y_prob = proba_df["1"].to_numpy()
        else:
            y_prob = proba_df.iloc[:, 1].to_numpy()
    except Exception as e:
        logger.warning(f"Failed to extract probabilities from AutoGluon: {e}")

    metrics = compute_automl_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    # 4. Feature importance if available
    try:
        fi_df = predictor.feature_importance(test_data, silent=True)
        if fi_df is not None and not fi_df.empty:
            fi_df = fi_df.reset_index().rename(columns={"index": "feature"})
            fi_df.to_csv(importance_save_path, index=False)
            logger.info(f"Saved AutoGluon feature importance to {importance_save_path}")
    except Exception as e:
        logger.warning(f"Could not compute feature importance from AutoGluon: {e}")

    metrics_summary = {
        "method": "AutoGluon Tabular",
        "tool": "AutoGluon",
        "best_model": str(best_model_name),
        "cv_or_validation_f1": round(val_f1, 4) if not np.isnan(val_f1) else np.nan,
        "test_f1": metrics["test_f1"],
        "test_roc_auc": metrics["test_roc_auc"],
        "test_pr_auc": metrics["test_pr_auc"],
        "test_precision": metrics["test_precision"],
        "test_recall": metrics["test_recall"],
        "test_accuracy": metrics["test_accuracy"],
        "training_time_seconds": round(actual_training_time, 2),
        "requested_time_budget_seconds": time_budget_seconds,
        "preprocessing_notes": "Automated multi-layer preprocessing (Type detection, imputations, categorical neural/tree encoding)",
        "interpretability_notes": "Low (Multi-layer ensemble stacking across GBDTs, neural nets, and weighted ensemble)",
    }

    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(metrics_save_path, index=False)
    logger.info(f"Saved AutoGluon metrics summary to {metrics_save_path}")

    metrics_summary["confusion_matrix"] = metrics["confusion_matrix"]
    metrics_summary["y_pred"] = metrics["y_pred"]
    metrics_summary["y_prob"] = metrics["y_prob"]
    metrics_summary["best_params"] = {"best_model": best_model_name}

    return predictor, metrics_summary
