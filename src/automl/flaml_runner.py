"""FLAML AutoML runner for Predictive Maintenance benchmarking."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from flaml import AutoML

from src.automl.common import compute_automl_metrics
from src.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.utils import set_random_seed, setup_logger

logger = setup_logger("flaml_runner")


def parse_flaml_log(log_path: Path) -> pd.DataFrame:
    """Parses FLAML trial log file into a structured leaderboard DataFrame."""
    records = []
    if not log_path.exists():
        return pd.DataFrame()

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "record_id" in data and "logged_metric" in data:
                    metric_val = 1.0 - data["loss"] if "loss" in data else np.nan
                    records.append({
                        "trial_id": data.get("record_id"),
                        "model": data.get("current_learner", "unknown"),
                        "validation_f1": round(float(metric_val), 4) if not np.isnan(metric_val) else np.nan,
                        "trial_time_s": round(float(data.get("trial_time", 0.0)), 3),
                        "total_time_s": round(float(data.get("total_time", 0.0)), 2),
                        "config": str(data.get("hyperparameters", {})),
                    })
            except Exception:
                continue

    if records:
        df = pd.DataFrame(records)
        df = df.sort_values(by="validation_f1", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df
    return pd.DataFrame()


def build_fallback_flaml_leaderboard(automl: AutoML) -> pd.DataFrame:
    """Builds structured summary table from AutoML search states if log is sparse."""
    records = []
    if hasattr(automl, "search_states"):
        for learner_name, state in automl.search_states.items():
            best_loss = getattr(state, "best_loss", None)
            val_f1 = (1.0 - best_loss) if (best_loss is not None and best_loss < float("inf")) else np.nan
            time_used = getattr(state, "total_time", 0.0)
            records.append({
                "model": learner_name,
                "validation_f1": round(float(val_f1), 4) if not np.isnan(val_f1) else np.nan,
                "total_time_s": round(float(time_used), 2),
                "is_winner": (learner_name == automl.best_estimator),
            })
    if records:
        df = pd.DataFrame(records).sort_values(by="validation_f1", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df
    # Fallback to single best model entry
    return pd.DataFrame([{
        "rank": 1,
        "model": str(automl.best_estimator),
        "validation_f1": round(float(1.0 - automl.best_loss), 4),
        "total_time_s": round(float(automl.time_to_find_best_model), 2),
        "is_winner": True,
    }])


def run_flaml_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    time_budget_seconds: int = 60,
    random_state: int = RANDOM_SEED,
    model_save_path: Path = MODELS_DIR / "flaml_model.pkl",
    leaderboard_save_path: Path = RESULTS_DIR / "flaml_leaderboard.csv",
    best_config_save_path: Path = RESULTS_DIR / "flaml_best_config.json",
    metrics_save_path: Path = RESULTS_DIR / "flaml_metrics.csv",
) -> Tuple[AutoML, Dict[str, Any]]:
    """Executes FLAML automated model search with fixed time budget."""
    set_random_seed(random_state)
    logger.info(f"Starting FLAML AutoML search (time_budget={time_budget_seconds}s, metric=f1)...")

    log_file_path = RESULTS_DIR / "flaml_log.txt"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    automl = AutoML()
    settings = {
        "time_budget": time_budget_seconds,
        "metric": "f1",
        "task": "classification",
        "seed": random_state,
        "log_file_name": str(log_file_path),
        "log_type": "all",
        "verbose": 0,
    }

    start_time = time.perf_counter()
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    actual_training_time = time.perf_counter() - start_time

    best_estimator = automl.best_estimator
    best_val_f1 = float(1.0 - automl.best_loss)
    best_config = automl.best_config

    logger.info(
        f"FLAML search completed in {actual_training_time:.2f}s | "
        f"Winner: {best_estimator} | Best Val F1: {best_val_f1:.4f}"
    )

    # 1. Parse or build leaderboard
    leaderboard_df = parse_flaml_log(log_file_path)
    if leaderboard_df.empty:
        leaderboard_df = build_fallback_flaml_leaderboard(automl)
    leaderboard_df.to_csv(leaderboard_save_path, index=False)
    logger.info(f"Saved FLAML leaderboard to {leaderboard_save_path}")

    # 2. Save best config JSON
    best_config_clean = {k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v)) for k, v in best_config.items()}
    best_config_meta = {
        "best_estimator": best_estimator,
        "best_validation_f1": round(best_val_f1, 4),
        "time_to_find_best_model_seconds": round(float(automl.time_to_find_best_model), 3),
        "hyperparameters": best_config_clean,
    }
    with open(best_config_save_path, "w", encoding="utf-8") as f:
        json.dump(best_config_meta, f, indent=4)
    logger.info(f"Saved FLAML best config to {best_config_save_path}")

    # 3. Save serialized model
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(automl, model_save_path)
    logger.info(f"Saved serialized FLAML model to {model_save_path}")

    # 4. Evaluate once on held-out test data
    y_pred = automl.predict(X_test)
    y_prob = None
    if hasattr(automl, "predict_proba"):
        try:
            probas = automl.predict_proba(X_test)
            y_prob = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
        except Exception as e:
            logger.warning(f"Failed to get probabilities from FLAML: {e}")

    metrics = compute_automl_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
    metrics_summary = {
        "method": "FLAML AutoML",
        "tool": "FLAML",
        "best_model": str(best_estimator),
        "cv_or_validation_f1": round(best_val_f1, 4),
        "test_f1": metrics["test_f1"],
        "test_roc_auc": metrics["test_roc_auc"],
        "test_pr_auc": metrics["test_pr_auc"],
        "test_precision": metrics["test_precision"],
        "test_recall": metrics["test_recall"],
        "test_accuracy": metrics["test_accuracy"],
        "training_time_seconds": round(actual_training_time, 2),
        "requested_time_budget_seconds": time_budget_seconds,
        "preprocessing_notes": "Native tabular encoding with tree learners (LightGBM, XGBoost, CatBoost, ExtraTrees, RF)",
        "interpretability_notes": "Moderate (Gradient Boosted Tree / Tree Ensemble)",
    }

    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(metrics_save_path, index=False)
    logger.info(f"Saved FLAML metrics summary to {metrics_save_path}")

    # Append test predictions and confusion matrix for downstream analysis
    metrics_summary["confusion_matrix"] = metrics["confusion_matrix"]
    metrics_summary["y_pred"] = metrics["y_pred"]
    metrics_summary["y_prob"] = metrics["y_prob"]
    metrics_summary["best_params"] = best_config_clean
    metrics_summary["model"] = automl

    return automl, metrics_summary
