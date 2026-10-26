"""Manual Baseline Model Implementation: Random Forest with class_weight='balanced'."""

import time
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import MODELS_DIR, RANDOM_SEED
from src.models.evaluation import evaluate_model, save_model
from src.utils import set_random_seed, setup_logger

logger = setup_logger("baseline")


def train_baseline_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 2,
    n_jobs: int = -1,
    random_state: int = RANDOM_SEED,
    model_save_path: Path = MODELS_DIR / "baseline_rf.pkl",
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Trains and cross-validates the manual baseline Random Forest model."""
    set_random_seed(random_state)
    logger.info(f"Training Baseline RandomForestClassifier with class_weight='balanced' (cv_folds={cv_folds})...")

    # Instantiate baseline Random Forest model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # Cross-validation on training partition only
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    start_time = time.perf_counter()
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1", n_jobs=n_jobs)
    cv_f1_mean = float(np.mean(cv_scores))
    cv_f1_std = float(np.std(cv_scores))

    # Fit on full training set
    rf.fit(X_train, y_train)
    end_time = time.perf_counter()
    training_time_seconds = end_time - start_time

    logger.info(
        f"Baseline RF fitted in {training_time_seconds:.2f}s | "
        f"CV F1 ({cv_folds}-fold): {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})"
    )

    # Save model artifact
    save_model(rf, model_save_path)

    # Evaluate on held-out test set
    baseline_params = {
        "n_estimators": 100,
        "max_depth": None,
        "class_weight": "balanced",
        "random_state": random_state,
    }
    metrics = evaluate_model(
        model=rf,
        X_test=X_test,
        y_test=y_test,
        method_name="Baseline Random Forest",
        cv_f1_mean=cv_f1_mean,
        cv_f1_std=cv_f1_std,
        training_time_seconds=training_time_seconds,
        best_params=baseline_params,
    )

    return rf, metrics
