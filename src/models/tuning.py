"""Hyperparameter tuning module implementing Grid Search, Random Search, and Optuna."""

import time
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from src.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.models.evaluation import evaluate_model, save_model
from src.utils import set_random_seed, setup_logger

logger = setup_logger("tuning")

# Suppress overly verbose Optuna logs in console
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_search_spaces(mode: str = "fast") -> Dict[str, Any]:
    """Returns parameter grids calibrated for 'fast' or 'final' mode."""
    if mode == "fast":
        grid_params = {
            "n_estimators": [100],
            "max_depth": [5, 10],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
        }
        random_space = {
            "n_estimators": [100, 300],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        optuna_config = {
            "n_estimators": [100],
            "max_depth": [5, 10],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "n_trials": 5,
        }
        random_iter = 5
    else:  # final mode
        grid_params = {
            "n_estimators": [100, 300],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        random_space = grid_params
        optuna_config = {
            "n_estimators": [100, 300],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "n_trials": 30,
        }
        random_iter = 20

    return {
        "grid_params": grid_params,
        "random_space": random_space,
        "random_iter": random_iter,
        "optuna_config": optuna_config,
    }


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mode: str = "fast",
    cv_folds: int = 2,
    n_jobs: int = -1,
    random_state: int = RANDOM_SEED,
    model_save_path: Path = MODELS_DIR / "grid_rf.pkl",
    results_save_path: Path = RESULTS_DIR / "grid_search_results.csv",
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Executes GridSearchCV tuning on Random Forest."""
    set_random_seed(random_state)
    param_grid = get_search_spaces(mode)["grid_params"]
    logger.info(f"Starting GridSearchCV (mode={mode}, cv_folds={cv_folds}, grid={param_grid})...")

    rf = RandomForestClassifier(class_weight="balanced", random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        return_train_score=False,
        n_jobs=n_jobs,
        refit=True,
    )

    start_time = time.perf_counter()
    grid.fit(X_train, y_train)
    duration = time.perf_counter() - start_time

    # Save CV results
    results_save_path.parent.mkdir(parents=True, exist_ok=True)
    cv_results_df = pd.DataFrame(grid.cv_results_)
    cv_results_df.to_csv(results_save_path, index=False)
    logger.info(f"Saved Grid Search CV table to {results_save_path}")

    # Extract CV scores for best model
    best_idx = grid.best_index_
    cv_f1_mean = float(cv_results_df.loc[best_idx, "mean_test_score"])
    cv_f1_std = float(cv_results_df.loc[best_idx, "std_test_score"])

    # Save serialized model
    save_model(grid.best_estimator_, model_save_path)

    # Evaluate on held-out test data
    metrics = evaluate_model(
        model=grid.best_estimator_,
        X_test=X_test,
        y_test=y_test,
        method_name="Grid Search Random Forest",
        cv_f1_mean=cv_f1_mean,
        cv_f1_std=cv_f1_std,
        training_time_seconds=duration,
        best_params=grid.best_params_,
    )

    return grid.best_estimator_, metrics


def run_random_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mode: str = "fast",
    cv_folds: int = 2,
    n_jobs: int = -1,
    random_state: int = RANDOM_SEED,
    model_save_path: Path = MODELS_DIR / "random_rf.pkl",
    results_save_path: Path = RESULTS_DIR / "random_search_results.csv",
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Executes RandomizedSearchCV tuning on Random Forest."""
    set_random_seed(random_state)
    config = get_search_spaces(mode)
    param_distributions = config["random_space"]
    n_iter = config["random_iter"]

    logger.info(f"Starting RandomizedSearchCV (mode={mode}, n_iter={n_iter}, cv_folds={cv_folds})...")

    rf = RandomForestClassifier(class_weight="balanced", random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=random_state,
        return_train_score=False,
        n_jobs=n_jobs,
        refit=True,
    )

    start_time = time.perf_counter()
    random_search.fit(X_train, y_train)
    duration = time.perf_counter() - start_time

    # Save CV results
    results_save_path.parent.mkdir(parents=True, exist_ok=True)
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    cv_results_df.to_csv(results_save_path, index=False)
    logger.info(f"Saved Random Search CV table to {results_save_path}")

    # Extract CV scores for best model
    best_idx = random_search.best_index_
    cv_f1_mean = float(cv_results_df.loc[best_idx, "mean_test_score"])
    cv_f1_std = float(cv_results_df.loc[best_idx, "std_test_score"])

    # Save serialized model
    save_model(random_search.best_estimator_, model_save_path)

    # Evaluate on held-out test data
    metrics = evaluate_model(
        model=random_search.best_estimator_,
        X_test=X_test,
        y_test=y_test,
        method_name="Random Search Random Forest",
        cv_f1_mean=cv_f1_mean,
        cv_f1_std=cv_f1_std,
        training_time_seconds=duration,
        best_params=random_search.best_params_,
    )

    return random_search.best_estimator_, metrics


def run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mode: str = "fast",
    cv_folds: int = 2,
    n_jobs: int = -1,
    random_state: int = RANDOM_SEED,
    model_save_path: Path = MODELS_DIR / "optuna_rf.pkl",
    trials_save_path: Path = RESULTS_DIR / "optuna_trials.csv",
    results_save_path: Path = RESULTS_DIR / "optuna_results.csv",
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Executes Bayesian Optimization using Optuna TPESampler."""
    set_random_seed(random_state)
    optuna_cfg = get_search_spaces(mode)["optuna_config"]
    n_trials = optuna_cfg["n_trials"]

    logger.info(f"Starting Optuna Bayesian Optimization (mode={mode}, n_trials={n_trials}, cv_folds={cv_folds})...")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_categorical("n_estimators", optuna_cfg["n_estimators"])
        max_depth = trial.suggest_categorical("max_depth", optuna_cfg["max_depth"])
        min_samples_split = trial.suggest_categorical("min_samples_split", optuna_cfg["min_samples_split"])
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", optuna_cfg["min_samples_leaf"])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=n_jobs)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    start_time = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    optimization_duration = time.perf_counter() - start_time

    # Save trials dataframe
    trials_save_path.parent.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_save_path, index=False)
    logger.info(f"Saved Optuna trials table to {trials_save_path}")

    # Retrain best model on full training set
    best_params = study.best_params
    best_cv_f1 = float(study.best_value)

    # Compute cv_f1_std for best params
    best_rf = RandomForestClassifier(
        **best_params,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    best_scores = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring="f1", n_jobs=n_jobs)
    cv_f1_std = float(np.std(best_scores))

    # Fit on complete training data
    refit_start = time.perf_counter()
    best_rf.fit(X_train, y_train)
    total_duration = optimization_duration + (time.perf_counter() - refit_start)

    # Save serialized model
    save_model(best_rf, model_save_path)

    # Evaluate on held-out test data
    metrics = evaluate_model(
        model=best_rf,
        X_test=X_test,
        y_test=y_test,
        method_name="Optuna Random Forest",
        cv_f1_mean=best_cv_f1,
        cv_f1_std=cv_f1_std,
        training_time_seconds=total_duration,
        best_params=best_params,
    )

    # Save optuna results summary CSV
    results_save_path.parent.mkdir(parents=True, exist_ok=True)
    optuna_summary_df = pd.DataFrame([
        {
            "method": "Optuna Random Forest",
            "best_cv_f1": best_cv_f1,
            "cv_f1_std": cv_f1_std,
            "test_f1": metrics["test_f1"],
            "test_roc_auc": metrics["test_roc_auc"],
            "test_precision": metrics["test_precision"],
            "test_recall": metrics["test_recall"],
            "test_accuracy": metrics["test_accuracy"],
            "training_time_seconds": total_duration,
            "best_params": str(best_params),
        }
    ])
    optuna_summary_df.to_csv(results_save_path, index=False)
    logger.info(f"Saved Optuna summary table to {results_save_path}")

    return best_rf, metrics
