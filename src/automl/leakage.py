"""Deliberate Data Leakage Experiment Module for Phase 3."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flaml import AutoML
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from src.automl.common import compute_automl_metrics
from src.config import DATA_PROCESSED_DIR, PLOTS_DIR, RANDOM_SEED, RESULTS_DIR, TARGET_COLUMN
from src.utils import set_random_seed, setup_logger

logger = setup_logger("leakage_experiment")


def inject_leaky_feature(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    p_pos: float = 0.92,
    p_neg: float = 0.04,
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Injects synthetic 'post_failure_inspection_flag' target leakage feature."""
    set_random_seed(seed)
    rng = np.random.RandomState(seed)
    df_leaky = df.copy()

    targets = df_leaky[target_col].to_numpy()
    probs = np.where(targets == 1, p_pos, p_neg)
    random_draws = rng.uniform(0.0, 1.0, size=len(df_leaky))
    df_leaky["post_failure_inspection_flag"] = (random_draws < probs).astype(int)

    logger.info(
        f"Injected 'post_failure_inspection_flag' (p_pos={p_pos}, p_neg={p_neg}) | "
        f"Positive target correlation = {df_leaky[['post_failure_inspection_flag', target_col]].corr().iloc[0, 1]:.3f}"
    )
    return df_leaky


def create_leaky_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Generates and writes train_automl_leak.csv and test_automl_leak.csv."""
    train_leaky = inject_leaky_feature(train_df, seed=seed)
    test_leaky = inject_leaky_feature(test_df, seed=seed + 1)

    train_leak_path = DATA_PROCESSED_DIR / "train_automl_leak.csv"
    test_leak_path = DATA_PROCESSED_DIR / "test_automl_leak.csv"

    train_leaky.to_csv(train_leak_path, index=False)
    test_leaky.to_csv(test_leak_path, index=False)

    logger.info(f"Saved leaky datasets: {train_leak_path} and {test_leak_path}")
    return train_leaky, test_leaky, train_leak_path, test_leak_path


def compute_leakage_feature_importances(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, str]:
    """Extracts feature importances via SHAP, native importance, or permutation importance."""
    method_used = "native"
    importances = None

    # 1. Try SHAP TreeExplainer
    try:
        import shap

        underlying = getattr(model, "model", model)
        estimator = getattr(underlying, "estimator", underlying)
        explainer = shap.TreeExplainer(estimator)
        # Sample test set for speed
        sample_X = X_test.head(100)
        shap_values = explainer.shap_values(sample_X)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            vals = np.abs(shap_values[1]).mean(axis=0)
        else:
            vals = np.abs(shap_values).mean(axis=0)
        if len(vals) == len(feature_names):
            importances = vals
            method_used = "SHAP TreeExplainer"
    except Exception as e:
        logger.debug(f"SHAP TreeExplainer unavailable for leakage importance: {e}")

    # 2. Try native feature_importances_
    if importances is None:
        try:
            underlying = getattr(model, "model", model)
            estimator = getattr(underlying, "estimator", underlying)
            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
                method_used = "Native Feature Importance"
        except Exception:
            pass

    # 3. Try permutation importance
    if importances is None:
        try:
            r = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=RANDOM_SEED, scoring="f1")
            importances = r.importances_mean
            method_used = "Permutation Importance"
        except Exception:
            pass

    # 4. Fallback: Surrogate Random Forest
    if importances is None or len(importances) != len(feature_names):
        logger.info("Using Surrogate Random Forest for leakage feature attribution.")
        surrogate = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight="balanced")
        surrogate.fit(X_test, y_test)
        importances = surrogate.feature_importances_
        method_used = "Surrogate Random Forest"

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_imp, method_used


def run_leakage_experiment(
    X_train_clean: pd.DataFrame,
    y_train: pd.Series,
    X_test_clean: pd.DataFrame,
    y_test: pd.Series,
    clean_metrics: Dict[str, Any],
    time_budget_seconds: int = 60,
    mode: str = "fast",
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Runs controlled experiment training FLAML on leaky vs. clean datasets."""
    set_random_seed(random_state)
    logger.info("=" * 60)
    logger.info("RUNNING CONTROLLED DATA LEAKAGE EXPERIMENT")
    logger.info("=" * 60)

    # Combine clean data to inject leak
    train_full = X_train_clean.copy()
    train_full[TARGET_COLUMN] = y_train.values

    test_full = X_test_clean.copy()
    test_full[TARGET_COLUMN] = y_test.values

    # Generate leaky datasets
    train_leaky, test_leaky, _, _ = create_leaky_datasets(train_full, test_full, seed=random_state)

    X_train_leak = train_leaky.drop(columns=[TARGET_COLUMN])
    y_train_leak = train_leaky[TARGET_COLUMN]
    X_test_leak = test_leaky.drop(columns=[TARGET_COLUMN])
    y_test_leak = test_leaky[TARGET_COLUMN]

    # Train FLAML on leaky data
    leaky_automl = AutoML()
    leaky_log = RESULTS_DIR / "flaml_leakage_log.txt"
    settings = {
        "time_budget": time_budget_seconds,
        "metric": "f1",
        "task": "classification",
        "seed": random_state,
        "log_file_name": str(leaky_log),
        "verbose": 0,
    }

    start_time = time.perf_counter()
    leaky_automl.fit(X_train=X_train_leak, y_train=y_train_leak, **settings)
    leak_train_time = time.perf_counter() - start_time

    # Evaluate leaky model on leaky test data
    y_pred_leak = leaky_automl.predict(X_test_leak)
    y_prob_leak = None
    if hasattr(leaky_automl, "predict_proba"):
        try:
            p = leaky_automl.predict_proba(X_test_leak)
            y_prob_leak = p[:, 1] if p.shape[1] > 1 else p[:, 0]
        except Exception:
            pass

    leaky_metrics = compute_automl_metrics(y_true=y_test_leak, y_pred=y_pred_leak, y_prob=y_prob_leak)

    # Compute feature importances
    feature_names = list(X_test_leak.columns)
    imp_df, imp_method = compute_leakage_feature_importances(leaky_automl, X_test_leak, y_test_leak, feature_names)

    top3_leaky = list(imp_df["feature"].head(3))
    while len(top3_leaky) < 3:
        top3_leaky.append("N/A")

    # Clean top features
    clean_imp_df, _ = compute_leakage_feature_importances(
        clean_metrics.get("model", leaky_automl),
        X_test_clean,
        y_test,
        list(X_test_clean.columns),
    )
    top3_clean = list(clean_imp_df["feature"].head(3))
    while len(top3_clean) < 3:
        top3_clean.append("N/A")

    # Construct results comparison table
    leakage_records = [
        {
            "experiment": "without_leaky_feature",
            "tool": "FLAML",
            "test_f1": clean_metrics["test_f1"],
            "test_roc_auc": clean_metrics["test_roc_auc"],
            "test_pr_auc": clean_metrics["test_pr_auc"],
            "test_accuracy": clean_metrics["test_accuracy"],
            "training_time_seconds": clean_metrics["training_time_seconds"],
            "top_feature_1": top3_clean[0],
            "top_feature_2": top3_clean[1],
            "top_feature_3": top3_clean[2],
        },
        {
            "experiment": "with_leaky_feature",
            "tool": "FLAML",
            "test_f1": leaky_metrics["test_f1"],
            "test_roc_auc": leaky_metrics["test_roc_auc"],
            "test_pr_auc": leaky_metrics["test_pr_auc"],
            "test_accuracy": leaky_metrics["test_accuracy"],
            "training_time_seconds": round(leak_train_time, 2),
            "top_feature_1": top3_leaky[0],
            "top_feature_2": top3_leaky[1],
            "top_feature_3": top3_leaky[2],
        },
    ]
    leakage_df = pd.DataFrame(leakage_records)
    leakage_csv_path = RESULTS_DIR / "leakage_results.csv"
    leakage_df.to_csv(leakage_csv_path, index=False)
    logger.info(f"Saved leakage results to {leakage_csv_path}")

    # Save metadata JSON
    metadata = {
        "leaky_feature_name": "post_failure_inspection_flag",
        "leak_probability_positive": 0.92,
        "leak_probability_negative": 0.04,
        "random_seed": random_state,
        "tool_used": "FLAML",
        "mode": mode,
        "importance_method_used": imp_method,
        "warning_message": (
            "CRITICAL WARNING: The feature 'post_failure_inspection_flag' was artificially synthesized to simulate post-event telemetry leakage. "
            "AutoML models exploit this feature heavily, producing artificially elevated F1-scores."
        ),
    }
    leakage_meta_path = RESULTS_DIR / "leakage_metadata.json"
    with open(leakage_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved leakage metadata to {leakage_meta_path}")

    # Generate plots
    # 1. Leakage Metric Comparison Bar Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics_to_plot = ["test_f1", "test_roc_auc", "test_pr_auc"]
    labels = ["Test F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    clean_vals = [clean_metrics["test_f1"], clean_metrics["test_roc_auc"], clean_metrics["test_pr_auc"]]
    leak_vals = [leaky_metrics["test_f1"], leaky_metrics["test_roc_auc"], leaky_metrics["test_pr_auc"]]

    bars1 = ax.bar(x - width / 2, clean_vals, width, label="Without Leakage", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, leak_vals, width, label="With Leaky Feature", color="#C44E52")

    ax.set_title("Impact of Data Leakage on Classification Metrics", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")

    for bar in list(bars1) + list(bars2):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plot_metric_path = PLOTS_DIR / "leakage_metric_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_metric_path, dpi=300)
    plt.close()

    # 2. Leakage Feature Importance Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ["#C44E52" if f == "post_failure_inspection_flag" else "#4C72B0" for f in imp_df["feature"]]
    sns.barplot(data=imp_df, x="importance", y="feature", palette=palette, ax=ax)
    ax.set_title(f"Feature Importance Under Leakage ({imp_method})", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Relative Importance / Attribution", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)

    plot_imp_path = PLOTS_DIR / "leakage_feature_importance.png"
    plt.tight_layout()
    plt.savefig(plot_imp_path, dpi=300)
    plt.close()

    return leakage_df, metadata, imp_df
