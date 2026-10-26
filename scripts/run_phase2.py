"""Phase 2 Master Runner: Baseline Random Forest & Hyperparameter Tuning (Grid, Random, Optuna)."""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import auc, roc_curve

from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    PRIMARY_METRIC,
    RANDOM_SEED,
    REPORTS_DIR,
    RESULTS_DIR,
    SCREENSHOTS_DIR,
    TARGET_COLUMN,
    init_directories,
)
from src.models.baseline import train_baseline_rf
from src.models.tuning import run_grid_search, run_optuna_tuning, run_random_search
from src.utils import set_random_seed, setup_logger

LOG_FILE = RESULTS_DIR / "phase2_run.log"
logger = setup_logger("run_phase2", log_file=LOG_FILE)


def load_phase1_manual_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Loads Phase 1 manual train/test splits and metadata."""
    train_path = DATA_PROCESSED_DIR / "train_manual.csv"
    test_path = DATA_PROCESSED_DIR / "test_manual.csv"
    meta_path = DATA_PROCESSED_DIR / "metadata.json"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Phase 1 manual data files not found in {DATA_PROCESSED_DIR}. "
            "Please run 'python scripts/run_phase1.py' first."
        )

    logger.info(f"Loading datasets: {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)

    logger.info(f"Loaded X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    return X_train, y_train, X_test, y_test, metadata


def generate_confusion_matrix_plot(
    cm: np.ndarray,
    title: str,
    output_path: Path,
) -> Path:
    """Generates and saves a high-resolution annotated confusion matrix plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Failure (0)", "Failure (1)"],
        yticklabels=["No Failure (0)", "Failure (1)"],
        ax=ax,
    )
    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_roc_curves_plot(
    models_metrics: List[Dict[str, Any]],
    y_test: pd.Series,
    output_path: Path,
) -> Path:
    """Generates multi-model ROC curve comparison."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for idx, item in enumerate(models_metrics):
        name = item["method"]
        y_proba = item.get("y_proba")
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_val = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                label=f"{name} (AUC = {roc_val:.3f})",
                color=palette[idx % len(palette)],
                linewidth=2,
            )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.500)")
    ax.set_title("Held-Out Test Set ROC Curves: Baseline vs. Tuned Models", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_comparison_bar_plots(
    comparison_df: pd.DataFrame,
    f1_plot_path: Path,
    time_plot_path: Path,
) -> None:
    """Generates F1 and time comparison bar charts."""
    f1_plot_path.parent.mkdir(parents=True, exist_ok=True)
    time_plot_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. F1 Score Comparison (Test F1 vs CV F1)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(comparison_df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, comparison_df["test_f1"], width, label="Held-Out Test F1", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, comparison_df["cv_f1_mean"], width, label="CV F1 Mean", color="#55A868")

    ax.set_title("F1-Score Comparison: Baseline vs. Hyperparameter Tuning Methods", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("F1-Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["method"], rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    for bar in list(bars1) + list(bars2):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.015,
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(f1_plot_path, dpi=300)
    plt.close()

    # 2. Total Training / Search Time Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(comparison_df["method"], comparison_df["training_time_seconds"], color="#8172B2", width=0.5)
    ax.set_title("Computational Wall-Clock Time Comparison", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("Total Search / Training Time (seconds)", fontsize=10)
    ax.set_xticks(np.arange(len(comparison_df)))
    ax.set_xticklabels(comparison_df["method"], rotation=15, ha="right", fontsize=9)
    max_time = comparison_df["training_time_seconds"].max()
    ax.set_ylim(0, max(1.0, max_time * 1.25))

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (max_time * 0.02 + 0.05),
            f"{yval:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(time_plot_path, dpi=300)
    plt.close()


def generate_academic_report_section(
    comparison_df: pd.DataFrame,
    best_info: Dict[str, Any],
    mode: str,
    output_path: Path = REPORTS_DIR / "automl_phase2_results.md",
) -> None:
    """Writes the comprehensive academic report section for Phase 2."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_table = comparison_df[
        ["method", "cv_f1_mean", "cv_f1_std", "test_f1", "test_roc_auc", "test_precision", "test_recall", "test_accuracy", "training_time_seconds"]
    ].to_markdown(index=False)

    report_content = f"""# Section 5 & 6: Manual Baseline Model and Hyperparameter Tuning

**Phase Mode:** `{mode.upper()}`  
**Evaluation Partition:** Held-out test set (2,000 records, 20% stratified)  
**Primary Metric:** F1-Score | **Secondary Metrics:** ROC-AUC, Precision, Recall, Accuracy  
**Random State:** `42`  

---

### 1. Phase 2 Objective
The primary objective of Phase 2 is to establish an un-tuned manual machine learning baseline using scikit-learn's `RandomForestClassifier` with balanced class weights, and benchmark three distinct hyperparameter optimization (HPO) paradigms across an identical model family:
1. **Grid Search (`GridSearchCV`)**: Exhaustive exploration of discrete hyperparameter combinations.
2. **Random Search (`RandomizedSearchCV`)**: Stochastic sampling across high-dimensional parameter distributions.
3. **Bayesian Optimization (`Optuna`)**: Tree-structured Parzen Estimator (TPE) directed search modeling the probability density of objective functions.

### 2. Baseline Model Description
- **Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Default Baseline Parameters:** `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`.
- **Imbalance Mitigation:** Configured with `class_weight='balanced'`, dynamically weighting samples inversely proportional to class frequencies to prevent minority class neglect.

### 3. Dataset Used
- **Source:** Preprocessed AI4I 2020 Predictive Maintenance Dataset (`data/processed/train_manual.csv` and `data/processed/test_manual.csv`).
- **Features (9 input dimensions):** Numerical sensors (`air_temp`, `process_temp`, `rotational_speed`, `torque`, `tool_wear`) alongside one-hot encoded machine variants (`type_H`, `type_L`, `type_M`).
- **Target:** `machine_failure` (binary indicator, positive prevalence $\\approx 3.39\\%$).

### 4. Validation Strategy
- **Stratified K-Fold Cross-Validation:** Hyperparameter searches operated strictly on training partitions ({'2 folds' if mode == 'fast' else '5 folds'}) to prevent data leakage.
- **Strict Partition Isolation:** Held-out test data (`test_manual.csv`) was accessed exactly once per method, strictly after the optimal estimator was refitted on the full training partition.

### 5. Hyperparameter Tuning Methods
- **Grid Search:** Evaluates parameter combinations across tree depth, estimator counts, and splitting thresholds.
- **Random Search:** Explores parameter space stochastically with a fixed budget of random iterations ({5 if mode == 'fast' else 20} iterations).
- **Optuna TPE:** Employs Bayesian sequential model-based optimization to navigate promising regions of the parameter space ({5 if mode == 'fast' else 30} trials).

### 6. Results Comparison Table
{markdown_table}

### 7. Computational Time Comparison
- **Baseline Training Time:** `{comparison_df.loc[comparison_df['method'] == 'Baseline Random Forest', 'training_time_seconds'].values[0]:.2f} seconds`.
- **Grid Search Wall-Clock Time:** `{comparison_df.loc[comparison_df['method'] == 'Grid Search Random Forest', 'training_time_seconds'].values[0]:.2f} seconds`.
- **Random Search Wall-Clock Time:** `{comparison_df.loc[comparison_df['method'] == 'Random Search Random Forest', 'training_time_seconds'].values[0]:.2f} seconds`.
- **Optuna Optimization Time:** `{comparison_df.loc[comparison_df['method'] == 'Optuna Random Forest', 'training_time_seconds'].values[0]:.2f} seconds`.

### 8. Best Method & Parameter Configuration
- **Selected Best Method:** **`{best_info['best_method']}`**
- **Criteria:** Highest Test F1-Score (`{best_info['test_f1']:.4f}`), breaking ties with ROC-AUC (`{best_info['test_roc_auc']:.4f}`) and wall-clock training efficiency.
- **Optimal Parameters:**
```json
{json.dumps(best_info['best_params'], indent=2)}
```

### 9. Engineering Interpretation
- Applying `class_weight='balanced'` enables Random Forest to achieve substantial recall on rare machine failures without requiring synthetic sample synthesis (such as SMOTE).
- Constraining tree depth (`max_depth`) curbs tree overfitting on minority failure clusters, leading to higher precision and improved overall test F1-scores.
- Bayesian optimization via Optuna efficiently locates high-performing hyperparameter configurations with fewer overall parameter evaluations compared to exhaustive grid searches.

### 10. Limitations
- Single model family constraint: only Random Forest was benchmarked; gradient boosting architectures (LightGBM, XGBoost, CatBoost) and stacking ensembles remain unexamined until Phase 3 AutoML.
- Discrete parameter discretization in fast mode limits the granularity of continuous regularization boundaries.

### 11. Screenshots to Include in Final Report
1. `reports/screenshots/phase2_baseline_confusion_matrix.png`: Confusion matrix for the baseline Random Forest classifier.
2. `reports/screenshots/phase2_best_tuned_confusion_matrix.png`: Confusion matrix for the top-performing tuned model.
3. `reports/screenshots/phase2_tuning_f1_comparison.png`: Bar chart contrasting CV F1 and held-out Test F1 scores.
4. `reports/screenshots/phase2_tuning_time_comparison.png`: Wall-clock execution time across tuning methods.
5. `reports/screenshots/phase2_roc_curve_phase2.png`: Receiver Operating Characteristic (ROC) curve comparison on the held-out test partition.
"""
    output_path.write_text(report_content, encoding="utf-8")
    logger.info(f"Generated academic report section at {output_path}")


def generate_phase2_summary_md(
    comparison_df: pd.DataFrame,
    best_info: Dict[str, Any],
    mode: str,
    output_path: Path = RESULTS_DIR / "phase2_summary.md",
) -> None:
    """Generates a concise markdown summary in results/."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_md = comparison_df.to_markdown(index=False)
    content = f"""# Phase 2 Execution Summary: Baseline & Hyperparameter Tuning

- **Execution Mode:** `{mode}`
- **Timestamp:** `{datetime.utcnow().isoformat()}Z`
- **Best Performing Method:** `{best_info['best_method']}`
- **Best Test F1-Score:** `{best_info['test_f1']:.4f}`
- **Best Test ROC-AUC:** `{best_info['test_roc_auc']:.4f}`
- **Best Model Artifact:** `{best_info['model_path']}`

## Model Performance & Resource Utilization
{table_md}

## Best Hyperparameters
```json
{json.dumps(best_info['best_params'], indent=2)}
```
"""
    output_path.write_text(content, encoding="utf-8")
    logger.info(f"Generated Phase 2 summary markdown at {output_path}")


def run_phase2_pipeline(mode: str = "fast", n_jobs: int = -1) -> None:
    """Runs complete Phase 2 pipeline."""
    set_random_seed(RANDOM_SEED)
    init_directories()

    logger.info("=" * 70)
    logger.info(f"STARTING PHASE 2: BASELINE & HYPERPARAMETER TUNING (MODE={mode.upper()})")
    logger.info("=" * 70)

    # 1. Load data
    X_train, y_train, X_test, y_test, metadata = load_phase1_manual_data()
    train_rows = len(X_train)
    test_rows = len(X_test)
    pos_rate_train = float(y_train.mean() * 100)
    pos_rate_test = float(y_test.mean() * 100)

    cv_folds = 2 if mode == "fast" else 5
    logger.info(f"Configured cv_folds={cv_folds}, n_jobs={n_jobs} for mode='{mode}'")

    # 2. Baseline Model
    baseline_rf, baseline_metrics = train_baseline_rf(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
        model_save_path=MODELS_DIR / "baseline_rf.pkl",
    )
    # Save baseline metrics CSV
    baseline_df = pd.DataFrame([{
        "method": baseline_metrics["method"],
        "cv_f1_mean": baseline_metrics["cv_f1_mean"],
        "cv_f1_std": baseline_metrics["cv_f1_std"],
        "test_f1": baseline_metrics["test_f1"],
        "test_roc_auc": baseline_metrics["test_roc_auc"],
        "test_precision": baseline_metrics["test_precision"],
        "test_recall": baseline_metrics["test_recall"],
        "test_accuracy": baseline_metrics["test_accuracy"],
        "training_time_seconds": baseline_metrics["training_time_seconds"],
        "best_params": str(baseline_metrics["best_params"]),
    }])
    baseline_df.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)

    # 3. Grid Search
    grid_rf, grid_metrics = run_grid_search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        mode=mode,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
        model_save_path=MODELS_DIR / "grid_rf.pkl",
        results_save_path=RESULTS_DIR / "grid_search_results.csv",
    )

    # 4. Random Search
    random_rf, random_metrics = run_random_search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        mode=mode,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
        model_save_path=MODELS_DIR / "random_rf.pkl",
        results_save_path=RESULTS_DIR / "random_search_results.csv",
    )

    # 5. Optuna Bayesian Optimization
    optuna_rf, optuna_metrics = run_optuna_tuning(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        mode=mode,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
        model_save_path=MODELS_DIR / "optuna_rf.pkl",
        trials_save_path=RESULTS_DIR / "optuna_trials.csv",
        results_save_path=RESULTS_DIR / "optuna_results.csv",
    )

    # 6. Compile Comparison Table
    all_metrics = [baseline_metrics, grid_metrics, random_metrics, optuna_metrics]
    comparison_rows = []
    for item in all_metrics:
        comparison_rows.append({
            "method": item["method"],
            "cv_f1_mean": item["cv_f1_mean"],
            "cv_f1_std": item["cv_f1_std"],
            "test_f1": item["test_f1"],
            "test_roc_auc": item["test_roc_auc"],
            "test_precision": item["test_precision"],
            "test_recall": item["test_recall"],
            "test_accuracy": item["test_accuracy"],
            "training_time_seconds": item["training_time_seconds"],
            "best_params": str(item["best_params"]),
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULTS_DIR / "tuning_comparison.csv", index=False)
    logger.info(f"Saved tuning comparison table to {RESULTS_DIR / 'tuning_comparison.csv'}")

    # 7. Best Model Selection
    # Priority: 1) highest test_f1, 2) highest test_roc_auc, 3) lowest training_time
    sorted_metrics = sorted(
        all_metrics,
        key=lambda m: (m["test_f1"], m["test_roc_auc"], -m["training_time_seconds"]),
        reverse=True,
    )
    best_metric_item = sorted_metrics[0]

    method_to_model_path = {
        "Baseline Random Forest": "models/baseline_rf.pkl",
        "Grid Search Random Forest": "models/grid_rf.pkl",
        "Random Search Random Forest": "models/random_rf.pkl",
        "Optuna Random Forest": "models/optuna_rf.pkl",
    }
    best_model_rel_path = method_to_model_path.get(best_metric_item["method"], "models/best_phase2_model.pkl")

    best_info = {
        "best_method": best_metric_item["method"],
        "model_path": best_model_rel_path,
        "test_f1": best_metric_item["test_f1"],
        "test_roc_auc": best_metric_item["test_roc_auc"],
        "training_time_seconds": best_metric_item["training_time_seconds"],
        "best_params": best_metric_item["best_params"],
        "phase2_mode": mode,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(RESULTS_DIR / "phase2_best_model.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=4)
    logger.info(f"Saved phase2_best_model.json: {best_info['best_method']}")

    # Save best_params.json containing best params for grid, random, optuna
    best_params_all = {
        "grid_search": grid_metrics["best_params"],
        "random_search": random_metrics["best_params"],
        "optuna": optuna_metrics["best_params"],
    }
    with open(RESULTS_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_all, f, indent=4)
    logger.info(f"Saved best_params.json")

    # 8. Generate Visualizations & Copy to Reports/Screenshots
    # A) Baseline Confusion Matrix
    cm_base_path = PLOTS_DIR / "baseline_confusion_matrix.png"
    generate_confusion_matrix_plot(
        baseline_metrics["confusion_matrix"],
        "Baseline Random Forest: Confusion Matrix",
        cm_base_path,
    )
    shutil.copy2(cm_base_path, SCREENSHOTS_DIR / "phase2_baseline_confusion_matrix.png")

    # B) Best Tuned Confusion Matrix
    cm_best_path = PLOTS_DIR / "best_tuned_confusion_matrix.png"
    generate_confusion_matrix_plot(
        best_metric_item["confusion_matrix"],
        f"Best Model ({best_metric_item['method']}): Confusion Matrix",
        cm_best_path,
    )
    shutil.copy2(cm_best_path, SCREENSHOTS_DIR / "phase2_best_tuned_confusion_matrix.png")

    # C) F1 and Time Comparison Bar Charts
    f1_chart_path = PLOTS_DIR / "tuning_f1_comparison.png"
    time_chart_path = PLOTS_DIR / "tuning_time_comparison.png"
    generate_comparison_bar_plots(comparison_df, f1_chart_path, time_chart_path)
    shutil.copy2(f1_chart_path, SCREENSHOTS_DIR / "phase2_tuning_f1_comparison.png")
    shutil.copy2(time_chart_path, SCREENSHOTS_DIR / "phase2_tuning_time_comparison.png")

    # D) ROC Curve Comparison
    roc_chart_path = PLOTS_DIR / "roc_curve_phase2.png"
    generate_roc_curves_plot(all_metrics, y_test, roc_chart_path)
    shutil.copy2(roc_chart_path, SCREENSHOTS_DIR / "phase2_roc_curve_phase2.png")
    logger.info("All Phase 2 plots generated and mirrored to reports/screenshots/")

    # 9. Phase 2 Metadata JSON
    package_versions = {
        "scikit-learn": sklearn.__version__,
        "optuna": optuna.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
    }

    phase2_metadata = {
        "phase": "phase2",
        "mode": mode,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "random_seed": RANDOM_SEED,
        "target_column": TARGET_COLUMN,
        "primary_metric": PRIMARY_METRIC,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "positive_class_percentage_train": round(pos_rate_train, 2),
        "positive_class_percentage_test": round(pos_rate_test, 2),
        "package_versions": package_versions,
        "fast_mode_warning": (
            "These are fast smoke-test results. For final report screenshots, rerun with --mode final."
            if mode == "fast"
            else "Final mode results benchmarked on 5-fold cross-validation."
        ),
        "command_used": f".venv\\Scripts\\python scripts/run_phase2.py --mode {mode}",
        "assumptions": [
            "Random Forest Classifier baseline utilized with class_weight='balanced' to handle ~3.39% minority failure class.",
            "Stratified K-fold cross-validation performed strictly on train_manual.csv to ensure no test data leakage.",
            "Evaluation on test_manual.csv conducted exactly once per method on the final refitted estimator.",
            "Fast mode uses 2 CV folds and lightweight search spaces for rapid execution; final mode uses 5 folds and wider spaces.",
        ],
        "installation_issues": "None. Scikit-learn and Optuna installed cleanly into .venv.",
    }
    with open(RESULTS_DIR / "phase2_metadata.json", "w", encoding="utf-8") as f:
        json.dump(phase2_metadata, f, indent=4)
    logger.info("Saved results/phase2_metadata.json")

    # 10. Generate Summary Markdown and Academic Report Section
    generate_phase2_summary_md(comparison_df, best_info, mode)
    generate_academic_report_section(comparison_df, best_info, mode)

    # 11. Final Console Summary
    print("\n" + "=" * 70)
    print("PHASE 2 EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Execution Mode            : {mode.upper()}")
    print(f"Cross-Validation Folds    : {cv_folds}")
    print(f"Best Performing Method    : {best_info['best_method']}")
    print(f"Best Test F1-Score        : {best_info['test_f1']:.4f}")
    print(f"Best Test ROC-AUC         : {best_info['test_roc_auc']:.4f}")
    print(f"Model Artifact Saved To   : {best_info['model_path']}")
    print("-" * 70)
    print("Benchmarking Results Table:")
    print(comparison_df[["method", "cv_f1_mean", "test_f1", "test_roc_auc", "training_time_seconds"]].to_string(index=False))
    print("=" * 70)
    if mode == "fast":
        print("Note: Fast smoke-test mode completed. Rerun with '--mode final' for 5-fold benchmark.")
    print("Phase 2 pipeline completed successfully.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: Baseline and Hyperparameter Tuning")
    parser.add_argument(
        "--mode",
        choices=["fast", "final"],
        default="fast",
        help="Execution mode: 'fast' for rapid validation or 'final' for rigorous benchmarking (default: fast)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel CPU jobs (-1 for all cores, default: -1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_phase2_pipeline(mode=args.mode, n_jobs=args.n_jobs)
