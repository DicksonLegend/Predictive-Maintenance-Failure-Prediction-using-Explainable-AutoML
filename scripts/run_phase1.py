"""Master Phase 1 Execution Script: Project Foundation, Data Pipeline, and EDA."""

import shutil
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    DATA_PROCESSED_DIR,
    NUMERIC_FEATURES,
    PLOTS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    SCREENSHOTS_DIR,
    TARGET_COLUMN,
    init_directories,
)
from src.data.load_data import load_ai4i_data
from src.data.preprocess import preprocess_and_save_all
from src.utils import set_random_seed, setup_logger

logger = setup_logger("run_phase1")


def generate_eda_tables(df: pd.DataFrame, results_dir: Path = RESULTS_DIR) -> dict[str, Path]:
    """Generates and writes all required Phase 1 EDA summary CSV files."""
    results_dir.mkdir(parents=True, exist_ok=True)
    table_paths = {}

    # 1. Dataset summary
    total_rows = len(df)
    total_cols = len(df.columns)
    target_positives = int(df[TARGET_COLUMN].sum())
    target_positive_rate = float(df[TARGET_COLUMN].mean() * 100)

    summary_records = [
        {"metric": "total_rows", "value": total_rows},
        {"metric": "total_columns", "value": total_cols},
        {"metric": "numeric_features_count", "value": len(NUMERIC_FEATURES)},
        {"metric": "categorical_features_count", "value": 1},
        {"metric": "target_column", "value": TARGET_COLUMN},
        {"metric": "failure_cases_count", "value": target_positives},
        {"metric": "non_failure_cases_count", "value": total_rows - target_positives},
        {"metric": "failure_rate_percent", "value": round(target_positive_rate, 2)},
        {"metric": "has_duplicates", "value": bool(df.duplicated().any())},
    ]
    summary_df = pd.DataFrame(summary_records)
    path_summary = results_dir / "dataset_summary.csv"
    summary_df.to_csv(path_summary, index=False)
    table_paths["dataset_summary"] = path_summary

    # 2. Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100
    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_counts.values,
        "missing_percentage": np.round(missing_pct.values, 2),
    })
    path_missing = results_dir / "missing_values.csv"
    missing_df.to_csv(path_missing, index=False)
    table_paths["missing_values"] = path_missing

    # 3. Class distribution
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    class_pct = (class_counts / total_rows) * 100
    class_df = pd.DataFrame({
        "class": class_counts.index,
        "label": ["No Failure (0)", "Machine Failure (1)"],
        "count": class_counts.values,
        "percentage": np.round(class_pct.values, 2),
    })
    path_class = results_dir / "class_distribution.csv"
    class_df.to_csv(path_class, index=False)
    table_paths["class_distribution"] = path_class

    # 4. Feature summary
    num_summary = df[NUMERIC_FEATURES].describe().T
    num_summary["skewness"] = df[NUMERIC_FEATURES].skew()
    num_summary["kurtosis"] = df[NUMERIC_FEATURES].kurt()
    num_summary = num_summary.reset_index().rename(columns={"index": "feature"})
    path_feature = results_dir / "feature_summary.csv"
    num_summary.to_csv(path_feature, index=False)
    table_paths["feature_summary"] = path_feature

    logger.info("EDA summary tables generated successfully in results/")
    return table_paths


def generate_eda_plots(
    df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
    screenshots_dir: Path = SCREENSHOTS_DIR,
) -> dict[str, Path]:
    """Generates all required Phase 1 EDA plots and mirrors them into reports/screenshots/."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Class Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    palette = ["#4C72B0", "#C44E52"]
    bars = ax.bar(["No Failure (0)", "Failure (1)"], class_counts.values, color=palette, width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (yval * 0.015) + 5,
            f"{yval:,} ({yval / len(df) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Machine Failure Class Distribution (Target Imbalance)", fontsize=13, pad=12)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_ylim(0, max(class_counts.values) * 1.15)
    path_class_plot = plots_dir / "class_distribution.png"
    plt.tight_layout()
    plt.savefig(path_class_plot, dpi=300)
    plt.close()
    plot_paths["class_distribution"] = path_class_plot

    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_cols = NUMERIC_FEATURES + [TARGET_COLUMN]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature & Target Pearson Correlation Matrix", fontsize=13, pad=12)
    path_corr_plot = plots_dir / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(path_corr_plot, dpi=300)
    plt.close()
    plot_paths["correlation_heatmap"] = path_corr_plot

    # 3. Missing Values Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    missing_series = df.isnull().sum()
    bars = ax.bar(missing_series.index, missing_series.values, color="#55A868")
    ax.set_title("Missing Values Count Across Features", fontsize=13, pad=12)
    ax.set_ylabel("Missing Count", fontsize=11)
    ax.set_ylim(0, max(10, missing_series.max() + 5))
    ax.tick_params(axis="x", rotation=30)
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.5,
            str(int(yval)),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    path_missing_plot = plots_dir / "missing_values.png"
    plt.tight_layout()
    plt.savefig(path_missing_plot, dpi=300)
    plt.close()
    plot_paths["missing_values"] = path_missing_plot

    # 4. Feature Distributions Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, feature in enumerate(NUMERIC_FEATURES):
        ax = axes[idx]
        sns.histplot(
            data=df,
            x=feature,
            hue=TARGET_COLUMN,
            kde=True,
            bins=30,
            palette={0: "#4C72B0", 1: "#C44E52"},
            ax=ax,
            stat="density",
            common_norm=False,
        )
        ax.set_title(f"Distribution: {feature}", fontsize=11)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)

    # 6th subplot for Categorical Type distribution
    ax_cat = axes[5]
    sns.countplot(
        data=df,
        x="type",
        hue=TARGET_COLUMN,
        palette={0: "#4C72B0", 1: "#C44E52"},
        ax=ax_cat,
        order=["L", "M", "H"],
    )
    ax_cat.set_title("Distribution: Machine Type (L/M/H)", fontsize=11)
    ax_cat.set_xlabel("Machine Type", fontsize=10)
    ax_cat.set_ylabel("Count", fontsize=10)

    path_dist_plot = plots_dir / "feature_distributions.png"
    plt.tight_layout()
    plt.savefig(path_dist_plot, dpi=300)
    plt.close()
    plot_paths["feature_distributions"] = path_dist_plot

    # Mirror plots to reports/screenshots/
    for name, p_path in plot_paths.items():
        dest = screenshots_dir / p_path.name
        shutil.copy2(p_path, dest)
        logger.info(f"Mirrored plot to screenshots: {dest}")

    logger.info("EDA plots generated successfully in plots/ and mirrored to reports/screenshots/")
    return plot_paths


def run_phase1_pipeline() -> None:
    """Executes the complete Phase 1 pipeline."""
    set_random_seed()
    init_directories()

    logger.info("=" * 60)
    logger.info("STARTING PHASE 1: FOUNDATION, DATA PIPELINE & EDA")
    logger.info("=" * 60)

    # 1. Load data (real or synthetic fallback)
    df, source = load_ai4i_data()
    logger.info(f"Step 1 Complete: Dataset loaded. Source='{source}', Shape={df.shape}")

    # 2. Preprocessing & stratified splits
    processed_files = preprocess_and_save_all(df)
    logger.info(f"Step 2 Complete: Stratified train/test splits created for Manual & AutoML.")

    # 3. EDA tables
    eda_tables = generate_eda_tables(df)
    logger.info(f"Step 3 Complete: EDA summary CSV tables generated.")

    # 4. EDA plots
    eda_plots = generate_eda_plots(df)
    logger.info(f"Step 4 Complete: EDA visualization plots generated.")

    # Summary report
    print("\n" + "=" * 60)
    print("PHASE 1 PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Dataset Source           : {source.upper()}")
    print(f"Dataset Shape            : {df.shape[0]} rows, {df.shape[1]} columns")
    failure_cnt = int(df[TARGET_COLUMN].sum())
    total_cnt = len(df)
    print(f"Target Distribution      : Failures={failure_cnt} ({failure_cnt / total_cnt * 100:.2f}%), Non-failures={total_cnt - failure_cnt}")
    print(f"Missing Values           : {int(df.isnull().sum().sum())} total missing cells")
    print("\nGenerated Processed Data:")
    for k, v in processed_files.items():
        print(f"  - {k:<15}: {v.relative_to(ROOT_DIR)}")
    print("\nGenerated EDA Tables:")
    for k, v in eda_tables.items():
        print(f"  - {k:<20}: {v.relative_to(ROOT_DIR)}")
    print("\nGenerated EDA Plots:")
    for k, v in eda_plots.items():
        print(f"  - {k:<20}: {v.relative_to(ROOT_DIR)}")
    print("=" * 60)
    print("Phase 1 pipeline completed successfully.\n")


if __name__ == "__main__":
    run_phase1_pipeline()
