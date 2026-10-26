"""Critical evaluation module: Overfitting audit, class imbalance analysis, and group fairness."""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score

from src.config import PLOTS_DIR, RESULTS_DIR
from src.utils import setup_logger

logger = setup_logger("critical_evaluation")


def evaluate_overfitting(
    models_data: List[Dict[str, Any]],
    gap_threshold: float = 0.08,
    save_path: Path = RESULTS_DIR / "overfitting_check.csv",
    plot_path: Path = PLOTS_DIR / "overfitting_check.png",
) -> pd.DataFrame:
    """Computes train/validation vs. test gaps and assigns overfitting diagnostic flags."""
    records = []
    for m in models_data:
        val_f1 = m.get("cv_or_validation_f1")
        test_f1 = m.get("test_f1")

        if val_f1 is None or np.isnan(val_f1):
            val_f1 = m.get("cv_f1_mean", np.nan)

        f1_gap = round(float(val_f1 - test_f1), 4) if (val_f1 is not None and not np.isnan(val_f1)) else np.nan

        if np.isnan(f1_gap):
            flag = "unverified"
            notes = "Validation F1 score not logged or unavailable."
        elif test_f1 - val_f1 > gap_threshold:
            flag = "unusual_distribution_shift_or_fast_mode_artifact"
            notes = f"Test F1 is higher than validation F1 by {abs(f1_gap):.4f} (typical in fast-mode small sample validation)."
        elif val_f1 - test_f1 > gap_threshold:
            flag = "possible_overfitting"
            notes = f"Validation F1 exceeds test F1 by {f1_gap:.4f}, suggesting generalization gap."
        else:
            flag = "stable"
            notes = f"Gap of {abs(f1_gap):.4f} is within stable tolerance (threshold={gap_threshold})."

        records.append({
            "method": m["method"],
            "validation_or_cv_f1": round(val_f1, 4) if not np.isnan(val_f1) else np.nan,
            "test_f1": test_f1,
            "f1_gap": f1_gap,
            "roc_auc_gap_if_available": np.nan,
            "overfitting_flag": flag,
            "notes": notes,
        })

    df = pd.DataFrame(records)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved overfitting check report to {save_path}")

    # Generate Overfitting Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width / 2, df["validation_or_cv_f1"], width, label="Validation / CV F1", color="#4C72B0")
    ax.bar(x + width / 2, df["test_f1"], width, label="Held-Out Test F1", color="#55A868")

    ax.set_title("Overfitting Assessment: Validation vs. Test Performance Gap", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return df


def generate_class_imbalance_analysis(
    y_train: pd.Series,
    y_test: pd.Series,
    best_model_metrics: Dict[str, Any],
    save_path: Path = RESULTS_DIR / "class_imbalance_report.csv",
) -> pd.DataFrame:
    """Generates class imbalance report and corresponding diagnostic plots."""
    train_pos = int(y_train.sum())
    train_neg = int(len(y_train) - train_pos)
    train_pos_pct = round(float(train_pos / len(y_train) * 100), 2)

    test_pos = int(y_test.sum())
    test_neg = int(len(y_test) - test_pos)
    test_pos_pct = round(float(test_pos / len(y_test) * 100), 2)

    cm = best_model_metrics.get("confusion_matrix")
    if cm is not None and cm.shape == (2, 2):
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    record = {
        "train_positive_count": train_pos,
        "train_negative_count": train_neg,
        "train_positive_percentage": train_pos_pct,
        "test_positive_count": test_pos,
        "test_negative_count": test_neg,
        "test_positive_percentage": test_pos_pct,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "selected_model_f1": best_model_metrics["test_f1"],
        "selected_model_recall": best_model_metrics["test_recall"],
        "selected_model_precision": best_model_metrics["test_precision"],
        "selected_model_pr_auc": best_model_metrics.get("test_pr_auc", np.nan),
    }

    df = pd.DataFrame([record])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved class imbalance report to {save_path}")

    # Plot 1: Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    categories = ["Train: Nominal (0)", "Train: Failure (1)", "Test: Nominal (0)", "Test: Failure (1)"]
    counts = [train_neg, train_pos, test_neg, test_pos]
    colors = ["#4C72B0", "#C44E52", "#55A868", "#E24A33"]

    bars = ax.bar(categories, counts, color=colors, width=0.5)
    ax.set_title("Severe Class Imbalance: Train vs. Test Partitions", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("Instance Count", fontsize=10)
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 15, f"{h:,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase3_class_distribution.png", dpi=300)
    plt.close()

    # Plot 2: Confusion Matrix for Best Model
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Failure (0)", "Failure (1)"],
                yticklabels=["No Failure (0)", "Failure (1)"], ax=ax)
    ax.set_title(f"Phase 3 Best Model Confusion Matrix\n({best_model_metrics['method']})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual True", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase3_confusion_matrix_best_model.png", dpi=300)
    plt.close()

    # Plot 3: Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(6.5, 5))
    y_prob = best_model_metrics.get("y_prob")
    if y_prob is not None:
        try:
            precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
            pr_val = auc(recalls, precisions)
            ax.plot(recalls, precisions, color="#C44E52", lw=2, label=f"PR Curve (AUC = {pr_val:.3f})")
            ax.axhline(test_pos / len(y_test), color="k", linestyle="--", alpha=0.6, label=f"No-Skill Baseline ({test_pos_pct}%)")
            ax.set_title("Precision-Recall Curve (Minority Failure Class)", fontsize=12, pad=12, fontweight="bold")
            ax.set_xlabel("Recall (Sensitivity)", fontsize=10)
            ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=10)
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.05])
            ax.legend(loc="upper right")
        except Exception as e:
            logger.warning(f"Could not render PR curve: {e}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase3_precision_recall_curve.png", dpi=300)
    plt.close()

    return df


def evaluate_group_bias(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_path: Path = RESULTS_DIR / "group_performance.csv",
    plot_path: Path = PLOTS_DIR / "group_performance.png",
) -> pd.DataFrame:
    """Evaluates performance disaggregated across machine variants (L, M, H)."""
    groups = ["L", "M", "H"]
    records = []

    eval_df = pd.DataFrame({
        "machine_type": test_df["type"].values,
        "y_true": y_true.values,
        "y_pred": y_pred,
    })

    for grp in groups:
        sub = eval_df[eval_df["machine_type"] == grp]
        total = len(sub)
        if total == 0:
            continue

        actual_fails = int(sub["y_true"].sum())
        actual_rate = round(float(actual_fails / total), 4)

        pred_fails = int(sub["y_pred"].sum())
        pred_rate = round(float(pred_fails / total), 4)

        cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        fpr = round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0
        fnr = round(float(fn / (fn + tp)), 4) if (fn + tp) > 0 else 0.0

        prec = round(float(precision_score(sub["y_true"], sub["y_pred"], zero_division=0)), 4)
        rec = round(float(recall_score(sub["y_true"], sub["y_pred"], zero_division=0)), 4)
        f1 = round(float(f1_score(sub["y_true"], sub["y_pred"], zero_division=0)), 4)

        records.append({
            "machine_type": grp,
            "total_samples": total,
            "actual_failure_count": actual_fails,
            "actual_failure_rate": actual_rate,
            "predicted_failure_count": pred_fails,
            "predicted_failure_rate": pred_rate,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

    group_df = pd.DataFrame(records)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    group_df.to_csv(save_path, index=False)
    logger.info(f"Saved group bias performance to {save_path}")

    # Generate Group Performance Comparison Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(group_df))
    width = 0.2

    ax.bar(x - 1.5 * width, group_df["actual_failure_rate"], width, label="Actual Failure Rate", color="#4C72B0")
    ax.bar(x - 0.5 * width, group_df["predicted_failure_rate"], width, label="Predicted Failure Rate", color="#55A868")
    ax.bar(x + 0.5 * width, group_df["false_positive_rate"], width, label="False Positive Rate", color="#8172B2")
    ax.bar(x + 1.5 * width, group_df["false_negative_rate"], width, label="False Negative Rate", color="#C44E52")

    ax.set_title("Group Fairness Analysis by Machine Variant (L, M, H)", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("Rate Proportion", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {g}" for g in group_df["machine_type"]], fontsize=10)
    ax.set_ylim(0, max(0.5, group_df[["false_negative_rate", "false_positive_rate"]].max().max() * 1.35))
    ax.legend(loc="upper right")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return group_df
