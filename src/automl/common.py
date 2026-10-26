"""Common utilities, dataset loading, and standardized evaluation metrics for Phase 3 AutoML."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import DATA_PROCESSED_DIR, RANDOM_SEED, TARGET_COLUMN
from src.utils import set_random_seed, setup_logger

logger = setup_logger("automl_common")


def load_automl_datasets(
    mode: str = "fast",
    fast_sample_rows: int = 3000,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Loads Phase 1 AutoML datasets.

    In fast mode, samples `fast_sample_rows` from training set using stratified sampling
    to satisfy execution time limits while keeping the held-out test set completely untouched.
    """
    set_random_seed(random_state)
    train_path = DATA_PROCESSED_DIR / "train_automl.csv"
    test_path = DATA_PROCESSED_DIR / "test_automl.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"AutoML datasets missing in {DATA_PROCESSED_DIR}. Run Phase 1 first."
        )

    logger.info(f"Loading AutoML datasets from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Clean string categorical type
    train_df["type"] = train_df["type"].astype(str)
    test_df["type"] = test_df["type"].astype(str)

    # In fast mode, optionally take a stratified subsample of training data
    if mode == "fast" and len(train_df) > fast_sample_rows:
        logger.info(f"Fast mode active: Stratified sampling {fast_sample_rows} rows from {len(train_df)} training instances...")
        split = StratifiedShuffleSplit(n_splits=1, train_size=fast_sample_rows, random_state=random_state)
        for sample_idx, _ in split.split(train_df, train_df[TARGET_COLUMN]):
            train_df = train_df.iloc[sample_idx].reset_index(drop=True)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)

    metadata = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_positive_rate": float(y_train.mean() * 100),
        "test_positive_rate": float(y_test.mean() * 100),
        "mode": mode,
        "features": list(X_train.columns),
    }

    logger.info(
        f"Datasets ready: Train shape={X_train.shape} (failures={y_train.sum()}, {metadata['train_positive_rate']:.2f}%), "
        f"Test shape={X_test.shape} (failures={y_test.sum()}, {metadata['test_positive_rate']:.2f}%)"
    )

    return X_train, y_train, X_test, y_test, metadata


def compute_automl_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculates standardized classification metrics including F1, ROC-AUC, and PR-AUC."""
    # Ensure binary discrete predictions
    if y_pred is None and y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # ROC-AUC and PR-AUC computation
    roc_auc = np.nan
    pr_auc = np.nan
    if y_prob is not None:
        try:
            # Handle multi-column probabilities
            if hasattr(y_prob, "ndim") and y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            roc_auc = float(roc_auc_score(y_true, y_prob))
            precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recalls, precisions))
        except Exception as e:
            logger.warning(f"Error calculating ROC-AUC or PR-AUC: {e}")

    return {
        "test_f1": round(f1, 4),
        "test_roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
        "test_pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
        "test_precision": round(precision, 4),
        "test_recall": round(recall, 4),
        "test_accuracy": round(accuracy, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
