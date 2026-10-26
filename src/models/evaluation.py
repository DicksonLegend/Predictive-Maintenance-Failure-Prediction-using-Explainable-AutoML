"""Model evaluation utilities, metrics computation, and serialization for Phase 2."""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import setup_logger

logger = setup_logger("evaluation")


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    method_name: str,
    cv_f1_mean: float,
    cv_f1_std: float,
    training_time_seconds: float,
    best_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluates a fitted model on held-out test data and returns standardized metrics."""
    logger.info(f"Evaluating {method_name} on held-out test partition ({len(X_test)} samples)...")

    # Predict discrete classes
    y_pred = model.predict(X_test)

    # Predict probabilities for ROC-AUC if supported
    y_proba = None
    test_roc_auc = 0.5
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_test)
            if probas.shape[1] > 1:
                y_proba = probas[:, 1]
            else:
                y_proba = probas[:, 0]
            test_roc_auc = float(roc_auc_score(y_test, y_proba))
        except Exception as e:
            logger.warning(f"Could not compute predict_proba / ROC-AUC for {method_name}: {e}")
            test_roc_auc = 0.5
    elif hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X_test)
            test_roc_auc = float(roc_auc_score(y_test, y_proba))
        except Exception as e:
            logger.warning(f"Could not compute decision_function / ROC-AUC for {method_name}: {e}")
            test_roc_auc = 0.5

    # Compute classification metrics
    test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    test_precision = float(precision_score(y_test, y_pred, zero_division=0))
    test_recall = float(recall_score(y_test, y_pred, zero_division=0))
    test_accuracy = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    logger.info(
        f"[{method_name}] Results -> CV F1: {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f}) | "
        f"Test F1: {test_f1:.4f} | ROC-AUC: {test_roc_auc:.4f} | "
        f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | "
        f"Time: {training_time_seconds:.2f}s"
    )

    return {
        "method": method_name,
        "cv_f1_mean": round(float(cv_f1_mean), 4),
        "cv_f1_std": round(float(cv_f1_std), 4),
        "test_f1": round(float(test_f1), 4),
        "test_roc_auc": round(float(test_roc_auc), 4),
        "test_precision": round(float(test_precision), 4),
        "test_recall": round(float(test_recall), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "training_time_seconds": round(float(training_time_seconds), 4),
        "best_params": best_params or {},
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def save_model(model: Any, filepath: Path) -> Path:
    """Serializes a model object using joblib to filepath."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Saved serialized model to {filepath}")
    return filepath


def load_model(filepath: Path) -> Any:
    """Loads a serialized model object from filepath."""
    if not filepath.exists():
        raise FileNotFoundError(f"Model file does not exist: {filepath}")
    model = joblib.load(filepath)
    logger.info(f"Loaded serialized model from {filepath}")
    return model
