"""Feature importance and SHAP attribution utilities for Phase 3 AutoML models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.config import PLOTS_DIR, RANDOM_SEED
from src.utils import set_random_seed, setup_logger

logger = setup_logger("feature_importance")


def extract_model_feature_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[List[str]] = None,
    max_features: int = 15,
) -> Tuple[pd.DataFrame, str]:
    """Extracts feature importance using SHAP, native importances, or permutation importance."""
    set_random_seed(RANDOM_SEED)
    if feature_names is None:
        feature_names = list(X_test.columns)

    importances = None
    method_used = "native"

    # 1. SHAP TreeExplainer
    try:
        import shap

        underlying = getattr(model, "model", model)
        estimator = getattr(underlying, "estimator", underlying)
        explainer = shap.TreeExplainer(estimator)
        # Sample for fast computation
        sample_size = min(100, len(X_test))
        sample_X = X_test.head(sample_size)
        shap_values = explainer.shap_values(sample_X)

        if isinstance(shap_values, list) and len(shap_values) > 1:
            vals = np.abs(shap_values[1]).mean(axis=0)
        else:
            vals = np.abs(shap_values).mean(axis=0)

        if len(vals) == len(feature_names):
            importances = vals
            method_used = "SHAP TreeExplainer"
    except Exception as e:
        logger.debug(f"SHAP TreeExplainer not applicable: {e}")

    # 2. Native feature_importances_
    if importances is None:
        try:
            underlying = getattr(model, "model", model)
            estimator = getattr(underlying, "estimator", underlying)
            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
                method_used = "Native Model Importance"
        except Exception:
            pass

    # 3. AutoGluon feature_importance method
    if importances is None and hasattr(model, "feature_importance"):
        try:
            test_data = X_test.copy()
            test_data["machine_failure"] = y_test.values
            fi_df = model.feature_importance(test_data, silent=True)
            if fi_df is not None and not fi_df.empty:
                df_out = fi_df.reset_index().rename(columns={"index": "feature", "importance": "importance"})
                return df_out[["feature", "importance"]].sort_values(by="importance", ascending=False), "AutoGluon Permutation Importance"
        except Exception as e:
            logger.debug(f"AutoGluon native feature importance failed: {e}")

    # 4. Scikit-learn Permutation Importance
    if importances is None:
        try:
            r = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=RANDOM_SEED, scoring="f1")
            importances = r.importances_mean
            method_used = "Permutation Importance"
        except Exception as e:
            logger.debug(f"Permutation importance failed: {e}")

    # Fallback default uniform if all else fails
    if importances is None or len(importances) != len(feature_names):
        importances = np.ones(len(feature_names)) / len(feature_names)
        method_used = "Uniform Fallback"

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df.head(max_features), method_used
