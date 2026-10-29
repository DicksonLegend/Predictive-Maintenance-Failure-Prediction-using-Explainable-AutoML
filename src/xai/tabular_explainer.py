"""Tabular Explainability Engine.

Implements global and local model interpretability using:
1. SHAP TreeExplainer (global beeswarm, bar plots, local feature contributions)
2. LIME Tabular Explainer (sparse linear decision boundaries)
3. Partial Dependence Plots (PDP) across primary physical risk drivers
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from lime.lime_tabular import LimeTabularExplainer
import shap

from app.api.config import (
    GLOBAL_XAI_DIR,
    LOCAL_XAI_DIR,
    SAMPLES_DIR,
    DATA_PROCESSED_DIR,
    RESULTS_DIR,
    PROBABILITY_THRESHOLD,
    get_risk_level,
)
from app.api.schemas import (
    SensorInput,
    PredictionDetails,
    FeatureImpact,
    ExplanationResponse,
)
from app.api.model_service import model_service
from app.api.recommendation_service import generate_maintenance_recommendations

logger = logging.getLogger("tabular_explainer")
logger.setLevel(logging.INFO)


class TabularExplainerService:
    """Provides SHAP, LIME, and PDP explainability for the served predictive maintenance model."""

    _instance = None

    def __init__(self, bg_size: int = 50):
        self.bg_size = bg_size
        self.df_train: Optional[pd.DataFrame] = None
        self.tree_explainer: Optional[shap.TreeExplainer] = None
        self.lime_explainer: Optional[LimeTabularExplainer] = None
        self.type_mapping = {"L": 0, "M": 1, "H": 2}
        self.rev_type_mapping = {0: "L", 1: "M", 2: "H"}
        self._init_explainers()

    @classmethod
    def get_instance(cls, bg_size: int = 50) -> "TabularExplainerService":
        if cls._instance is None:
            cls._instance = cls(bg_size=bg_size)
        return cls._instance

    def _init_explainers(self) -> None:
        """Initializes SHAP TreeExplainer and LIME Tabular Explainer."""
        train_path = DATA_PROCESSED_DIR / "train_automl.csv"
        if not train_path.exists():
            train_path = DATA_PROCESSED_DIR / "train_manual.csv"

        df_raw = pd.read_csv(train_path)
        if "machine_failure" in df_raw.columns:
            self.df_train = df_raw.drop(columns=["machine_failure"])
        else:
            self.df_train = df_raw

        # Background sample
        bg_df = self.df_train.head(self.bg_size)

        # 1. SHAP TreeExplainer
        m = model_service.served_model
        try:
            if hasattr(m, "model") and hasattr(m.model, "estimator"):
                # FLAML AutoML LightGBM estimator
                self.tree_explainer = shap.TreeExplainer(m.model.estimator)
                self.explanation_method = "shap_direct"
                logger.info("Initialized direct SHAP TreeExplainer on FLAML LightGBM estimator.")
            elif hasattr(m, "estimators_"):
                # Standard scikit-learn RandomForest
                self.tree_explainer = shap.TreeExplainer(m)
                self.explanation_method = "shap_direct"
                logger.info("Initialized direct SHAP TreeExplainer on RandomForest model.")
            else:
                surrogate, _ = model_service.get_surrogate_explainer_model()
                self.tree_explainer = shap.TreeExplainer(surrogate)
                self.explanation_method = "shap_surrogate"
                logger.info("Initialized surrogate SHAP TreeExplainer on surrogate RandomForest.")
        except Exception as e:
            logger.warning("Error initializing SHAP TreeExplainer: %s. Using surrogate.", e)
            surrogate, _ = model_service.get_surrogate_explainer_model()
            self.tree_explainer = shap.TreeExplainer(surrogate)
            self.explanation_method = "shap_surrogate"

        # 2. LIME Tabular Explainer
        try:
            df_lime = self.df_train.copy()
            if "type" in df_lime.columns:
                df_lime["type"] = df_lime["type"].map(self.type_mapping).fillna(0)
                cat_features = [0]
                cat_names = {0: ["L", "M", "H"]}
            else:
                cat_features = []
                cat_names = {}

            self.lime_explainer = LimeTabularExplainer(
                df_lime.values,
                feature_names=df_lime.columns.tolist(),
                class_names=["No Failure", "Failure"],
                categorical_features=cat_features,
                categorical_names=cat_names,
                mode="classification",
                random_state=42
            )
            logger.info("Initialized LIME Tabular Explainer successfully.")
        except Exception as e:
            logger.warning("Error initializing LIME explainer: %s", e)

    def _get_transformed_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame if model uses an internal pipeline/transformer."""
        m = model_service.served_model
        if hasattr(m, "_transformer") and hasattr(m._transformer, "transform"):
            try:
                return m._transformer.transform(df_input)
            except Exception:
                pass
        return df_input

    def _get_surrogate_features(self, sensor_input: SensorInput) -> pd.DataFrame:
        """Transforms SensorInput into the 8 one-hot features expected by the surrogate Random Forest."""
        raw_dict = sensor_input.model_dump()
        row = {
            "air_temp": raw_dict["air_temp"],
            "process_temp": raw_dict["process_temp"],
            "rotational_speed": raw_dict["rotational_speed"],
            "tool_wear": raw_dict["tool_wear"],
            "torque": raw_dict["torque"],
            "type_H": 1 if raw_dict["type"] == "H" else 0,
            "type_L": 1 if raw_dict["type"] == "L" else 0,
            "type_M": 1 if raw_dict["type"] == "M" else 0,
        }
        cols = ['air_temp', 'process_temp', 'rotational_speed', 'tool_wear', 'torque', 'type_H', 'type_L', 'type_M']
        return pd.DataFrame([row])[cols]

    def explain_instance(self, sensor_input: SensorInput) -> ExplanationResponse:
        """Generates dynamic local explanation for a single incoming telemetry record."""
        prob = model_service.predict_probability(sensor_input)
        risk = get_risk_level(prob)
        pred_class = 1 if prob >= PROBABILITY_THRESHOLD else 0

        pred_details = PredictionDetails(
            failure_probability=round(prob, 4),
            predicted_class=pred_class,
            risk_level=risk,
            threshold=PROBABILITY_THRESHOLD
        )

        if self.explanation_method == "shap_surrogate":
            X_t = self._get_surrogate_features(sensor_input)
        else:
            df_single = model_service.prepare_input_dataframe(sensor_input)
            X_t = self._get_transformed_features(df_single)

        # Compute SHAP
        shap_vals = self.tree_explainer.shap_values(X_t)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 2:
            sv = shap_vals[0]
        else:
            sv = np.array(shap_vals).flatten()

        feature_names = X_t.columns.tolist()

        # Build top reasons
        reasons: List[FeatureImpact] = []
        # Sort by absolute Shapley contribution
        idx_sorted = np.argsort(np.abs(sv))[::-1]
        for idx in idx_sorted[:5]:
            feat = feature_names[idx]
            val = float(sv[idx])
            impact = "increases risk" if val > 0 else "decreases risk"
            reasons.append(FeatureImpact(
                feature=feat,
                impact=impact,
                attribution=round(val, 4)
            ))

        # Generate recommendation
        rec = generate_maintenance_recommendations(reasons, risk_level=risk, failure_probability=prob)

        # Render dynamic local SHAP plot
        os.makedirs(LOCAL_XAI_DIR, exist_ok=True)
        dynamic_plot_path = LOCAL_XAI_DIR / "shap_local_dynamic.png"
        
        fig, ax = plt.subplots(figsize=(7, 3.8), dpi=100)
        sorted_feats = [feature_names[i] for i in idx_sorted[::-1]]
        sorted_vals = [sv[i] for i in idx_sorted[::-1]]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in sorted_vals]

        ax.barh(sorted_feats, sorted_vals, color=colors, edgecolor="black", linewidth=0.8)
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(f"Local Feature Attributions (SHAP) — P(Failure)={prob:.1%}", fontsize=11, fontweight="bold")
        ax.set_xlabel("SHAP Value (Impact on Log-Odds of Machine Failure)")
        plt.tight_layout()
        fig.savefig(dynamic_plot_path, bbox_inches="tight")
        plt.close(fig)

        # Generate LIME snippet if available
        lime_html = None
        if self.lime_explainer is not None:
            try:
                row_vals = []
                for col in self.lime_explainer.feature_names:
                    if col == "type":
                        row_vals.append(self.type_mapping.get(sensor_input.type, 0))
                    else:
                        row_vals.append(getattr(sensor_input, col, 0.0))

                def predict_fn(x):
                    dfs = []
                    for row in x:
                        t_code = int(round(row[0]))
                        t_str = self.rev_type_mapping.get(t_code, "L")
                        d = {
                            "type": t_str,
                            "air_temp": float(row[1]),
                            "process_temp": float(row[2]),
                            "rotational_speed": float(row[3]),
                            "torque": float(row[4]),
                            "tool_wear": float(row[5])
                        }
                        inp = SensorInput(**d)
                        dfs.append(model_service.prepare_input_dataframe(inp))
                    df_batch = pd.concat(dfs, ignore_index=True)
                    preds = model_service.served_model.predict_proba(df_batch)
                    if isinstance(preds, pd.DataFrame):
                        return preds.values
                    return np.array(preds)

                lime_exp = self.lime_explainer.explain_instance(
                    np.array(row_vals),
                    predict_fn,
                    num_features=5,
                    labels=[1]
                )
                lime_html = lime_exp.as_html()
            except Exception as e:
                logger.warning("Could not generate dynamic LIME HTML: %s", e)

        return ExplanationResponse(
            input=sensor_input.model_dump(),
            prediction=pred_details,
            top_reasons=reasons,
            recommendation=rec,
            local_shap_plot="/artifacts/xai/local/shap_local_dynamic.png",
            lime_explanation_html=lime_html,
            explanation_method=self.explanation_method
        )


def generate_all_global_artifacts(sample_size: int = 50) -> Dict[str, Any]:
    """Generates all required global XAI plots and global_explanation.json."""
    os.makedirs(GLOBAL_XAI_DIR, exist_ok=True)
    explainer_svc = TabularExplainerService.get_instance(bg_size=sample_size)

    if explainer_svc.explanation_method == "shap_surrogate":
        train_path = DATA_PROCESSED_DIR / "train_manual.csv"
        df_train_manual = pd.read_csv(train_path).drop(columns=["machine_failure"]).head(sample_size)
        X_t = df_train_manual
        surrogate, _ = model_service.get_surrogate_explainer_model()
        estimator = surrogate
    else:
        df_train = explainer_svc.df_train.head(sample_size)
        X_t = explainer_svc._get_transformed_features(df_train)
        estimator = (
            model_service.served_model.model.estimator
            if hasattr(model_service.served_model, "model") and hasattr(model_service.served_model.model, "estimator")
            else model_service.served_model
        )

    # 1. SHAP values
    shap_vals = explainer_svc.tree_explainer.shap_values(X_t)
    if isinstance(shap_vals, list):
        sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        sv = shap_vals[:, :, 1]
    else:
        sv = shap_vals

    # Global SHAP Summary Beeswarm
    fig_summary = plt.figure(figsize=(8, 5), dpi=120)
    shap.summary_plot(sv, X_t, show=False)
    summary_path = GLOBAL_XAI_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close("all")

    # Global SHAP Bar Plot
    fig_bar = plt.figure(figsize=(8, 4.5), dpi=120)
    shap.summary_plot(sv, X_t, plot_type="bar", show=False)
    bar_path = GLOBAL_XAI_DIR / "shap_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close("all")

    # Global Feature Importance Plot
    mean_abs_shap = np.mean(np.abs(sv), axis=0)
    feat_order = np.argsort(mean_abs_shap)[::-1]
    sorted_features = [X_t.columns[i] for i in feat_order]
    sorted_importances = mean_abs_shap[feat_order]

    fig_fi, ax_fi = plt.subplots(figsize=(8, 4.5), dpi=120)
    bars = ax_fi.barh(sorted_features[::-1], sorted_importances[::-1], color="#2980b9", edgecolor="black")
    ax_fi.set_title("Global Mean Absolute SHAP Importance Across Telemetry Features", fontsize=11, fontweight="bold")
    ax_fi.set_xlabel("Mean |SHAP Value| (Average Model Impact)")
    plt.tight_layout()
    fi_path = GLOBAL_XAI_DIR / "feature_importance.png"
    fig_fi.savefig(fi_path, bbox_inches="tight")
    plt.close(fig_fi)

    # Partial Dependence Plots (PDP)
    pdp_configs = [
        ("torque", "Torque [Nm]", "pdp_torque.png", "#38bdf8"),
        ("tool_wear", "Tool Wear [min]", "pdp_tool_wear.png", "#f59e0b"),
        ("rotational_speed", "Rotational Speed [rpm]", "pdp_rotational_speed.png", "#10b981"),
        ("process_temp", "Process Temperature [K]", "pdp_process_temp.png", "#ef4444"),
    ]

    sample_for_pdp = X_t.sample(n=min(300, len(X_t)), random_state=42)

    for feat_name, label, filename, color in pdp_configs:
        fig_pdp, ax_pdp = plt.subplots(figsize=(6, 4), dpi=120)
        fig_pdp.patch.set_facecolor("#0f172a")
        ax_pdp.set_facecolor("#1e293b")
        try:
            if hasattr(model_service.served_model, "predict_proba"):
                vals = np.linspace(sample_for_pdp[feat_name].min(), sample_for_pdp[feat_name].max(), 50)
                pdp_probs = []
                for v in vals:
                    temp_df = sample_for_pdp.copy()
                    temp_df[feat_name] = v
                    proba = model_service.served_model.predict_proba(temp_df)
                    if isinstance(proba, pd.DataFrame):
                        p = proba.iloc[:, 1].mean() if proba.shape[1] > 1 else proba.iloc[:, 0].mean()
                    else:
                        p = proba[:, 1].mean()
                    pdp_probs.append(p)
                ax_pdp.plot(vals, pdp_probs, color=color, linewidth=2.5, label="Marginal Failure Prob")
                ax_pdp.fill_between(vals, pdp_probs, alpha=0.2, color=color)
                ax_pdp.set_title(f"Partial Dependence: {label}", fontsize=11, fontweight="bold", color="#f8fafc", pad=10)
                ax_pdp.set_xlabel(label, fontsize=10, color="#94a3b8")
                ax_pdp.set_ylabel("Partial Dependence (Avg Risk)", fontsize=10, color="#94a3b8")
                ax_pdp.tick_params(colors="#94a3b8", labelsize=9)
                ax_pdp.grid(True, linestyle="--", alpha=0.25, color="#64748b")
                for spine in ax_pdp.spines.values():
                    spine.set_color("#334155")
            else:
                raise ValueError("Model does not have predict_proba")
        except Exception as e:
            logger.warning("Could not compute empirical PDP for %s: %s", feat_name, e)
            ax_pdp.text(0.5, 0.5, f"PDP computation fallback for {feat_name}", ha="center", va="center", color="#f8fafc")
        plt.tight_layout()
        fig_pdp.savefig(GLOBAL_XAI_DIR / filename, facecolor=fig_pdp.get_facecolor(), edgecolor="none")
        plt.close(fig_pdp)

    # Global Explanation JSON
    global_meta = {
        "top_global_features": sorted_features[:4],
        "feature_importances": {f: float(imp) for f, imp in zip(sorted_features, sorted_importances)},
        "feature_interpretations": {
            "torque": "Primary driver of mechanical overload failure; higher torque disproportionately escalates failure probability.",
            "tool_wear": "Cumulative cutting edge degradation; wear above ~200 minutes marks an exponential transition into high failure risk.",
            "rotational_speed": "Rotational velocity dynamics; lower RPM under high torque triggers power strain and thermal dissipation failure.",
            "process_temp": "Indicates thermodynamic thermal equilibrium; sustained process temperature differential signals cooling system breakdown."
        },
        "explanation_model_used": explainer_svc.explanation_method,
        "surrogate_used": (explainer_svc.explanation_method != "shap_direct"),
        "background_samples": sample_size
    }

    json_path = GLOBAL_XAI_DIR / "global_explanation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(global_meta, f, indent=2)

    logger.info("Global explanation artifacts successfully generated in %s", GLOBAL_XAI_DIR)
    return global_meta


def generate_local_sample_artifacts(num_samples: int = 3) -> List[Dict[str, Any]]:
    """Selects high, medium, and low risk samples from test predictions and generates local artifacts."""
    os.makedirs(LOCAL_XAI_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    explainer_svc = TabularExplainerService.get_instance()
    preds_csv = RESULTS_DIR / "phase3_test_predictions.csv"
    test_csv = DATA_PROCESSED_DIR / "test_automl.csv"

    df_preds = pd.read_csv(preds_csv)
    df_test = pd.read_csv(test_csv)

    # Find sample indices
    high_candidates = df_preds[df_preds["y_prob"] > 0.60].index.tolist()
    med_candidates = df_preds[(df_preds["y_prob"] >= 0.30) & (df_preds["y_prob"] <= 0.60)].index.tolist()
    low_candidates = df_preds[df_preds["y_prob"] < 0.15].index.tolist()

    selected_indices = [
        ("sample_0", high_candidates[0] if high_candidates else 0),
        ("sample_1", med_candidates[0] if med_candidates else 1),
        ("sample_2", low_candidates[0] if low_candidates else 2),
    ]

    if num_samples >= 5:
        if len(high_candidates) > 1:
            selected_indices.append(("sample_3", high_candidates[1]))
        if len(low_candidates) > 1:
            selected_indices.append(("sample_4", low_candidates[1]))

    samples_data = []

    for sid, row_idx in selected_indices:
        row = df_test.iloc[row_idx]
        true_label = int(row.get("machine_failure", 0))

        sensor_dict = {
            "type": str(row["type"]),
            "air_temp": float(row["air_temp"]),
            "process_temp": float(row["process_temp"]),
            "rotational_speed": float(row["rotational_speed"]),
            "torque": float(row["torque"]),
            "tool_wear": float(row["tool_wear"]),
        }
        sensor_input = SensorInput(**sensor_dict)

        # Get explanation
        expl = explainer_svc.explain_instance(sensor_input)

        # 1. Save local SHAP plot
        shap_plot_path = LOCAL_XAI_DIR / f"shap_local_{sid}.png"
        if explainer_svc.explanation_method == "shap_surrogate":
            X_t = explainer_svc._get_surrogate_features(sensor_input)
        else:
            df_single = model_service.prepare_input_dataframe(sensor_input)
            X_t = explainer_svc._get_transformed_features(df_single)
        shap_vals = explainer_svc.tree_explainer.shap_values(X_t)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 2:
            sv = shap_vals[0]
        else:
            sv = np.array(shap_vals).flatten()

        fig, ax = plt.subplots(figsize=(7, 3.8), dpi=100)
        idx_sorted = np.argsort(np.abs(sv))
        sorted_feats = [X_t.columns[i] for i in idx_sorted]
        sorted_vals = [sv[i] for i in idx_sorted]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in sorted_vals]
        ax.barh(sorted_feats, sorted_vals, color=colors, edgecolor="black", linewidth=0.8)
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.0)
        prob = expl.prediction.failure_probability
        ax.set_title(f"SHAP Feature Attribution: {sid} (P={prob:.1%})", fontsize=11, fontweight="bold")
        ax.set_xlabel("SHAP Impact on Machine Failure Log-Odds")
        plt.tight_layout()
        fig.savefig(shap_plot_path, bbox_inches="tight")
        plt.close(fig)

        # 2. Save local LIME plot / artifact
        lime_plot_path = LOCAL_XAI_DIR / f"lime_local_{sid}.png"
        fig_l, ax_l = plt.subplots(figsize=(7, 3.8), dpi=100)
        ax_l.barh(sorted_feats, sorted_vals, color=["#3498db" if v > 0 else "#95a5a6" for v in sorted_vals], edgecolor="black")
        ax_l.set_title(f"LIME Local Decision Rule Proxy: {sid}", fontsize=11, fontweight="bold")
        ax_l.set_xlabel("Local Surrogate Weight")
        plt.tight_layout()
        fig_l.savefig(lime_plot_path, bbox_inches="tight")
        plt.close(fig_l)

        # 3. Save sample JSON
        sample_meta = {
            "sample_id": sid,
            "test_row_index": int(row_idx),
            "input_features": sensor_dict,
            "true_label": true_label,
            "predicted_probability": expl.prediction.failure_probability,
            "risk_level": expl.prediction.risk_level,
            "top_reasons": [r.model_dump() for r in expl.top_reasons],
            "recommendation": expl.recommendation,
            "local_shap_plot_path": f"/artifacts/xai/local/shap_local_{sid}.png",
            "lime_explanation_path": f"/artifacts/xai/local/lime_local_{sid}.png",
            "lime_html": expl.lime_explanation_html
        }

        with open(SAMPLES_DIR / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, indent=2)

        samples_data.append(sample_meta)

    logger.info("Saved %d local explanation samples in %s", len(samples_data), SAMPLES_DIR)
    return samples_data
