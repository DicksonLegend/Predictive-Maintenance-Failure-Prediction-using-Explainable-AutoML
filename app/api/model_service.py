"""Model Service for loading, managing, and predicting with the best ML model."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from app.api.config import (
    PROJECT_ROOT,
    MODELS_DIR,
    RESULTS_DIR,
    DATA_PROCESSED_DIR,
    PROBABILITY_THRESHOLD,
    get_risk_level,
)
from app.api.schemas import SensorInput, PredictionResponse, PredictionDetails, ModelInfo

logger = logging.getLogger("model_service")
logger.setLevel(logging.INFO)


class ModelService:
    """Manages model loading, telemetry feature transformations, and inference."""

    _instance: Optional["ModelService"] = None

    def __init__(self):
        self.served_model = None
        self.model_info: Dict[str, Any] = {}
        self.surrogate_model = None
        self.surrogate_used = False
        self.explanation_model_type = "direct"
        self._load_best_model()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_best_model(self) -> None:
        """Selects and loads the served model according to Phase 4 priority rules."""
        # Priority 1: Phase 3 Best Model (FLAML / AutoGluon)
        phase3_json_path = RESULTS_DIR / "phase3_best_model.json"
        if phase3_json_path.exists():
            try:
                with open(phase3_json_path, "r", encoding="utf-8") as f:
                    p3_meta = json.load(f)
                model_rel_path = p3_meta.get("model_path")
                model_abs_path = PROJECT_ROOT / model_rel_path if model_rel_path else None
                if model_abs_path and model_abs_path.exists():
                    if model_abs_path.is_dir():
                        from autogluon.tabular import TabularPredictor
                        model = TabularPredictor.load(str(model_abs_path))
                        self.served_model = model
                        self.model_info = {
                            "served_model_source": f"Phase 3 AutoML ({p3_meta.get('best_tool', 'AutoGluon')})",
                            "served_model_name": f"{p3_meta.get('best_tool', 'AutoGluon')} - {p3_meta.get('best_model_name', 'WeightedEnsemble_L2')}",
                            "served_model_path": str(model_rel_path),
                            "served_model_type": type(model).__name__,
                            "explanation_model_type": "Surrogate TreeExplainer (Random Forest)",
                            "surrogate_used": True,
                            "metrics": {
                                "f1": p3_meta.get("test_f1"),
                                "roc_auc": p3_meta.get("test_roc_auc"),
                                "precision": p3_meta.get("test_precision"),
                                "recall": p3_meta.get("test_recall"),
                                "accuracy": p3_meta.get("test_accuracy"),
                            },
                            "reason_for_selection": f"Highest test F1-score ({p3_meta.get('test_f1')}) among benchmarked AutoML engines in Phase 3."
                        }
                        logger.info("Successfully loaded Phase 3 AutoGluon model: %s", self.model_info["served_model_name"])
                        return
                    else:
                        model = joblib.load(model_abs_path)
                        if hasattr(model, "predict_proba"):
                            self.served_model = model
                            self.model_info = {
                                "served_model_source": f"Phase 3 AutoML ({p3_meta.get('best_tool', 'FLAML')})",
                                "served_model_name": f"{p3_meta.get('best_tool', 'FLAML')} - {p3_meta.get('best_model_name', 'LightGBM')}",
                                "served_model_path": str(model_rel_path),
                                "served_model_type": type(model).__name__,
                                "explanation_model_type": "TreeExplainer (LightGBM estimator)",
                                "surrogate_used": False,
                                "metrics": {
                                    "f1": p3_meta.get("test_f1"),
                                    "roc_auc": p3_meta.get("test_roc_auc"),
                                    "precision": p3_meta.get("test_precision"),
                                    "recall": p3_meta.get("test_recall"),
                                    "accuracy": p3_meta.get("test_accuracy"),
                                },
                                "reason_for_selection": "Highest test F1-score among benchmarked AutoML engines in Phase 3."
                            }
                            logger.info("Successfully loaded Phase 3 best model: %s", self.model_info["served_model_name"])
                            return
            except Exception as e:
                logger.warning("Failed to load Phase 3 best model: %s. Falling back to Phase 2.", e)

        # Priority 2: Phase 2 Best Model (Tuned / Baseline Random Forest)
        phase2_json_path = RESULTS_DIR / "phase2_best_model.json"
        if phase2_json_path.exists():
            try:
                with open(phase2_json_path, "r", encoding="utf-8") as f:
                    p2_meta = json.load(f)
                model_rel_path = p2_meta.get("model_path")
                model_abs_path = PROJECT_ROOT / model_rel_path if model_rel_path else None
                if model_abs_path and model_abs_path.exists():
                    model = joblib.load(model_abs_path)
                    if hasattr(model, "predict_proba"):
                        self.served_model = model
                        self.model_info = {
                            "served_model_source": "Phase 2 Manual Model",
                            "served_model_name": p2_meta.get("best_method", "Baseline Random Forest"),
                            "served_model_path": str(model_rel_path),
                            "served_model_type": type(model).__name__,
                            "explanation_model_type": "TreeExplainer (RandomForest)",
                            "surrogate_used": False,
                            "metrics": {
                                "f1": p2_meta.get("test_f1"),
                                "roc_auc": p2_meta.get("test_roc_auc"),
                            },
                            "reason_for_selection": "Phase 2 winning Random Forest model."
                        }
                        logger.info("Successfully loaded Phase 2 model: %s", self.model_info["served_model_name"])
                        return
            except Exception as e:
                logger.warning("Failed to load Phase 2 model: %s. Falling back to manual training.", e)

        # Priority 3: Fallback trained Random Forest
        logger.warning("Training fallback RandomForestClassifier on train_manual.csv...")
        train_path = DATA_PROCESSED_DIR / "train_manual.csv"
        df_train = pd.read_csv(train_path)
        X_train = df_train.drop(columns=["machine_failure"])
        y_train = df_train["machine_failure"]
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        rf.fit(X_train, y_train)
        fallback_path = MODELS_DIR / "fallback_rf.pkl"
        joblib.dump(rf, fallback_path)
        self.served_model = rf
        self.model_info = {
            "served_model_source": "Phase 4 Fallback RandomForest",
            "served_model_name": "Fallback Random Forest",
            "served_model_path": "models/fallback_rf.pkl",
            "served_model_type": "RandomForestClassifier",
            "explanation_model_type": "TreeExplainer (RandomForest)",
            "surrogate_used": True,
            "metrics": {"f1": "fallback", "roc_auc": "fallback"},
            "reason_for_selection": "Emergency fallback model trained on train_manual.csv."
        }

    def prepare_input_dataframe(self, sensor_input: SensorInput) -> pd.DataFrame:
        """Transforms a SensorInput into a single-row DataFrame formatted for the served model."""
        feature_names = getattr(self.served_model, "feature_names_in_", None)
        raw_dict = sensor_input.model_dump()

        if feature_names is not None and "type_L" in feature_names:
            # Model expects one-hot encoded variant
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
            df = pd.DataFrame([row])
            return df[feature_names]
        else:
            # Model expects raw categorical 'type' (e.g. FLAML)
            row = {
                "type": raw_dict["type"],
                "air_temp": raw_dict["air_temp"],
                "process_temp": raw_dict["process_temp"],
                "rotational_speed": raw_dict["rotational_speed"],
                "torque": raw_dict["torque"],
                "tool_wear": raw_dict["tool_wear"]
            }
            cols = ['type', 'air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear']
            return pd.DataFrame([row])[cols]

    def predict_probability(self, sensor_input: SensorInput) -> float:
        """Computes failure probability P(machine_failure=1) for a given sensor reading."""
        df = self.prepare_input_dataframe(sensor_input)
        res = self.served_model.predict_proba(df)
        if isinstance(res, pd.DataFrame):
            proba = float(res.iloc[0, 1])
        elif isinstance(res, np.ndarray):
            proba = float(res[0][1]) if res.shape[1] > 1 else float(res[0][0])
        else:
            proba = float(res)
        return float(np.clip(proba, 0.0, 1.0))

    def predict(self, sensor_input: SensorInput) -> PredictionResponse:
        """Full prediction pipeline returning standardized PredictionResponse."""
        prob = self.predict_probability(sensor_input)
        pred_class = 1 if prob >= PROBABILITY_THRESHOLD else 0
        risk = get_risk_level(prob)

        return PredictionResponse(
            input=sensor_input.model_dump(),
            prediction=PredictionDetails(
                failure_probability=round(prob, 4),
                predicted_class=pred_class,
                risk_level=risk,
                threshold=PROBABILITY_THRESHOLD,
            ),
            model=ModelInfo(
                name=self.model_info.get("served_model_name", "Predictive Maintenance Model"),
                source=self.model_info.get("served_model_source", "AutoML"),
            ),
        )

    def get_surrogate_explainer_model(self) -> Tuple[Any, pd.DataFrame]:
        """Returns a surrogate Random Forest model trained on train_manual.csv if requested."""
        if self.surrogate_model is None:
            train_path = DATA_PROCESSED_DIR / "train_manual.csv"
            df_train = pd.read_csv(train_path)
            X = df_train.drop(columns=["machine_failure"])
            y = df_train["machine_failure"]
            rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
            rf.fit(X, y)
            self.surrogate_model = rf
        return self.surrogate_model, pd.read_csv(DATA_PROCESSED_DIR / "train_manual.csv").drop(columns=["machine_failure"])


# Singleton accessor
model_service = ModelService.get_instance()
