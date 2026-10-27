"""Text Explainability Engine for Industrial Maintenance Notes.

Uses TF-IDF + Logistic Regression classification and LIME Text Explainer
to attribute technician note tokens to risk diagnoses.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer

from app.api.config import PROJECT_ROOT, MODELS_DIR, DATA_TEXT_DIR, TEXT_DEMO_DIR
from app.api.schemas import TextExplainResponse, WordImpact

logger = logging.getLogger("text_explainer")
logger.setLevel(logging.INFO)

NOTES_CSV_PATH = DATA_TEXT_DIR / "maintenance_notes.csv"
TFIDF_MODEL_PATH = MODELS_DIR / "text_demo_tfidf.pkl"
LOGREG_MODEL_PATH = MODELS_DIR / "text_demo_logreg.pkl"

SYNTHETIC_MAINTENANCE_NOTES = [
    # Urgent samples
    ("Machine making loud screeching noise and vibrating heavily during milling cycle.", "urgent"),
    ("Tool wear reached critical threshold, urgent cutter head replacement needed immediately.", "urgent"),
    ("Process temperature exceeded safe threshold with cooling system failure suspected.", "urgent"),
    ("Rotational speed fluctuating wildly under load with heavy shaft runout.", "urgent"),
    ("Smoke observed near drive spindle; thermal cutoff triggered emergency stop.", "urgent"),
    ("Torque surge detected causing rapid stall and potential gearbox gear tooth fracture.", "urgent"),
    ("Excessive tool chatter and severe surface pitting observed on high-tolerance workpiece.", "urgent"),
    ("Coolant pressure dropped to zero during high-speed feed cycle; spindle overheating.", "urgent"),
    ("Hydraulic pressure loss causing chuck slippage and uncontrolled tool engagement.", "urgent"),
    ("Bearing temperature spike accompanied by high-frequency acoustic emissions.", "urgent"),
    ("Spindle motor drawing excessive current; thermal overload warning active.", "urgent"),
    ("Emergency stop tripped due to severe axis positioning error and mechanical binding.", "urgent"),
    ("Cutter insert chipped during heavy facing pass, immediate shutdown required.", "urgent"),
    ("Gearbox oil level empty with visible metal debris in sump strainer.", "urgent"),
    ("Drive belt snapped on auxiliary blower causing immediate heat accumulation.", "urgent"),
    ("Severe vibration spike detected on main shaft accelerometer during 2000 rpm test.", "urgent"),
    ("Tool wear progression accelerating exponentially; cutting edge delamination detected.", "urgent"),
    ("Main bearing race showing severe spalling and overheating discoloration.", "urgent"),
    ("Hydraulic fluid leaking profusely onto shop floor near motor housing.", "urgent"),
    ("Sudden torque overload triggered drive protection circuit breaker.", "urgent"),
    ("Spindle runout exceeds 0.05mm tolerance limit; severe mechanical wear confirmed.", "urgent"),
    ("Pneumatic pressure dropped below interlock threshold; tool clamp unseated.", "urgent"),
    ("Thermal imaging shows hotspot exceeding 95C on primary bearing housing.", "urgent"),
    ("Loud clanking noise originating from planetary reducer under heavy load.", "urgent"),
    ("Vibration sensor reading 4x above baseline; immediate maintenance shutdown required.", "urgent"),
    ("Cooling jacket clogged with scale; process temperature cannot be regulated.", "urgent"),
    ("Machine frame resonance observed at nominal operating frequency; structural inspection needed.", "urgent"),
    ("Tool holder taper worn and slipping inside spindle socket under axial thrust.", "urgent"),
    ("Lubrication pump failed; dry friction detected on linear guide ways.", "urgent"),
    ("Motor winding insulation breakdown indicated by ground fault warning.", "urgent"),

    # Normal samples
    ("Routine inspection completed; all operational telemetry parameters within normal limits.", "normal"),
    ("Machine running smoothly at standard rotational velocity with no vibration detected.", "normal"),
    ("Replaced cutting insert during scheduled maintenance interval; calibration verified.", "normal"),
    ("Coolant flow rate and temperature nominal throughout 8-hour production shift.", "normal"),
    ("Torque readings stable and well within manufacturer safety envelopes.", "normal"),
    ("Cleaned chip conveyor and topped off hydraulic oil reservoir to nominal mark.", "normal"),
    ("Standard tool wear progression observed; 120 minutes remaining on cutter lifespan.", "normal"),
    ("Bearing acoustic emissions baseline checked; no abnormal frequency peaks detected.", "normal"),
    ("Spindle alignment verified with dial indicator; runout within 0.005mm spec.", "normal"),
    ("Daily pre-shift checklist completed; safety interlocks and limit switches functional.", "normal"),
    ("Thermal equilibrium achieved at 308K with optimal ambient heat dissipation.", "normal"),
    ("Lubrication cycle completed successfully; rail wipers and seals in good condition.", "normal"),
    ("Surface finish on finished components meets quality specification H-tier standards.", "normal"),
    ("Checked pneumatic regulator; line pressure holding steady at 6.2 bar.", "normal"),
    ("Preventive maintenance inspection logged; motor belt tension adjusted to spec.", "normal"),
    ("Workpiece dimension tolerances verified; machine repeatability confirmed.", "normal"),
    ("Coolant filtration system backwashed; fluid clarity and pH level within spec.", "normal"),
    ("Operator logged routine shift handover; no abnormal telemetry alerts noted.", "normal"),
    ("Spindle motor power consumption steady at 4.2 kW under continuous duty cycle.", "normal"),
    ("Vibration spectra match baseline healthy machine signature across all axes.", "normal"),
    ("Tool wear index measured at 45 minutes; edge geometry intact and sharp.", "normal"),
    ("Ambient air temperature stable at 298K with positive pressure ventilation active.", "normal"),
    ("Drive cabinet cooling fans operating normally with dust filters recently cleaned.", "normal"),
    ("Axis backlash measurement within nominal tolerance; servo gains verified.", "normal"),
    ("Completed scheduled oil analysis; no particulate contamination detected.", "normal"),
    ("Chuck clamping force tested with force gauge; pressure certified for shift.", "normal"),
    ("Tool turret indexing smoothly with precise repeatability on all 12 stations.", "normal"),
    ("Minor cosmetic scratch on enclosure panel; zero impact on machine performance.", "normal"),
    ("Machine completed batch run of 500 units without single telemetry warning.", "normal"),
    ("Scheduled monthly calibration logged; telemetry sensors verified against reference standards.", "normal"),
]


class TextExplainerService:
    """Manages the NLP dataset, classifier model, and LIME explanations."""

    _instance = None

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.explainer: Optional[LimeTextExplainer] = None
        self._ensure_dataset_and_models()

    @classmethod
    def get_instance(cls) -> "TextExplainerService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_dataset_and_models(self) -> None:
        """Initializes dataset, trains/loads TF-IDF and Logistic Regression."""
        os.makedirs(DATA_TEXT_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(TEXT_DEMO_DIR, exist_ok=True)

        # 1. Dataset
        if not NOTES_CSV_PATH.exists():
            df_notes = pd.DataFrame(SYNTHETIC_MAINTENANCE_NOTES, columns=["text", "label"])
            df_notes.to_csv(NOTES_CSV_PATH, index=False)
            logger.info("Created synthetic maintenance notes dataset: %s (%d rows)", NOTES_CSV_PATH, len(df_notes))
        else:
            df_notes = pd.read_csv(NOTES_CSV_PATH)

        # 2. Train or Load Models
        if TFIDF_MODEL_PATH.exists() and LOGREG_MODEL_PATH.exists():
            try:
                self.vectorizer = joblib.load(TFIDF_MODEL_PATH)
                self.classifier = joblib.load(LOGREG_MODEL_PATH)
                logger.info("Loaded pre-trained text classifier and vectorizer.")
            except Exception as e:
                logger.warning("Error loading text models: %s. Retraining...", e)
                self._train_models(df_notes)
        else:
            self._train_models(df_notes)

        # 3. Initialize LIME
        self.explainer = LimeTextExplainer(
            class_names=["normal", "urgent"],
            random_state=42
        )

    def _train_models(self, df_notes: pd.DataFrame) -> None:
        """Fits TF-IDF and LogisticRegression and saves them."""
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, stop_words="english")
        X = vectorizer.fit_transform(df_notes["text"])
        y = (df_notes["label"] == "urgent").astype(int)

        clf = LogisticRegression(random_state=42, class_weight="balanced")
        clf.fit(X, y)

        joblib.dump(vectorizer, TFIDF_MODEL_PATH)
        joblib.dump(clf, LOGREG_MODEL_PATH)
        self.vectorizer = vectorizer
        self.classifier = clf
        logger.info("Trained and serialized text models to %s and %s", TFIDF_MODEL_PATH, LOGREG_MODEL_PATH)

    def _predict_proba_fn(self, texts: List[str]) -> np.ndarray:
        """Helper probability prediction function for LIME."""
        X = self.vectorizer.transform(texts)
        proba_urgent = self.classifier.predict_proba(X)  # shape (N, 2): [P(normal), P(urgent)]
        return proba_urgent

    def explain(self, text: str) -> TextExplainResponse:
        """Predicts risk label and generates token-level attribution highlights."""
        probas = self._predict_proba_fn([text])[0]
        prob_urgent = float(probas[1])
        predicted_label = "urgent" if prob_urgent >= 0.50 else "normal"

        # LIME explanation
        exp = self.explainer.explain_instance(
            text,
            self._predict_proba_fn,
            num_features=6,
            labels=[1]
        )

        raw_weights = exp.as_list(label=1)
        important_words: List[WordImpact] = []
        for word, weight in raw_weights:
            impact_tag = "urgent" if weight > 0 else "normal"
            important_words.append(WordImpact(word=word, impact=impact_tag, weight=round(weight, 4)))

        # Generate HTML explanation snippet
        lime_html = exp.as_html()

        return TextExplainResponse(
            text=text,
            predicted_label=predicted_label,
            probability=round(prob_urgent, 4),
            important_words=important_words,
            lime_html=lime_html
        )

    def generate_demo_artifacts(self) -> None:
        """Generates saved demo samples in artifacts/xai/text_demo/."""
        samples = [
            ("Machine making loud noise and vibrating heavily near drive spindle.", "sample_0"),
            ("Process temperature exceeded safe threshold with cooling system failure suspected.", "sample_1"),
            ("Routine inspection completed; all operational telemetry parameters within normal limits.", "sample_2"),
        ]

        for text, sid in samples:
            resp = self.explain(text)
            json_path = TEXT_DEMO_DIR / f"text_{sid}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(resp.model_dump(), f, indent=2)

        # Also save HTML snippet for sample_2 or sample_0
        resp_urgent = self.explain(samples[0][0])
        html_path = TEXT_DEMO_DIR / "text_sample_0.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(resp_urgent.lime_html or "")

        logger.info("Text demo artifacts generated under %s", TEXT_DEMO_DIR)


# Singleton
text_explainer_service = TextExplainerService.get_instance()
