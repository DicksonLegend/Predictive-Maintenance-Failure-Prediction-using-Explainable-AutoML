# Standardized Model Card: Predictive Maintenance Failure Classifier

**Model Name:** FLAML AutoML LightGBM Classifier  
**Target Domain:** Industry 4.0 CNC Milling Machine Predictive Maintenance  
**Version:** 1.0.0 (Phase 4 Deployment Release)  
**Date:** 2026-10-27  

---

## 1. Model Overview
The served machine learning model is an optimized LightGBM gradient boosted decision tree classifier discovered via **FLAML AutoML** under a fixed 60-second optimization budget. It predicts impending machine failure ($y \in \{0, 1\}$) from real-time sensory telemetry before catastrophic breakdown occurs.

---

## 2. Intended Use & Target Users
- **Intended Use:** Operational decision support for factory maintenance technicians, reliability engineers, and plant managers. Provides real-time failure probability estimation coupled with local attribution explanations and prescriptive work order recommendations.
- **Out-of-Scope Use:** Autonomous unverified machine shutdown without operator validation; application to non-rotational tooling systems (e.g., additive manufacturing, chemical reactors).

---

## 3. Training & Validation Data
- **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (10,000 instances).
- **Split:** Stratified 80/20 train/test split locked to `RANDOM_SEED = 42`.
  - **Training Set:** 8,000 records (7,729 nominal, 271 failures $\rightarrow$ 3.39% failure incidence).
  - **Test Set:** 2,000 records (1,932 nominal, 68 failures).
- **Telemetry Features:**
  - `type`: Machine quality variant (`L` = 50%, `M` = 30%, `H` = 20%).
  - `air_temp`: Ambient factory air temperature ($[K]$).
  - `process_temp`: Internal operating process temperature ($[K]$).
  - `rotational_speed`: Spindle rotational velocity ($[rpm]$).
  - `torque`: Drive shaft mechanical torque ($[Nm]$).
  - `tool_wear`: Cumulative tool operation duration ($[min]$).

---

## 4. Evaluation Metrics
Metrics achieved on the held-out 2,000-instance stratified test set:

| Metric | Score | Assessment |
| :--- | :--- | :--- |
| **F1-Score (Primary)** | **0.6281** | Winning AutoML model (outperforming AutoGluon 0.3721) |
| **ROC-AUC** | **0.9171** | High discriminative ranking capability |
| **PR-AUC** | **0.6866** | Robust performance under severe class imbalance |
| **Precision** | **0.7170** | High operator confidence (low false alarm rate) |
| **Recall** | **0.5588** | Captures majority of failures at default 0.50 cutoff |
| **Accuracy** | **0.9775** | Baseline majority accuracy reference |

*Benchmark Note:* The Phase 2 manual Random Forest with `class_weight="balanced"` achieved test F1 = 0.7132 with higher recall (0.6765). The FLAML LightGBM model offers rapid inference latency (<1ms) and direct TreeExplainer compatibility.

---

## 5. Explainability Architecture
1. **Global SHAP (TreeExplainer):** Identifies global feature influence. `torque` and `tool_wear` account for >70% of total variance in failure log-odds.
2. **Local SHAP Waterfall:** Deconstructs specific telemetry records into additive feature contributions with directional risk indicators.
3. **LIME Decision Rules:** Provides sparse local linear surrogates for transparent operator verification.
4. **Partial Dependence Plots (PDP):** Visualizes non-linear margin effects across torque, tool wear, rotational speed, and process temperature.
5. **Counterfactual What-If Simulation:** Interactively calculates required parameter reductions (e.g., torque relief, tool renewal) to revert machines to low risk (<30%).
6. **Multi-Modal Demonstration:**
   - NLP LIME Token Explainer for technician shift handover notes (`urgent` vs. `normal`).
   - Saliency defect localization maps for tool blade micro-cracks, bearing spalling, and stator thermal hotspots.

---

## 6. Fairness & Group Bias Findings
Performance was audited across machine quality tiers:
- **Variant L (Low, 50%):** F1 = 0.6176, False Positive Rate = 1.1%
- **Variant M (Medium, 30%):** F1 = 0.6667, False Positive Rate = 0.8%
- **Variant H (High, 20%):** F1 = 0.5455, False Positive Rate = 1.3%

*Conclusion:* Disparate impact is minimal ($FPR < 1.5\%$ across all variants). The lower F1 on Variant H is driven by sample scarcity (only 21 failure instances in the test set).

---

## 7. Data Leakage Audit Findings
In Phase 3, a synthetic post-hoc inspection flag (`post_failure_inspection_flag`) was introduced in a controlled trial:
- Test F1 inflated from 0.6281 to **0.8244** (+19.6%).
- Feature importance analysis revealed the leaky flag captured **62.4%** of all model attribution.
- *Mitigation:* Strict feature pipeline segregation enforced in `src/data/preprocess.py` to prevent post-event variables from entering production pipelines.

---

## 8. Prescriptive Maintenance Recommendations
The dashboard embeds a deterministic recommendation engine that translates top Shapley drivers into specific mechanical work orders:
- **Torque:** Reduce feed rate and mechanical cutting depth; inspect planetary gearbox and drive coupling.
- **Tool Wear:** Initiate immediate cutting tool insert replacement; review tool lifespan history.
- **Process Temperature:** Inspect heat exchanger cooling coils and ventilation filters; verify thermal equilibrium.
- **Rotational Speed:** Check spindle dynamic balance and high-frequency bearing vibration.

---

## 9. Known Limitations & Safe Operating Limits
- **Imbalance Threshold Sensitivity:** In high-consequence production environments, the classification threshold should be adjusted to 0.30 to prioritize recall over precision.
- **Sensor Range Drift:** Telemetry values exceeding the training envelopes ($torque > 100 Nm$, $tool\_wear > 300 min$) represent out-of-distribution operation and require engineering review.
- **Vibration Resolution:** Relies on rotational speed proxy rather than continuous high-frequency accelerometer spectrograms.
