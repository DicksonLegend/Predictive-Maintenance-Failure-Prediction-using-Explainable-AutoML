# Standardized Model Card: Predictive Maintenance Failure Classifier

**Model Name:** AutoGluon Tabular WeightedEnsemble_L2 Classifier  
**Target Domain:** Industry 4.0 CNC Milling Machine Predictive Maintenance  
**Version:** 2.0.0 (Final Mode Benchmark & Deployment Release)  
**Date:** 2026-10-27  

---

## 1. Model Overview
The served machine learning model is an optimized multi-layer stacked ensemble (**AutoGluon WeightedEnsemble_L2**) discovered under a 300-second optimization budget over the full 8,000-instance training corpus. It predicts impending machine failure ($y \in \{0, 1\}$) from real-time sensory telemetry before catastrophic breakdown occurs. Explainability is backed by high-fidelity surrogate TreeExplainer SHAP and LIME algorithms.

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
| **F1-Score (Primary)** | **0.7797** | Winning Production Model (outperforming FLAML 0.7680 & RF 0.7132) |
| **ROC-AUC** | **0.9735** | Exceptional discriminative ranking capability |
| **PR-AUC** | **0.8490** | Outstanding precision-recall balance under severe 3.4% imbalance |
| **Precision** | **0.9200** | Ultra-high operator confidence (only 4 false alarms out of 2,000 tests) |
| **Recall** | **0.6765** | Captures >2/3 of all impending failures at default 0.50 cutoff |
| **Accuracy** | **0.9870** | Near-optimal classification across 2,000 held-out instances |

*Benchmark Note:* In Final Mode, AutoGluon's multi-layer `WeightedEnsemble_L2` overtook all competitors, advancing from fast mode ($F_1 = 0.3721$) to **0.7797** in final mode, outperforming FLAML AutoML ($F_1 = 0.7680$) and manual Random Forest tuning ($F_1 = 0.7132$).

---

## 5. Explainability Architecture
1. **Global SHAP (TreeExplainer via Surrogate):** Identifies global feature influence. `torque`, `rotational_speed`, and `tool_wear` account for >75% of total variance in failure log-odds.
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
- **Variant L (Low, 58.5% test partition, n=1170):** Precision = 0.8929, Recall = 0.6579, F1 = **0.7576**, FPR = **0.0027** (0.27%)
- **Variant M (Medium, 30.8% test partition, n=616):** Precision = 0.9500, Recall = 0.7600, F1 = **0.8444**, FPR = **0.0017** (0.17%)
- **Variant H (High, 10.7% test partition, n=214):** Precision = 1.0000, Recall = 0.4000, F1 = **0.5714**, FPR = **0.0000** (0.00%)

*Conclusion:* Disparate impact is negligible with false positive rates strictly under 0.30% across all tiers. The lower recall on Variant H is driven by sample scarcity (only 5 positive failure instances in the test set).

---

## 7. Data Leakage Audit Findings
In Phase 3, a synthetic post-hoc inspection flag (`post_failure_inspection_flag`) was evaluated under a controlled leakage test:
- Test F1 inflated from 0.7680 to **0.8722** (+10.42 percentage points).
- Feature importance analysis revealed that the leaky flag dominated the attribution landscape.
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
