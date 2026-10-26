# AutoML Benchmarking for Predictive Maintenance Failure Prediction
## Academic Project Report Outline & Specification

**Author / Candidate:** Senior ML Engineering & Academic Project  
**Course Combination:** AutoML & Explainable AI (XAI)  
**Target Domain:** Smart Manufacturing & Predictive Maintenance  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI / Kaggle)  
**Primary Metric:** F1-Score | **Secondary Metrics:** ROC-AUC, Precision, Recall, Accuracy  

---

### 1. Abstract
- Executive summary of the study: comparing human-crafted machine learning workflows against state-of-the-art automated machine learning (AutoML) frameworks for industrial equipment failure prognosis.
- Key findings on performance (F1-score, ROC-AUC), time efficiency, pipeline complexity, data leakage vulnerability, and interpretability.

### 2. Problem Statement
- Industrial context: unscheduled downtime in computerized numerical control (CNC) and manufacturing machinery leads to catastrophic economic loss and worker hazards.
- Mathematical formulation: binary classification task predicting whether a machine will fail ($y \in \{0, 1\}$) given sensor telemetry streams under severe class imbalance (~3.4% failure prevalence).

### 3. Dataset Description
- Origin: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository).
- Records: 10,000 synthetic/real telemetry records.
- Telemetry features:
  - `type`: Machine quality variant (Low [L, 60%], Medium [M, 30%], High [H, 10%]).
  - `air_temp`: Ambient air temperature ($K$).
  - `process_temp`: Internal operating process temperature ($K$).
  - `rotational_speed`: Spindle rotational velocity ($rpm$).
  - `torque`: Spindle resistance/torque ($Nm$).
  - `tool_wear`: Cumulative duration of tool usage in cutting operations ($min$).
- Underlying failure modes: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), Random Network Failure (RNF).

### 4. Data Preprocessing
- Missing value imputation strategy (median for numeric, mode for categorical).
- Feature encoding: One-hot encoding for manual ML (`type_L`, `type_M`, `type_H`) versus native categorical handling for AutoML.
- Stratified train-test split (80% train, 20% test, fixed random seed 42) ensuring uniform failure representation without synthetic oversampling during initial ingestion.

### 5. Manual Baseline Model
- Development of standard scikit-learn classification baselines:
  - Dummy Classifier (majority class baseline).
  - Logistic Regression with standard scaling.
  - Random Forest Classifier (default parameters).
  - HistGradientBoostingClassifier.
- Baseline performance benchmarking against primary metric (F1) and secondary metrics.

### 6. Hyperparameter Tuning
- Methodological comparison of search paradigms across identical parameter spaces:
  - **Grid Search:** Exhaustive discrete parameter exploration with 5-fold cross-validation.
  - **Random Search:** Stochastic parameter sampling over continuous and discrete distributions.
  - **Optuna:** Bayesian optimization utilizing Tree-structured Parzen Estimator (TPE) with pruning.
- Analysis of search duration, wall-clock time, convergence efficiency, and final cross-validation F1-score gains.

### 7. AutoML Tools Used
- Justification and architectural breakdown of selected AutoML frameworks:
  - **FLAML (Fast and Lightweight AutoML):** Cost-effective search focusing on low computational overhead and randomized search over hyperparameter estimators.
  - **AutoGluon-Tabular:** Multi-layer ensemble stacking, deep tree cascades, and automated feature transformation.
- Unified benchmarking constraints: fixed execution time budget ($T_{agent} = 60s$ for quick CI/CD verification; $T_{final} = 900s$ for formal reporting).

### 8. FLAML Experiment
- Optimization configuration: `time_budget = 900`, `metric = 'f1'`, `task = 'classification'`.
- Search trajectory: estimators evaluated (LightGBM, XGBoost, CatBoost, ExtraTrees, RandomForest).
- Best estimator selected, hyperparameter configuration, and resource consumption profile.

### 9. AutoGluon Experiment
- Optimization configuration: `eval_metric = 'f1'`, `time_limit = 900`, `presets = 'best_quality'` (or `'high_quality'`).
- Multi-layer stack ensembling analysis: Level 1 base models, Level 2 stacking regressors/classifiers, and weighted ensembling.
- Leaderboard analysis of individual models vs. stacked ensemble.

### 10. Model Comparison
- Systematic comparative evaluation table across:
  - Manual Baseline vs. Optuna-tuned vs. FLAML vs. AutoGluon.
  - Primary metric (F1) and secondary metrics (ROC-AUC, Precision, Recall, Accuracy).
  - Computational wall-clock time and memory consumption.
  - Pipeline complexity, code footprint, and model artifact sizes.

### 11. Cross-Validation Results
- 5-fold stratified cross-validation performance distributions across all competing approaches.
- Variance, standard deviation, and stability analysis across folds.

### 12. Held-Out Test Results
- Final generalization assessment on the pristine 20% held-out test partition.
- Confusion matrices, Precision-Recall Curves (PRC), and Receiver Operating Characteristic (ROC) curves.

### 13. Leakage Experiment
- Injection of a deliberately leaky feature (e.g., target-correlated failure proxy or post-event telemetry flag).
- Comparative evaluation: does AutoML naively exploit the leakage versus manual pipeline detection?
- Post-leakage diagnosis, SHAP feature importance shifts, and defensive engineering practices.

### 14. Class Imbalance Analysis
- Impact of the ~3.4% failure class distribution on model training.
- Comparison of mitigation techniques: Class weighting (`class_weight='balanced'`), threshold-moving via Precision-Recall curve tuning, and focal loss.

### 15. Bias / Group Analysis
- Disaggregated performance evaluation across machine variants: High (H), Medium (M), and Low (L) quality variants.
- Disparate impact, false positive rate disparities, and maintenance fairness considerations across equipment tiers.

### 16. Explainability using SHAP
- TreeSHAP and KernelSHAP integration on the top-performing AutoML pipeline.
- Global feature attribution: Summary Beeswarm plots and Mean Absolute SHAP bar charts.
- Local instance explanations: Waterfall plots and Force plots for high-risk machine alerts.

### 17. Deployment
- Architecture of interactive deployment: Streamlit graphical user interface and lightweight REST endpoints.
- Interactive telemetry simulator: sliders for temperature, torque, tool wear, and speed with real-time failure prediction and risk score.

### 18. Model Card
- Comprehensive model documentation (intended use, training dataset, input features, metrics, out-of-scope applications, ethical risks, and operational guidelines).

### 19. Limitations
- Synthetic data fidelity vs. physical plant operating conditions.
- Sensor degradation, calibration drift, and unobserved environmental confounders.
- Inference latency constraints on embedded PLC hardware.

### 20. Conclusion
- Synthesis of findings: strengths and weaknesses of AutoML in mission-critical industrial predictive maintenance.
- Recommended operational workflow for manufacturing practitioners combining AutoML efficiency with human-in-the-loop XAI validation.

### 21. References
- Scholarly publications on AI4I predictive maintenance, AutoML frameworks (FLAML, AutoGluon), hyperparameter optimization (Optuna), and model explainability (SHAP).

### 22. Appendix: Screenshots and Tables
- Leaderboards, cross-validation logs, ROC/PR curves, confusion matrices, and GUI screenshots.
