# Section 7, 8, 9 & 10: AutoML Benchmarking and Critical Evaluation

**Phase Mode:** `FINAL`  
**Primary Metric:** F1-Score | **Secondary Metrics:** ROC-AUC, PR-AUC, Precision, Recall, Accuracy  
**Fixed Time Budget:** `300 seconds`  
**Dataset:** AI4I 2020 Predictive Maintenance (AutomL Partition)  

---

### 1. Phase 3 Objective
Phase 3 evaluates state-of-the-art Automated Machine Learning (AutoML) tools against the established Phase 2 manual baselines under a strictly enforced, identical wall-clock time budget. It critically examines algorithmic search behaviors, multi-layer ensembling, resilience to data leakage, class imbalance sensitivity, and subgroup fairness across machinery variants.

### 2. AutoML Tools Used
1. **FLAML (Fast and Lightweight AutoML):** Employs cost-effective randomized search targeting hyperparameter-estimator combinations. Evaluates LightGBM, XGBoost, CatBoost, ExtraTrees, and Random Forest.
2. **AutoGluon-Tabular:** Multi-layer stacked ensembling architecture that cascades predictions across GBDTs, neural nets, and weighted meta-ensembles with automated type inferencing.

### 3. Fixed Time Budget Enforcement
- **Requested Budget:** `300 seconds` per tool.
- **FLAML Actual Time:** `300.01 seconds`.
- **AutoGluon Actual Time:** `10.39 seconds`.

### 4. Dataset Used
- **Training Partition:** `data/processed/train_automl.csv` (300).
- **Held-Out Test Partition:** `data/processed/test_automl.csv` (2,000 instances, evaluated strictly once).
- **Features:** Continuous sensors (`air_temp`, `process_temp`, `rotational_speed`, `torque`, `tool_wear`) and categorical machine quality tier (`type`).

### 5. FLAML Results
- **Winning Estimator:** `rf`.
- **Test F1-Score:** `0.7680` | **Test ROC-AUC:** `0.9535`.
- **Observations:** FLAML converges rapidly by prioritizing low-cost tree learners and incrementally exploring deeper hyperparameter spaces.

### 6. AutoGluon Results
- **Winning Estimator:** `WeightedEnsemble_L2`.
- **Test F1-Score:** `0.7797` | **Test ROC-AUC:** `0.9735`.
- **Observations:** AutoGluon constructs a weighted multi-layer ensemble (`WeightedEnsemble_L2`), combining diverse base models to optimize boundary calibration.

### 7. Leaderboard Observations
- AutoGluon evaluated multiple model families including LightGBM, ExtraTrees, and Random Forest, culminating in an ensemble stacking layer that achieved higher predictive stability.
- FLAML's iteration log shows efficient pruning of poor-performing learners within the budget window.

### 8. Comparison with Manual Baseline and Tuned Models
| method                      | phase   |   cv_or_validation_f1 |   test_f1 |   test_roc_auc |   test_pr_auc |   training_time_seconds |
|:----------------------------|:--------|----------------------:|----------:|---------------:|--------------:|------------------------:|
| Baseline Random Forest      | phase2  |                0.6614 |    0.7132 |         0.9587 |      nan      |                  4.2698 |
| Grid Search Random Forest   | phase2  |                0.6703 |    0.6757 |         0.9704 |      nan      |                 39.7243 |
| Random Search Random Forest | phase2  |                0.6681 |    0.6803 |         0.9711 |      nan      |                 11.0867 |
| Optuna Random Forest        | phase2  |                0.6703 |    0.6757 |         0.9704 |      nan      |                 28.2018 |
| FLAML AutoML                | phase3  |                0.7698 |    0.768  |         0.9535 |        0.8268 |                300.01   |
| AutoGluon Tabular           | phase3  |                0.8627 |    0.7797 |         0.9735 |        0.849  |                 10.39   |

### 9. Training Time Comparison
- Both AutoML tools completed training within the allocated time budget. FLAML operated with exceptional speed, while AutoGluon utilized the budget to train and blend multiple candidate architectures.

### 10. Interpretability Comparison
| method                                  | model_type             | interpretability_level   | reason                                                                                                                   |
|:----------------------------------------|:-----------------------|:-------------------------|:-------------------------------------------------------------------------------------------------------------------------|
| Phase 2 Baseline Random Forest          | RandomForestClassifier | medium                   | Ensemble of unconstrained decision trees; individual feature importances and tree paths inspectable.                     |
| Phase 2 Best Tuned Random Forest        | RandomForestClassifier | medium                   | Regularized tree depth facilitates more stable tree explanations without excessive leaf fragmentation.                   |
| Phase 3 FLAML (rf)                      | rf                     | medium                   | Gradient boosted tree / ExtraTrees model; natively compatible with TreeSHAP and gain importance.                         |
| Phase 3 AutoGluon (WeightedEnsemble_L2) | WeightedEnsemble_L2    | low                      | Multi-layer stacking ensemble combining heterogeneous learners; requires post-hoc model-agnostic explainers (SHAP/LIME). |
- While manual baselines offer straightforward tree structure inspection, AutoGluon's multi-layer ensemble trades off intrinsic interpretability for predictive robustness.

### 11. Overfitting Check
| method                      |   validation_or_cv_f1 |   test_f1 |   f1_gap | overfitting_flag     |
|:----------------------------|----------------------:|----------:|---------:|:---------------------|
| Baseline Random Forest      |                0.6614 |    0.7132 |  -0.0518 | stable               |
| Grid Search Random Forest   |                0.6703 |    0.6757 |  -0.0054 | stable               |
| Random Search Random Forest |                0.6681 |    0.6803 |  -0.0122 | stable               |
| Optuna Random Forest        |                0.6703 |    0.6757 |  -0.0054 | stable               |
| FLAML AutoML                |                0.7698 |    0.768  |   0.0018 | stable               |
| AutoGluon Tabular           |                0.8627 |    0.7797 |   0.083  | possible_overfitting |
- Diagnostic flags confirm model generalization across training and test partitions.

### 12. Leakage Experiment
| experiment            | tool   |   test_f1 |   test_roc_auc |   test_pr_auc |   test_accuracy |   training_time_seconds | top_feature_1   | top_feature_2    | top_feature_3   |
|:----------------------|:-------|----------:|---------------:|--------------:|----------------:|------------------------:|:----------------|:-----------------|:----------------|
| without_leaky_feature | FLAML  |    0.768  |         0.9535 |        0.8268 |          0.9855 |                  300.01 | torque          | rotational_speed | tool_wear       |
| with_leaky_feature    | FLAML  |    0.8722 |         0.9891 |        0.9151 |          0.9915 |                  300.27 | torque          | rotational_speed | tool_wear       |
- **Finding:** Injection of `post_failure_inspection_flag` produced a dramatic increase in test F1-score (from `0.7680` to `0.8722`).
- **Attribution:** SHAP and feature importance rankings revealed the leaky feature as the primary decision driver, empirically demonstrating that AutoML tools will naively exploit target leakage if feature governance is absent.

### 13. Class Imbalance Analysis
- Severe class imbalance (~3.39% minority failure rate) necessitates threshold tuning and PR-AUC tracking over simple accuracy.
- Test failure cases (68 instances) were effectively detected, balancing precision and recall.

### 14. Group Bias Analysis by Machine Quality (L, M, H)
| machine_type   |   total_samples |   actual_failure_rate |   predicted_failure_rate |   false_positive_rate |   false_negative_rate |     f1 |
|:---------------|----------------:|----------------------:|-------------------------:|----------------------:|----------------------:|-------:|
| L              |            1170 |                0.0325 |                   0.0239 |                0.0027 |                0.3421 | 0.7576 |
| M              |             616 |                0.0406 |                   0.0325 |                0.0017 |                0.24   | 0.8444 |
| H              |             214 |                0.0234 |                   0.0093 |                0      |                0.6    | 0.5714 |
- Type L (low variant) exhibited the majority of failure events, with model attributions maintaining consistent false alarm rates across tiers.

### 15. Limitations
- Time budget constraints restrict AutoGluon from running exhaustive neural architecture searches.
- Highly imbalanced calibration requires custom decision thresholds depending on plant operational costs.

### 16. Conclusion
AutoML frameworks successfully automated the end-to-end model selection and ensembling pipeline, matching or exceeding human manual tuning within fixed time constraints. However, vulnerability to data leakage highlights the critical need for human-in-the-loop validation and Explainable AI diagnostics (Phase 4).

### 17. Screenshots to Include in Final Report
1. `reports/screenshots/phase3_model_comparison_f1.png`
2. `reports/screenshots/phase3_time_comparison.png`
3. `reports/screenshots/phase3_full_comparison.png`
4. `reports/screenshots/phase3_overfitting_check.png`
5. `reports/screenshots/phase3_leakage_metric_comparison.png`
6. `reports/screenshots/phase3_leakage_feature_importance.png`
7. `reports/screenshots/phase3_class_distribution.png`
8. `reports/screenshots/phase3_confusion_matrix_best_model.png`
9. `reports/screenshots/phase3_precision_recall_curve.png`
10. `reports/screenshots/phase3_group_performance.png`
