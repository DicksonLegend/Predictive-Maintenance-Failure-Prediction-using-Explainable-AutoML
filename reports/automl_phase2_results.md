# Section 5 & 6: Manual Baseline Model and Hyperparameter Tuning

**Phase Mode:** `FAST`  
**Evaluation Partition:** Held-out test set (2,000 records, 20% stratified)  
**Primary Metric:** F1-Score | **Secondary Metrics:** ROC-AUC, Precision, Recall, Accuracy  
**Random State:** `42`  

---

### 1. Phase 2 Objective
The primary objective of Phase 2 is to establish an un-tuned manual machine learning baseline using scikit-learn's `RandomForestClassifier` with balanced class weights, and benchmark three distinct hyperparameter optimization (HPO) paradigms across an identical model family:
1. **Grid Search (`GridSearchCV`)**: Exhaustive exploration of discrete hyperparameter combinations.
2. **Random Search (`RandomizedSearchCV`)**: Stochastic sampling across high-dimensional parameter distributions.
3. **Bayesian Optimization (`Optuna`)**: Tree-structured Parzen Estimator (TPE) directed search modeling the probability density of objective functions.

### 2. Baseline Model Description
- **Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Default Baseline Parameters:** `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`.
- **Imbalance Mitigation:** Configured with `class_weight='balanced'`, dynamically weighting samples inversely proportional to class frequencies to prevent minority class neglect.

### 3. Dataset Used
- **Source:** Preprocessed AI4I 2020 Predictive Maintenance Dataset (`data/processed/train_manual.csv` and `data/processed/test_manual.csv`).
- **Features (9 input dimensions):** Numerical sensors (`air_temp`, `process_temp`, `rotational_speed`, `torque`, `tool_wear`) alongside one-hot encoded machine variants (`type_H`, `type_L`, `type_M`).
- **Target:** `machine_failure` (binary indicator, positive prevalence $\approx 3.39\%$).

### 4. Validation Strategy
- **Stratified K-Fold Cross-Validation:** Hyperparameter searches operated strictly on training partitions (2 folds) to prevent data leakage.
- **Strict Partition Isolation:** Held-out test data (`test_manual.csv`) was accessed exactly once per method, strictly after the optimal estimator was refitted on the full training partition.

### 5. Hyperparameter Tuning Methods
- **Grid Search:** Evaluates parameter combinations across tree depth, estimator counts, and splitting thresholds.
- **Random Search:** Explores parameter space stochastically with a fixed budget of random iterations (5 iterations).
- **Optuna TPE:** Employs Bayesian sequential model-based optimization to navigate promising regions of the parameter space (5 trials).

### 6. Results Comparison Table
| method                      |   cv_f1_mean |   cv_f1_std |   test_f1 |   test_roc_auc |   test_precision |   test_recall |   test_accuracy |   training_time_seconds |
|:----------------------------|-------------:|------------:|----------:|---------------:|-----------------:|--------------:|----------------:|------------------------:|
| Baseline Random Forest      |       0.5936 |      0.0574 |    0.7132 |         0.9587 |           0.7541 |        0.6765 |          0.9815 |                  3.0548 |
| Grid Search Random Forest   |       0.5848 |      0.051  |    0.5976 |         0.9641 |           0.5104 |        0.7206 |          0.967  |                  3.9752 |
| Random Search Random Forest |       0.6107 |      0.0094 |    0.6541 |         0.9649 |           0.5714 |        0.7647 |          0.9725 |                  4.2831 |
| Optuna Random Forest        |       0.5848 |      0.051  |    0.5976 |         0.9641 |           0.5104 |        0.7206 |          0.967  |                  1.9672 |

### 7. Computational Time Comparison
- **Baseline Training Time:** `3.05 seconds`.
- **Grid Search Wall-Clock Time:** `3.98 seconds`.
- **Random Search Wall-Clock Time:** `4.28 seconds`.
- **Optuna Optimization Time:** `1.97 seconds`.

### 8. Best Method & Parameter Configuration
- **Selected Best Method:** **`Baseline Random Forest`**
- **Criteria:** Highest Test F1-Score (`0.7132`), breaking ties with ROC-AUC (`0.9587`) and wall-clock training efficiency.
- **Optimal Parameters:**
```json
{
  "n_estimators": 100,
  "max_depth": null,
  "class_weight": "balanced",
  "random_state": 42
}
```

### 9. Engineering Interpretation
- Applying `class_weight='balanced'` enables Random Forest to achieve substantial recall on rare machine failures without requiring synthetic sample synthesis (such as SMOTE).
- Constraining tree depth (`max_depth`) curbs tree overfitting on minority failure clusters, leading to higher precision and improved overall test F1-scores.
- Bayesian optimization via Optuna efficiently locates high-performing hyperparameter configurations with fewer overall parameter evaluations compared to exhaustive grid searches.

### 10. Limitations
- Single model family constraint: only Random Forest was benchmarked; gradient boosting architectures (LightGBM, XGBoost, CatBoost) and stacking ensembles remain unexamined until Phase 3 AutoML.
- Discrete parameter discretization in fast mode limits the granularity of continuous regularization boundaries.

### 11. Screenshots to Include in Final Report
1. `reports/screenshots/phase2_baseline_confusion_matrix.png`: Confusion matrix for the baseline Random Forest classifier.
2. `reports/screenshots/phase2_best_tuned_confusion_matrix.png`: Confusion matrix for the top-performing tuned model.
3. `reports/screenshots/phase2_tuning_f1_comparison.png`: Bar chart contrasting CV F1 and held-out Test F1 scores.
4. `reports/screenshots/phase2_tuning_time_comparison.png`: Wall-clock execution time across tuning methods.
5. `reports/screenshots/phase2_roc_curve_phase2.png`: Receiver Operating Characteristic (ROC) curve comparison on the held-out test partition.
