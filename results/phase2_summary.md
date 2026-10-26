# Phase 2 Execution Summary: Baseline & Hyperparameter Tuning

- **Execution Mode:** `fast`
- **Timestamp:** `2026-10-26T17:00:16.376587Z`
- **Best Performing Method:** `Baseline Random Forest`
- **Best Test F1-Score:** `0.7132`
- **Best Test ROC-AUC:** `0.9587`
- **Best Model Artifact:** `models/baseline_rf.pkl`

## Model Performance & Resource Utilization
| method                      |   cv_f1_mean |   cv_f1_std |   test_f1 |   test_roc_auc |   test_precision |   test_recall |   test_accuracy |   training_time_seconds | best_params                                                                              |
|:----------------------------|-------------:|------------:|----------:|---------------:|-----------------:|--------------:|----------------:|------------------------:|:-----------------------------------------------------------------------------------------|
| Baseline Random Forest      |       0.5936 |      0.0574 |    0.7132 |         0.9587 |           0.7541 |        0.6765 |          0.9815 |                  3.0548 | {'n_estimators': 100, 'max_depth': None, 'class_weight': 'balanced', 'random_state': 42} |
| Grid Search Random Forest   |       0.5848 |      0.051  |    0.5976 |         0.9641 |           0.5104 |        0.7206 |          0.967  |                  3.9752 | {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}    |
| Random Search Random Forest |       0.6107 |      0.0094 |    0.6541 |         0.9649 |           0.5714 |        0.7647 |          0.9725 |                  4.2831 | {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': None}  |
| Optuna Random Forest        |       0.5848 |      0.051  |    0.5976 |         0.9641 |           0.5104 |        0.7206 |          0.967  |                  1.9672 | {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1}    |

## Best Hyperparameters
```json
{
  "n_estimators": 100,
  "max_depth": null,
  "class_weight": "balanced",
  "random_state": 42
}
```
