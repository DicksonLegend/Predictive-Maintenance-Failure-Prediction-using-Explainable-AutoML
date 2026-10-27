# Phase 2 Execution Summary: Baseline & Hyperparameter Tuning

- **Execution Mode:** `final`
- **Timestamp:** `2026-10-27T09:28:41.468133Z`
- **Best Performing Method:** `Baseline Random Forest`
- **Best Test F1-Score:** `0.7132`
- **Best Test ROC-AUC:** `0.9587`
- **Best Model Artifact:** `models/baseline_rf.pkl`

## Model Performance & Resource Utilization
| method                      |   cv_f1_mean |   cv_f1_std |   test_f1 |   test_roc_auc |   test_precision |   test_recall |   test_accuracy |   training_time_seconds | best_params                                                                              |
|:----------------------------|-------------:|------------:|----------:|---------------:|-----------------:|--------------:|----------------:|------------------------:|:-----------------------------------------------------------------------------------------|
| Baseline Random Forest      |       0.6614 |      0.0472 |    0.7132 |         0.9587 |           0.7541 |        0.6765 |          0.9815 |                  4.2698 | {'n_estimators': 100, 'max_depth': None, 'class_weight': 'balanced', 'random_state': 42} |
| Grid Search Random Forest   |       0.6703 |      0.0155 |    0.6757 |         0.9704 |           0.625  |        0.7353 |          0.976  |                 39.7243 | {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}  |
| Random Search Random Forest |       0.6681 |      0.0185 |    0.6803 |         0.9711 |           0.6329 |        0.7353 |          0.9765 |                 11.0867 | {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 20}    |
| Optuna Random Forest        |       0.6703 |      0.0155 |    0.6757 |         0.9704 |           0.625  |        0.7353 |          0.976  |                 28.2018 | {'n_estimators': 300, 'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 1}  |

## Best Hyperparameters
```json
{
  "n_estimators": 100,
  "max_depth": null,
  "class_weight": "balanced",
  "random_state": 42
}
```
