# Results Directory

This directory stores quantitative benchmarking results, evaluation metrics, and EDA summary tables saved as CSV files.

Phase 1 Deliverables:
- `dataset_summary.csv`: Core dataset statistics (sample count, feature types, imbalance ratio).
- `missing_values.csv`: Missing value count and percentages per feature.
- `class_distribution.csv`: Counts and percentages of machine failure vs. nominal states.
- `feature_summary.csv`: Descriptive statistics (mean, std, percentiles, skewness, kurtosis).

Subsequent Phase Deliverables:
- `manual_baseline_results.csv` (Phase 2)
- `hpo_comparison_results.csv` (Phase 2)
- `automl_leaderboard_flaml.csv` (Phase 3)
- `automl_leaderboard_autogluon.csv` (Phase 3)
- `leakage_experiment_results.csv` (Phase 3)
- `bias_group_fairness.csv` (Phase 3)
