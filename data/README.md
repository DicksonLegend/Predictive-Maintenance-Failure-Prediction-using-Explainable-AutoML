# Data Directory

This directory manages all raw telemetry data and preprocessed datasets.

- `raw/`: Stores the raw AI4I 2020 Predictive Maintenance CSV (`ai4i2020.csv`) or the synthetic fallback (`synthetic_ai4i_smoke_test.csv`).
- `processed/`: Contains stratified 80/20 train/test splits for both modeling paradigms:
  - `train_manual.csv` & `test_manual.csv`: Imputed missing values and one-hot encoded machine types (`type_H`, `type_L`, `type_M`).
  - `train_automl.csv` & `test_automl.csv`: Native continuous features and categorical string representations for AutoML frameworks.
  - `metadata.json`: Dataset metadata, shape, source flag, and target distribution summary.
