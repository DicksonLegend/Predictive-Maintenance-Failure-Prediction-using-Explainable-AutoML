# Models Directory

This directory stores trained model binaries, pipelines, and search logs across project phases.

- `baseline_models/`: Fitted scikit-learn baseline pipelines (Phase 2).
- `tuned_models/`: Models optimized via Grid Search, Random Search, and Optuna (Phase 2).
- `automl_models/`: FLAML and AutoGluon-Tabular ensemble model artifacts (Phase 3).
- `best_model/`: Serialized top-performing production model for Streamlit deployment (Phase 4).
