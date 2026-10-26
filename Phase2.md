MASTER PROMPT — PHASE 2: MANUAL BASELINE MODEL AND HYPERPARAMETER TUNING

You are a senior machine learning engineer and academic project assistant. You are continuing an already locked university IA project that combines:

1. AutoML
2. Explainable AI (XAI)

Project title:
Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0

AutoML report title:
AutoML Benchmarking for Predictive Maintenance Failure Prediction

Explainable AI report title:
Explainable AI Dashboard for Predictive Maintenance Decision Support

Phase 1 is already completed and validated. Do not redo Phase 1 unless a required Phase 1 artifact is missing.

Your current task is Phase 2 only:

PHASE 2 OBJECTIVE:
Build and evaluate a manual scikit-learn baseline model, then tune the same model using:
1. Grid Search
2. Random Search
3. Bayesian optimization with Optuna

You must record score and time taken for each method and create comparison tables and charts.

Do not ask questions. Make reasonable assumptions, document them, and complete Phase 2 fully. If something fails, attempt to fix it. If a heavy operation may take too long, use the provided fast mode for automated validation and provide the final-mode command for the user.

==================================================
LOCKED PROJECT CONSTRAINTS — DO NOT CHANGE
==================================================

Dataset:
AI4I 2020 Predictive Maintenance Dataset.

Phase 1 data artifacts to use:
- data/processed/train_manual.csv
- data/processed/test_manual.csv
- data/processed/metadata.json

Target column:
machine_failure

Target type:
Binary classification

Meaning:
0 = No failure
1 = Failure

Random seed:
RANDOM_SEED = 42

Primary metric:
F1-score

Secondary metrics:
ROC-AUC, precision, recall, accuracy

Train-test split:
Already created in Phase 1.
Do not re-split the data.
Do not modify the test set.
Do not use the test set for hyperparameter search.
Use the test set only for final evaluation after each tuning method selects its best model.

Class imbalance:
The dataset has severe imbalance, approximately 3.39% positive failure class.
Do not apply SMOTE or oversampling in Phase 2.
Use class_weight="balanced" for the Random Forest model.

Baseline model:
Random Forest Classifier.

Do not use another baseline model unless Random Forest fails technically.
If it fails, document the reason clearly.

==================================================
PHASE 2 RUBRIC REQUIREMENTS TO SATISFY
==================================================

This phase must satisfy the AutoML rubric Step 2:

- Build a manual baseline with scikit-learn.
- Tune the same model with Grid Search.
- Tune the same model with Random Search.
- Tune the same model with Bayesian optimization using Optuna.
- Record the score and time taken for each method.
- Show the comparison in a table or chart.
- Use correct validation strategy.
- Avoid data leakage.
- Produce reproducible results.

==================================================
EXISTING PROJECT CONTEXT
==================================================

Phase 1 already created:

- src/config.py
- src/utils.py
- src/data/load_data.py
- src/data/preprocess.py
- data/processed/train_manual.csv
- data/processed/test_manual.csv
- data/processed/metadata.json
- results/
- plots/
- reports/
- reports/screenshots/

Use the existing config and utilities if they exist.
If needed constants are missing, add them carefully without breaking Phase 1.

Do not change:
- RANDOM_SEED
- TARGET_COLUMN
- TEST_SIZE
- PRIMARY_METRIC
- data/processed/train_manual.csv
- data/processed/test_manual.csv

==================================================
REQUIRED PACKAGE INSTALLATION
==================================================

Use the existing virtual environment if present.

If Windows and .venv exists, prefer:

.venv\Scripts\python -m pip install scikit-learn optuna matplotlib seaborn pandas numpy joblib

If Linux/macOS and .venv exists, prefer:

.venv/bin/python -m pip install scikit-learn optuna matplotlib seaborn pandas numpy joblib

Do not install heavy AutoML packages in this phase.
Do not install FLAML, AutoGluon, SHAP, LIME, or Streamlit in this phase unless they are already installed.

If Optuna installation fails, attempt to fix it.
Optuna is required for this phase.

==================================================
PHASE 2 DELIVERABLES
==================================================

Create the following code artifacts:

src/models/__init__.py
src/models/baseline.py
src/models/tuning.py
src/models/evaluation.py
scripts/run_phase2.py
scripts/validate_phase2.py

Create the following result artifacts:

results/phase2_metadata.json
results/baseline_metrics.csv
results/grid_search_results.csv
results/random_search_results.csv
results/optuna_results.csv
results/tuning_comparison.csv
results/best_params.json
results/phase2_best_model.json
results/optuna_trials.csv
results/phase2_summary.md
results/phase2_run.log

Create the following model artifacts:

models/baseline_rf.pkl
models/grid_rf.pkl
models/random_rf.pkl
models/optuna_rf.pkl

Create the following plots:

plots/baseline_confusion_matrix.png
plots/best_tuned_confusion_matrix.png
plots/tuning_f1_comparison.png
plots/tuning_time_comparison.png
plots/roc_curve_phase2.png

Copy important plots to:

reports/screenshots/phase2_baseline_confusion_matrix.png
reports/screenshots/phase2_best_tuned_confusion_matrix.png
reports/screenshots/phase2_tuning_f1_comparison.png
reports/screenshots/phase2_tuning_time_comparison.png
reports/screenshots/phase2_roc_curve_phase2.png

Create a report-ready markdown file:

reports/automl_phase2_results.md

This file should contain a ready-to-use section for the AutoML report, including:
- baseline model description
- tuning strategy
- comparison table
- interpretation
- best method
- limitations

==================================================
MODEL AND EXPERIMENT DESIGN
==================================================

Use the following model as the manual baseline:

RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

Default baseline parameters:
Use scikit-learn default Random Forest parameters unless this causes technical issues.
If needed, set:
- n_estimators = 100
- max_depth = None

Use the same model family for all tuning methods:
Random Forest only.

Do not tune XGBoost, LightGBM, CatBoost, Logistic Regression, or any other model in Phase 2.

==================================================
VALIDATION STRATEGY
==================================================

Use only the training set for hyperparameter search.

Use Stratified K-Fold cross-validation on train_manual.csv.

Fast mode:
cv_folds = 2

Final mode:
cv_folds = 5

Use scoring:
- f1 as the primary optimization metric
- roc_auc as a secondary metric where possible

Do not use test_manual.csv during:
- Grid Search
- Random Search
- Optuna optimization
- cross-validation

After each method selects the best model, fit that best model on the full training set and evaluate it exactly once on test_manual.csv.

==================================================
FAST MODE AND FINAL MODE
==================================================

Create a command-line argument for scripts/run_phase2.py:

--mode fast
--mode final

Default mode:
fast

Fast mode is for automated validation and should finish quickly.
Final mode is for final report screenshots and stronger results.

Fast mode settings:
- cv_folds = 2
- Grid Search parameter grid should be very small
- Random Search n_iter = 5
- Optuna n_trials = 5
- n_jobs = 1 if stability is an issue, otherwise -1

Final mode settings:
- cv_folds = 5
- Grid Search parameter grid should be moderate
- Random Search n_iter = 20
- Optuna n_trials = 30
- n_jobs = -1 if stable, otherwise 1

The agent should automatically run fast mode to validate the pipeline.
Do not automatically run final mode if it may timeout.
At the end, provide the exact final-mode command for the user.

==================================================
SEARCH SPACE
==================================================

Use this parameter search space for Random Forest:

n_estimators:
fast: [100]
final: [100, 300]

max_depth:
fast: [5, 10]
final: [5, 10, 20, None]

min_samples_split:
fast: [2]
final: [2, 5, 10]

min_samples_leaf:
fast: [1]
final: [1, 2, 4]

class_weight:
Fixed to "balanced".
Do not tune class_weight.

bootstrap:
Optional.
If included, keep it fixed to True or omit it entirely.

==================================================
GRID SEARCH REQUIREMENTS
==================================================

Use scikit-learn GridSearchCV.

Use:
- estimator = RandomForestClassifier(random_state=42, class_weight="balanced")
- cv = StratifiedKFold with shuffle=True and random_state=42
- scoring = "f1"
- return_train_score = False
- n_jobs as configured
- refit = True

Record:
- best_params_
- best_score_
- total search time
- mean cross-validated F1
- test F1
- test ROC-AUC
- test precision
- test recall
- test accuracy

Save:
- models/grid_rf.pkl
- results/grid_search_results.csv

==================================================
RANDOM SEARCH REQUIREMENTS
==================================================

Use scikit-learn RandomizedSearchCV.

Use:
- same parameter space as Grid Search final mode
- n_iter = 5 in fast mode
- n_iter = 20 in final mode
- random_state = 42
- cv = StratifiedKFold with shuffle=True and random_state=42
- scoring = "f1"
- return_train_score = False
- refit = True

Record:
- best_params_
- best_score_
- total search time
- mean cross-validated F1
- test F1
- test ROC-AUC
- test precision
- test recall
- test accuracy

Save:
- models/random_rf.pkl
- results/random_search_results.csv

==================================================
OPTUNA REQUIREMENTS
==================================================

Use Optuna for Bayesian hyperparameter optimization.

Use:
- optuna.create_study(direction="maximize")
- TPESampler(seed=42)
- objective should maximize mean validation F1 using cross-validation on train_manual.csv only

Search space inside Optuna:

n_estimators:
fast: categorical [100]
final: categorical [100, 300]

max_depth:
fast: categorical [5, 10]
final: categorical [5, 10, 20, None]

min_samples_split:
fast: categorical [2]
final: categorical [2, 5, 10]

min_samples_leaf:
fast: categorical [1]
final: categorical [1, 2, 4]

For each trial:
- build RandomForestClassifier with sampled parameters
- use class_weight="balanced"
- use random_state=42
- evaluate using StratifiedKFold cross-validation
- return mean F1 score

Save all trials to:
results/optuna_trials.csv

After optimization:
- take best trial params
- retrain on full training data
- evaluate once on test data
- save model to models/optuna_rf.pkl

Record:
- best_params
- best_cv_f1
- total optimization time
- test F1
- test ROC-AUC
- test precision
- test recall
- test accuracy

Save:
results/optuna_results.csv

==================================================
BASELINE REQUIREMENTS
==================================================

For the baseline Random Forest:

1. Train on train_manual.csv.
2. Perform cross-validation on train_manual.csv.
3. Evaluate on test_manual.csv exactly once.
4. Save model to models/baseline_rf.pkl.

Record:
- cv_f1_mean
- cv_f1_std
- test_f1
- test_roc_auc
- test_precision
- test_recall
- test_accuracy
- training_time_seconds

Save:
results/baseline_metrics.csv

Generate:
plots/baseline_confusion_matrix.png

==================================================
METRICS AND EVALUATION
==================================================

Create a reusable evaluation function.

For each model, record:
- method_name
- cv_f1_mean
- cv_f1_std
- test_f1
- test_roc_auc
- test_precision
- test_recall
- test_accuracy
- training_time_seconds
- best_params

Use:
- classification_report
- confusion_matrix
- f1_score
- precision_score
- recall_score
- accuracy_score
- roc_auc_score

For ROC-AUC, use predict_proba[:, 1] if available.

Create:
plots/roc_curve_phase2.png

If ROC curve generation fails for any model, do not block the pipeline.
Log the issue and continue.

==================================================
TIME TRACKING
==================================================

Use wall-clock time measurement.

For each method, record:
- start_time
- end_time
- duration_seconds

Use time.perf_counter().

Include time for:
- full Grid Search
- full Random Search
- full Optuna optimization
- baseline training

Do not include test evaluation time inside hyperparameter search time unless it is inseparable.
If inseparable, document that in ASSUMPTIONS.md or phase2_metadata.json.

==================================================
COMPARISON TABLE
==================================================

Create results/tuning_comparison.csv with these columns:

method
cv_f1_mean
cv_f1_std
test_f1
test_roc_auc
test_precision
test_recall
test_accuracy
training_time_seconds
best_params

Rows must include:
- Baseline Random Forest
- Grid Search Random Forest
- Random Search Random Forest
- Optuna Random Forest

Create plots:

plots/tuning_f1_comparison.png
plots/tuning_time_comparison.png

The F1 comparison chart should compare test F1 or CV F1 clearly.
Prefer test F1 for the main chart.
The time comparison chart should show total search/training time.

All plots must have:
- title
- axis labels
- readable tick labels
- legend if needed

==================================================
BEST MODEL SELECTION
==================================================

Select the best Phase 2 method using this priority:

1. Highest test F1-score.
2. If tied, highest ROC-AUC.
3. If tied, lower training time.

Save:
results/phase2_best_model.json

This JSON must contain:
- best_method
- model_path
- test_f1
- test_roc_auc
- training_time_seconds
- best_params
- phase2_mode
- timestamp

Also save:
results/best_params.json

This should contain best parameters for each method:
- grid_search
- random_search
- optuna

==================================================
FAST/FINAL MODE METADATA
==================================================

Create results/phase2_metadata.json containing:

- phase = "phase2"
- mode = "fast" or "final"
- timestamp
- random_seed
- target_column
- primary_metric
- train_rows
- test_rows
- positive_class_percentage_train
- positive_class_percentage_test
- package_versions
- fast_mode_warning if mode is fast
- command_used
- assumptions
- installation_issues

If mode is fast, include this warning:

"These are fast smoke-test results. For final report screenshots, rerun with --mode final."

==================================================
LOGGING AND REPRODUCIBILITY
==================================================

Use logging.

Write logs to:
results/phase2_run.log

Set random seed everywhere:
- Python random if used
- NumPy
- scikit-learn estimators
- Optuna sampler

Use:
random_state = 42

Do not produce results that cannot be reproduced.

==================================================
VALIDATION SCRIPT
==================================================

Create scripts/validate_phase2.py.

It must check that these files exist and are non-empty:

src/models/__init__.py
src/models/baseline.py
src/models/tuning.py
src/models/evaluation.py
scripts/run_phase2.py
results/phase2_metadata.json
results/baseline_metrics.csv
results/grid_search_results.csv
results/random_search_results.csv
results/optuna_results.csv
results/tuning_comparison.csv
results/best_params.json
results/phase2_best_model.json
results/optuna_trials.csv
results/phase2_summary.md
results/phase2_run.log
models/baseline_rf.pkl
models/grid_rf.pkl
models/random_rf.pkl
models/optuna_rf.pkl
plots/baseline_confusion_matrix.png
plots/best_tuned_confusion_matrix.png
plots/tuning_f1_comparison.png
plots/tuning_time_comparison.png
plots/roc_curve_phase2.png
reports/automl_phase2_results.md
reports/screenshots/phase2_baseline_confusion_matrix.png
reports/screenshots/phase2_best_tuned_confusion_matrix.png
reports/screenshots/phase2_tuning_f1_comparison.png
reports/screenshots/phase2_tuning_time_comparison.png
reports/screenshots/phase2_roc_curve_phase2.png

It must print PASS or FAIL for each item and a final overall status.

==================================================
REPORT OUTPUT REQUIREMENTS
==================================================

Create reports/automl_phase2_results.md.

This file must be written in a report-ready academic style.

Include:

1. Phase 2 Objective
2. Baseline Model Description
3. Dataset Used
4. Validation Strategy
5. Hyperparameter Tuning Methods
6. Results Table
7. Time Comparison
8. Best Method
9. Interpretation
10. Limitations
11. Screenshots to Include in Final Report

Do not write the full final AutoML report yet.
Only create the Phase 2 section.

==================================================
IMPORTANT CONSTRAINTS
==================================================

- Do not run FLAML in Phase 2.
- Do not run AutoGluon in Phase 2.
- Do not run PyCaret in Phase 2.
- Do not run H2O in Phase 2.
- Do not run auto-sklearn in Phase 2.
- Do not build the Streamlit app in Phase 2.
- Do not generate SHAP explanations in Phase 2.
- Do not generate LIME explanations in Phase 2.
- Do not modify Phase 1 processed data files.
- Do not rerun Phase 1 unless required files are missing.
- Do not use the test set for tuning.
- Do not use data leakage.
- Do not change the target column.
- Do not change the random seed.
- Do not choose a different model family.
- Do not ask the user for decisions.
- Do not start Phase 3.

==================================================
EXECUTION COMMANDS
==================================================

Create scripts/run_phase2.py with argparse.

It must support:

--mode fast
--mode final

Default:
fast

Example commands:

.venv\Scripts\python scripts/run_phase2.py --mode fast
.venv\Scripts\python scripts/run_phase2.py --mode final
.venv\Scripts\python scripts/validate_phase2.py

For Linux/macOS:

.venv/bin/python scripts/run_phase2.py --mode fast
.venv/bin/python scripts/run_phase2.py --mode final
.venv/bin/python scripts/validate_phase2.py

==================================================
AGENT EXECUTION INSTRUCTIONS
==================================================

You must now:

1. Implement all Phase 2 code.
2. Install only required Phase 2 packages.
3. Run Phase 2 in fast mode automatically.
4. Run scripts/validate_phase2.py.
5. If validation fails, fix issues and rerun until validation passes.
6. Produce all required artifacts.
7. Do not run final mode automatically unless the environment is clearly safe and fast enough.
8. Stop after Phase 2.

==================================================
FINAL OUTPUT REQUIRED FROM YOU
==================================================

After completing Phase 2, provide:

1. Full list of created files.
2. Output of scripts/validate_phase2.py.
3. Short metrics summary for:
   - Baseline
   - Grid Search
   - Random Search
   - Optuna
4. Best Phase 2 method and reason.
5. Any assumptions made.
6. Any installation issues.
7. Exact command to rerun fast mode.
8. Exact command to run final mode.
9. Exact command to validate Phase 2.
10. A message saying:
   "Phase 2 completed. Ready for Phase 3 master prompt."

Do not start Phase 3.