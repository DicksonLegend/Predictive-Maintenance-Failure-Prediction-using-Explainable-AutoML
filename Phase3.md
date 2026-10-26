MASTER PROMPT — PHASE 3: AUTOML TOOL COMPARISON AND CRITICAL EVALUATION

You are a senior machine learning engineer and academic project assistant. You are continuing an already locked university IA project that combines:

1. AutoML
2. Explainable AI (XAI)

Project title:
Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0

AutoML report title:
AutoML Benchmarking for Predictive Maintenance Failure Prediction

Explainable AI report title:
Explainable AI Dashboard for Predictive Maintenance Decision Support

Phase 1 and Phase 2 are already completed and validated.

Phase 1 created:
- dataset loading
- preprocessing
- train/test splits
- EDA artifacts

Phase 2 created:
- manual Random Forest baseline
- Grid Search tuning
- Random Search tuning
- Optuna tuning
- comparison tables
- saved models
- plots
- Phase 2 report section

Your current task is Phase 3 only.

Do not ask questions. Make reasonable engineering decisions, document every decision, and complete Phase 3 fully. If a tool fails, use the specified fallback rules. Do not fabricate results. Do not skip failed steps silently. Log every issue.

==================================================
LOCKED PROJECT CONSTRAINTS — DO NOT CHANGE
==================================================

Dataset:
AI4I 2020 Predictive Maintenance Dataset.

Phase 1 data artifacts to use for AutoML:
- data/processed/train_automl.csv
- data/processed/test_automl.csv
- data/processed/metadata.json

Do not use train_manual.csv or test_manual.csv for training AutoML tools.
You may read Phase 2 result files for comparison only.

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
- ROC-AUC
- Precision-Recall AUC
- Precision
- Recall
- Accuracy

Class imbalance:
The dataset has severe imbalance.
Do not use accuracy alone.
Do not apply SMOTE unless explicitly required later.
Use metrics that handle imbalance.

Train-test split:
Already created in Phase 1.
Do not re-split the data.
Do not modify test_automl.csv.
Do not use the test set for model selection or hyperparameter search.
Use the test set only for final evaluation after each AutoML tool completes training.

==================================================
PHASE 3 OBJECTIVE
==================================================

Phase 3 has two major parts.

PART A: AutoML implementation and tool comparison.

You must:
1. Run FLAML on the same dataset.
2. Run AutoGluon Tabular on the same dataset.
3. Give both tools the same fixed time budget.
4. Save each tool leaderboard or equivalent model search results.
5. Record which models were tried.
6. Record which model won.
7. Record preprocessing or feature engineering done by each tool if available.
8. Compare AutoML results with Phase 2 manual baseline and tuned Random Forest models.
9. Compare performance, training time, and interpretability.

PART B: Critical evaluation.

You must:
1. Check overfitting using internal validation or cross-validation and held-out test results.
2. Create one deliberately leaky feature.
3. Show whether an AutoML tool exploits the leaky feature.
4. Use feature importance or SHAP to expose the leaky feature.
5. Analyze class imbalance.
6. Analyze group behavior using machine type: L, M, H.
7. Save limitations.
8. Produce report-ready markdown output.

==================================================
PHASE 3 RUBRIC REQUIREMENTS TO SATISFY
==================================================

This phase must satisfy the AutoML rubric Step 3 and Step 4.

Step 3 requirements:
- Run at least two AutoML frameworks.
- Give every tool the same fixed time budget.
- Study each leaderboard.
- Identify models tried and winning model.
- Identify preprocessing and feature engineering applied.
- Compare AutoML results with manual baseline on F1 or accuracy.
- Compare training time.
- Compare interpretability.

Step 4 requirements:
- Check for data leakage and overfitting using cross-validation and a held-out test set.
- Deliberately add one leaky feature and show whether the AutoML tool exploits it.
- Check for bias or class imbalance.
- Explain model decisions with SHAP or an equivalent method.
- Write limitations.

==================================================
EXISTING PROJECT CONTEXT
==================================================

Use existing artifacts:

Data:
- data/processed/train_automl.csv
- data/processed/test_automl.csv
- data/processed/metadata.json

Phase 2 results:
- results/phase2_metadata.json
- results/baseline_metrics.csv
- results/grid_search_results.csv
- results/random_search_results.csv
- results/optuna_results.csv
- results/tuning_comparison.csv
- results/phase2_best_model.json
- models/baseline_rf.pkl
- models/grid_rf.pkl
- models/random_rf.pkl
- models/optuna_rf.pkl

If any required Phase 2 file is missing, stop and report the missing file.
Do not redo Phase 2 automatically.

==================================================
PACKAGE INSTALLATION
==================================================

Use the existing virtual environment if present.

If Windows and .venv exists, prefer:

.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install flaml autogluon.tabular shap matplotlib seaborn pandas numpy scikit-learn joblib

If Linux/macOS and .venv exists, prefer:

.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install flaml autogluon.tabular shap matplotlib seaborn pandas numpy scikit-learn joblib

Do not uninstall Phase 2 packages.

If AutoGluon installation fails:
1. Capture the full error in results/tool_error_log.json.
2. Retry once.
3. If it still fails, install PyCaret as fallback second AutoML tool:

.venv\Scripts\python -m pip install pycaret

or

.venv/bin/python -m pip install pycaret

If PyCaret is used, set:

second_tool_name = "PyCaret"
second_tool_fallback = true

and document that PyCaret does not enforce a strict native time budget as cleanly as AutoGluon.

If both AutoGluon and PyCaret fail, stop and report the blocker. Do not fake results.

==================================================
FAST MODE AND FINAL MODE
==================================================

Create scripts/run_phase3.py with command-line argument:

--mode fast
--mode final

Default mode:
fast

Fast mode is for automated validation and should finish quickly.
Final mode is for final report screenshots and stronger results.

Fast mode settings:
- AUTOML_TIME_BUDGET_SECONDS = 60
- FAST_MODE_TRAIN_SAMPLE_ROWS = 3000
- CV_FOLDS = 2 where applicable
- SHAP_BACKGROUND_SAMPLE_SIZE = 50
- SHAP_EXPLANATION_SAMPLE_SIZE = 5

Final mode settings:
- AUTOML_TIME_BUDGET_SECONDS = 900
- Use full train_automl.csv
- CV_FOLDS = 5 where applicable
- SHAP_BACKGROUND_SAMPLE_SIZE = 200
- SHAP_EXPLANATION_SAMPLE_SIZE = 20

If fast mode still cannot complete because of resource limits, reduce FAST_MODE_TRAIN_SAMPLE_ROWS to 1500 and record this in metadata.

The agent should automatically run fast mode.
Do not automatically run final mode unless the environment is clearly safe and fast enough.
At the end, provide the exact final-mode command.

==================================================
CREATE REQUIRED CODE STRUCTURE
==================================================

Create these code artifacts:

src/automl/__init__.py
src/automl/common.py
src/automl/flaml_runner.py
src/automl/autogluon_runner.py
src/automl/fallback_runner.py
src/automl/leakage.py
src/automl/critical_evaluation.py
src/automl/feature_importance.py

scripts/run_phase3.py
scripts/validate_phase3.py

==================================================
DATA HANDLING RULES
==================================================

Load:

train_automl.csv
test_automl.csv

Expected columns:

type
air_temp
process_temp
rotational_speed
torque
tool_wear
machine_failure

The column type is categorical.
Do not one-hot encode it for AutoML tools unless the tool requires it.
Let the AutoML tool handle preprocessing if possible.

If fast mode sampling is used:
- sample from train_automl.csv only
- use stratified sampling by machine_failure
- keep test_automl.csv unchanged
- record sampled train row count in metadata

Do not overwrite:
- data/processed/train_automl.csv
- data/processed/test_automl.csv

==================================================
AUTOML TOOL 1: FLAML
==================================================

Use FLAML as the first AutoML tool.

Requirements:
- task = classification
- metric = f1
- time_budget_s = AUTOML_TIME_BUDGET_SECONDS
- seed = 42
- use cross-validation if supported
- do not pass the test set for training or model selection
- evaluate the final FLAML model once on test_automl.csv

Save:
- results/flaml_leaderboard.csv
- results/flaml_best_config.json
- results/flaml_metrics.csv
- models/flaml_model.pkl
- results/flaml_log.txt if available

FLAML leaderboard or equivalent output must include as much as possible:
- model name
- validation score
- config or hyperparameters if available
- training time if available
- rank if available

If FLAML does not provide a full leaderboard, save the best model information and any available searched model summary.

==================================================
AUTOML TOOL 2: AUTOGLUON TABULAR
==================================================

Use AutoGluon Tabular as the second AutoML tool.

Requirements:
- label = machine_failure
- eval_metric = f1
- time_limit = AUTOML_TIME_BUDGET_SECONDS
- use fixed random seed where supported
- do not pass test data during fit
- after fit, generate leaderboard using the held-out test set exactly once
- save the AutoGluon predictor directory

Save:
- results/autogluon_leaderboard.csv
- results/autogluon_metrics.csv
- results/autogluon_feature_importance.csv if available
- models/autogluon_predictor/ directory

The AutoGluon leaderboard must include:
- model
- score_test if available
- score_val if available
- pred_time_test if available
- fit_time if available
- stack_level if available
- can_infer if available

If AutoGluon creates large model artifacts, do not fail validation because of size.
Only verify that the predictor directory exists and is non-empty.

==================================================
FALLBACK TOOL: PYCARET
==================================================

Use PyCaret only if AutoGluon installation or execution fails.

Requirements:
- use train_automl.csv
- target = machine_failure
- session_id = 42
- use classification setup
- compare models
- use fold = 2 in fast mode and fold = 5 in final mode
- record wall-clock time
- evaluate best model on test_automl.csv exactly once

Save:
- results/pycaret_leaderboard.csv
- results/pycaret_metrics.csv
- models/pycaret_model.pkl

Mark clearly in metadata:

second_tool_name = "PyCaret"
second_tool_fallback = true
time_budget_enforcement = "approximate"

==================================================
SAME FIXED TIME BUDGET RULE
==================================================

Both AutoML tools must receive the same requested time budget.

Use:

FAST MODE:
60 seconds

FINAL MODE:
900 seconds

Record:
- requested_time_budget_seconds
- actual_training_time_seconds

If a tool finishes early, record actual time.
If a tool times out, record actual time and status.
Do not give one tool more time than the other.

==================================================
METRICS AND EVALUATION
==================================================

For every model or tool, record:

method
tool
best_model
cv_or_validation_f1
test_f1
test_roc_auc
test_pr_auc
test_precision
test_recall
test_accuracy
training_time_seconds
requested_time_budget_seconds
preprocessing_notes
interpretability_notes

Use threshold:
0.5

For ROC-AUC and PR-AUC, use predict_proba if available.
If predict_proba is unavailable, use decision_function if available.
If neither is available, set the value to NaN and log the issue.

Create:

results/automl_comparison.csv

This file must contain only Phase 3 AutoML tools.

Create:

results/phase3_full_comparison.csv

This file must combine:
- Phase 2 baseline
- Phase 2 Grid Search
- Phase 2 Random Search
- Phase 2 Optuna
- Phase 3 FLAML
- Phase 3 AutoGluon or fallback tool

Include an extra column:

phase
mode
source

Example values:

phase = phase2 or phase3
mode = fast or final
source = phase2_results or phase3_run

==================================================
OVERFITTING CHECK
==================================================

Create results/overfitting_check.csv.

For each method, include:

method
validation_or_cv_f1
test_f1
f1_gap
roc_auc_gap_if_available
overfitting_flag
notes

Rules:
- If test F1 is much higher than validation F1, mark unusual_distribution_shift_or_fast_mode_artifact.
- If validation F1 is much higher than test F1, mark possible_overfitting.
- If both are close, mark stable.

Use a gap threshold of 0.08 for flagging, but always show the raw values.

==================================================
LEAKAGE EXPERIMENT
==================================================

Create a deliberate leakage experiment.

Create new leaky datasets:

data/processed/train_automl_leak.csv
data/processed/test_automl_leak.csv

Do not overwrite original files.

Add one leaky feature:

post_failure_inspection_flag

Generate it deterministically using seed 42.

Logic:
- If machine_failure == 1, set post_failure_inspection_flag = 1 with probability 0.92.
- If machine_failure == 0, set post_failure_inspection_flag = 1 with probability 0.04.

Apply the same logic separately to train and test using their own target values.

Run the leakage experiment using FLAML only, unless FLAML fails.
If FLAML fails, use the second AutoML tool.

Use the same mode time budget.

Evaluate:
- without leaky feature
- with leaky feature

Save:

results/leakage_results.csv

Columns:

experiment
tool
test_f1
test_roc_auc
test_pr_auc
test_accuracy
training_time_seconds
top_feature_1
top_feature_2
top_feature_3

Save:

results/leakage_metadata.json

Include:
- leaky_feature_name
- leak_probability_positive
- leak_probability_negative
- random_seed
- tool_used
- mode
- warning message

Generate:

plots/leakage_metric_comparison.png

This should compare F1, ROC-AUC, and PR-AUC with and without leakage.

Generate:

plots/leakage_feature_importance.png

Use feature importance or SHAP.

Priority order for explaining leakage:
1. SHAP on the AutoML model if possible.
2. Native feature importance from the AutoML model if possible.
3. Permutation importance on the AutoML model if possible.
4. Surrogate Random Forest trained on the leaky dataset if none of the above works.

If a surrogate model is used, clearly mark it in the plot title and in the report.

The leaky feature should appear among the top features if leakage was exploited.

==================================================
CLASS IMBALANCE ANALYSIS
==================================================

Create results/class_imbalance_report.csv.

Include:
- train positive count
- train negative count
- train positive percentage
- test positive count
- test negative count
- test positive percentage
- selected model test confusion matrix values
- selected model F1
- selected model recall
- selected model precision
- selected model PR-AUC

Generate:

plots/phase3_class_distribution.png
plots/phase3_confusion_matrix_best_model.png
plots/phase3_precision_recall_curve.png

If precision-recall curve generation fails, log the issue and continue.

==================================================
GROUP BIAS ANALYSIS
==================================================

Use the column type from test_automl.csv.

Groups:
L
M
H

Using the selected best Phase 3 model, compute group-wise performance on the held-out test set.

Create results/group_performance.csv.

Columns:

machine_type
total_samples
actual_failure_count
actual_failure_rate
predicted_failure_count
predicted_failure_rate
true_positives
false_positives
true_negatives
false_negatives
false_positive_rate
false_negative_rate
precision
recall
f1

Generate:

plots/group_performance.png

The plot should compare at least:
- false positive rate
- false negative rate
- predicted failure rate
- actual failure rate

Write a short interpretation in the Phase 3 report file.

==================================================
BEST MODEL SELECTION
==================================================

Select the best Phase 3 model using this priority:

1. Highest test F1-score.
2. If tied, highest test PR-AUC.
3. If tied, highest ROC-AUC.
4. If tied, lower training time.

Save:

results/phase3_best_model.json

Include:
- best_tool
- best_model_name
- model_path
- test_f1
- test_roc_auc
- test_pr_auc
- test_precision
- test_recall
- test_accuracy
- training_time_seconds
- requested_time_budget_seconds
- mode
- timestamp
- notes

Save test predictions:

results/phase3_test_predictions.csv

Columns:

row_index
machine_type
y_true
y_pred
y_prob

If y_prob is unavailable, set it to NaN and log.

==================================================
INTERPRETABILITY NOTES
==================================================

Create results/interpretability_notes.csv.

Columns:

method
model_type
interpretability_level
reason

Include rows for:
- Phase 2 Baseline Random Forest
- Phase 2 Best Tuned Random Forest
- FLAML best model
- AutoGluon best model or fallback best model

Use interpretability_level:
- high
- medium
- low

Guidance:
- Simple decision tree: high
- Random Forest: medium
- Single linear model: high
- Gradient boosting: medium to low
- Weighted ensemble or stacker: low

==================================================
REPORT OUTPUT
==================================================

Create:

reports/automl_phase3_results.md

Write it in report-ready academic style.

Include:

1. Phase 3 Objective
2. AutoML Tools Used
3. Fixed Time Budget
4. Dataset Used
5. FLAML Results
6. AutoGluon Results or Fallback Tool Results
7. Leaderboard Observations
8. Comparison with Manual Baseline and Tuned Models
9. Training Time Comparison
10. Interpretability Comparison
11. Overfitting Check
12. Leakage Experiment
13. Class Imbalance Analysis
14. Group Bias Analysis
15. Limitations
16. Conclusion
17. Screenshots to Include in Final Report

Do not write the full final AutoML report yet.
Only create the Phase 3 section.

==================================================
METADATA AND LOGGING
==================================================

Create results/phase3_metadata.json.

Include:
- phase = phase3
- mode = fast or final
- timestamp
- random_seed
- target_column
- primary_metric
- requested_time_budget_seconds
- actual_flaml_time_seconds
- actual_second_tool_time_seconds
- train_rows_used
- test_rows_used
- positive_class_percentage_train
- positive_class_percentage_test
- phase2_mode if detectable
- phase2_warning if phase2 was fast mode
- second_tool_name
- second_tool_fallback
- package_versions
- assumptions
- installation_issues
- execution_issues

Create logs:

results/phase3_run.log
results/tool_error_log.json

If there are no errors, tool_error_log.json should contain an empty list.

Use logging.
Use matplotlib Agg backend if display is unavailable.

==================================================
PLOTS AND SCREENSHOTS
==================================================

Create these plots:

plots/phase3_model_comparison_f1.png
plots/phase3_time_comparison.png
plots/phase3_full_comparison.png
plots/overfitting_check.png
plots/leakage_metric_comparison.png
plots/leakage_feature_importance.png
plots/phase3_class_distribution.png
plots/phase3_confusion_matrix_best_model.png
plots/phase3_precision_recall_curve.png
plots/group_performance.png

All plots must have:
- title
- axis labels
- readable tick labels
- legend if needed

Copy important plots to reports/screenshots/ with phase3 prefix:

reports/screenshots/phase3_model_comparison_f1.png
reports/screenshots/phase3_time_comparison.png
reports/screenshots/phase3_full_comparison.png
reports/screenshots/phase3_overfitting_check.png
reports/screenshots/phase3_leakage_metric_comparison.png
reports/screenshots/phase3_leakage_feature_importance.png
reports/screenshots/phase3_class_distribution.png
reports/screenshots/phase3_confusion_matrix_best_model.png
reports/screenshots/phase3_precision_recall_curve.png
reports/screenshots/phase3_group_performance.png

==================================================
VALIDATION SCRIPT
==================================================

Create scripts/validate_phase3.py.

It must dynamically check the required artifacts.

Always check:

src/automl/__init__.py
src/automl/common.py
src/automl/flaml_runner.py
src/automl/leakage.py
src/automl/critical_evaluation.py
src/automl/feature_importance.py
scripts/run_phase3.py
results/phase3_metadata.json
results/phase3_run.log
results/tool_error_log.json
results/flaml_leaderboard.csv
results/flaml_metrics.csv
results/flaml_best_config.json
models/flaml_model.pkl
results/automl_comparison.csv
results/phase3_full_comparison.csv
results/overfitting_check.csv
results/leakage_results.csv
results/leakage_metadata.json
results/class_imbalance_report.csv
results/group_performance.csv
results/interpretability_notes.csv
results/phase3_best_model.json
results/phase3_test_predictions.csv
results/phase3_summary.md
reports/automl_phase3_results.md
plots/phase3_model_comparison_f1.png
plots/phase3_time_comparison.png
plots/phase3_full_comparison.png
plots/overfitting_check.png
plots/leakage_metric_comparison.png
plots/leakage_feature_importance.png
plots/phase3_class_distribution.png
plots/phase3_confusion_matrix_best_model.png
plots/phase3_precision_recall_curve.png
plots/group_performance.png

Also check second tool artifacts dynamically:

If second_tool_name is AutoGluon:
- results/autogluon_leaderboard.csv
- results/autogluon_metrics.csv
- models/autogluon_predictor/

If second_tool_name is PyCaret:
- results/pycaret_leaderboard.csv
- results/pycaret_metrics.csv
- models/pycaret_model.pkl

The validator must print PASS or FAIL for each item and a final overall status.

==================================================
IMPORTANT CONSTRAINTS
==================================================

- Do not modify Phase 1 data files.
- Do not modify Phase 2 result files.
- Do not rerun Phase 2.
- Do not overwrite train_automl.csv or test_automl.csv.
- Do not use test data for training or model selection.
- Do not give different time budgets to different AutoML tools.
- Do not fabricate leaderboard results.
- Do not run Streamlit in Phase 3.
- Do not build the final XAI dashboard in Phase 3.
- Do not deploy the model in Phase 3.
- Do not start Phase 4.
- Do not ask the user for decisions.
- Do not choose a different dataset.
- Do not change the target column.
- Do not change the random seed.

==================================================
EXECUTION COMMANDS
==================================================

Create scripts/run_phase3.py with argparse.

It must support:

--mode fast
--mode final

Default:
fast

Example commands:

.venv\Scripts\python scripts/run_phase3.py --mode fast
.venv\Scripts\python scripts/run_phase3.py --mode final
.venv\Scripts\python scripts/validate_phase3.py

For Linux/macOS:

.venv/bin/python scripts/run_phase3.py --mode fast
.venv/bin/python scripts/run_phase3.py --mode final
.venv/bin/python scripts/validate_phase3.py

==================================================
AGENT EXECUTION INSTRUCTIONS
==================================================

You must now:

1. Implement all Phase 3 code.
2. Install required Phase 3 packages.
3. Run Phase 3 in fast mode automatically.
4. Run scripts/validate_phase3.py.
5. If validation fails, fix issues and rerun until validation passes.
6. Produce all required artifacts.
7. Do not run final mode automatically unless the environment is clearly safe and fast enough.
8. Stop after Phase 3.

==================================================
FINAL OUTPUT REQUIRED FROM YOU
==================================================

After completing Phase 3, provide:

1. Full list of created files.
2. Output of scripts/validate_phase3.py.
3. Short metrics summary for:
   - FLAML
   - AutoGluon or fallback tool
   - Phase 2 baseline comparison
4. Best Phase 3 model and reason.
5. Leakage experiment finding.
6. Overfitting finding.
7. Class imbalance finding.
8. Group bias finding.
9. Any assumptions made.
10. Any installation issues.
11. Any tool execution issues.
12. Exact command to rerun fast mode.
13. Exact command to run final mode.
14. Exact command to validate Phase 3.
15. A message saying:
   "Phase 3 completed. Ready for Phase 4 master prompt."

Do not start Phase 4.