MASTER PROMPT — PHASE 1: PROJECT FOUNDATION, SPEC, DATA PIPELINE, AND EDA

You are a senior machine learning engineer and academic project assistant. Your job is to create the complete foundation for a university IA project that combines two subjects:

1. AutoML
2. Explainable AI (XAI)

You must not ask me questions. You must make reasonable assumptions, document them, and complete Phase 1 fully. If something is missing, create a placeholder or fallback and continue. Do not stop for small issues.

==================================================
PROJECT LOCKED DECISIONS — DO NOT CHANGE THESE
==================================================

Project domain:
Predictive Maintenance in Smart Manufacturing.

Final project idea:
Build an Explainable AutoML system that predicts machine failure using sensor telemetry data, compares manual ML with AutoML tools, critically evaluates leakage and imbalance, and provides explainable predictions through a dashboard.

Final problem statement:
Given machine sensor readings, predict whether the machine will fail or not, and explain the root cause of the failure risk using Explainable AI methods.

Dataset:
AI4I 2020 Predictive Maintenance Dataset.

Dataset source:
Search for “AI4I 2020 Predictive Maintenance Dataset” from UCI Machine Learning Repository or Kaggle.

If the real dataset is not available in the environment:
- Create a synthetic smoke-test dataset with the same schema.
- Clearly mark it as synthetic.
- Do not treat synthetic data as final submission data.
- Create a file called DOWNLOAD_INSTRUCTIONS.md explaining that the final run must use the real AI4I dataset.

Target variable:
Create a binary target column named:

machine_failure

Meaning:
0 = No failure
1 = Failure

If the dataset contains a column called “Target”, use it.
If the dataset contains failure type columns such as TWF, HDF, PWF, OSF, RNF, create machine_failure as 1 if any failure type is 1, otherwise 0.

Main features:
- type
- air_temp
- process_temp
- rotational_speed
- torque
- tool_wear

Machine type values:
L, M, H

Fixed random seed:
RANDOM_SEED = 42

Train-test split:
80% train, 20% test, stratified by target.

Primary metric:
F1-score

Secondary metrics:
ROC-AUC, precision, recall, accuracy

Class imbalance:
The failure class is expected to be imbalanced. Do not oversample in Phase 1. Only record the class distribution.

==================================================
RUBRICS TO SATISFY
==================================================

AutoML rubric requirements:
- Real-world problem and dataset with at least a few thousand rows.
- Manual baseline with scikit-learn.
- Hyperparameter tuning using Grid Search, Random Search, and Optuna.
- At least two AutoML frameworks, preferably FLAML and AutoGluon.
- Same fixed time budget for AutoML tools.
- Compare leaderboard, models tried, preprocessing, performance, training time, and interpretability.
- Check leakage, overfitting, bias/class imbalance.
- Add one deliberately leaky feature and show whether AutoML exploits it.
- Use SHAP or equivalent explanation.
- Deploy using Streamlit or FastAPI.
- Provide model card.
- Provide report, slides, and reproducible code.

Explainable AI rubric requirements:
- GUI-based application.
- Clear understanding of Explainable AI concepts.
- Apply explanation methods to tabular data, and also demonstrate image and text explanation at a small demo level.
- Visualization and interpretation.
- Presentation clarity.
- Innovation.
- Documentation.

This Phase 1 foundation must support both reports:

AutoML Report Title:
AutoML Benchmarking for Predictive Maintenance Failure Prediction

Explainable AI Report Title:
Explainable AI Dashboard for Predictive Maintenance Decision Support

==================================================
PHASE 1 OBJECTIVES
==================================================

Phase 1 must create the complete project foundation. Do not train final ML models yet. Do not run AutoML yet. Do not build the full Streamlit app yet.

Phase 1 must do the following:

1. Create the full project folder structure.
2. Create PROJECT_SPEC.md containing the full project plan and locked decisions.
3. Create PLAN.md describing all 4 phases.
4. Create requirements.txt.
5. Create README.md.
6. Create .gitignore.
7. Create ASSUMPTIONS.md.
8. Create data loading utilities.
9. Create synthetic fallback data generator if real data is unavailable.
10. Create preprocessing utilities.
11. Create train/test splits.
12. Create EDA plots and summary tables.
13. Create report outlines for both AutoML and XAI reports.
14. Create placeholder model card.
15. Create a Phase 1 validation script.
16. Run the Phase 1 pipeline successfully.
17. Print a final Phase 1 completion checklist.

==================================================
REQUIRED FOLDER STRUCTURE
==================================================

Create this exact structure from the project root:

predictive_maintenance_xai_automl/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   └── data/
│       ├── __init__.py
│       ├── load_data.py
│       ├── generate_synthetic.py
│       └── preprocess.py
│
├── scripts/
│   ├── run_phase1.py
│   └── validate_phase1.py
│
├── models/
│   └── README.md
│
├── results/
│   └── README.md
│
├── plots/
│   └── README.md
│
├── app/
│   └── README.md
│
├── reports/
│   ├── automl_report_outline.md
│   ├── xai_report_outline.md
│   └── screenshots/
│       └── README.md
│
├── slides/
│   └── README.md
│
├── PROJECT_SPEC.md
├── PLAN.md
├── ASSUMPTIONS.md
├── DOWNLOAD_INSTRUCTIONS.md
├── README.md
├── requirements.txt
└── .gitignore

==================================================
CONFIG REQUIREMENTS
==================================================

Create src/config.py with these constants:

RANDOM_SEED = 42
TARGET_COLUMN = "machine_failure"
TEST_SIZE = 0.2

PRIMARY_METRIC = "f1"
SECONDARY_METRICS = ["roc_auc", "precision", "recall", "accuracy"]

AUTOML_TIME_BUDGET_SECONDS_AGENT = 60
AUTOML_TIME_BUDGET_SECONDS_FINAL = 900

Paths:
ROOT_DIR
DATA_RAW_DIR
DATA_PROCESSED_DIR
RESULTS_DIR
PLOTS_DIR
MODELS_DIR
REPORTS_DIR
SCREENSHOTS_DIR

Important:
Use AUTOML_TIME_BUDGET_SECONDS_AGENT = 60 for automated agent testing in later phases.
Use AUTOML_TIME_BUDGET_SECONDS_FINAL = 900 for final human-run experiments and screenshots.

==================================================
DATA LOADING REQUIREMENTS
==================================================

Create src/data/load_data.py.

It must:
1. Look for the real AI4I dataset in data/raw/.
2. Accept common file names such as:
   - ai4i_2020.csv
   - ai4i2020.csv
   - AI4I_2020.csv
   - ai4i_data.csv
3. If found, load it.
4. Standardize column names to:
   - type
   - air_temp
   - process_temp
   - rotational_speed
   - torque
   - tool_wear
   - machine_failure
5. Handle common original column names such as:
   - Type
   - Air temperature [K]
   - Process temperature [K]
   - Rotational speed [rpm]
   - Torque [Nm]
   - Tool wear [min]
   - Target
   - Machine failure
6. If failure type columns exist, create machine_failure from them if needed.
7. Drop unnecessary columns such as UDI or Product ID.
8. Return a clean pandas DataFrame.
9. Save metadata to data/processed/metadata.json.

Metadata must include:
- dataset_source: "real" or "synthetic"
- number_of_rows
- number_of_columns
- target_distribution
- missing_value_count
- created_timestamp
- warning if synthetic data is used

==================================================
SYNTHETIC FALLBACK REQUIREMENTS
==================================================

If no real dataset is found, create src/data/generate_synthetic.py.

Generate 10000 rows.

Feature ranges should be approximately:
- type: L, M, H
- air_temp: 295 to 305 Kelvin
- process_temp: 305 to 315 Kelvin
- rotational_speed: 1000 to 3000 rpm
- torque: 10 to 80 Nm
- tool_wear: 0 to 250 minutes

Target logic:
Failure probability should increase when:
- torque is high
- tool wear is high
- process temperature is high
- rotational speed is abnormal

Target imbalance:
Failure class should be around 3% to 6% of rows.

Save synthetic file to:
data/raw/synthetic_ai4i_smoke_test.csv

Important:
If synthetic data is used, create a warning file called USE_REAL_DATA.md saying:
“This pipeline is currently using synthetic data for smoke testing only. Replace data/raw/ with the real AI4I 2020 dataset before final submission.”

==================================================
PREPROCESSING REQUIREMENTS
==================================================

Create src/data/preprocess.py.

It must create two versions of the data:

1. Manual ML version:
   - Numeric features cleaned and imputed.
   - Missing numeric values filled with median.
   - Missing categorical values filled with mode.
   - Machine type one-hot encoded.
   - Target column remains machine_failure.
   - Save as:
     data/processed/train_manual.csv
     data/processed/test_manual.csv

2. AutoML version:
   - Keep numeric features.
   - Keep type as categorical string.
   - Do not one-hot encode.
   - Save as:
     data/processed/train_automl.csv
     data/processed/test_automl.csv

Use stratified train-test split:
test_size = 0.2
random_state = 42

All outputs must include the target column machine_failure.

==================================================
EDA REQUIREMENTS
==================================================

Create EDA outputs inside scripts/run_phase1.py.

Generate the following results:

CSV files:
results/dataset_summary.csv
results/missing_values.csv
results/class_distribution.csv
results/feature_summary.csv

Plot files:
plots/class_distribution.png
plots/correlation_heatmap.png
plots/missing_values.png
plots/feature_distributions.png

Use matplotlib Agg backend if display is unavailable.

Use seaborn and matplotlib.

Save copies of important plots into:
reports/screenshots/

==================================================
DOCUMENTATION REQUIREMENTS
==================================================

Create PROJECT_SPEC.md containing:
- Final project idea
- Combined subjects
- Dataset
- Target
- Metrics
- Random seed
- Train-test split
- AutoML tools
- Explanation methods
- Deployment plan
- Report titles
- Rubric mapping

Create PLAN.md with these 4 phases:

Phase 1:
Foundation, specification, data loading, preprocessing, EDA.

Phase 2:
Manual baseline, Grid Search, Random Search, Optuna tuning, comparison tables.

Phase 3:
FLAML and AutoGluon AutoML experiments, leaderboard analysis, leakage experiment, bias/imbalance analysis.

Phase 4:
SHAP, LIME, partial dependence, what-if explanation, Streamlit GUI, image XAI demo, text XAI demo, model card, final report artifacts.

Create README.md containing:
- Project title
- Problem statement
- Setup instructions
- Phase 1 run command
- Folder structure explanation
- Warning about synthetic data if used

Create ASSUMPTIONS.md and record all assumptions made during Phase 1.

Create DOWNLOAD_INSTRUCTIONS.md explaining how to download the real AI4I 2020 Predictive Maintenance Dataset from UCI or Kaggle.

Create model card placeholder inside PROJECT_SPEC.md or a separate section in README.md with these fields:
- Model name
- Dataset
- Intended use
- Target variable
- Metrics
- Limitations
- Ethical considerations

==================================================
REPORT OUTLINE REQUIREMENTS
==================================================

Create reports/automl_report_outline.md with these sections:

1. Abstract
2. Problem Statement
3. Dataset Description
4. Data Preprocessing
5. Manual Baseline Model
6. Hyperparameter Tuning
7. AutoML Tools Used
8. FLAML Experiment
9. AutoGluon Experiment
10. Model Comparison
11. Cross-validation Results
12. Held-out Test Results
13. Leakage Experiment
14. Class Imbalance Analysis
15. Bias / Group Analysis
16. Explainability using SHAP
17. Deployment
18. Model Card
19. Limitations
20. Conclusion
21. References
22. Appendix: Screenshots and Tables

Create reports/xai_report_outline.md with these sections:

1. Abstract
2. Introduction to Explainable AI
3. Need for Explainability in Predictive Maintenance
4. Case Study Description
5. Dataset Description
6. Explainability Methods Used
7. Global Explanation using SHAP
8. Local Explanation using SHAP
9. Local Explanation using LIME
10. Partial Dependence Analysis
11. What-If Explanation
12. Maintenance Recommendation System
13. Leakage Detection using SHAP
14. Bias and Fairness Explanation
15. Image Explainability Demo
16. Text Explainability Demo
17. GUI Application Design
18. Limitations
19. Conclusion
20. References
21. Appendix: Screenshots

==================================================
PACKAGE REQUIREMENTS
==================================================

Create requirements.txt with these packages:

pandas
numpy
scikit-learn
matplotlib
seaborn
shap
lime
flaml
autogluon.tabular
optuna
streamlit

For Phase 1 execution, install only these lightweight packages if possible:

pandas
numpy
scikit-learn
matplotlib
seaborn

Do not block Phase 1 if heavy packages like autogluon.tabular fail to install. Record installation issues in ASSUMPTIONS.md.

==================================================
EXECUTION REQUIREMENTS
==================================================

Create scripts/run_phase1.py.

This script must:
1. Load real data if available.
2. If real data is unavailable, generate synthetic smoke-test data.
3. Preprocess data.
4. Create manual and AutoML train/test CSV files.
5. Generate EDA tables and plots.
6. Save metadata.
7. Print a summary.

Create scripts/validate_phase1.py.

This script must check whether these files exist:

PROJECT_SPEC.md
PLAN.md
README.md
ASSUMPTIONS.md
requirements.txt
.gitignore
data/processed/train_manual.csv
data/processed/test_manual.csv
data/processed/train_automl.csv
data/processed/test_automl.csv
data/processed/metadata.json
results/dataset_summary.csv
results/missing_values.csv
results/class_distribution.csv
results/feature_summary.csv
plots/class_distribution.png
plots/correlation_heatmap.png
plots/missing_values.png
plots/feature_distributions.png
reports/automl_report_outline.md
reports/xai_report_outline.md

It must print PASS or FAIL for each item and a final overall status.

==================================================
IMPORTANT CONSTRAINTS
==================================================

- Do not choose a different problem.
- Do not choose a different dataset.
- Do not change the target column name.
- Do not run AutoML in Phase 1.
- Do not train final models in Phase 1.
- Do not build the full Streamlit app in Phase 1.
- Do not generate final PDF reports yet.
- Do not ask me for decisions.
- Use fixed random seed 42 everywhere.
- Make all code runnable from the project root.
- Use pathlib for file paths.
- Use logging instead of random print statements where appropriate.
- If a command fails, try to fix it.
- If a package is unavailable, document it and continue with the remaining work.
- All generated plots should be saved as PNG files.
- All generated tables should be saved as CSV files.
- Keep the code modular and clean for Phase 2, Phase 3, and Phase 4.

==================================================
FINAL OUTPUT REQUIRED FROM YOU
==================================================

After completing Phase 1, you must provide:

1. The full list of created files.
2. The output of scripts/validate_phase1.py.
3. A short summary of dataset shape, target distribution, and missing values.
4. Any assumptions made.
5. Any installation issues.
6. The exact command to run Phase 1 again.
7. A message saying:
   “Phase 1 completed. Ready for Phase 2 master prompt.”

Do not start Phase 2. Stop after Phase 1 validation.