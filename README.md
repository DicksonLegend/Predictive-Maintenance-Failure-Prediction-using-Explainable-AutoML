# Explainable AutoML for Predictive Maintenance

An industrial machine learning decision support system uniting **Automated Machine Learning (AutoML)** and **Explainable AI (XAI)** to predict equipment failure from sensor telemetry and deliver transparent, actionable root-cause explanations.

---

## 1. Problem Statement
In modern smart manufacturing, unexpected machinery breakdowns result in massive financial losses, supply chain bottlenecks, and severe safety hazards. 

This project tackles the **AI4I 2020 Predictive Maintenance** challenge: given telemetry measurements (air temperature, process temperature, spindle rotational speed, torque, and cumulative tool wear), predict whether the equipment is prone to imminent failure (`machine_failure` = 1) versus nominal operation (`machine_failure` = 0). Crucially, the system uncovers the underlying physical failure mechanisms (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure) via game-theoretic Shapley attributions (SHAP), local linear surrogates (LIME), partial dependence curves, and interactive what-if counterfactual analysis.

---

## 2. Quickstart & Setup Instructions

### Environment Setup
1. Open PowerShell / Command Prompt in the project root:
   ```powershell
   cd d:\AutoML_XAI
   ```
2. Activate the pre-configured Python 3.11 virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   *(For Phase 1 foundation runs, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn` are pre-installed in `.venv`)*.

---

## 3. Phase 1 Execution & Validation Commands

### Run Complete Phase 1 Pipeline
Executes data discovery (or synthetic generation), stratified train/test split, manual and AutoML preprocessing, and full EDA summary tables and plots:
```powershell
python scripts/run_phase1.py
```

### Validate Phase 1 Deliverables
Verifies that all 21 mandatory project specifications, data splits, EDA tables, plots, and documentation files exist and are non-empty:
```powershell
python scripts/validate_phase1.py
```

---

## 4. Directory Structure Explanation

```
predictive_maintenance_xai_automl/
│
├── data/
│   ├── raw/                  # Raw AI4I 2020 dataset or synthetic smoke-test CSV
│   ├── processed/            # Stratified train/test splits for manual & AutoML pipelines, and metadata.json
│   └── README.md             # Dataset documentation and column schema
│
├── notebooks/                # Exploratory and prototyping Jupyter notebooks
│   └── README.md
│
├── src/                      # Production modular source code
│   ├── __init__.py
│   ├── config.py             # Global constants, paths, seed=42, metric definitions
│   ├── utils.py              # Structured logging and random seed utilities
│   └── data/                 # Data engineering package
│       ├── __init__.py
│       ├── load_data.py      # Raw data ingestion, column canonicalization, metadata generation
│       ├── generate_synthetic.py # Physics-based synthetic data generator (~3.8% failure rate)
│       └── preprocess.py     # Stratified 80/20 split, median/mode imputation, one-hot encoding
│
├── scripts/                  # Executable operational scripts
│   ├── run_phase1.py         # End-to-end Phase 1 execution
│   └── validate_phase1.py    # Automated Phase 1 artifact validation checklist
│
├── models/                   # Serialized baseline, tuned, and AutoML model artifacts
│   └── README.md
│
├── results/                  # Generated CSV summary tables (dataset summary, missing values, class distributions)
│   └── README.md
│
├── plots/                    # High-resolution EDA visualization figures
│   └── README.md
│
├── app/                      # Streamlit graphical user interface and deployment modules
│   └── README.md
│
├── reports/                  # Academic report specifications, outlines, and presentation assets
│   ├── automl_report_outline.md # 22-section outline for AutoML benchmarking study
│   ├── xai_report_outline.md    # 21-section outline for Explainable AI study
│   └── screenshots/          # Embedded visualization and UI screenshot assets
│       └── README.md
│
├── slides/                   # Academic presentation slide decks
│   └── README.md
│
├── PROJECT_SPEC.md           # Locked architectural decisions, rubric mapping, and problem spec
├── PLAN.md                   # Complete 4-phase academic project roadmap
├── ASSUMPTIONS.md            # Comprehensive log of engineering assumptions and package status
├── DOWNLOAD_INSTRUCTIONS.md  # Official download links and placement guide for real AI4I dataset
├── requirements.txt          # Complete dependency requirements
├── .gitignore                # Git exclusion rules for environments and temporary artifacts
└── README.md                 # Primary project overview (this file)
```

---

## 5. Synthetic Data Warning
> [!WARNING]
> If `USE_REAL_DATA.md` is present in the project root, the pipeline is currently operating on **synthetic smoke-test data** generated by `src/data/generate_synthetic.py`.
> While the synthetic data mirrors the exact AI4I schema, physical relationships ($P = \tau \omega, \Delta T$), and ~3.8% failure imbalance, it is intended strictly for smoke-testing. Replace `data/raw/` with the official AI4I 2020 dataset as detailed in [`DOWNLOAD_INSTRUCTIONS.md`](file:///d:/AutoML_XAI/DOWNLOAD_INSTRUCTIONS.md) prior to final evaluation.

---

## 6. Model Card Placeholder
```yaml
Model Details:
  Model Name: "AutoML-Predictive-Maintenance-Ensemble"
  Model Version: "1.0.0-rc (Phase 1 Baseline)"
  Model Type: "Binary Classifier (Ensemble of Gradient Boosted Decision Trees & Tabular Stacks)"
  Frameworks: "Scikit-Learn, FLAML, AutoGluon-Tabular"
  License: "MIT Open Source Academic License"

Dataset & Inputs:
  Training Dataset: "AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)"
  Number of Instances: "10,000 telemetry records (8,000 Train / 2,000 Test)"
  Input Features:
    - type: Categorical [L, M, H]
    - air_temp: Continuous (Kelvin)
    - process_temp: Continuous (Kelvin)
    - rotational_speed: Continuous (rpm)
    - torque: Continuous (Nm)
    - tool_wear: Continuous (minutes)
  Target Variable: "machine_failure (0: Nominal, 1: Failure)"

Intended Use:
  Primary Application: "Industrial machine health prognosis and early warning anomaly detection."
  Intended Users: "Shop-floor maintenance technicians, reliability engineers, and plant operations managers."
  Out-of-Scope Uses: "Direct closed-loop autonomous machine shutdown without human-in-the-loop review."

Evaluation & Performance Metrics:
  Primary Evaluation Metric: "Macro F1-Score on held-out test partition"
  Secondary Metrics: "ROC-AUC, Precision, Recall, Overall Accuracy"

Limitations & Failure Modes:
  - Synthetic Data Limitations: AI4I contains synthetic distributions derived from physical twin models; live field telemetry may exhibit unmodeled vibrational harmonics.
  - Sensor Drift: Uncalibrated thermocouples or torque sensors can induce spurious drift in Shapley attributions.
  - Rare Failure Types: Random Network Failures (RNF) possess zero telemetry correlation and cannot be anticipated by sensor models.

Ethical Considerations:
  - Safety & Accountability: XAI outputs are assistive recommendations; final operational sign-offs remain with qualified manufacturing engineers.
  - Machine Type Equity: Verification that budget machine variants (Type L) are not disproportionately flagged with false alarms due to higher vibration tolerance thresholds.
```
