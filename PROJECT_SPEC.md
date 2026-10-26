# Project Specification: Explainable AutoML for Predictive Maintenance

## 1. Project Overview & Final Project Idea
- **Project Title:** Explainable AutoML System for Industrial Predictive Maintenance in Smart Manufacturing
- **Combined Academic Subjects:**
  1. **Automated Machine Learning (AutoML):** Systematic pipeline automation, hyperparameter search (Grid, Random, Optuna), framework benchmarking (FLAML vs. AutoGluon-Tabular), leakage evaluation, and time-budgeted optimization.
  2. **Explainable Artificial Intelligence (XAI):** Game-theoretic feature attributions (SHAP), local surrogate models (LIME), partial dependence profiling (PDP/ICE), counterfactual/what-if simulation, and multi-modal XAI demonstrations (image thermal crack Grad-CAM, text maintenance log token attributions).
- **Core Problem Statement:** Given continuous machine sensor telemetry readings (temperatures, spindle rotational velocities, torques, cumulative tool wear), predict whether the equipment is facing imminent failure ($0 = \text{No Failure}, 1 = \text{Failure}$), and provide factory floor operators with actionable, transparent root-cause explanations and counterfactual remedies.

---

## 2. Locked Project Decisions
| Specification Dimension | Locked Decision | Rationale & Constraint |
| :--- | :--- | :--- |
| **Domain** | Predictive Maintenance in Smart Manufacturing | High-stakes industrial setting where failure prediction and explanation are equally vital. |
| **Dataset** | AI4I 2020 Predictive Maintenance Dataset (UCI / Kaggle) | Real-world benchmark reflecting realistic manufacturing equipment telemetry and failure modes. |
| **Target Variable** | `machine_failure` ($0 =$ Nominal, $1 =$ Failure) | Standardized binary target column. Derived from `Target` or unified failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`). |
| **Primary Metric** | **F1-Score** | Critical for imbalanced failure detection (~3.4% positive rate) balancing precision and recall. |
| **Secondary Metrics** | **ROC-AUC, Precision, Recall, Accuracy** | Comprehensive threshold-independent and threshold-dependent assessment. |
| **Random Seed** | `RANDOM_SEED = 42` | Strictly locked across all splits, initializations, and search algorithms for reproducible science. |
| **Train-Test Split** | **80% Train, 20% Test (Stratified by Target)** | Stratification preserves exact class imbalance ratio without data leakage. |
| **AutoML Frameworks** | **FLAML & AutoGluon-Tabular** | Contrasts fast, lightweight budget-constrained optimization against multi-layer stacked ensembling. |
| **AutoML Time Budgets**| `AGENT = 60s`, `FINAL = 900s` (15 min) | Fast smoke-testing during development, rigorous exploration for formal benchmarking. |
| **Manual Baseline** | Scikit-learn (Dummy, Logistic Regression, Random Forest, HistGradientBoosting) | Solid reference point before tuning and AutoML comparisons. |
| **HPO Methods** | Grid Search, Random Search, Optuna (TPE) | Direct comparison of brute force, stochastic, and Bayesian optimization strategies. |
| **XAI Methods** | SHAP (Summary, Waterfall, Force), LIME, PDP, What-If counterfactuals | Comprehensive coverage across global, local, marginal, and actionable interpretability. |
| **Multi-Modal XAI** | Tabular (primary) + Image (Grad-CAM demo) + Text (Token highlights demo) | Satisfies academic curriculum requirement for multi-modal explanation capabilities. |
| **Deployment** | Streamlit Graphical User Interface | Operator-facing dashboard with telemetry simulator, risk gauges, and dynamic explanations. |

---

## 3. Dataset Schema & Canonical Features
The raw AI4I dataset contains 10,000 telemetry readings with 14 attributes. The canonical feature set processed by this system is:
- **`type`** *(Categorical: 'L' [60%], 'M' [30%], 'H' [10%])*: Machine variant representing Low, Medium, and High quality variants.
- **`air_temp`** *(Continuous, Kelvin)*: Ambient room air temperature (nominal range: 295 - 305 K).
- **`process_temp`** *(Continuous, Kelvin)*: Internal operating temperature of the machine process (nominal range: 305 - 315 K).
- **`rotational_speed`** *(Continuous, rpm)*: Rotational speed of the spindle shaft (nominal range: 1,100 - 2,900 rpm).
- **`torque`** *(Continuous, Nm)*: Spindle shaft torque (nominal range: 10 - 80 Nm).
- **`tool_wear`** *(Continuous, min)*: Cumulative tool wear time in minutes (nominal range: 0 - 250 min).
- **`machine_failure`** *(Binary Target: 0 or 1)*: 1 indicates machine breakdown due to tool wear (TWF), heat dissipation (HDF), power overload (PWF), overstrain (OSF), or random failure (RNF).

---

## 4. Academic Rubric Mapping

### A. AutoML Rubric Mapping
| Rubric Requirement | Project Implementation |
| :--- | :--- |
| **Real-world problem & $\ge$ few thousand rows** | AI4I 2020 dataset (10,000 rows) in industrial manufacturing. |
| **Manual baseline with scikit-learn** | Logistic Regression, Random Forest, and HistGradientBoosting baselines in Phase 2. |
| **Hyperparameter tuning** | Grid Search, Random Search, and Optuna Bayesian optimization with convergence curves. |
| **At least 2 AutoML frameworks** | FLAML (lightweight HPO) and AutoGluon-Tabular (multi-layer ensemble stacking). |
| **Fixed time budget benchmarking** | Both tools evaluated under exact same fixed wall-clock budget (60s agent / 900s final). |
| **Multi-dimensional comparison** | Leaderboard, candidate models tried, preprocessing, metrics, training time, and inference latency. |
| **Leakage & Bias analysis** | Injection of deliberately leaky feature to test AutoML robustness; group parity analysis across machine types (L/M/H). |
| **Model Card & Deployment** | Standardized Hugging Face style model card and interactive Streamlit web dashboard. |

### B. Explainable AI (XAI) Rubric Mapping
| Rubric Requirement | Project Implementation |
| :--- | :--- |
| **GUI-based application** | Modern interactive Streamlit dashboard featuring live telemetry sliders and explanation views. |
| **Clear XAI conceptual understanding** | Detailed mathematical and conceptual breakdown of Shapley values, local surrogates, and marginal distributions. |
| **Tabular explanations** | Global SHAP beeswarm, local SHAP waterfalls, LIME decision rules, and PDP/ICE response curves. |
| **Multi-modal XAI demos** | Image crack detection with Grad-CAM heatmap demo; technician log classification with token attribution demo. |
| **Actionable Decision Support** | Translation of Shapley attributions into concrete maintenance standard operating procedures (SOPs). |
| **Counterfactual What-If** | Minimal parameter perturbation simulator allowing operators to see how setting adjustments resolve failure risks. |

---

## 5. Report Titles
1. **AutoML Report Title:**  
   `AutoML Benchmarking for Predictive Maintenance Failure Prediction`
2. **Explainable AI Report Title:**  
   `Explainable AI Dashboard for Predictive Maintenance Decision Support`

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
  Baseline Expected F1: ">= 0.82 on held-out test set"

Limitations & Failure Modes:
  - Synthetic Data Limitations: AI4I contains synthetic distributions derived from physical twin models; live field telemetry may exhibit unmodeled vibrational harmonics.
  - Sensor Drift: Uncalibrated thermocouples or torque sensors can induce spurious drift in Shapley attributions.
  - Rare Failure Types: Random Network Failures (RNF) possess zero telemetry correlation and cannot be anticipated by sensor models.

Ethical Considerations:
  - Safety & Accountability: XAI outputs are assistive recommendations; final operational sign-offs remain with qualified manufacturing engineers.
  - Machine Type Equity: Verification that budget machine variants (Type L) are not disproportionately flagged with false alarms due to higher vibration tolerance thresholds.
```
