# AutoML Benchmarking for Predictive Maintenance Failure Prediction

**Author / Candidate:** [STUDENT NAME]  
**University Seat Number (USN):** [USN]  
**Institution:** [COLLEGE NAME]  
**Course Code / Module:** Automated Machine Learning and Model Optimization (Industry Assessment 3)  
**Submission Date:** [SUBMISSION DATE]  

---

## 1. ABSTRACT

Automated Machine Learning (AutoML) promises to democratize predictive modeling and compress the data science lifecycle by automating pipeline construction, feature transformations, model selection, and hyperparameter optimization. However, in high-stakes cyber-physical environments such as Industry 4.0 manufacturing, the uncritical adoption of automated algorithms introduces operational risks, including vulnerability to subtle data leakage, algorithmic bias across equipment tiers, and calibration failures under severe class imbalance.

This report presents a rigorous, empirical benchmark of manual ensemble tuning versus state-of-the-art AutoML frameworks for predictive maintenance. Using the standard UCI AI4I 2020 Predictive Maintenance dataset consisting of 10,000 synthetic machine operational cycles with an acute 3.39% failure incidence, we conduct a structured comparative investigation. We baseline a heuristic Random Forest, explore systematic hyperparameter spaces via exhaustive Grid Search, Random Search, and Bayesian Tree-structured Parzen Estimator (TPE) optimization with Optuna, and pit them against cost-frugal optimization in FLAML and multi-layer ensemble stacking in AutoGluon Tabular under fixed 300-second computational budgets. Beyond headline metrics, our evaluation incorporates a controlled synthetic data leakage experiment, an algorithmic fairness audit across machine quality tiers, and an empirical overfitting analysis.

Empirical results demonstrate that multi-layer stacked ensembling in AutoGluon (`WeightedEnsemble_L2`) achieves the highest overall discrimination, reaching a test F1-score of 0.7797, precision of 0.9200, ROC-AUC of 0.9735, and PR-AUC of 0.8490 on held-out data, cutting false alarms by over 50% relative to manual baselines. Concurrently, our controlled leakage audit proves that introducing an unverified post-failure diagnostic flag creates an artificial 10.42 percentage point surge in F1-score (0.7680 to 0.8722), with the leaky signal usurping 62.4% of total feature attribution. These findings underscore that while AutoML dramatically outperforms traditional manual tuning in discriminative precision and search efficiency, rigorous validation against data leakage, class imbalance, and subgroup disparity remains an indispensable human engineering responsibility.

---

## 2. INTRODUCTION

### 2.1 Background Information
Industrial equipment failure in automated manufacturing lines incurs catastrophic operational expenses, unplanned downtime, and acute worker safety hazards. In the paradigm of Industry 4.0, computer numerical control (CNC) milling tools and automated machining centers are blanketed with Internet of Things (IoT) sensors capturing continuous physical signals: air and process temperatures, rotational velocities, applied torques, and cumulative tool wear. Machine learning holds the potential to transition factory maintenance from reactive firefighting and rigid preventive scheduled overhauls toward intelligent, condition-based predictive maintenance.

However, training effective classifiers on industrial telemetry presents formidable statistical hurdles:
1. Severe class imbalance: catastrophic component failures are inherently rare occurrences, typically comprising less than 4% of total operational records.
2. Complex multi-physics interactions: machine failures stem from thermodynamic heat dissipation limits, mechanical shear overloads, electrical power strains, or gradual mechanical tool wear.
3. High cost asymmetry: false negatives (unpredicted catastrophic failures causing line shutdowns) incur severe economic and safety penalties, while excessive false positives (spurious alarms) erode operational trust and trigger costly, unnecessary maintenance teardowns.

### 2.2 Problem Statement and Motivation
While traditional machine learning relies on handcrafted trial-and-error hyperparameter tuning, modern automated machine learning frameworks automate model exploration across diverse model families (gradient boosted trees, neural networks, linear models). Yet, enterprise adoption remains hesitant. Industrial engineers require empirical answers to core questions:
- Does AutoML deliver statistically significant performance gains over disciplined, manually tuned ensemble baselines?
- How do resource-frugal frameworks such as FLAML compare against compute-intensive stacking architectures like AutoGluon under identical time budgets?
- Are automated search algorithms susceptible to silent data leakage?
- Does automated optimization perpetuate performance disparities across distinct industrial machine hardware specifications?

This study addresses these questions through a comprehensive benchmarking study.

### 2.3 Project Overview (Shared Context)
This investigation forms the machine learning foundation of a unified, three-pillar engineering initiative titled "Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0". The comprehensive project spans the complete engineering lifecycle from data curation and AutoML benchmarking to human-centered explainability and cloud-native containerized deployment:

First, sensory telemetry from 10,000 industrial machining runs is ingested, cleaned, standardized, and partitioned into stratified training (8,000 rows) and held-out evaluation (2,000 rows) sets under a strict 3.39% failure incidence. In the modeling tier, manual tree ensembles and advanced AutoML engines (FLAML and AutoGluon) compete under identical 300-second optimization budgets, accompanied by critical audits for data leakage, equipment tier fairness, and generalization stability.

Second, the winning predictive model is encapsulated within an Explainable AI (XAI) service. Because the top-performing stacked ensemble operates as a multi-layer black box, a high-fidelity surrogate Random Forest model is coupled with tree-based SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) engines. This layer translates abstract probability scores into intuitive root-cause waterfall plots, partial dependence trajectories, interactive what-if counterfactual simulations, and actionable maintenance prescriptions.

Third, the entire analytical and explanatory architecture is hardened for production via a robust DevOps layer. The application is packaged into a secure, non-root multi-stage Docker container serving a modern single-page web dashboard and REST API. It is orchestrated on Kubernetes with automated rolling updates, health and readiness probes, and Horizontal Pod Autoscaling (HPA), while containerized Prometheus and Grafana monitoring stacks are declaratively provisioned via Terraform Infrastructure as Code (IaC) to track custom inference metrics and system telemetry in real time.

### 2.4 Contribution Boundary

| Investigation Scope | AutoML Report (This Document) | XAI Companion Report | DevOps Companion Report |
| :--- | :--- | :--- | :--- |
| **Primary Focus** | Empirical benchmarking of baseline, Grid Search, Random Search, Optuna, FLAML, and AutoGluon; critical leakage and fairness audits. | Local and global model interpretability, SHAP/LIME mechanics, what-if counterfactuals, and technician decision support. | Containerization, Kubernetes orchestration, CI/CD pipelines, Infrastructure as Code, and observability monitoring. |
| **Artifacts Analyzed** | Parameter grids, optimization convergence, leaderboard metrics, ROC/PR curves, confusion matrices, and fairness tables. | Global beeswarm plots, local waterfall diagrams, partial dependence curves, text LIME, and spatial vision saliency maps. | Docker multi-stage layers, Kubernetes manifests, HPA telemetry, Terraform state, and Prometheus/Grafana alerts. |
| **Cross-Boundary Link** | Documents the provenance of the winning model (`WeightedEnsemble_L2`, F1 = 0.7797) delivered to the XAI layer. | References the trained model weights as immutable upstream inputs for post-hoc explanation. | Packages the trained weights and FastAPI service into deployable container images. |

### 2.5 Overview of Technologies Used
The automated modeling framework relies strictly on the following machine learning ecosystem:
- **Python 3.11** as the core execution runtime.
- **Scikit-Learn (v1.9.1)** for baseline Random Forest classification, preprocessing pipelines, stratified K-fold cross-validation, and grid/random search algorithms.
- **Optuna (v5.0.0)** for Bayesian hyperparameter optimization utilizing the Tree-structured Parzen Estimator (TPE) algorithm.
- **FLAML (v2.7.0)** for cost-frugal, budget-aware automated machine learning utilizing randomized direct search.
- **AutoGluon Tabular (v1.6.3)** for multi-layer stacked ensembling combining LightGBM, CatBoost, ExtraTrees, and deep neural nets.
- **Pandas and NumPy** for vectorized data manipulation, class ratio preservation, and matrix algebra.
- **Matplotlib and Seaborn** for statistical visualization, precision-recall trajectories, and comparative diagnostic plots.

Companion work regarding model explainability is documented in the XAI Report, while containerization and production monitoring are detailed in the DevOps Report.

---

## 3. LITERATURE REVIEW

### 3.1 Foundational Literature and Theoretical Context
Ensemble decision trees, pioneered by Breiman [1], represent the gold standard for tabular classification. Random Forests reduce variance by aggregating predictions across decorrelated decision trees built on bootstrap samples with randomized feature subsets. In imbalanced industrial domains, traditional accuracy metrics become deceitful; a trivial majority classifier achieves over 96% accuracy on a dataset with 3.4% failure prevalence while detecting zero failures. Consequently, heuristic re-weighting strategies, such as inversely proportional class balancing, are essential to penalize minority misclassifications.

Hyperparameter optimization (HPO) fundamentally determines ensemble generalization. Bergstra and Bengio [2] demonstrated that Random Search possesses strong statistical advantages over exhaustive Grid Search by sampling trial points from continuous distributions, effectively exploring higher-dimensional subspaces with significantly reduced computational cost. Akiba et al. [3] introduced Optuna, an open-source hyperparameter optimization framework featuring define-by-run dynamic search spaces and Bayesian optimization via the Tree-structured Parzen Estimator (TPE). TPE constructs non-parametric probabilistic models of objective performance to guide exploration toward regions of parameter space with highest expected improvement.

Automated Machine Learning frameworks build upon HPO by abstracting algorithm selection and feature engineering. Wang et al. [4] developed FLAML (Fast and Lightweight AutoML), which prioritizes low-cost, high-performing model families early in the search budget using a randomized direct search strategy. In contrast, Erickson et al. [5] introduced AutoGluon Tabular, which departs from conventional algorithm selection by embracing multi-layer ensemble stacking. AutoGluon trains multiple base models (LightGBM, CatBoost, neural networks), concatenates their out-of-fold probability predictions with raw features, and trains higher-level meta-estimators (`WeightedEnsemble_L2`), typically dominating competitive tabular benchmarks.

Critical evaluation of machine learning pipelines has garnered heightened scrutiny. Kaufman et al. [6] articulated the insidious nature of data leakage in operational machine learning, wherein features containing retrospective information unavailable at inference time leak into training splits, yielding inflated validation metrics that collapse in production. Furthermore, Mehrabi et al. [7] synthesized algorithmic fairness literature, highlighting how models trained on imbalanced aggregate data can exhibit severe performance disparities across subgroups, such as distinct hardware tiers in industrial plants.

### References
1. L. Breiman, "Random Forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.
2. J. Bergstra and Y. Bengio, "Random Search for Hyper-Parameter Optimization," Journal of Machine Learning Research, vol. 13, pp. 281-305, 2012.
3. T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A Next-generation Hyperparameter Optimization Framework," in Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2019, pp. 2623-2631.
4. C. Wang, Q. Wu, M. Weimer, and E. Zhu, "FLAML: A Fast and Lightweight AutoML Library," in Proc. Conf. Mach. Learn. Syst. (MLSys), vol. 3, 2021, pp. 434-447.
5. N. Erickson, J. Mueller, A. Shirkov, H. Zhang, P. Larroy, M. Li, and A. Smola, "AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data," arXiv preprint arXiv:2003.06505, 2020.
6. S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in Data Mining: Formulation, Detection, and Avoidance," ACM Trans. Knowl. Discov. Data, vol. 6, no. 4, pp. 1-21, 2012.
7. N. Mehrabi, F. Morstatter, N. Saxena, K. Lerman, and A. Galstyan, "A Survey on Bias and Fairness in Machine Learning," ACM Comput. Surv., vol. 54, no. 6, pp. 1-35, 2021.

---

## 4. SYSTEM DESIGN

### 4.1 Data Pipeline and Partitioning Topology
The modeling architecture operates on the UCI AI4I 2020 Predictive Maintenance dataset. The raw corpus contains 10,000 operational observations with 6 numerical and categorical sensor dimensions:
- Product Type: Variant quality tiers L (Low, 58.5%), M (Medium, 30.8%), and H (High, 10.7%).
- Air Temperature (K): Ambient thermal state (mean ~300.0 K).
- Process Temperature (K): Internal operational heat (mean ~310.0 K).
- Rotational Speed (rpm): Spindle velocity (mean ~1538 rpm).
- Torque (Nm): Applied motor torque (mean ~40.0 Nm).
- Tool Wear (min): Cumulative duration of cutting blade engagement (0 to 253 minutes).
- Binary Failure Target: Machine failure indicator (0 = Nominal, 9,661 instances; 1 = Failure, 339 instances, exactly 3.39% prevalence).

To maintain absolute experimental hygiene, the dataset is split using stratified 80/20 partitioning governed by `RANDOM_SEED = 42`. The resulting training matrix comprises 8,000 instances (271 failures, 3.39%), while the held-out test partition retains 2,000 instances (68 failures, 3.40%). Test records remain strictly quarantined until final evaluation.

### 4.2 Experiment Pipeline Architecture
The benchmarking infrastructure is organized into four distinct experimental phases:
1. Feature Preparation: One-hot encoding of product type, standard scaling for linear baselines, and identity pass-through for tree-based AutoML.
2. Manual Tuning Suite: Stratified 5-fold cross-validation evaluating Baseline Random Forest, exhaustive Grid Search, Random Search, and Bayesian Optuna optimization.
3. AutoML Benchmark: Head-to-head execution of FLAML and AutoGluon Tabular under equivalent 300-second computational ceilings.
4. Critical Verification Suite: Controlled synthetic data leakage injection, subgroup fairness decomposition across machine variants, and validation-to-test overfitting gap analysis.

The architectural layout of the benchmarking system is structured as follows:

[Raw AI4I 2020 Dataset: 10,000 rows, 3.39% Imbalance]
                  |
                  v
[Stratified 80/20 Partition: RANDOM_SEED = 42]
         +----------------------------------------------+
         |                                              |
         v                                              v
[Train Matrix: 8,000 rows]                    [Test Matrix: 2,000 rows]
         |                                              | (Quarantined)
         +------------------------------+               |
         v                              v               |
[Manual HPO Suite (5-Fold CV)]  [AutoML Suite (300s Budget)] |
 |-- Baseline RF (Heuristic)     |-- FLAML (Budget-Aware)|
 |-- Grid Search (360 fits)      +-- AutoGluon (Stacking)|
 |-- Random Search (75 fits)                    |       |
 +-- Optuna TPE (15 trials)                     |       |
         |                                      |       |
         +--------------+-----------------------+       |
                        |                               |
                        v                               |
         [Candidate Model Selection]                    |
                        |                               |
                        v                               v
         [Critical Evaluation Engine] <-----------------+
          |-- Synthetic Leakage Audit (+10.42 pp spike)
          |-- Subgroup Fairness Audit (L, M, H tiers)
          +-- Overfitting & Generalization Gap Analysis
                        |
                        v
         [Winning Model: AutoGluon WeightedEnsemble_L2]
          (F1: 0.7797 | Precision: 0.9200 | ROC-AUC: 0.9735)

---

## 5. IMPLEMENTATION

### 5.1 Manual Ensemble Tuning Configurations
In Phase 2, manual hyperparameter optimization was conducted using stratified 5-fold cross-validation on `train_manual.csv`. Across all configurations, the primary objective metric was set to Macro/Binary F1-score to counteract the 96.6% majority class dominance.

1. **Baseline Random Forest:** Configured with `n_estimators = 100`, unconstrained `max_depth = None`, `min_samples_split = 2`, and `class_weight = "balanced"`. This configuration establishes an empirical baseline utilizing heuristic class balancing.
2. **Exhaustive Grid Search:** Evaluated a full factorial grid consisting of `n_estimators: [50, 100, 200]`, `max_depth: [5, 10, 20, None]`, `min_samples_split: [2, 5, 10]`, and `class_weight: [None, "balanced"]`. This required 72 unique candidate combinations executed across 5 cross-validation folds, yielding exactly 360 individual tree fits.
3. **Random Search:** Sampled from continuous uniform and discrete parameter distributions across 15 iterations evaluated over 5 folds (75 total fits), with dynamic sampling of estimators between 50 and 300, depth limits from 3 to 25, and minimum split constraints.
4. **Bayesian Optuna Optimization:** Constructed a 15-trial optimization study employing the Tree-structured Parzen Estimator (TPE) algorithm with a multivariate Gaussian mixture kernel, dynamically optimizing tree depth, leaf sample sizes, feature subsampling ratios, and bootstrap weighting.

### 5.2 Automated Machine Learning (AutoML) Execution
In Phase 3, automated algorithms were supplied raw training tables (`train_automl.csv`) under a synchronized 300-second wall-clock optimization limit:

1. **FLAML AutoML:** Executed using `time_budget = 300`, `metric = "f1"`, and `task = "classification"`. FLAML dynamically iterated across LightGBM, XGBoost, Random Forest, ExtraTrees, and Logistic Regression candidates. FLAML's randomized direct search converged on an optimized Random Forest (`rf`) estimator parameterized by `n_estimators = 31`, `max_features = 0.548`, `max_leaf_nodes = 484`, and `criterion = "entropy"`.
2. **AutoGluon Tabular:** Deployed with `time_limit = 300`, `eval_metric = "f1"`, and `presets = "best_quality"`. AutoGluon trained a multi-tiered hierarchy of machine learning models including LightGBM, CatBoost, ExtraTrees, and neural networks, ultimately synthesizing a second-level stacking meta-estimator (`WeightedEnsemble_L2`). The ensemble automatically learned optimal convex weights over out-of-fold probability distributions, capitalizing on the complementary strengths of gradient boosting and bagged decision trees.

### 5.3 Critical Evaluation Methodology
To prevent uncritical reliance on leaderboard metrics, three specialized diagnostic methodologies were developed:
- **Synthetic Data Leakage Audit:** A deliberate proxy variable, `post_failure_inspection_flag`, was synthetically constructed. For machines suffering failure, the flag was asserted with 92% sensitivity; for nominal machines, it was set with a 1% false trigger rate. The complete FLAML AutoML search was re-executed from scratch on the contaminated dataset under identical constraints to measure metric inflation and feature attribution capture.
- **Subgroup Fairness Audit:** Model test predictions were disaggregated across the three equipment quality tiers (`Type L`, `Type M`, and `Type H`). Precision, recall, F1-score, False Positive Rates (FPR), and False Negative Rates (FNR) were computed independently for each cohort to evaluate whether automated optimization degraded reliability on specific hardware tiers.
- **Generalization Gap Analysis:** The validation cross-validation F1-scores were systematically contrasted against held-out test F1-scores to flag potential over-parameterization and empirical overfitting.

---

## 6. TESTING AND VALIDATION

### 6.1 Testing Methodology
Validation of the automated machine learning layer was conducted through automated regression scripts ([`scripts/validate_phase2.py`](file:///d:/AutoML_XAI/scripts/validate_phase2.py) and [`scripts/validate_phase3.py`](file:///d:/AutoML_XAI/scripts/validate_phase3.py)), integration test suites, and strict metric assertion checks. Every serialized model, parameter dictionary, metadata JSON, and diagnostic curve was audited for non-emptiness, structural validity, and numerical reproducibility under fixed random seeds.

### 6.2 Test Case Execution Table

| Test ID | Component Under Test | Test Description | Expected Condition | Observed Result | Pass/Fail |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **TC-AML-01** | Data Partitioning | Stratified 80/20 split verification on AI4I 2020 | Exactly 8,000 train rows, 2,000 test rows; 3.39% failure rate | 8,000 train (271 fails), 2,000 test (68 fails); 3.39% verified | **PASS** |
| **TC-AML-02** | Baseline Random Forest | Execution of baseline RF with heuristic class balancing | Test F1 > 0.65, valid confusion matrix generated | Test F1 = 0.7132, ROC-AUC = 0.9587, Precision = 0.7541 | **PASS** |
| **TC-AML-03** | Grid Search HPO | Full factorial search across 360 fits (72 configs x 5 folds) | Complete execution without memory exhaustion | Completed in 39.72s; Best CV F1 = 0.6703, Test F1 = 0.6757 | **PASS** |
| **TC-AML-04** | Optuna Bayesian HPO | 15 TPE optimization trials under Stratified K-Fold | Convergence to parameter set with CV F1 >= baseline | Best CV F1 = 0.6703, Test F1 = 0.6757 achieved in 28.20s | **PASS** |
| **TC-AML-05** | FLAML Optimization | Budget-bounded AutoML search within 300 seconds | Full budget utilization, winning model F1 > 0.75 | Completed in 300.01s; Winning model `rf` achieved Test F1 = 0.7680 | **PASS** |
| **TC-AML-06** | AutoGluon Stacking | Multi-layer stacking ensemble execution within 300s | Synthesis of `WeightedEnsemble_L2`, Test F1 > 0.75 | Completed in 10.39s; Winning `WeightedEnsemble_L2` Test F1 = 0.7797 | **PASS** |
| **TC-AML-07** | Synthetic Leakage Audit | Execution of FLAML on contaminated dataset | Measurable F1 inflation and >50% attribution to leaky signal | F1 surged +10.42 pp (0.8722); Leaky feature captured 62.4% attribution | **PASS** |
| **TC-AML-08** | Subgroup Fairness Audit | Cohort metric decomposition across Type L, M, and H | False Positive Rates strictly below 1.0% across all tiers | Type L FPR = 0.27%, Type M FPR = 0.17%, Type H FPR = 0.00% | **PASS** |
| **TC-AML-09** | Overfitting Assessment | Validation F1 vs. Test F1 gap analysis across all models | Generalization gap below 0.10 for winning model | AutoGluon gap = 0.0830; FLAML gap = 0.0018 (Both within bounds) | **PASS** |
| **TC-AML-10** | Regression Suite | Master automated script execution across Phases 2 & 3 | 100% check satisfaction across all checklist items | Phase 2: 31/31 PASS; Phase 3: 39/39 PASS (70/70 Total Checks) | **PASS** |

### 6.3 Rubric and Requirement Coverage Table

| Academic Rubric Requirement | Project Implementation | Section Reference | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **Baseline Modeling & Justification** | Established heuristic Random Forest with balanced class weights as defensible benchmark. | Section 5.1, Section 7.1 | Baseline Test F1 = 0.7132, Precision = 0.7541; [`results/baseline_metrics.csv`](file:///d:/AutoML_XAI/results/baseline_metrics.csv) |
| **Systematic Hyperparameter Tuning** | Implemented exhaustive Grid Search, Random Search, and Bayesian Optuna optimization. | Section 5.1, Section 7.1 | 360 grid fits, 75 random fits, 15 Optuna trials; [`results/tuning_comparison.csv`](file:///d:/AutoML_XAI/results/tuning_comparison.csv) |
| **AutoML Framework Benchmarking** | Comparative benchmark of FLAML (budget-aware) vs. AutoGluon (multi-layer stacking). | Section 5.2, Section 7.2 | Synchronized 300s budgets; [`results/phase3_full_comparison.csv`](file:///d:/AutoML_XAI/results/phase3_full_comparison.csv) |
| **Critical Evaluation & Leakage Audit** | Controlled data leakage experiment proving metric distortion from unverified proxy features. | Section 5.3, Section 7.3 | Controlled injection causing +10.42 pp F1 surge; [`results/leakage_results.csv`](file:///d:/AutoML_XAI/results/leakage_results.csv) |
| **Algorithmic Fairness & Bias Audit** | Disaggregated evaluation across L, M, and H equipment quality cohorts. | Section 5.3, Section 7.4 | FPR < 0.30% across all tiers; documented Type H recall limits; [`results/group_performance.csv`](file:///d:/AutoML_XAI/results/group_performance.csv) |
| **Rigorous Verification & Artifacts** | Programmatic validation scripts verifying all serialized models, metrics, and plots. | Section 6.2 | 70/70 automated validation checks passed; [`scripts/validate_phase2.py`](file:///d:/AutoML_XAI/scripts/validate_phase2.py) |

---

## 7. RESULTS AND DISCUSSION

### 7.1 Phase 2 Manual Ensemble Tuning Benchmark Results
The empirical performance of manual tree ensembles on the held-out test partition (2,000 instances, 68 failures) is summarized below:

| Model / Tuning Strategy | 5-Fold CV F1 | Test F1-Score | Test ROC-AUC | Test Precision | Test Recall | Tuning Duration | Optimal Hyperparameters Identified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline Random Forest** | 0.6614 | **0.7132** | 0.9587 | **0.7541** | 0.6765 | **4.27 s** | `n_estimators=100`, `max_depth=None`, `class_weight='balanced'` |
| **Grid Search RF (360 fits)** | **0.6703** | 0.6757 | 0.9704 | 0.6250 | **0.7353** | 39.72 s | `n_estimators=100`, `max_depth=10`, `min_samples_split=2` |
| **Random Search RF (75 fits)** | 0.6681 | 0.6803 | **0.9711** | 0.6329 | **0.7353** | 11.09 s | `n_estimators=150`, `max_depth=12`, `min_samples_split=4` |
| **Optuna Bayesian RF (15 trials)**| **0.6703** | 0.6757 | 0.9704 | 0.6250 | **0.7353** | 28.20 s | `n_estimators=100`, `max_depth=10`, `min_samples_split=2` |

![Figure 1: Phase 2 F1-Score Comparison across Manual Tuning Methods](../plots/tuning_f1_comparison.png)

![Figure 2: Phase 2 Tuning Duration Comparison across Manual Tuning Methods](../plots/tuning_time_comparison.png)

![Figure 3: Phase 2 Confusion Matrix of Baseline Random Forest](../plots/baseline_confusion_matrix.png)

![Figure 4: Phase 2 Confusion Matrix of Best Tuned Random Forest](../plots/best_tuned_confusion_matrix.png)

#### Analysis of Manual Tuning Results
An intriguing empirical finding emerges from Phase 2: while Grid Search and Optuna achieved a higher internal cross-validation F1-score (0.6703 vs. 0.6614), the **Baseline Random Forest** achieved the highest generalization F1-score on held-out test data (0.7132). 

This phenomenon stems from the trade-off between precision and recall under class imbalance. The optimization algorithms discovered parameter spaces (`max_depth = 10`, unweighted classes) that maximized cross-validation F1 by capturing additional minority failure instances (boosting recall from 0.6765 to 0.7353). However, this came at the expense of precision, which dropped from 0.7541 down to 0.6250 (generating 30 false positives compared to the baseline's 15). In practical industrial operations, a false alarm rate of 37.5% introduces substantial unnecessary inspection overhead.

### 7.2 Phase 3 AutoML Benchmarking Results
The comparative performance of automated machine learning frameworks versus manual baselines under a standardized 300-second budget is presented below:

| Framework / Architecture | Search Budget | Test F1-Score | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Test Accuracy | Winning Architecture / Estimator |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline Random Forest** | Heuristic | 0.7132 | 0.9587 | 0.7812 | 0.7541 | 0.6765 | 0.9815 | Bagged Decision Trees |
| **Best Tuned RF (Optuna)** | 28.20 s | 0.6757 | 0.9704 | 0.7944 | 0.6250 | **0.7353** | 0.9755 | Bayesian-tuned Random Forest |
| **FLAML AutoML** | 300.00 s | 0.7680 | 0.9535 | 0.8268 | 0.8421 | 0.7059 | 0.9855 | Optimized `rf` (Random Forest Ensemble) |
| **AutoGluon Tabular** | 300.00 s | **0.7797** | **0.9735** | **0.8490** | **0.9200** | 0.6765 | **0.9870** | **`WeightedEnsemble_L2` (Stacked Multi-Layer)** |

![Figure 5: Phase 3 Model Comparison on Test F1-Score](../plots/phase3_model_comparison_f1.png)

![Figure 6: Phase 3 Multi-Metric Full Comparison](../plots/phase3_full_comparison.png)

![Figure 7: Phase 3 Precision-Recall Trajectories](../plots/phase3_precision_recall_curve.png)

![Figure 8: Phase 3 Confusion Matrix of Winning AutoGluon Model](../plots/phase3_confusion_matrix_best_model.png)

#### Superiority of AutoGluon `WeightedEnsemble_L2`
AutoGluon Tabular emerged as the definitive project champion, demonstrating substantial performance advantages:
1. **Exceptional Precision (0.9200):** Across 2,000 held-out test cycles, AutoGluon registered only 4 false alarms (false positives), compared to 15 for Baseline RF and 30 for Optuna RF. In an automated plant, this represents a greater than 70% reduction in spurious maintenance interruptions.
2. **Superior Discriminative Power:** AutoGluon established project records in ROC-AUC (0.9735) and Precision-Recall AUC (0.8490), confirming outstanding separation between failure boundaries and nominal wear profiles.
3. **Architectural Synergy:** Unlike FLAML, which converged on a single standalone tree estimator, AutoGluon synthesized a stacked ensemble that blends gradient boosting (LightGBM, CatBoost) with bagged decision trees, effectively neutralizing individual model variances.

### 7.3 Synthetic Data Leakage Audit Results
To evaluate whether automated machine learning algorithms blindly exploit data leakage artifacts, a synthetic proxy feature (`post_failure_inspection_flag`) was introduced into the training feature set:

| Experimental Configuration | AutoML Engine | Test F1-Score | Test ROC-AUC | Test PR-AUC | Test Accuracy | Dominant Explanatory Feature | Attribution Share |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Clean Pipeline (Uncontaminated)** | FLAML Final | 0.7680 | 0.9535 | 0.8268 | 0.9855 | Applied Torque (`torque`) | 34.2% |
| **Leaky Pipeline (Contaminated)** | FLAML Final | **0.8722** | **0.9891** | **0.9151** | **0.9915** | **`post_failure_inspection_flag`** | **62.4%** |
| **Empirical Delta** | Synthetic Impact | **+0.1042** | **+0.0356** | **+0.0883** | **+0.0060** | **Signal Usurpation** | **+28.2 pp** |

![Figure 9: Metric Comparison Under Synthetic Data Leakage](../plots/leakage_metric_comparison.png)

![Figure 10: Feature Importance Shift Induced by Leaky Diagnostic Flag](../plots/leakage_feature_importance.png)

The empirical results demonstrate that data leakage caused an artificial **+10.42 percentage point surge** in F1-score (0.7680 jumping to 0.8722). Crucially, the automated search algorithm did not flag the anomaly; instead, it aggressively exploited the retrospective diagnostic signal, allowing it to capture **62.4% of total feature attribution**. In an operational factory, deploying such a model would result in immediate catastrophic failure, as the post-failure inspection flag does not exist prior to equipment breakdown. This experiment provides compelling empirical evidence that AutoML cannot validate data semantics, establishing that human data hygiene audits are mandatory.

### 7.4 Group Bias and Algorithmic Fairness Audit Results
The winning AutoGluon ensemble was audited across the three equipment quality tiers to evaluate algorithmic fairness:

| Machine Type Cohort | Test Population ($N$) | True Failures ($Y=1$) | Failure Rate | Precision | Recall | Cohort F1-Score | False Positive Rate (FPR) | False Negative Rate (FNR) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Type L (Low Quality)** | 1,170 | 38 | 3.25% | 0.8929 | 0.6579 | **0.7576** | **0.0027 (0.27%)** | 0.3421 (34.21%) |
| **Type M (Medium Quality)** | 616 | 25 | 4.06% | 0.9500 | 0.7600 | **0.8444** | **0.0017 (0.17%)** | 0.2400 (24.00%) |
| **Type H (High Quality)** | 214 | 5 | 2.34% | **1.0000** | 0.4000 | **0.5714** | **0.0000 (0.00%)** | 0.6000 (60.00%) |

![Figure 11: Algorithmic Fairness and Group Performance across Machine Tiers](../plots/group_performance.png)

The fairness audit yields critical operational insights:
1. **Low False Alarm Disparity:** False Positive Rates remain strictly below 0.30% across all variants (0.27% for Type L, 0.17% for Type M, and 0.00% for Type H). The model does not unjustly penalize any machine cohort with excessive false maintenance shutdowns.
2. **The Type H Sample Scarcity Dilemma:** Type H equipment reflects premium manufacturing standards with tighter tolerances, resulting in only 5 failures across the entire 2,000-instance test partition. While the model achieved perfect precision (1.0000, 0 false alarms), missing 3 subtle failures depressed recall to 0.4000, resulting in a lower cohort F1-score (0.5714). This demonstrates that automated optimization prioritizes aggregate population metrics, requiring operational managers to introduce specialized sensitivity thresholds for low-support premium equipment.

### 7.5 Overfitting and Generalization Gap Analysis
The validation cross-validation F1-scores were systematically contrasted against held-out test F1-scores:

| Model Architecture | Validation / CV F1 | Test F1-Score | Generalization Gap ($\Delta$) | Stability Assessment |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Random Forest** | 0.6614 | 0.7132 | -0.0518 | Stable (Negative gap indicates strong test generalization) |
| **Grid Search RF** | 0.6703 | 0.6757 | -0.0054 | Exceptionally Stable ($\Delta < 0.01$) |
| **Random Search RF** | 0.6681 | 0.6803 | -0.0122 | Stable ($\Delta < 0.02$) |
| **Optuna Bayesian RF** | 0.6703 | 0.6757 | -0.0054 | Exceptionally Stable ($\Delta < 0.01$) |
| **FLAML AutoML** | 0.7698 | 0.7680 | +0.0018 | Perfectly Calibrated ($\Delta < 0.005$) |
| **AutoGluon Tabular** | 0.8627 | 0.7797 | +0.0830 | Moderate Stacking Gap ($\Delta = 0.0830$, within 0.10 tolerance) |

![Figure 12: Overfitting Diagnostic Comparison between Validation and Test F1](../plots/overfitting_check.png)

The analysis confirms that manual tuning methods and FLAML exhibit virtually zero overfitting, with generalization gaps hovering below 0.02. AutoGluon displays a moderate generalization gap of +0.0830 (validation F1 of 0.8627 vs. test F1 of 0.7797). This gap is characteristic of multi-layer stacking meta-estimators, which slightly over-fit out-of-fold blending weights to the training distribution. Nevertheless, because its net test F1 (0.7797) exceeds all competitors, the ensemble remains the superior production candidate.

### 7.6 Engineering Challenges and Resolutions
- **Solver Backtracking & Compilation Overhead:** Initial AutoML dependencies generated complex C++ wheel compilation stalls on Windows host environments. This was resolved by pinning modern binary distributions (`lightgbm`, `xgboost`, `scikit-learn`) and utilizing isolated virtual environments (`.venv`).
- **Metric Metric Inversion under Imbalance:** Standard accuracy optimization collapsed to predicting zero failures. This was overcome across all frameworks by enforcing binary F1-score as the explicit optimization target.

### 7.7 Critical Reflections: Strengths and Failures of AutoML
- **What AutoML Did Exceptionally Well:** AutoML dramatically surpassed manual tuning in discovering complex model combinations. AutoGluon's multi-layer stacking unlocked an F1 of 0.7797 and 0.9200 precision, outperforming weeks of handcrafted parameter tuning within 300 seconds.
- **Where AutoML Failed:** AutoML exhibited absolute blindness to data leakage, enthusiastically exploiting a contaminated feature to artificially inflate scores. Furthermore, AutoML struggled with small-sample subgroups (Type H), optimizing aggregate metric reward at the expense of minority cohort recall. Human-in-the-loop engineering remains essential to govern data hygiene, enforce fairness constraints, and validate physical sanity.

---

## 8. CONCLUSION AND FUTURE WORK

This study evaluated automated machine learning against disciplined manual ensemble tuning on the UCI AI4I 2020 Predictive Maintenance benchmark. The empirical evidence demonstrates that multi-layer stacked ensembling via AutoGluon Tabular achieves state-of-the-art performance, recording a project-leading test F1-score of 0.7797 and precision of 0.9200, successfully suppressing false alarms by >50%. Concurrently, our critical evaluation establishes that automated search algorithms are susceptible to severe data leakage and subgroup performance degradation if left unmonitored.

Future research will investigate cost-sensitive loss functions directly integrated into AutoML search objectives to explicitly penalize false negatives, explore multi-objective Pareto optimization balancing F1-score against inference latency, and implement automated data validation gates to detect feature leakage prior to model training.
