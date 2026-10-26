# Phase 1: Recorded Assumptions & Operational Log

This document records all engineering decisions, architectural assumptions, and operational constraints adopted during **Phase 1: Project Foundation, Specification, Data Pipeline, and EDA**.

---

## 1. Environment & Package Management Assumptions
- **Virtual Environment (`.venv`):** In accordance with user feedback and software engineering best practices, a dedicated virtual environment `.venv` was initialized in the project root (`d:\AutoML_XAI\.venv`) using Python 3.11.9.
- **Lightweight Package Installation:** For Phase 1 execution, only the core foundational packages were installed in `.venv`:
  - `pandas` (v3.1.0)
  - `numpy` (v2.4.6)
  - `scikit-learn` (v1.9.1)
  - `matplotlib` (v3.11.2)
  - `seaborn` (v0.13.2)
- **Heavy AutoML & XAI Packages:** Packages listed in `requirements.txt` such as `autogluon.tabular`, `flaml`, `optuna`, `shap`, `lime`, and `streamlit` are documented for later phases. To ensure non-blocking, rapid execution of Phase 1 foundation tasks, their installation is scheduled for Phase 2, 3, and 4 when those specific capabilities are exercised.

---

## 2. Dataset & Target Definition Assumptions
- **Canonical Target Column (`machine_failure`):**
  - Binary target variable where $0 =$ Nominal Machine Operation, and $1 =$ Machine Failure.
  - In real AI4I datasets where a column named `Target` or `Machine failure` is provided, that value is directly mapped.
  - If sub-failure flags are present (`TWF` - Tool Wear Failure, `HDF` - Heat Dissipation Failure, `PWF` - Power Failure, `OSF` - Overstrain Failure, `RNF` - Random Network Failure), `machine_failure` is set to $1$ if $\sum(\text{failure modes}) > 0$, else $0$.
- **Identifier Features:** Columns such as `UDI` (Unique Device Identifier) and `Product ID` are dropped from the active modeling matrix to prevent spurious index memorization and artificial leakage.
- **Continuous Features:**
  - `air_temp` ($K$): Ambient air temperature.
  - `process_temp` ($K$): Operating process temperature.
  - `rotational_speed` ($rpm$): Spindle angular velocity.
  - `torque` ($Nm$): Spindle resistance torque.
  - `tool_wear` ($min$): Cumulative milling operation duration.
- **Categorical Feature:**
  - `type`: Machine variant ('L' = Low quality, 'M' = Medium quality, 'H' = High quality).

---

## 3. Data Imbalance & Splitting Assumptions
- **No Resampling in Phase 1:** Severe class imbalance (~3.4% failure prevalence) is intrinsic to industrial predictive maintenance. In Phase 1, no oversampling (e.g., SMOTE, ADASYN) or undersampling is applied. The raw distribution is preserved to ensure that exploratory data analysis and baseline metrics reflect true operational conditions.
- **Stratified Splitting:** Train/test splitting utilizes an $80\% / 20\%$ stratified split (`test_size=0.2`, `random_state=42`) using `machine_failure` as the stratification key. This guarantees identical failure representation across both training and held-out test partitions.

---

## 4. Preprocessing & Leakage Prevention Assumptions
- **Leakage-Free Imputation:**
  - In `src/data/preprocess.py`, missing numeric values are imputed using the **median of the training split**, and missing categorical values are imputed using the **mode of the training split**. The fitted statistics from the training set are subsequently applied to the test set, preventing test-set information leakage.
- **Dual Representation for Modeling Paradigms:**
  - **Manual ML (`train_manual.csv`, `test_manual.csv`):** Imputed values and one-hot encoded `type` features (`type_H`, `type_L`, `type_M`).
  - **AutoML (`train_automl.csv`, `test_automl.csv`):** Preserves native continuous columns and string categorical `type`, permitting AutoML frameworks (FLAML, AutoGluon) to employ their proprietary categorical encoders and neural/tree embeddings.

---

## 5. Synthetic Data Generation Assumptions
- When the real AI4I 2020 dataset is not pre-populated in `data/raw/`, `src/data/generate_synthetic.py` generates a 10,000-row synthetic smoke-test dataset.
- Synthetic generation is governed by physical relationships:
  - Mechanical power equation: $P = \tau \cdot \left(\frac{2\pi \cdot \omega}{60}\right)$.
  - Thermal gradient dissipation: $\Delta T = T_{\text{process}} - T_{\text{air}}$.
  - Overstrain failure: $\text{OSF} \sim \text{torque} \times \text{tool\_wear}$.
- The resulting synthetic failure rate is calibrated to $\sim 3.8\%$, providing realistic class imbalance for testing data pipelines and baseline scripts prior to real data replacement.
- An explicit warning file (`USE_REAL_DATA.md`) and metadata flag (`dataset_source: "synthetic"`) are automatically emitted when synthetic data is active.
