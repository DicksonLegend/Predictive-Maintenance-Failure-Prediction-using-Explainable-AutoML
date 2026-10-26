# Explainable AI Dashboard for Predictive Maintenance Decision Support
## Academic Project Report Outline & Specification

**Author / Candidate:** Senior ML Engineering & Academic Project  
**Course Combination:** Explainable AI (XAI) & AutoML  
**Target Domain:** Smart Manufacturing & Predictive Maintenance  
**System Title:** Explainable Predictive Maintenance Decision Support System  
**Methods Evaluated:** SHAP (Kernel/Tree), LIME, Partial Dependence (PDP), Counterfactual / What-If Analysis, Grad-CAM (Image Demo), Integrated Gradients / LIME (Text Demo)  

---

### 1. Abstract
- Executive summary of the Explainable AI (XAI) investigation: bridging the semantic gap between black-box failure forecasting models and domain shop-floor operators.
- Highlights: multi-modal XAI capabilities across tabular telemetry, thermal machine inspection images, and technician incident logs, coupled with interactive Streamlit decision support.

### 2. Introduction to Explainable AI
- Theoretical foundations of interpretability versus explainability in artificial intelligence.
- Taxonomy of XAI: Intrinsic vs. Post-hoc, Model-Agnostic vs. Model-Specific, Global vs. Local, Perturbation vs. Attribution methods.
- Regulatory and industrial standards requiring explainability (e.g., ISO safety guidelines, EU AI Act conformity for high-impact industrial machinery).

### 3. Need for Explainability in Predictive Maintenance
- High-stakes cost of false alarms (unnecessary production stoppage) vs. false negatives (catastrophic tool fracture, spindle seizure).
- Why manufacturing engineers and plant supervisors cannot rely on opaque risk scores: the necessity of actionable root-cause diagnoses to schedule targeted mechanical repairs.

### 4. Case Study Description
- Operational environment: CNC milling and turning machines subjected to variable cutting loads, spindle velocity cycles, and ambient climate shifts.
- Diagnostic objectives: distinguish between heat dissipation failure (HDF), power overload (PWF), mechanical overstrain (OSF), and cumulative tool wear failure (TWF).

### 5. Dataset Description
- Analysis of the AI4I 2020 Predictive Maintenance Dataset schema:
  - Input telemetry: `type` (L, M, H), `air_temp` ($K$), `process_temp` ($K$), `rotational_speed` ($rpm$), `torque` ($Nm$), `tool_wear` ($min$).
  - Target: `machine_failure` ($0 =$ Nominal, $1 =$ Impending Failure).
  - Physics-based relationships: mechanical power ($P = \tau \cdot \omega$), thermal dissipation gradient ($\Delta T = T_{process} - T_{air}$).

### 6. Explainability Methods Used
- Theoretical principles and mathematical formulations of chosen methods:
  - **SHAP (Shapley Additive Explanations):** Cooperative game-theoretic Shapley value attribution with local accuracy, missingness, and consistency properties.
  - **LIME (Local Interpretable Model-agnostic Explanations):** Local surrogate linear models fit over perturbed neighborhood samples.
  - **Partial Dependence Plots (PDP) & Individual Conditional Expectation (ICE):** Marginal effect of continuous variables on predicted failure probability.
  - **Counterfactual / What-If Analysis:** Minimal perturbation required to reverse a predicted failure state back into nominal operating thresholds.

### 7. Global Explanation using SHAP
- Fleet-level insights into failure drivers:
  - Global Feature Importance (Mean Absolute Shapley values $|\phi_i|$).
  - Summary Beeswarm plots highlighting directional impact: how high torque and high tool wear systematically propel risk upwards.
  - SHAP interaction value analysis (e.g., non-linear compounding effects between rotational speed and torque).

### 8. Local Explanation using SHAP
- Single-machine incident drilldowns for operators:
  - SHAP Waterfall plots decomposing individual risk scores: baseline expectation $E[f(X)]$ plus additive feature contributions $\sum \phi_i$.
  - SHAP Force plots showing positive (pushing towards failure) and negative (stabilizing) forces on active machines.

### 9. Local Explanation using LIME
- Independent validation of local attributions using sparse linear surrogate models:
  - LIME explanation bar charts showing feature rule weights (e.g., `tool_wear > 195 min` adding +0.42 to failure probability).
  - Comparison of LIME vs. SHAP local ranking alignment and fidelity score.

### 10. Partial Dependence Analysis
- 1D and 2D Partial Dependence Plots:
  - Effect of `tool_wear` on failure probability across [0, 250] minutes (identifying the wear threshold inflection point).
  - Joint 2D PDP interaction between `rotational_speed` and `torque` capturing the operational envelope of the electric drive motor.

### 11. What-If Explanation
- Interactive counterfactual exploration:
  - "If we decrease spindle rotational speed by 250 rpm and reduce torque by 12 Nm, does the machine return to a safe operational envelope?"
  - Automated calculation of minimum intervention vectors to avoid unscheduled downtime.

### 12. Maintenance Recommendation System
- Translation of XAI attributions into standardized maintenance actions:
  - If `tool_wear` is the dominant positive contributor $\rightarrow$ "Schedule immediate cutting tool insert replacement (Standard Work Order #TW-101)."
  - If `process_temp - air_temp` thermal margin failure $\rightarrow$ "Flush coolant circulation system and inspect heat exchanger fins."
  - If `torque` & `rotational_speed` power anomaly $\rightarrow$ "Recalibrate spindle drive gearbox and check load sensor."

### 13. Leakage Detection using SHAP
- Using SHAP as an adversarial diagnostic tool:
  - How an injected target-correlated feature dominates Shapley values, exposing synthetic data artifacts or target leakage before production deployment.

### 14. Bias and Fairness Explanation
- Evaluating explanation equity across equipment tiers:
  - Comparing feature importance distributions and attribution consistency between Low (L), Medium (M), and High (H) quality variants.
  - Ensuring model decisions do not unfairly penalize budget machine variants without physical basis.

### 15. Image Explainability Demo
- Multi-modal demonstration of XAI for industrial vision:
  - Thermal camera / surface crack inspection imagery demo.
  - Saliency mapping using Grad-CAM / Integrated Gradients to localize thermal hot-spots or surface micro-fractures on cutting bits.

### 16. Text Explainability Demo
- Multi-modal demonstration of XAI for maintenance logs:
  - Technician shift handover notes and work order log sentiment/failure categorization.
  - LIME / Attention-weight token highlights demonstrating which phrases (e.g., "excessive vibration", "oil burnt odor") triggered an automated maintenance ticket.

### 17. GUI Application Design
- Architecture and user experience flow of the Streamlit Explainable Dashboard:
  - Tab 1: Machine Telemetry Overview & Real-Time Risk Monitor.
  - Tab 2: Global Fleet XAI (SHAP Beeswarm, PDP curves, feature rankings).
  - Tab 3: Local Diagnostic Workbench (SHAP waterfall, LIME explanations, What-If simulator).
  - Tab 4: Multi-Modal XAI Demos (Image thermal crack Grad-CAM & Text log highlights).

### 18. Limitations
- Computational cost of exact Shapley value computation for complex ensembles.
- Explanation fidelity vs. human cognitive load on factory floors.
- Potential misinterpretation of correlation as causation in post-hoc attributions.

### 19. Conclusion
- Summary of XAI utility in industrial decision support: boosting operator trust, expediting root-cause isolation, and preventing catastrophic equipment breakdowns.

### 20. References
- Core references in explainable artificial intelligence (Lundberg & Lee, Ribeiro et al., Molnar), predictive maintenance standards, and industrial human-AI interaction.

### 21. Appendix: Screenshots
- Interactive GUI views, multi-modal explainability visualizations, SHAP waterfall plots, and dashboard interface captures.
