MASTER PROMPT — PHASE 4: EXPLAINABLE AI DASHBOARD WITH FASTAPI AND REACT

You are a senior machine learning engineer, Explainable AI engineer, and full-stack developer. You are continuing an already locked university IA project that combines:

1. AutoML
2. Explainable AI (XAI)
3. Future DevOps deployment

Project title:
Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0

AutoML report title:
AutoML Benchmarking for Predictive Maintenance Failure Prediction

Explainable AI report title:
Explainable AI Dashboard for Predictive Maintenance Decision Support

Phase 1, Phase 2, and Phase 3 are completed.

Phase 1 created:
- dataset loading
- preprocessing
- train/test splits
- EDA artifacts

Phase 2 created:
- manual Random Forest baseline
- Grid Search
- Random Search
- Optuna
- saved tuned models
- Phase 2 comparison tables

Phase 3 created:
- FLAML AutoML results
- AutoGluon AutoML results
- leakage experiment
- overfitting check
- class imbalance analysis
- group bias analysis
- saved AutoML models
- Phase 3 comparison tables

Your current task is Phase 4 only.

PHASE 4 OBJECTIVE:
Build an Explainable AI dashboard using:

1. FastAPI backend
2. React frontend
3. Saved ML model
4. Saved XAI artifacts
5. Local prediction and explanation endpoints
6. Model card
7. Image XAI demo
8. Text XAI demo
9. What-if explanation
10. Maintenance recommendation engine

This phase must produce a working local web application that can later be containerized and deployed in Phase 5 DevOps.

Do not ask questions. Make reasonable engineering decisions, document every decision, and complete Phase 4 fully. If a tool fails, use the specified fallback rules. Do not fabricate outputs. Do not skip failed steps silently. Log every issue.

==================================================
LOCKED PROJECT CONSTRAINTS — DO NOT CHANGE
==================================================

Dataset:
AI4I 2020 Predictive Maintenance Dataset.

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
ROC-AUC, PR-AUC, precision, recall, accuracy

Risk thresholds:
Low risk: probability < 0.30
Medium risk: 0.30 <= probability <= 0.60
High risk: probability > 0.60

Use existing artifacts:
- data/processed/train_automl.csv
- data/processed/test_automl.csv
- data/processed/train_manual.csv
- data/processed/test_manual.csv
- data/processed/metadata.json
- results/phase2_best_model.json
- results/phase3_best_model.json
- results/phase3_test_predictions.csv
- models/
- plots/
- results/

Do not rerun Phase 1, Phase 2, or Phase 3 unless a required artifact is missing.
Do not modify Phase 1, Phase 2, or Phase 3 result files.
Do not start DevOps Phase 5.
Do not create Docker, Kubernetes, GitHub Actions, Terraform, or cloud deployment in this phase.
However, structure the app so that DevOps Phase 5 can be added easily later.

==================================================
PHASE 4 RUBRIC REQUIREMENTS TO SATISFY
==================================================

This phase must satisfy the Explainable AI rubric and the AutoML deployment requirement.

Explainable AI requirements:
- GUI-based application
- Clear understanding of Explainable AI concepts
- Apply explanation methods to tabular data
- Demonstrate image explanation at a small level
- Demonstrate text explanation at a small level
- Visualization and interpretation
- Innovation
- Documentation

AutoML deployment requirements:
- Export the best model
- Serve it through FastAPI or Streamlit
- Provide a working prediction demo
- Provide a short model card covering:
  - data used
  - metrics
  - intended use
  - known limits

This phase will use FastAPI and React.

==================================================
APPLICATION ARCHITECTURE
==================================================

Create this architecture:

predictive_maintenance_xai_automl/
│
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── model_service.py
│   │   ├── xai_service.py
│   │   ├── text_demo_service.py
│   │   ├── image_demo_service.py
│   │   └── recommendation_service.py
│   │
│   ├── frontend/
│   │   ├── package.json
│   │   ├── vite.config.js
│   │   ├── index.html
│   │   └── src/
│   │       ├── main.jsx
│   │       ├── App.jsx
│   │       ├── api.js
│   │       ├── styles.css
│   │       └── components/
│   │           ├── Layout.jsx
│   │           ├── MetricCard.jsx
│   │           ├── PredictionForm.jsx
│   │           ├── ExplanationPanel.jsx
│   │           ├── PlotImage.jsx
│   │           └── RiskBadge.jsx
│   │
│   └── static_fallback/
│       └── index.html
│
├── src/
│   ├── xai/
│   │   ├── __init__.py
│   │   ├── tabular_explainer.py
│   │   ├── text_explainer.py
│   │   ├── image_explainer.py
│   │   └── what_if.py
│
├── artifacts/
│   ├── xai/
│   │   ├── global/
│   │   ├── local/
│   │   ├── text_demo/
│   │   ├── image_demo/
│   │   └── samples/
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
├── scripts/
│   ├── generate_xai_artifacts.py
│   ├── build_frontend.py
│   ├── run_phase4.py
│   └── validate_phase4.py
│
├── reports/
│   ├── xai_phase4_results.md
│   ├── automl_phase4_deployment.md
│   ├── model_card.md
│   └── screenshots/
│
└── README.md

==================================================
REQUIRED PACKAGE INSTALLATION
==================================================

Use the existing virtual environment if present.

Install required Python packages:

.venv\Scripts\python -m pip install fastapi uvicorn pydantic scikit-learn pandas numpy matplotlib seaborn shap lime joblib

If Linux/macOS:

.venv/bin/python -m pip install fastapi uvicorn pydantic scikit-learn pandas numpy matplotlib seaborn shap lime joblib

Do not uninstall previous packages.

Optional image explanation:
Attempt to use torch and torchvision only if they are already installed or if installation is fast and stable.
Do not block Phase 4 if torch cannot be installed.
If torch is unavailable, use the specified image demo fallback.

Frontend:
Use Node.js and npm if available.
If npm is unavailable, create a static HTML/JavaScript fallback frontend and log this clearly.

==================================================
FAST MODE AND FINAL MODE
==================================================

Create scripts/run_phase4.py with command-line argument:

--mode fast
--mode final

Default mode:
fast

Fast mode settings:
- SHAP background sample size = 50
- Number of local explanation samples = 3
- LIME explanations limited to 3 samples
- Text demo dataset size small
- Image demo fallback allowed
- Do not perform heavy image model download

Final mode settings:
- SHAP background sample size = 200
- Number of local explanation samples = 5
- LIME explanations limited to 5 samples
- More detailed text demo examples
- Attempt real Grad-CAM image demo only if torch and sample images are available

The agent should automatically run fast mode.
Do not automatically run final mode unless the environment is clearly safe and fast enough.
At the end, provide the exact final-mode command.

==================================================
MODEL SELECTION FOR SERVING
==================================================

Create app/api/model_service.py.

Select the served model using this priority:

1. Read results/phase3_best_model.json.
2. If the Phase 3 best model loads successfully and supports predict_proba, use it.
3. If it fails, read results/phase2_best_model.json and use the Phase 2 best model.
4. If that also fails, train a fallback RandomForestClassifier on data/processed/train_manual.csv with random_state=42 and class_weight="balanced".

Save the selected model information in:

results/phase4_metadata.json

Include:
- served_model_source
- served_model_path
- served_model_type
- explanation_model_type
- whether a surrogate explanation model is used
- selected test metrics if available
- reason for selection

If the served model is not easily explainable, create a surrogate explanation model.

Surrogate explanation model rules:
- Use RandomForestClassifier trained on train_manual.csv.
- Use class_weight="balanced".
- Use random_state=42.
- Clearly mark in metadata and UI that the explanation is from a surrogate model if this happens.

Do not silently explain a different model without disclosure.

==================================================
INPUT SCHEMA AND PREDICTION API
==================================================

Create app/api/schemas.py.

Define a Pydantic schema:

SensorInput:
- type: string, must be one of L, M, H
- air_temp: float
- process_temp: float
- rotational_speed: float
- torque: float
- tool_wear: float

Add basic validation:
- numeric fields must be finite
- type must be L, M, or H
- values should be within reasonable industrial sensor ranges, but do not reject uncommon values too aggressively

Create POST /predict endpoint.

It must return:

{
  "input": {...},
  "prediction": {
    "failure_probability": float,
    "predicted_class": 0 or 1,
    "risk_level": "Low" or "Medium" or "High",
    "threshold": 0.5
  },
  "model": {
    "name": string,
    "source": string
  }
}

Use threshold 0.5 for predicted_class.
Use risk thresholds:
- Low: < 0.30
- Medium: 0.30 to 0.60
- High: > 0.60

==================================================
EXPLANATION API
==================================================

Create app/api/xai_service.py.

Create POST /explain endpoint.

It accepts SensorInput and returns:

{
  "input": {...},
  "prediction": {...},
  "top_reasons": [
    {"feature": "torque", "impact": "increases risk"},
    {"feature": "tool_wear", "impact": "increases risk"}
  ],
  "recommendation": string,
  "local_shap_plot": "/artifacts/xai/local/shap_local_dynamic.png" or base64,
  "lime_explanation_html": string or null,
  "explanation_method": "shap_direct" or "shap_surrogate" or "lime_surrogate"
}

For dynamic explanations:
- If computation is fast, compute SHAP values for the input.
- If computation is too slow, use the closest precomputed sample explanation and clearly say:
  "Explanation approximated using nearest precomputed case."

Do not block the API for more than a few seconds per request.

==================================================
GLOBAL EXPLANATION ARTIFACTS
==================================================

Create scripts/generate_xai_artifacts.py.

Generate and save global explanation artifacts under artifacts/xai/global/.

Required global artifacts:

1. SHAP summary plot:
artifacts/xai/global/shap_summary.png

2. SHAP bar plot:
artifacts/xai/global/shap_bar.png

3. Feature importance plot:
artifacts/xai/global/feature_importance.png

4. Partial dependence plots:
artifacts/xai/global/pdp_torque.png
artifacts/xai/global/pdp_tool_wear.png
artifacts/xai/global/pdp_rotational_speed.png
artifacts/xai/global/pdp_process_temp.png

5. Global explanation JSON:
artifacts/xai/global/global_explanation.json

The JSON must include:
- top global features
- short interpretation for each top feature
- explanation model used
- whether surrogate model was used

Use matplotlib Agg backend.

==================================================
LOCAL EXPLANATION SAMPLES
==================================================

Select sample test cases from results/phase3_test_predictions.csv and data/processed/test_automl.csv.

Choose:
- one high-risk sample
- one medium-risk sample
- one low-risk sample

In final mode, choose five samples if possible.

For each sample, create:

artifacts/xai/samples/sample_0.json
artifacts/xai/samples/sample_1.json
artifacts/xai/samples/sample_2.json

Each JSON must contain:
- sample_id
- input features
- true_label
- predicted_probability
- risk_level
- top_reasons
- recommendation
- local_shap_plot_path
- lime_explanation_path if available

Generate local SHAP plots:

artifacts/xai/local/shap_local_sample_0.png
artifacts/xai/local/shap_local_sample_1.png
artifacts/xai/local/shap_local_sample_2.png

Generate LIME explanations:

artifacts/xai/local/lime_local_sample_0.png
artifacts/xai/local/lime_local_sample_1.png
artifacts/xai/local/lime_local_sample_2.png

If LIME fails for any sample, log the issue and continue.

Create GET /samples endpoint returning these sample cases.

==================================================
WHAT-IF EXPLANATION
==================================================

Create src/xai/what_if.py.

Create POST /what-if endpoint.

It accepts:

{
  "original_input": SensorInput,
  "modified_input": SensorInput
}

It returns:

{
  "original_probability": float,
  "modified_probability": float,
  "probability_change": float,
  "original_risk_level": string,
  "modified_risk_level": string,
  "interpretation": string,
  "recommendation": string
}

Example interpretation:

"Reducing torque from 70 Nm to 45 Nm decreases failure probability from 0.84 to 0.31."

Do not require SHAP for what-if if it is too slow.
Use the served model directly for probability changes.

==================================================
MAINTENANCE RECOMMENDATION ENGINE
==================================================

Create app/api/recommendation_service.py.

Generate simple rule-based maintenance recommendations.

Rules:

If torque is among top risk contributors:
- Reduce mechanical load
- Check torque limits
- Inspect drive system

If tool wear is among top risk contributors:
- Replace tool immediately
- Schedule maintenance
- Check tool usage history

If process temperature or air temperature is among top risk contributors:
- Check cooling system
- Inspect ventilation
- Monitor temperature trend

If rotational speed is among top risk contributors:
- Check motor alignment
- Inspect bearing condition
- Verify load balance

If no strong contributor is found:
- Continue monitoring
- Schedule routine inspection

Save recommendation logic in a readable function.
Do not use an external LLM for this unless already available and stable.

==================================================
TEXT EXPLAINABILITY DEMO
==================================================

Create src/xai/text_explainer.py.

Build a small text explanation demo.

Create a small synthetic maintenance-note dataset if no real dataset is available.

Save it to:

data/text/maintenance_notes.csv

Columns:
- text
- label

Labels:
- urgent
- normal

Create at least 50 short maintenance notes.

Examples:

"Machine making loud noise and vibrating heavily."
"Tool wear reached maximum limit, urgent inspection needed."
"Temperature normal, no visible issue."
"Rotational speed unstable under high load."
"All parameters within normal operating range."
"Cooling system failure suspected due to high process temperature."

Train a simple model:
- TF-IDF vectorizer
- Logistic Regression

Use random_state=42.

Save:
models/text_demo_tfidf.pkl
models/text_demo_logreg.pkl

Create POST /text-explain endpoint.

It accepts:

{
  "text": string
}

It returns:

{
  "text": string,
  "predicted_label": "urgent" or "normal",
  "probability": float,
  "important_words": [
    {"word": "vibrating", "impact": "urgent"},
    {"word": "noise", "impact": "urgent"}
  ],
  "lime_html": string
}

Use LIME text explainer.

Save example outputs:

artifacts/xai/text_demo/text_sample_0.json
artifacts/xai/text_demo/text_sample_1.json
artifacts/xai/text_demo/text_sample_2.png or html file

The React frontend must allow the user to type maintenance text and see the explanation.

==================================================
IMAGE EXPLAINABILITY DEMO
==================================================

Create src/xai/image_explainer.py.

The image demo must demonstrate image explainability, but it must not block Phase 4.

Preferred method:
If torch and torchvision are available and stable:
- use a pretrained MobileNetV2 or ResNet18
- use Grad-CAM on two or three sample images
- save original image, heatmap, and overlay

Save:
artifacts/xai/image_demo/image_0_original.png
artifacts/xai/image_demo/image_0_heatmap.png
artifacts/xai/image_demo/image_0_overlay.png
artifacts/xai/image_demo/image_0_metadata.json

Fallback method:
If torch is unavailable or image model download fails:
- create a lightweight illustrative image explanation demo
- use synthetic machine-part-like images generated with matplotlib or OpenCV
- create saliency-style heatmap overlays
- clearly mark in metadata:
  "image_demo_mode": "illustrative_fallback"
- explain in the report that Grad-CAM is the recommended method and the fallback is for demonstration only

Create GET /image-demo endpoint.

It returns:

{
  "image_demo_mode": "gradcam" or "illustrative_fallback",
  "items": [
    {
      "title": string,
      "original": "/artifacts/xai/image_demo/image_0_original.png",
      "heatmap": "/artifacts/xai/image_demo/image_0_heatmap.png",
      "overlay": "/artifacts/xai/image_demo/image_0_overlay.png",
      "predicted_class": string,
      "explanation": string
    }
  ]
}

Do not make the image demo depend on external internet access during API runtime.

==================================================
FASTAPI APPLICATION
==================================================

Create app/api/main.py.

The FastAPI app must include these endpoints:

GET /
GET /health
GET /version
GET /model-card
GET /model/info
GET /global-explanations
GET /samples
GET /image-demo
POST /predict
POST /explain
POST /what-if
POST /text-explain
GET /artifacts/{file_path:path}

Requirements:
- Use FastAPI APIRouter if useful.
- Serve artifacts using StaticFiles mounted at /artifacts.
- Serve the built React frontend if available.
- If React build is unavailable, serve app/static_fallback/index.html.
- Enable CORS for local development.
- Use logging.
- Handle errors cleanly with JSON error responses.
- Do not expose arbitrary file system paths.
- Only serve files under artifacts/ and frontend build/static fallback.

GET /health must return:

{
  "status": "ok",
  "model_loaded": true or false,
  "timestamp": string
}

GET /version must return:

{
  "app_name": "Predictive Maintenance XAI Dashboard",
  "api_version": "1.0.0",
  "phase": "phase4",
  "frontend_mode": "react" or "static_fallback",
  "model_served": string
}

GET /model/info must return:
- served model source
- model type
- metrics if available
- explanation model used
- feature list

GET /model-card must return the model card JSON.

==================================================
MODEL CARD
==================================================

Create reports/model_card.md and also create a JSON version loadable by the API.

Save JSON version:

artifacts/model_card.json

The model card must include:

1. Model name
2. Dataset used
3. Target variable
4. Intended use
5. Out-of-scope use
6. Training data summary
7. Metrics
8. Explanation methods
9. Fairness findings
10. Leakage findings
11. Known limitations
12. Deployment notes
13. Maintenance recommendations
14. Ethical considerations

Use results from Phase 2 and Phase 3.
Do not invent metrics.
If a metric is unavailable, write not_available.

==================================================
REACT FRONTEND
==================================================

Create a minimal React frontend using Vite.

Use JavaScript, not TypeScript.
Do not use Redux.
Do not use complex state management.
Do not use authentication.
Do not use heavy UI frameworks.
Simple CSS is enough.

The frontend must have these tabs or pages:

1. Dashboard
2. Predict
3. Explain Sample
4. Global Insights
5. What-If Analysis
6. Text XAI Demo
7. Image XAI Demo
8. Model Card

Dashboard page must show:
- project title
- problem statement
- model name
- model metrics
- class distribution plot from Phase 3 if available
- risk threshold explanation

Predict page must show:
- sensor input form
- predict button
- failure probability
- risk badge
- top reasons
- recommendation

Explain Sample page must show:
- list of saved samples
- selected sample explanation
- SHAP local plot
- LIME explanation if available
- true label and predicted probability

Global Insights page must show:
- SHAP summary plot
- SHAP bar plot
- feature importance plot
- partial dependence plots
- global interpretation text

What-If Analysis page must show:
- original input form
- modified input form
- probability comparison
- risk change
- interpretation

Text XAI Demo page must show:
- text area
- analyze button
- predicted label
- important words
- LIME HTML explanation

Image XAI Demo page must show:
- image demo items
- original image
- heatmap
- overlay
- explanation

Model Card page must show:
- model card details in readable format

Create app/frontend/src/api.js to centralize API calls.

Use fetch.
Handle API errors gracefully.
Show loading states.
Show fallback messages if an endpoint fails.

Build the frontend into:

app/frontend/dist

Configure FastAPI to serve this folder if it exists.

==================================================
STATIC FALLBACK FRONTEND
==================================================

If npm or React build fails, create a static fallback frontend:

app/static_fallback/index.html

This fallback must:
- call the FastAPI endpoints
- display prediction form
- display explanation sample
- display model card
- display text demo
- display image demo

It does not need to be as polished as React.
But it must work.

Record frontend_mode in metadata:
- react
- static_fallback

==================================================
TESTS
==================================================

Create tests/test_api.py.

Use fastapi.testclient.TestClient.

Test:

1. GET /health returns 200 and status ok.
2. GET /version returns 200.
3. GET /model-card returns 200 and contains dataset field.
4. GET /global-explanations returns 200.
5. GET /samples returns 200 and at least one sample.
6. POST /predict with valid input returns failure_probability between 0 and 1.
7. POST /explain with valid input returns top_reasons or explanation_method.
8. POST /what-if with valid input returns original_probability and modified_probability.
9. POST /text-explain with sample text returns predicted_label.
10. GET /image-demo returns 200 and items list.

Do not start a long-running uvicorn server during validation.
Use TestClient only.

==================================================
METADATA AND LOGGING
==================================================

Create results/phase4_metadata.json.

Include:
- phase = phase4
- mode = fast or final
- timestamp
- frontend_mode
- image_demo_mode
- served_model_source
- served_model_path
- explanation_model_type
- surrogate_used
- shap_background_sample_size
- number_of_local_samples
- text_demo_samples
- endpoints_tested
- package_versions
- assumptions
- installation_issues
- execution_issues

Create logs:

results/phase4_run.log
results/phase4_error_log.json

If there are no errors, phase4_error_log.json should contain an empty list.

==================================================
REPORT OUTPUTS
==================================================

Create reports/xai_phase4_results.md.

Write it in report-ready academic style.

Include:

1. Phase 4 Objective
2. GUI Application Architecture
3. FastAPI Backend Design
4. React Frontend Design
5. Tabular Explainability Features
6. SHAP Global Explanation
7. SHAP Local Explanation
8. LIME Explanation
9. Partial Dependence Analysis
10. What-If Explanation
11. Maintenance Recommendation Engine
12. Text Explainability Demo
13. Image Explainability Demo
14. Model Card Summary
15. Innovation Points
16. Limitations
17. Screenshots to Capture for Final Report

Create reports/automl_phase4_deployment.md.

Include:

1. Deployment Objective
2. Model Export
3. FastAPI Prediction Service
4. Health and Version Endpoints
5. Local Running Instructions
6. Prediction Demo Example
7. Model Card Summary
8. Limitations
9. Preparation for DevOps Phase 5

Create or update reports/model_card.md.

==================================================
SCREENSHOT INSTRUCTIONS
==================================================

Create reports/screenshots/phase4_ui_screenshot_instructions.md.

List the screenshots the user must capture after running the app:

1. React dashboard home page
2. Prediction page with high-risk result
3. Explanation page with SHAP local plot
4. Global insights page
5. What-if analysis page
6. Text XAI demo page
7. Image XAI demo page
8. Model card page
9. FastAPI /docs Swagger UI
10. Terminal showing API health check

If a headless browser screenshot tool is available, attempt to capture screenshots.
If not, do not block Phase 4. Just create the instructions.

==================================================
README UPDATE
==================================================

Update README.md with Phase 4 instructions.

Include:

1. How to generate XAI artifacts:
.venv\Scripts\python scripts/generate_xai_artifacts.py --mode fast

2. How to build frontend:
.venv\Scripts\python scripts/build_frontend.py

3. How to run API:
.venv\Scripts\python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000

4. How to run tests:
.venv\Scripts\python -m pytest tests/test_api.py

5. How to validate Phase 4:
.venv\Scripts\python scripts/validate_phase4.py

Also include Linux/macOS equivalents.

==================================================
VALIDATION SCRIPT
==================================================

Create scripts/validate_phase4.py.

It must check that these files exist and are non-empty:

app/__init__.py
app/api/main.py
app/api/config.py
app/api/schemas.py
app/api/model_service.py
app/api/xai_service.py
app/api/text_demo_service.py
app/api/image_demo_service.py
app/api/recommendation_service.py
src/xai/tabular_explainer.py
src/xai/text_explainer.py
src/xai/image_explainer.py
src/xai/what_if.py
scripts/generate_xai_artifacts.py
scripts/build_frontend.py
scripts/run_phase4.py
scripts/validate_phase4.py
tests/test_api.py
results/phase4_metadata.json
results/phase4_run.log
results/phase4_error_log.json
artifacts/model_card.json
reports/model_card.md
reports/xai_phase4_results.md
reports/automl_phase4_deployment.md
reports/screenshots/phase4_ui_screenshot_instructions.md
artifacts/xai/global/shap_summary.png
artifacts/xai/global/shap_bar.png
artifacts/xai/global/feature_importance.png
artifacts/xai/global/pdp_torque.png
artifacts/xai/global/pdp_tool_wear.png
artifacts/xai/global/pdp_rotational_speed.png
artifacts/xai/global/pdp_process_temp.png
artifacts/xai/global/global_explanation.json
artifacts/xai/samples/sample_0.json
artifacts/xai/local/shap_local_sample_0.png
artifacts/xai/local/lime_local_sample_0.png
artifacts/xai/text_demo/text_sample_0.json
artifacts/xai/image_demo/image_0_metadata.json
models/text_demo_tfidf.pkl
models/text_demo_logreg.pkl

It must also dynamically check frontend mode:

If frontend_mode is react:
- app/frontend/dist/index.html must exist

If frontend_mode is static_fallback:
- app/static_fallback/index.html must exist

It must also run the API tests using TestClient or pytest.
If tests fail, validation must fail.

The validator must print PASS or FAIL for each item and a final overall status.

==================================================
IMPORTANT CONSTRAINTS
==================================================

- Do not start Phase 5 DevOps.
- Do not create Dockerfile yet.
- Do not create Kubernetes manifests yet.
- Do not create GitHub Actions workflows yet.
- Do not deploy to cloud.
- Do not modify Phase 1, Phase 2, or Phase 3 artifacts.
- Do not rerun AutoML tools.
- Do not fabricate XAI plots.
- Do not use the test set for training new explanation models unless absolutely necessary for surrogate explanation and clearly documented.
- Do not require external paid APIs.
- Do not require user login.
- Do not use a database.
- Do not ask the user for decisions.
- Do not build an overly large React app.
- Keep the UI clean, simple, and academic.

==================================================
EXECUTION COMMANDS
==================================================

Create scripts/run_phase4.py with argparse.

It must support:

--mode fast
--mode final

Default:
fast

Example commands:

.venv\Scripts\python scripts/run_phase4.py --mode fast
.venv\Scripts\python scripts/run_phase4.py --mode final
.venv\Scripts\python scripts/validate_phase4.py

For Linux/macOS:

.venv/bin/python scripts/run_phase4.py --mode fast
.venv/bin/python scripts/run_phase4.py --mode final
.venv/bin/python scripts/validate_phase4.py

==================================================
AGENT EXECUTION INSTRUCTIONS
==================================================

You must now:

1. Implement all Phase 4 code.
2. Install required Phase 4 packages.
3. Generate XAI artifacts in fast mode.
4. Build the React frontend if npm is available.
5. If React build fails, create static fallback frontend.
6. Run API tests using TestClient.
7. Run scripts/validate_phase4.py.
8. If validation fails, fix issues and rerun until validation passes.
9. Produce all required artifacts.
10. Do not run final mode automatically unless the environment is clearly safe and fast enough.
11. Stop after Phase 4.

==================================================
FINAL OUTPUT REQUIRED FROM YOU
==================================================

After completing Phase 4, provide:

1. Full list of created files.
2. Output of scripts/validate_phase4.py.
3. API test summary.
4. Frontend mode used:
   - react
   - or static_fallback
5. Image demo mode used:
   - gradcam
   - illustrative_fallback
6. Served model summary.
7. Explanation model summary.
8. Any assumptions made.
9. Any installation issues.
10. Any endpoint issues.
11. Exact command to run the API locally.
12. Exact command to rerun fast mode.
13. Exact command to run final mode.
14. Exact command to validate Phase 4.
15. A message saying:
   "Phase 4 completed. Ready for Phase 5 DevOps master prompt."

Do not start Phase 5.