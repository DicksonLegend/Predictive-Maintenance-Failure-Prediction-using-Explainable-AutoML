# Phase 4 UI Screenshot Capture Instructions

This guide provides step-by-step instructions for capturing the ten required screenshots of the Explainable AI Dashboard for inclusion in the final academic reports and presentation slides.

---

## Prerequisites: Start the Local Dashboard

Open PowerShell or terminal and execute:
```powershell
.venv\Scripts\python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```
Open your web browser and navigate to:
```
http://localhost:8000/
```

---

## 10 Required Screenshots

| # | Screenshot Name | File Target | Action to Capture |
|---|---|---|---|
| 1 | **React Dashboard Home Page** | `reports/screenshots/ui_01_dashboard.png` | Navigate to `1. Dashboard` tab. Capture top summary cards, model metrics (F1 = 0.6281), operational risk bands, and global SHAP bar plot. |
| 2 | **Prediction Page (High Risk)** | `reports/screenshots/ui_02_prediction_high_risk.png` | Navigate to `2. Telemetry Predict & Explain`. Enter values: `Type: L`, `Air Temp: 303.5`, `Process Temp: 312.0`, `Speed: 1360`, `Torque: 68.0`, `Wear: 220`. Click submit. Capture high risk badge and probability (>80%). |
| 3 | **Explanation Page (SHAP Waterfall)** | `reports/screenshots/ui_03_explanation_shap.png` | On the prediction results panel, capture the dynamic local SHAP horizontal bar chart and prescriptive work order box. |
| 4 | **Global Insights Page** | `reports/screenshots/ui_04_global_insights.png` | Navigate to `4. Global Insights`. Capture the global SHAP beeswarm summary plot and Partial Dependence curves for Torque and Tool Wear. |
| 5 | **What-If Analysis Page** | `reports/screenshots/ui_05_what_if.png` | Navigate to `5. What-If Simulation`. Click "Simulate Parameter Adjustment". Capture baseline vs. modified probabilities, probability delta, and narrative interpretation. |
| 6 | **Text XAI Demo Page** | `reports/screenshots/ui_06_text_xai.png` | Navigate to `6. Text XAI Demo`. Click "Classify Urgency & Highlight Keywords". Capture the `URGENT WORK ORDER` badge and the highlighted attributed tokens (`"screeching"`, `"vibrating"`). |
| 7 | **Image XAI Demo Page** | `reports/screenshots/ui_07_image_xai.png` | Navigate to `7. Image XAI Demo`. Capture the three component inspection items: Tool Blade Micro-Crack, Bearing Race Spalling, and Stator Thermal Hotspot with saliency overlays. |
| 8 | **Model Card Page** | `reports/screenshots/ui_08_model_card.png` | Navigate to `8. Model Card`. Capture the formatted JSON model governance viewer displaying data lineage and metrics. |
| 9 | **FastAPI Swagger UI** | `reports/screenshots/ui_09_swagger_docs.png` | Navigate to `http://localhost:8000/docs`. Capture the OpenAPI interactive endpoint documentation showing `/predict`, `/explain`, `/what-if`, etc. |
| 10 | **Terminal API Health Check** | `reports/screenshots/ui_10_terminal_health.png` | In a terminal, execute `curl http://localhost:8000/health` and `.venv\Scripts\python -m pytest tests/test_api.py`. Capture the terminal showing 10 passed tests. |

---

## Visual Presentation Guidelines
- Capture screenshots in full resolution (minimum 1920x1080 display scaling).
- Crop browser chrome if desired, or keep address bar visible to confirm localhost operation.
- Save image files in PNG format under `reports/screenshots/`.
