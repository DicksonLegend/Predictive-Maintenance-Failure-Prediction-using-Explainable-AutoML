import React, { useState, useEffect } from 'react';
import { api } from './api';
import Layout from './components/Layout';
import MetricCard from './components/MetricCard';
import PredictionForm from './components/PredictionForm';
import ExplanationPanel from './components/ExplanationPanel';
import PlotImage from './components/PlotImage';
import RiskBadge from './components/RiskBadge';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [appInfo, setAppInfo] = useState({ servedModel: 'FLAML LightGBM', mode: 'fast' });
  const [modelInfo, setModelInfo] = useState(null);
  const [modelCard, setModelCard] = useState(null);
  const [globalData, setGlobalData] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [imageDemo, setImageDemo] = useState(null);

  // Predict state
  const [telemetry, setTelemetry] = useState({
    type: 'L',
    air_temp: 300.5,
    process_temp: 310.2,
    rotational_speed: 1420.0,
    torque: 62.4,
    tool_wear: 215.0
  });
  const [predictResult, setPredictResult] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);

  // What-If state
  const [whatIfOriginal, setWhatIfOriginal] = useState({
    type: 'L',
    air_temp: 302.0,
    process_temp: 311.0,
    rotational_speed: 1350.0,
    torque: 68.0,
    tool_wear: 220.0
  });
  const [whatIfModified, setWhatIfModified] = useState({
    type: 'L',
    air_temp: 300.0,
    process_temp: 310.0,
    rotational_speed: 1550.0,
    torque: 42.0,
    tool_wear: 15.0
  });
  const [whatIfResult, setWhatIfResult] = useState(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);

  // Text Demo state
  const [textInput, setTextInput] = useState('Machine making loud screeching noise and vibrating heavily near primary spindle drive.');
  const [textResult, setTextResult] = useState(null);
  const [textLoading, setTextLoading] = useState(false);

  useEffect(() => {
    // Initial fetch
    api.getModelInfo().then(d => {
      setModelInfo(d);
      setAppInfo(prev => ({ ...prev, servedModel: d.model_name || 'FLAML LightGBM' }));
    }).catch(console.error);

    api.getGlobalExplanations().then(setGlobalData).catch(console.error);
    api.getSamples().then(s => {
      setSamples(s);
      if (s.length > 0) setSelectedSample(s[0]);
    }).catch(console.error);
    api.getImageDemo().then(setImageDemo).catch(console.error);
    api.getModelCard().then(setModelCard).catch(console.error);
  }, []);

  const handlePredictSubmit = async (e) => {
    e.preventDefault();
    setPredictLoading(true);
    try {
      const res = await api.explain(telemetry);
      setPredictResult(res);
    } catch (err) {
      alert(`Prediction failed: ${err.message}`);
    } finally {
      setPredictLoading(false);
    }
  };

  const handleWhatIfSubmit = async () => {
    setWhatIfLoading(true);
    try {
      const res = await api.whatIf(whatIfOriginal, whatIfModified);
      setWhatIfResult(res);
    } catch (err) {
      alert(`Simulation failed: ${err.message}`);
    } finally {
      setWhatIfLoading(false);
    }
  };

  const handleTextExplainSubmit = async () => {
    setTextLoading(true);
    try {
      const res = await api.explainText(textInput);
      setTextResult(res);
    } catch (err) {
      alert(`Text explanation failed: ${err.message}`);
    } finally {
      setTextLoading(false);
    }
  };

  return (
    <Layout activeTab={activeTab} onTabChange={setActiveTab} appInfo={appInfo}>
      {/* 1. DASHBOARD */}
      {activeTab === 'dashboard' && (
        <div>
          <div className="card">
            <h2>Explainable AutoML for Predictive Maintenance</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '14px', lineHeight: '1.6' }}>
              Industrial manufacturing plants lose billions annually to unpredicted catastrophic machine breakdowns.
              This system deploys a high-performance AutoML classification engine trained on multi-sensor CNC telemetry
              (temperatures, rotational velocity, torque, tool wear), paired with multi-modal Explainable AI (SHAP, LIME,
              Partial Dependence, What-If counterfactuals, and NLP note analysis) to transform black-box predictions into
              prescriptive technician actions.
            </p>
          </div>

          <div className="grid-4" style={{ marginBottom: '24px' }}>
            <MetricCard
              label="Primary Metric"
              value={modelInfo?.metrics?.f1 ? modelInfo.metrics.f1 : '0.6281'}
              subtext="Test F1-Score (AutoML winner)"
              highlight={true}
            />
            <MetricCard
              label="ROC-AUC"
              value={modelInfo?.metrics?.roc_auc ? modelInfo.metrics.roc_auc : '0.9171'}
              subtext="Discriminative Ranking"
            />
            <MetricCard
              label="Precision"
              value={modelInfo?.metrics?.precision ? modelInfo.metrics.precision : '0.7170'}
              subtext="Low False Alarm Rate"
            />
            <MetricCard
              label="Recall"
              value={modelInfo?.metrics?.recall ? modelInfo.metrics.recall : '0.5588'}
              subtext="Failure Capture Rate"
            />
          </div>

          <div className="grid-2">
            <div className="card">
              <h2>Operational Risk Framework</h2>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '16px', fontSize: '0.9rem' }}>
                Failure probabilities $P(\text{failure} \mid \mathbf{x})$ are mapped to standard operational risk bands:
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ padding: '12px', background: '#121929', borderRadius: '6px', borderLeft: '4px solid #10b981' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <strong style={{ color: '#6ee7b7' }}>LOW RISK</strong>
                    <RiskBadge level="Low" />
                  </div>
                  <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '4px' }}>
                    $P &lt; 0.30$ — Nominal operating parameters; routine telemetry sampling continues.
                  </p>
                </div>
                <div style={{ padding: '12px', background: '#121929', borderRadius: '6px', borderLeft: '4px solid #f59e0b' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <strong style={{ color: '#fde68a' }}>MEDIUM RISK</strong>
                    <RiskBadge level="Medium" />
                  </div>
                  <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '4px' }}>
                    $0.30 \le P \le 0.60$ — Elevated tool wear or temperature differential; alert shift technician.
                  </p>
                </div>
                <div style={{ padding: '12px', background: '#121929', borderRadius: '6px', borderLeft: '4px solid #ef4444' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <strong style={{ color: '#fca5a5' }}>HIGH RISK</strong>
                    <RiskBadge level="High" />
                  </div>
                  <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '4px' }}>
                    $P &gt; 0.60$ — Impending catastrophic breakdown; immediate work order dispatched.
                  </p>
                </div>
              </div>
            </div>

            <div className="card">
              <h2>Global Feature Influence (SHAP)</h2>
              <PlotImage
                src="/artifacts/xai/global/shap_bar.png"
                alt="Global SHAP Bar Plot"
                caption="Global mean absolute Shapley values ranking primary mechanical failure drivers."
              />
            </div>
          </div>
        </div>
      )}

      {/* 2. PREDICT */}
      {activeTab === 'predict' && (
        <div className="grid-2">
          <div className="card">
            <h2>Machine Telemetry Ingestion</h2>
            <PredictionForm
              values={telemetry}
              onChange={setTelemetry}
              onSubmit={handlePredictSubmit}
              loading={predictLoading}
            />
          </div>
          <div className="card">
            <h2>Diagnostic Inference & Root Cause</h2>
            <ExplanationPanel result={predictResult} />
          </div>
        </div>
      )}

      {/* 3. EXPLAIN SAMPLES */}
      {activeTab === 'samples' && (
        <div>
          <div className="card">
            <h2>Benchmark Test Case Diagnostics</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '16px', fontSize: '0.9rem' }}>
              Audited sample instances from the held-out test set illustrating model reasoning on high-risk, medium-risk, and nominal machines.
            </p>
            <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', flexWrap: 'wrap' }}>
              {samples.map((s, idx) => (
                <button
                  key={s.sample_id}
                  className={`nav-tab ${selectedSample?.sample_id === s.sample_id ? 'active' : ''}`}
                  style={{ borderRadius: '6px', border: '1px solid #273553' }}
                  onClick={() => setSelectedSample(s)}
                >
                  {s.sample_id.toUpperCase()} ({s.risk_level} Risk)
                </button>
              ))}
            </div>

            {selectedSample && (
              <div className="grid-2">
                <div>
                  <h3>Input Telemetry & Metadata</h3>
                  <div style={{ background: '#121929', padding: '16px', borderRadius: '8px', marginBottom: '16px' }}>
                    <p><strong>Sample ID:</strong> {selectedSample.sample_id.toUpperCase()}</p>
                    <p><strong>Ground Truth Failure:</strong> {selectedSample.true_label === 1 ? 'Failure (1)' : 'Nominal (0)'}</p>
                    <p><strong>Predicted Probability:</strong> {(selectedSample.predicted_probability * 100).toFixed(1)}%</p>
                    <p style={{ marginTop: '8px' }}><RiskBadge level={selectedSample.risk_level} /></p>
                  </div>
                  <h3>Input Telemetry Values:</h3>
                  <pre style={{ background: '#121929', padding: '12px', borderRadius: '6px', fontSize: '0.85rem' }}>
                    {JSON.stringify(selectedSample.input_features, null, 2)}
                  </pre>
                  <h3 style={{ marginTop: '16px' }}>Work Order Recommendation:</h3>
                  <div className="work-order-box">{selectedSample.recommendation}</div>
                </div>

                <div>
                  <h3>Local SHAP Waterfall Attribution</h3>
                  <PlotImage
                    src={selectedSample.local_shap_plot_path}
                    alt="Sample SHAP Plot"
                    caption="Additive feature contributions for this specific machine record."
                  />
                  {selectedSample.lime_explanation_path && (
                    <div style={{ marginTop: '16px' }}>
                      <h3>Local LIME Rule Surrogate</h3>
                      <PlotImage
                        src={selectedSample.lime_explanation_path}
                        alt="Sample LIME Plot"
                      />
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 4. GLOBAL INSIGHTS */}
      {activeTab === 'global' && (
        <div>
          <div className="card">
            <h2>Fleet-Wide Global Model Interpretability</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '20px', fontSize: '0.9rem' }}>
              Comprehensive model interpretability across the entire training manifold using SHAP TreeExplainer and Partial Dependence Plots.
            </p>
            <div className="grid-2">
              <div>
                <h3>SHAP Beeswarm Summary Plot</h3>
                <PlotImage
                  src="/artifacts/xai/global/shap_summary.png"
                  alt="SHAP Beeswarm"
                  caption="Feature value impact on model output log-odds across all samples."
                />
              </div>
              <div>
                <h3>Global Mean Absolute SHAP Importance</h3>
                <PlotImage
                  src="/artifacts/xai/global/shap_bar.png"
                  alt="SHAP Bar"
                  caption="Average absolute impact of each telemetry feature across all predictions."
                />
              </div>
            </div>

            <h2 style={{ marginTop: '24px' }}>Partial Dependence Curves (PDP)</h2>
            <div className="grid-2">
              <div>
                <h3>Partial Dependence: Torque [Nm]</h3>
                <PlotImage
                  src="/artifacts/xai/global/pdp_torque.png"
                  alt="PDP Torque"
                  caption="Marginal effect of torque on failure probability."
                />
              </div>
              <div>
                <h3>Partial Dependence: Tool Wear [min]</h3>
                <PlotImage
                  src="/artifacts/xai/global/pdp_tool_wear.png"
                  alt="PDP Tool Wear"
                  caption="Marginal effect of cumulative tool wear on failure risk."
                />
              </div>
              <div>
                <h3>Partial Dependence: Rotational Speed [rpm]</h3>
                <PlotImage
                  src="/artifacts/xai/global/pdp_rotational_speed.png"
                  alt="PDP Rotational Speed"
                  caption="Marginal effect of spindle RPM on failure probability."
                />
              </div>
              <div>
                <h3>Partial Dependence: Process Temperature [K]</h3>
                <PlotImage
                  src="/artifacts/xai/global/pdp_process_temp.png"
                  alt="PDP Process Temp"
                  caption="Marginal effect of process temperature on failure probability."
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 5. WHAT-IF ANALYSIS */}
      {activeTab === 'whatif' && (
        <div className="card">
          <h2>Counterfactual Parameter Sensitivity Simulator</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '20px', fontSize: '0.9rem' }}>
            Simulate operational parameter modifications to observe how targeted engineering interventions
            (e.g., torque relief, speed adjustment, cutting tool renewal) transition machines from high to low risk.
          </p>
          <div className="grid-2">
            <div>
              <h3>Baseline Operating State (Stressed)</h3>
              <div className="form-group">
                <label className="form-label">Torque [Nm]</label>
                <input
                  type="number"
                  step="0.1"
                  className="form-input"
                  value={whatIfOriginal.torque}
                  onChange={e => setWhatIfOriginal({ ...whatIfOriginal, torque: parseFloat(e.target.value) || 0 })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Tool Wear [min]</label>
                <input
                  type="number"
                  step="1"
                  className="form-input"
                  value={whatIfOriginal.tool_wear}
                  onChange={e => setWhatIfOriginal({ ...whatIfOriginal, tool_wear: parseFloat(e.target.value) || 0 })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Rotational Speed [rpm]</label>
                <input
                  type="number"
                  step="1"
                  className="form-input"
                  value={whatIfOriginal.rotational_speed}
                  onChange={e => setWhatIfOriginal({ ...whatIfOriginal, rotational_speed: parseFloat(e.target.value) || 0 })}
                />
              </div>
            </div>

            <div>
              <h3>Modified Counterfactual State (Intervention)</h3>
              <div className="form-group">
                <label className="form-label">Modified Torque [Nm]</label>
                <input
                  type="number"
                  step="0.1"
                  className="form-input"
                  value={whatIfModified.torque}
                  onChange={e => setWhatIfModified({ ...whatIfModified, torque: parseFloat(e.target.value) || 0 })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Modified Tool Wear [min]</label>
                <input
                  type="number"
                  step="1"
                  className="form-input"
                  value={whatIfModified.tool_wear}
                  onChange={e => setWhatIfModified({ ...whatIfModified, tool_wear: parseFloat(e.target.value) || 0 })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Modified Rotational Speed [rpm]</label>
                <input
                  type="number"
                  step="1"
                  className="form-input"
                  value={whatIfModified.rotational_speed}
                  onChange={e => setWhatIfModified({ ...whatIfModified, rotational_speed: parseFloat(e.target.value) || 0 })}
                />
              </div>
            </div>
          </div>

          <button
            className="btn-primary"
            style={{ marginTop: '16px' }}
            onClick={handleWhatIfSubmit}
            disabled={whatIfLoading}
          >
            {whatIfLoading ? 'Computing Counterfactual...' : 'Run What-If Sensitivity Simulation'}
          </button>

          {whatIfResult && (
            <div style={{ marginTop: '24px', background: '#121929', padding: '20px', borderRadius: '8px', border: '1px solid #273553' }}>
              <div style={{ display: 'flex', gap: '24px', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap' }}>
                <div>
                  <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Original Risk</span>
                  <div style={{ fontSize: '1.4rem', fontWeight: '700', color: '#fff' }}>
                    {(whatIfResult.original_probability * 100).toFixed(1)}% <RiskBadge level={whatIfResult.original_risk_level} />
                  </div>
                </div>
                <div style={{ fontSize: '1.5rem', color: '#64748b' }}>➔</div>
                <div>
                  <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Modified Risk</span>
                  <div style={{ fontSize: '1.4rem', fontWeight: '700', color: '#fff' }}>
                    {(whatIfResult.modified_probability * 100).toFixed(1)}% <RiskBadge level={whatIfResult.modified_risk_level} />
                  </div>
                </div>
                <div>
                  <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Probability Delta</span>
                  <div style={{ fontSize: '1.4rem', fontWeight: '700', color: whatIfResult.probability_change < 0 ? '#10b981' : '#ef4444' }}>
                    {whatIfResult.probability_change > 0 ? '+' : ''}{(whatIfResult.probability_change * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <h3>Simulation Interpretation:</h3>
              <p style={{ background: '#0a0f1d', padding: '12px', borderRadius: '6px', margin: '8px 0 16px 0', border: '1px solid #273553' }}>
                {whatIfResult.interpretation}
              </p>

              <h3>Updated Prescriptive Recommendation:</h3>
              <div className="work-order-box">{whatIfResult.recommendation}</div>
            </div>
          )}
        </div>
      )}

      {/* 6. TEXT XAI DEMO */}
      {activeTab === 'textdemo' && (
        <div className="card">
          <h2>Multi-Modal NLP: Maintenance Shift Notes Token Attribution</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '16px', fontSize: '0.9rem' }}>
            Technicians log unstructured text notes during shift handovers. This NLP classifier uses TF-IDF and
            LIME Text Explainer to highlight which specific symptom keywords drove an urgent or normal diagnosis.
          </p>

          <div className="form-group">
            <label className="form-label">Technician Maintenance Note</label>
            <textarea
              rows={3}
              className="form-textarea"
              value={textInput}
              onChange={e => setTextInput(e.target.value)}
            />
          </div>

          <button
            className="btn-primary"
            onClick={handleTextExplainSubmit}
            disabled={textLoading}
          >
            {textLoading ? 'Analyzing Text Tokens...' : 'Classify Urgency & Highlight Keywords (LIME)'}
          </button>

          {textResult && (
            <div style={{ marginTop: '24px', background: '#121929', padding: '20px', borderRadius: '8px', border: '1px solid #273553' }}>
              <div style={{ display: 'flex', gap: '16px', alignItems: 'center', marginBottom: '16px' }}>
                <span className={`risk-badge ${textResult.predicted_label === 'urgent' ? 'high' : 'low'}`}>
                  {textResult.predicted_label.toUpperCase()} WORK ORDER
                </span>
                <span style={{ fontSize: '0.9rem', color: '#94a3b8' }}>
                  Confidence: <strong>{(textResult.probability * 100).toFixed(1)}%</strong>
                </span>
              </div>

              <h3>Attributed Diagnostic Tokens:</h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', margin: '12px 0' }}>
                {textResult.important_words.map((w, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: '#1a233b',
                      border: `1px solid ${w.impact === 'urgent' ? 'rgba(239, 68, 68, 0.5)' : 'rgba(16, 185, 129, 0.5)'}`,
                      padding: '8px 14px',
                      borderRadius: '6px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '2px'
                    }}
                  >
                    <strong style={{ color: '#fff' }}>"{w.word}"</strong>
                    <span style={{ fontSize: '0.75rem', color: w.impact === 'urgent' ? '#ef4444' : '#10b981' }}>
                      {w.impact.toUpperCase()} ({w.weight > 0 ? '+' : ''}{w.weight})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* 7. IMAGE XAI DEMO */}
      {activeTab === 'imagedemo' && (
        <div className="card">
          <h2>Multi-Modal Spatial Saliency Inspection (Grad-CAM Demonstration)</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '20px', fontSize: '0.9rem' }}>
            Spatial attribution saliency heatmaps highlighting localized physical defect initiation on CNC components
            (tool blade micro-cracks, spindle bearing race flaking, and motor stator thermal hotspots).
          </p>

          <div className="grid-3">
            {imageDemo?.items?.map((it, idx) => (
              <div key={idx} className="card" style={{ background: '#121929', margin: 0 }}>
                <h3 style={{ color: '#fff' }}>{it.title}</h3>
                <span className="risk-badge medium" style={{ margin: '8px 0 12px 0' }}>
                  {it.predicted_class}
                </span>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
                  <div>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Original Scan</span>
                    <PlotImage src={it.original} alt={`${it.title} Original`} />
                  </div>
                  <div>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Saliency Overlay</span>
                    <PlotImage src={it.overlay} alt={`${it.title} Overlay`} />
                  </div>
                </div>

                <p style={{ fontSize: '0.8rem', color: '#94a3b8', lineHeight: '1.4' }}>
                  {it.explanation}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 8. MODEL CARD */}
      {activeTab === 'modelcard' && (
        <div className="card">
          <h2>Standardized AI Model Card</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '20px', fontSize: '0.9rem' }}>
            Hugging Face style standardized model governance documentation detailing intended use, data lineage,
            performance metrics, fairness audits, and known operational limitations.
          </p>
          <pre style={{
            background: '#0a0f1d',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #273553',
            overflowX: 'auto',
            fontSize: '0.85rem',
            lineHeight: '1.5'
          }}>
            {modelCard ? JSON.stringify(modelCard, null, 2) : 'Loading model card...'}
          </pre>
        </div>
      )}
    </Layout>
  );
}
