import React from 'react';

export default function Layout({ activeTab, onTabChange, appInfo, children }) {
  const tabs = [
    { id: 'dashboard', label: '1. Dashboard' },
    { id: 'predict', label: '2. Telemetry Predict & Explain' },
    { id: 'samples', label: '3. Explain Samples' },
    { id: 'global', label: '4. Global Insights' },
    { id: 'whatif', label: '5. What-If Simulation' },
    { id: 'textdemo', label: '6. Text XAI Demo' },
    { id: 'imagedemo', label: '7. Image XAI Demo' },
    { id: 'modelcard', label: '8. Model Card' },
  ];

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-title">
          <h1>Explainable AI Predictive Maintenance Dashboard</h1>
          <p>AutoML Machine Failure Diagnostics & Root-Cause Explainability for Industry 4.0</p>
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <span className="status-pill">
            <span style={{ fontSize: '8px' }}>●</span>
            {appInfo.servedModel || 'FLAML LightGBM'}
          </span>
          <span className="status-pill" style={{ borderColor: '#10b981', color: '#10b981' }}>
            FastAPI Live
          </span>
        </div>
      </header>

      <nav className="nav-bar">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => onTabChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="main-content">
        {children}
      </main>

      <footer className="footer">
        Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0 | Antigravity Phase 4
      </footer>
    </div>
  );
}
