import React from 'react';
import RiskBadge from './RiskBadge';
import PlotImage from './PlotImage';

export default function ExplanationPanel({ result }) {
  if (!result) {
    return (
      <div style={{
        padding: '60px 20px',
        textAlign: 'center',
        color: '#64748b',
        background: '#121929',
        borderRadius: '8px',
        border: '1px dashed #273553'
      }}>
        Adjust machine telemetry parameters and submit to view failure risk diagnostics and SHAP attribution waterfall.
      </div>
    );
  }

  const { prediction, top_reasons, recommendation, local_shap_plot, explanation_method } = result;

  return (
    <div>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '16px',
        paddingBottom: '16px',
        borderBottom: '1px solid #273553'
      }}>
        <div>
          <span style={{ fontSize: '0.85rem', color: '#94a3b8', display: 'block' }}>Failure Probability</span>
          <span style={{ fontSize: '2rem', fontWeight: '700', color: '#fff', fontFamily: 'var(--font-mono)' }}>
            {(prediction.failure_probability * 100).toFixed(1)}%
          </span>
        </div>
        <div>
          <RiskBadge level={prediction.risk_level} />
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '0.9rem', color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>
          Top Attributed Risk Drivers ({explanation_method})
        </h3>
        {top_reasons && top_reasons.map((r, i) => (
          <div
            key={i}
            className={`reason-card ${r.impact.includes('increases') ? 'increase' : 'decrease'}`}
          >
            <div>
              <strong style={{ color: '#fff', textTransform: 'uppercase' }}>{r.feature.replace('_', ' ')}</strong>
              <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>{r.impact}</div>
            </div>
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontWeight: '700',
              color: r.impact.includes('increases') ? '#ef4444' : '#10b981'
            }}>
              {r.attribution !== undefined && r.attribution !== null ? `${r.attribution > 0 ? '+' : ''}${r.attribution}` : ''}
            </span>
          </div>
        ))}
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '0.9rem', color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>
          Prescriptive Maintenance Work Order
        </h3>
        <div className="work-order-box">
          {recommendation}
        </div>
      </div>

      {local_shap_plot && (
        <div>
          <h3 style={{ fontSize: '0.9rem', color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>
            Local Shapley Attribution Breakdown
          </h3>
          <PlotImage
            src={`${local_shap_plot}?t=${Date.now()}`}
            alt="Dynamic Local SHAP Waterfall"
            caption="Dynamic local SHAP values quantifying feature contributions to prediction log-odds."
          />
        </div>
      )}
    </div>
  );
}
