import React from 'react';

export default function MetricCard({ label, value, subtext, highlight = false }) {
  return (
    <div className="metric-box">
      <div className="metric-label">{label}</div>
      <div className="metric-val" style={highlight ? { color: '#38bdf8' } : {}}>
        {value}
      </div>
      {subtext && <div className="metric-sub">{subtext}</div>}
    </div>
  );
}
