import React from 'react';

export default function RiskBadge({ level = 'Low', probability = null }) {
  const normLevel = (level || 'Low').toLowerCase();
  return (
    <span className={`risk-badge ${normLevel}`}>
      <span style={{ fontSize: '10px' }}>●</span>
      {level} Risk {probability !== null ? `(${(probability * 100).toFixed(1)}%)` : ''}
    </span>
  );
}
