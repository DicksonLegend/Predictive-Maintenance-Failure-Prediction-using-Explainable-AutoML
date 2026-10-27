import React, { useState } from 'react';

export default function PlotImage({ src, alt, caption }) {
  const [error, setError] = useState(false);

  return (
    <div style={{ marginBottom: '16px' }}>
      {error ? (
        <div style={{
          padding: '24px',
          background: '#131b2e',
          border: '1px dashed #273553',
          borderRadius: '8px',
          textAlign: 'center',
          color: '#94a3b8',
          fontSize: '0.85rem'
        }}>
          Plot artifact not found or loading... ({alt})
        </div>
      ) : (
        <img
          src={src}
          alt={alt}
          className="plot-img"
          onError={() => setError(true)}
        />
      )}
      {caption && (
        <p style={{
          fontSize: '0.8rem',
          color: '#94a3b8',
          marginTop: '6px',
          textAlign: 'center',
          fontStyle: 'italic'
        }}>
          {caption}
        </p>
      )}
    </div>
  );
}
