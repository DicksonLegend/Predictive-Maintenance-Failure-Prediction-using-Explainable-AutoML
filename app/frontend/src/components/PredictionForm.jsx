import React from 'react';

export default function PredictionForm({ values, onChange, onSubmit, loading }) {
  const handleChange = (e) => {
    const { name, value } = e.target;
    onChange({
      ...values,
      [name]: name === 'type' ? value : parseFloat(value) || 0
    });
  };

  return (
    <form onSubmit={onSubmit}>
      <div className="form-group">
        <label className="form-label">Machine Quality Variant (Type)</label>
        <select
          name="type"
          value={values.type}
          onChange={handleChange}
          className="form-select"
        >
          <option value="L">L (50% Production Share - Low Quality)</option>
          <option value="M">M (30% Production Share - Medium Quality)</option>
          <option value="H">H (20% Production Share - High Quality)</option>
        </select>
      </div>

      <div className="form-group">
        <label className="form-label">Air Temperature [K] (Nominal: 295 - 304 K)</label>
        <input
          type="number"
          step="0.1"
          name="air_temp"
          value={values.air_temp}
          onChange={handleChange}
          className="form-input"
          required
        />
      </div>

      <div className="form-group">
        <label className="form-label">Process Temperature [K] (Nominal: 305 - 314 K)</label>
        <input
          type="number"
          step="0.1"
          name="process_temp"
          value={values.process_temp}
          onChange={handleChange}
          className="form-input"
          required
        />
      </div>

      <div className="form-group">
        <label className="form-label">Rotational Speed [rpm] (Nominal: 1300 - 1800 rpm)</label>
        <input
          type="number"
          step="1"
          name="rotational_speed"
          value={values.rotational_speed}
          onChange={handleChange}
          className="form-input"
          required
        />
      </div>

      <div className="form-group">
        <label className="form-label">Drive Torque [Nm] (High risk: &gt; 60 Nm)</label>
        <input
          type="number"
          step="0.1"
          name="torque"
          value={values.torque}
          onChange={handleChange}
          className="form-input"
          required
        />
      </div>

      <div className="form-group">
        <label className="form-label">Tool Wear [min] (High risk: &gt; 200 min)</label>
        <input
          type="number"
          step="1"
          name="tool_wear"
          value={values.tool_wear}
          onChange={handleChange}
          className="form-input"
          required
        />
      </div>

      <button type="submit" className="btn-primary" disabled={loading}>
        {loading ? 'Evaluating Model & SHAP Explainer...' : 'Run Diagnostics & Explain Failure Risk'}
      </button>
    </form>
  );
}
