/**
 * Centralized API client for Predictive Maintenance XAI Dashboard.
 */

const API_BASE = '';

async function request(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {})
      }
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      const message = errData.detail || errData.message || `HTTP ${res.status}: ${res.statusText}`;
      throw new Error(message);
    }

    return await res.json();
  } catch (err) {
    console.error(`API Error on [${endpoint}]:`, err);
    throw err;
  }
}

export const api = {
  getHealth: () => request('/health'),
  getVersion: () => request('/version'),
  getModelInfo: () => request('/model/info'),
  getModelCard: () => request('/model-card'),
  getGlobalExplanations: () => request('/global-explanations'),
  getSamples: () => request('/samples'),
  getImageDemo: () => request('/image-demo'),

  predict: (sensorInput) =>
    request('/predict', {
      method: 'POST',
      body: JSON.stringify(sensorInput)
    }),

  explain: (sensorInput) =>
    request('/explain', {
      method: 'POST',
      body: JSON.stringify(sensorInput)
    }),

  whatIf: (originalInput, modifiedInput) =>
    request('/what-if', {
      method: 'POST',
      body: JSON.stringify({
        original_input: originalInput,
        modified_input: modifiedInput
      })
    }),

  explainText: (text) =>
    request('/text-explain', {
      method: 'POST',
      body: JSON.stringify({ text })
    })
};
