export default function SpoofAnalysisPanel({ result }) {
  const spoof = result?.spoof_detection || {};
  const models = spoof.models || {};
  const heuristics = spoof.heuristics || {};
  const weights = spoof.effective_weights || {};

  return (
    <section className="card">
      <div className="section-header">
        <span className="eyebrow">Ensemble Telemetry & Signal Features</span>
        <span className={`pill ${spoof.status || 'incomplete'}`}>{spoof.model_status || spoof.status || 'Waiting'}</span>
      </div>

      <ul>
        <li><strong>Architecture:</strong> Three-Model Ensemble + Heuristic Fusion</li>
        <li><strong>Decision Band:</strong> <span style={{ fontWeight: 600, color: spoof.decision_band === 'HIGH_SPOOF_RISK_REVIEW' ? '#f87171' : spoof.decision_band === 'REVIEW_REQUIRED' ? '#fbbf24' : '#4ade80' }}>{spoof.decision_band || 'N/A'}</span></li>
        <li><strong>Composite Index:</strong> {spoof.ensemble_spoof_risk != null ? `${spoof.ensemble_spoof_risk} / 100` : 'N/A'} (Provisional)</li>
        <li><strong>Threshold Bands:</strong> Review: {spoof.review_threshold || 70.0} · High Risk: {spoof.high_risk_threshold || 75.0}</li>
        <li><strong>Effective Weights:</strong> {Object.entries(weights).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join(' · ') || 'N/A'}</li>
        <li><strong>Latency:</strong> {spoof.inference_latency_ms != null ? `${spoof.inference_latency_ms} ms` : 'N/A'}</li>
        {spoof.is_degraded && (
          <li><strong>Degraded State:</strong> <span style={{ color: '#fbbf24' }}>Offline: {spoof.missing_models?.join(', ') || 'Model A'}</span></li>
        )}
      </ul>

      {/* Heuristic Features summary */}
      {heuristics.supporting_features && (
        <div style={{ marginTop: '0.75rem', background: 'rgba(15, 23, 42, 0.4)', padding: '0.5rem', borderRadius: '6px', fontSize: '0.73rem', color: '#94a3b8' }}>
          <div style={{ fontWeight: 600, color: '#e2e8f0', marginBottom: '0.2rem' }}>Acoustic Heuristic Features:</div>
          <div>Dynamic Range: {heuristics.supporting_features.dynamic_range_db} dB · ZCR: {heuristics.supporting_features.zero_crossing_rate_mean}</div>
          <div>Spectral Centroid: {heuristics.supporting_features.spectral_centroid_hz} Hz · Flatness: {heuristics.supporting_features.spectral_flatness_mean}</div>
          {heuristics.supporting_features.pitch_f0_mean_hz && (
            <div>F0 Mean: {heuristics.supporting_features.pitch_f0_mean_hz} Hz · F0 Std: {heuristics.supporting_features.pitch_f0_std_hz} Hz</div>
          )}
        </div>
      )}

      <p className="notice" style={{ marginTop: '0.75rem', fontSize: '0.73rem' }}>
        Ensemble outputs combine acoustic phase embeddings, self-supervised representations, and spectral heuristics. Results indicate statistical anti-spoofing evidence, not proof of individual speaker identity.
      </p>
    </section>
  );
}
