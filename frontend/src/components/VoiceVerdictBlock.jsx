export default function VoiceVerdictBlock({ result }) {
  const spoof = result?.spoof_detection || {};
  const status = spoof.status;
  const decisionBand = spoof.decision_band || spoof.predicted_class;
  const ensembleRisk = spoof.ensemble_spoof_risk != null ? Number(spoof.ensemble_spoof_risk) : null;
  const reviewThresh = spoof.review_threshold || 70.0;
  const highRiskThresh = spoof.high_risk_threshold || 75.0;
  const isDegraded = Boolean(spoof.is_degraded);
  const models = spoof.models || {};
  const heuristics = spoof.heuristics || {};

  const isLowRisk = decisionBand === 'LOWER_MODEL_ESTIMATED_SPOOF_RISK' || decisionBand === 'MODEL_PREDICTS_BONA_FIDE';
  const isReview = decisionBand === 'REVIEW_REQUIRED' || decisionBand === 'INDETERMINATE';
  const isHighRisk = decisionBand === 'HIGH_SPOOF_RISK_REVIEW' || decisionBand === 'MODEL_PREDICTS_SPOOF';
  const isUnavailable = decisionBand === 'MODEL_UNAVAILABLE' || status === 'unavailable';
  const isError = decisionBand === 'PROCESSING_ERROR' || status === 'error';

  const getBorderColor = () => {
    if (isLowRisk) return 'rgba(34, 197, 94, 0.4)';
    if (isHighRisk) return 'rgba(239, 68, 68, 0.4)';
    if (isReview) return 'rgba(245, 158, 11, 0.4)';
    return 'rgba(148, 163, 184, 0.2)';
  };

  const getBgColor = () => {
    if (isLowRisk) return 'rgba(34, 197, 94, 0.12)';
    if (isHighRisk) return 'rgba(239, 68, 68, 0.12)';
    if (isReview) return 'rgba(245, 158, 11, 0.12)';
    return 'rgba(148, 163, 184, 0.1)';
  };

  const getTextColor = () => {
    if (isLowRisk) return '#4ade80';
    if (isHighRisk) return '#f87171';
    if (isReview) return '#fbbf24';
    return '#94a3b8';
  };

  const getHeadline = () => {
    if (isLowRisk) return '🟢 LOWER ESTIMATED SPOOF RISK';
    if (isReview) return '🟡 REVIEW REQUIRED';
    if (isHighRisk) return '🔴 HIGH SPOOF RISK REVIEW';
    if (isUnavailable) return '⚪ DETECTOR UNAVAILABLE';
    return '🔴 INDETERMINATE / ERROR';
  };

  return (
    <section className="card" style={{ border: `1px solid ${getBorderColor()}` }}>
      <div className="section-header">
        <span className="eyebrow">Voice Anti-Spoofing Verdict</span>
        {decisionBand && (
          <span className={`pill ${isLowRisk ? 'pass' : isHighRisk ? 'fail' : 'incomplete'}`}>
            {decisionBand}
          </span>
        )}
      </div>

      {!result ? (
        <div style={{ color: '#94a3b8', fontSize: '0.95rem', padding: '0.5rem 0' }}>
          Record or upload audio to run three-model anti-spoofing verification.
        </div>
      ) : status === 'completed' || status === 'degraded' ? (
        <div>
          {/* Main Verdict Banner */}
          <div
            style={{
              padding: '1rem',
              borderRadius: '12px',
              backgroundColor: getBgColor(),
              border: `1px solid ${getBorderColor()}`,
              marginBottom: '1rem',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: getTextColor(), marginBottom: '0.25rem' }}>
              Multi-Model Ensemble Decision Band
            </div>
            <div style={{ fontSize: '1.25rem', fontWeight: '800', color: getTextColor(), letterSpacing: '0.02em' }}>
              {getHeadline()}
            </div>
            <div style={{ fontSize: '0.8rem', color: '#cbd5e1', marginTop: '0.4rem', lineHeight: '1.4' }}>
              {spoof.decision_explanation || (
                isLowRisk ? 'Model evidence did not cross the review threshold. Not a guarantee of human origin.' :
                isReview ? 'Spoof risk lies in the ambiguous margin [70.0, 75.0); secondary review required.' :
                'Synthetic or vocoder attack patterns identified across multiple detectors.'
              )}
            </div>
            {isDegraded && (
              <div style={{ marginTop: '0.4rem', fontSize: '0.72rem', color: '#fbbf24' }}>
                ⚠️ Running in Degraded Mode ({spoof.missing_models?.length || 1} model offline; weights re-normalized)
              </div>
            )}
          </div>

          {/* Composite Index & Threshold Range Meter */}
          <div style={{ background: 'rgba(30, 41, 59, 0.7)', padding: '0.85rem', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.15)', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
              <span style={{ fontSize: '0.78rem', textTransform: 'uppercase', color: '#94a3b8', fontWeight: 600 }}>
                Ensemble Spoof-Risk Index
              </span>
              <span style={{ fontSize: '1.15rem', fontWeight: '800', color: getTextColor() }}>
                {ensembleRisk != null ? `${ensembleRisk.toFixed(1)} / 100` : 'N/A'}
              </span>
            </div>

            {/* Threshold progress meter */}
            <div style={{ background: '#1e293b', height: '10px', borderRadius: '5px', overflow: 'hidden', position: 'relative', marginTop: '0.2rem' }}>
              <div
                style={{
                  height: '100%',
                  width: `${Math.min(100, Math.max(0, ensembleRisk || 0))}%`,
                  backgroundColor: isHighRisk ? '#ef4444' : isReview ? '#f59e0b' : '#22c55e',
                  transition: 'width 0.4s ease',
                }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#64748b', marginTop: '0.35rem' }}>
              <span>0 (Lower Risk)</span>
              <span>Review: {reviewThresh}</span>
              <span>High Risk: {highRiskThresh}</span>
              <span>100 (Max Risk)</span>
            </div>
            <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '0.3rem', fontStyle: 'italic' }}>
              * Provisional uncalibrated score index. Low risk indicates sub-threshold evidence, not certified human proof.
            </div>
          </div>

          {/* Multi-Model Breakdown Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '0.75rem' }}>
            {/* Model B: AASIST */}
            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '0.6rem', borderRadius: '8px', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
              <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Model B (AASIST)</div>
              <div style={{ fontSize: '0.95rem', fontWeight: '700', color: models.model_b?.spoof_score >= 50 ? '#f87171' : '#4ade80', marginTop: '0.1rem' }}>
                {models.model_b?.spoof_score != null ? `${models.model_b.spoof_score}% Spoof` : 'Unavailable'}
              </div>
              <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '0.15rem' }}>
                LLR: {models.model_b?.raw_score != null ? models.model_b.raw_score.toFixed(2) : 'N/A'}
              </div>
            </div>

            {/* Model C: Wav2Vec2 */}
            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '0.6rem', borderRadius: '8px', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
              <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Model C (Wav2Vec2)</div>
              <div style={{ fontSize: '0.95rem', fontWeight: '700', color: models.model_c?.spoof_score >= 50 ? '#f87171' : '#4ade80', marginTop: '0.1rem' }}>
                {models.model_c?.spoof_score != null ? `${models.model_c.spoof_score}% Spoof` : 'Unavailable'}
              </div>
              <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '0.15rem' }}>
                Margin: {models.model_c?.raw_score != null ? models.model_c.raw_score.toFixed(2) : 'N/A'}
              </div>
            </div>

            {/* Model A: W2V2-AASIST */}
            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '0.6rem', borderRadius: '8px', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
              <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Model A (W2V2-AASIST)</div>
              <div style={{ fontSize: '0.85rem', fontWeight: '600', color: models.model_a?.status === 'available' ? '#cbd5e1' : '#f59e0b', marginTop: '0.1rem' }}>
                {models.model_a?.status === 'available' ? `${models.model_a.spoof_score}% Spoof` : 'Uncached (~1.2GB)'}
              </div>
              <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '0.15rem' }}>
                Weight re-allocated
              </div>
            </div>

            {/* Heuristics */}
            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '0.6rem', borderRadius: '8px', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
              <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Signal Heuristics</div>
              <div style={{ fontSize: '0.95rem', fontWeight: '700', color: '#cbd5e1', marginTop: '0.1rem' }}>
                {heuristics.heuristic_risk_score != null ? `${heuristics.heuristic_risk_score}/100 Risk` : 'N/A'}
              </div>
              <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '0.15rem' }}>
                Flags: {heuristics.quality_flags?.length || 0}
              </div>
            </div>
          </div>

          {/* Heuristic Quality Flags */}
          {heuristics.quality_flags?.length > 0 && (
            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '0.5rem 0.75rem', borderRadius: '8px', fontSize: '0.72rem', color: '#94a3b8' }}>
              <strong>Acoustic Quality Indicators:</strong>{' '}
              {heuristics.quality_flags.map((flag) => (
                <span
                  key={flag}
                  style={{
                    display: 'inline-block',
                    background: flag.includes('SILENCE') || flag.includes('CLIPPING') ? 'rgba(239, 68, 68, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                    color: flag.includes('SILENCE') || flag.includes('CLIPPING') ? '#fca5a5' : '#93c5fd',
                    padding: '1px 6px',
                    borderRadius: '4px',
                    margin: '2px 4px 2px 0',
                    fontSize: '0.68rem',
                  }}
                >
                  {flag}
                </span>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div style={{ color: '#fcd34d', fontSize: '0.85rem', padding: '0.5rem 0' }}>
          <strong>Status: {decisionBand || status}</strong>
          <p style={{ margin: '0.4rem 0 0', color: '#cbd5e1' }}>
            {spoof.reason || 'Detector not ready or audio quality failed.'}
          </p>
        </div>
      )}
    </section>
  );
}
