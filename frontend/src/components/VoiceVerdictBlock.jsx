export default function VoiceVerdictBlock({ result }) {
  const spoof = result?.spoof_detection || {};
  const status = spoof.status;
  const predictedClass = spoof.predicted_class; // 'genuine' | 'spoof'
  const genuineScore = spoof.genuine_score;
  const spoofScore = spoof.spoof_score;

  // Format score nicely as percentage and raw float
  const formatScore = (val) => {
    if (val === null || val === undefined) return 'N/A';
    return `${(val * 100).toFixed(1)}% (${Number(val).toFixed(4)})`;
  };

  // Determine state
  const isHuman = predictedClass === 'genuine';
  const isAi = predictedClass === 'spoof';

  return (
    <section className="card" style={{ border: isHuman ? '1px solid rgba(34, 197, 94, 0.4)' : isAi ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid rgba(148, 163, 184, 0.2)' }}>
      <div className="section-header">
        <span className="eyebrow">Voice Classification</span>
        {status && (
          <span className={`pill ${isHuman ? 'pass' : isAi ? 'fail' : 'incomplete'}`}>
            {status}
          </span>
        )}
      </div>

      {!result ? (
        <div style={{ color: '#94a3b8', fontSize: '0.95rem', padding: '0.5rem 0' }}>
          Record and submit audio to verify whether it is a human voice or AI generated.
        </div>
      ) : status === 'completed' ? (
        <div>
          {/* Main Verdict Text Block */}
          <div
            style={{
              padding: '1rem',
              borderRadius: '12px',
              backgroundColor: isHuman ? 'rgba(34, 197, 94, 0.12)' : 'rgba(239, 68, 68, 0.12)',
              border: `1px solid ${isHuman ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
              marginBottom: '1rem',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: isHuman ? '#86efac' : '#fca5a5', marginBottom: '0.25rem' }}>
              Result
            </div>
            <div
              style={{
                fontSize: '1.6rem',
                fontWeight: '800',
                color: isHuman ? '#4ade80' : '#f87171',
                letterSpacing: '0.02em',
              }}
            >
              {isHuman ? '🟢 HUMAN VOICE' : '🔴 AI GENERATED (DEEPFAKE)'}
            </div>
          </div>

          {/* Text-based Scores */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '0.75rem' }}>
            <div
              style={{
                background: 'rgba(30, 41, 59, 0.7)',
                padding: '0.75rem',
                borderRadius: '10px',
                border: '1px solid rgba(148, 163, 184, 0.15)',
              }}
            >
              <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>Human Score</div>
              <div style={{ fontSize: '1.25rem', fontWeight: '700', color: isHuman ? '#4ade80' : '#e2e8f0', marginTop: '0.25rem' }}>
                {formatScore(genuineScore)}
              </div>
            </div>

            <div
              style={{
                background: 'rgba(30, 41, 59, 0.7)',
                padding: '0.75rem',
                borderRadius: '10px',
                border: '1px solid rgba(148, 163, 184, 0.15)',
              }}
            >
              <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>AI / Deepfake Score</div>
              <div style={{ fontSize: '1.25rem', fontWeight: '700', color: isAi ? '#f87171' : '#e2e8f0', marginTop: '0.25rem' }}>
                {formatScore(spoofScore)}
              </div>
            </div>
          </div>

          <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
            Detector: <span style={{ color: '#94a3b8' }}>{spoof.model_name || 'AASIST'}</span>
            {spoof.processing_duration_ms ? ` · Processed in ${spoof.processing_duration_ms} ms` : ''}
          </div>
        </div>
      ) : (
        <div style={{ color: '#fcd34d', fontSize: '0.9rem', padding: '0.5rem 0' }}>
          <strong>Status: {status}</strong>
          <p style={{ margin: '0.4rem 0 0', color: '#cbd5e1' }}>{spoof.reason || 'Detector not ready or audio quality failed.'}</p>
        </div>
      )}
    </section>
  );
}
