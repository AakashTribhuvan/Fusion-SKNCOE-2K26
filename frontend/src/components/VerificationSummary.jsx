export default function VerificationSummary({ result }) {
  const riskCategory = result?.risk_category;
  const overallStatus = result?.overall_status;
  const reasons = result?.reasons || [];

  const getPillClass = () => {
    if (overallStatus === 'pass') return 'pass';
    if (overallStatus === 'fail') return 'fail';
    if (overallStatus === 'review_required') return 'suspicious';
    return 'incomplete';
  };

  return (
    <section className="card summary-card">
      <div className="section-header">
        <span className="eyebrow">Overall Session Risk Assessment</span>
        {overallStatus && <span className={`pill ${getPillClass()}`}>{overallStatus}</span>}
      </div>

      {result ? (
        <>
          <div className="status-row" style={{ marginBottom: '0.5rem' }}>
            <strong>Risk Assessment:</strong>{' '}
            <span style={{
              fontWeight: 700,
              color: riskCategory === 'LOW_RISK' ? '#4ade80' : riskCategory === 'HIGH_RISK' ? '#f87171' : '#fcd34d'
            }}>
              {riskCategory || 'INDETERMINATE'}
            </span>
          </div>

          <div className="status-row" style={{ fontSize: '0.85rem', marginBottom: '0.35rem' }}>
            <strong>Processing Duration:</strong> {result.processing_time_ms ?? 'N/A'} ms
          </div>

          <div className="status-row" style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>
            <strong>Usable Speech:</strong>{' '}
            {result.speech_detection?.speech_detected ? 'Yes (VAD confirmed)' : 'No / Low Signal'}
          </div>

          {reasons.length > 0 && (
            <div style={{ marginTop: '0.75rem', background: 'rgba(15, 23, 42, 0.6)', padding: '0.6rem 0.8rem', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', color: '#94a3b8', marginBottom: '0.25rem' }}>
                Rule Findings:
              </div>
              <ul style={{ margin: 0, paddingLeft: '1.1rem', fontSize: '0.78rem', color: '#cbd5e1' }}>
                {reasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      ) : (
        <p style={{ color: '#64748b', fontSize: '0.85rem' }}>Awaiting audio recording and submission.</p>
      )}
    </section>
  );
}
