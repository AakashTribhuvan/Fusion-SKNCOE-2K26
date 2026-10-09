export default function VerificationSummary({ result }) {
  return (
    <section className="card summary-card">
      <div className="section-header">
        <span className="eyebrow">Verification summary</span>
      </div>
      {result ? (
        <>
          <div className="status-row">
            <strong>Overall checks:</strong> <span className={`pill ${result.overall_status || 'incomplete'}`}>{result.overall_status}</span>
          </div>
          <div className="status-row">
            <strong>Processing time:</strong> {result.processing_time_ms ?? 'N/A'} ms
          </div>
          <div className="status-row">
            <strong>Speech detected:</strong> {result.speech_detection?.speech_detected !== undefined ? String(result.speech_detection.speech_detected) : 'N/A'}
          </div>
        </>
      ) : (
        <p>No verification result yet.</p>
      )}
    </section>
  );
}
