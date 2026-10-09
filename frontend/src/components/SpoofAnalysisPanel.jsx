export default function SpoofAnalysisPanel({ result }) {
  const spoof = result?.spoof_detection || {};
  return (
    <section className="card">
      <div className="section-header">
        <span className="eyebrow">Spoof detector</span>
      </div>
      <div className="status-row">
        <strong>Status:</strong> <span className={`pill ${spoof.status || 'incomplete'}`}>{spoof.status || 'incomplete'}</span>
      </div>
      <ul>
        <li>Model: {spoof.model_name || 'N/A'}</li>
        <li>Version: {spoof.model_version || 'N/A'}</li>
        <li>Predicted class: {spoof.predicted_class || 'N/A'}</li>
        <li>Genuine score: {spoof.genuine_score ?? 'N/A'}</li>
        <li>Spoof score: {spoof.spoof_score ?? 'N/A'}</li>
        <li>Threshold: {spoof.threshold ?? 'N/A'}</li>
        <li>Reason: {spoof.reason || 'No detector output yet'}</li>
      </ul>
      <p className="notice">Audio spoof signal only—not speaker identity or a final user verdict. Scores are not calibrated.</p>
    </section>
  );
}
