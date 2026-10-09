export default function AudioQualityPanel({ result }) {
  const quality = result?.audio_quality || {};
  return (
    <section className="card">
      <div className="section-header">
        <span className="eyebrow">Audio quality</span>
      </div>
      <div className="status-row">
        <strong>Status:</strong> <span className={`pill ${quality.status || 'incomplete'}`}>{quality.status || 'incomplete'}</span>
      </div>
      <ul>
        <li>Duration: {quality.duration_seconds ?? 'N/A'} s</li>
        <li>Sample rate: {quality.sample_rate ?? 'N/A'} Hz</li>
        <li>Speech usable: {quality.speech_usable !== undefined ? String(quality.speech_usable) : 'N/A'}</li>
        <li>Reason: {quality.reason || 'Not available yet'}</li>
      </ul>
    </section>
  );
}
