export default function SynchronizationPanel({ result }) {
  const sync = result?.audio_video_sync || {};
  return (
    <section className="card">
      <div className="section-header">
        <span className="eyebrow">Audio/video synchronization</span>
      </div>
      <div className="status-row">
        <strong>Status:</strong> <span className={`pill ${sync.status || 'incomplete'}`}>{sync.status || 'incomplete'}</span>
      </div>
      <ul>
        <li>Timing offset: {sync.timing_offset_seconds ?? 'N/A'} s</li>
        <li>Alignment correlation: {sync.alignment_correlation ?? 'N/A'}</li>
        <li>Evidence quality: {sync.evidence_quality || 'N/A'}</li>
        <li>Reason: {sync.reason || 'Not available yet'}</li>
      </ul>
      {sync.status === 'completed' && (
        <p className="notice">Experimental timing evidence only. A mismatch does not prove manipulation.</p>
      )}
    </section>
  );
}
