export default function AudioRecorder({
  status,
  permissionState,
  duration,
  isRecording,
  audioUrl,
  onStart,
  onStop,
  onReplay,
  onReset,
  onSubmit,
  submitting,
}) {
  return (
    <section className="card recorder-card">
      <div className="section-header">
        <span className="eyebrow">Microphone input</span>
        <span className={`pill ${permissionState === 'granted' ? 'pass' : permissionState === 'denied' ? 'fail' : 'incomplete'}`}>
          {permissionState === 'granted' ? 'Permission granted' : permissionState === 'denied' ? 'Permission denied' : 'Awaiting permission'}
        </span>
      </div>

      <div className="timer-box">{duration}s</div>
      <div className="status-line">{status}</div>

      <div className="button-row">
        {!isRecording ? (
          <button onClick={onStart}>Start Recording</button>
        ) : (
          <button className="warn" onClick={onStop}>Stop Recording</button>
        )}
        <button className="secondary" onClick={onReplay} disabled={!audioUrl}>Playback Recording</button>
        <button className="secondary" onClick={onReset}>Record Again</button>
        <button className="primary" onClick={onSubmit} disabled={submitting || !audioUrl}>Submit for Verification</button>
      </div>

      {audioUrl && (
        <audio controls src={audioUrl} className="audio-player" />
      )}
    </section>
  );
}
