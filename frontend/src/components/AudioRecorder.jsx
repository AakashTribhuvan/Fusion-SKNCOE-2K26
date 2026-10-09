import { useRef } from 'react';

export default function AudioRecorder({
  status,
  permissionState,
  duration,
  isRecording,
  audioUrl,
  audioFileName,
  onStart,
  onStop,
  onReplay,
  onReset,
  onSubmit,
  onFileSelect,
  submitting,
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && onFileSelect) {
      onFileSelect(file);
    }
  };

  return (
    <section className="card recorder-card">
      <div className="section-header">
        <span className="eyebrow">Audio Input & File Upload</span>
        <span className={`pill ${audioUrl ? 'pass' : permissionState === 'granted' ? 'pass' : permissionState === 'denied' ? 'fail' : 'incomplete'}`}>
          {audioFileName ? `File: ${audioFileName}` : permissionState === 'granted' ? 'Microphone Ready' : permissionState === 'denied' ? 'Permission Denied' : 'Awaiting Input'}
        </span>
      </div>

      <div className="timer-box">{duration}s</div>
      <div className="status-line">{status}</div>

      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        accept="audio/*,.wav,.mp3,.mpeg,.m4a,.ogg,.webm"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      <div className="button-row">
        {!isRecording ? (
          <button onClick={onStart}>Start Recording</button>
        ) : (
          <button className="warn" onClick={onStop}>Stop Recording</button>
        )}
        <button
          type="button"
          className="secondary"
          onClick={() => fileInputRef.current?.click()}
          disabled={isRecording}
        >
          📂 Upload Audio File
        </button>
        <button className="secondary" onClick={onReplay} disabled={!audioUrl}>Playback Audio</button>
        <button className="secondary" onClick={onReset}>Clear / Reset</button>
        <button className="primary" onClick={onSubmit} disabled={submitting || !audioUrl}>Submit for Verification</button>
      </div>

      {audioUrl && (
        <div style={{ marginTop: '1rem' }}>
          {audioFileName && (
            <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>
              Loaded File: <strong style={{ color: '#e2e8f0' }}>{audioFileName}</strong>
            </div>
          )}
          <audio controls src={audioUrl} className="audio-player" style={{ width: '100%' }} />
        </div>
      )}
    </section>
  );
}
