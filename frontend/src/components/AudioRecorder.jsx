import { useRef, useState, useEffect } from 'react';

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
  const [isDragOver, setIsDragOver] = useState(false);
  const [showPasteModal, setShowPasteModal] = useState(false);
  const [pasteText, setPasteText] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && onFileSelect) {
      onFileSelect(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && onFileSelect) {
      onFileSelect(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const processPastedData = async (dataTransfer) => {
    // 1. Check for files
    const items = dataTransfer.items ? Array.from(dataTransfer.items) : [];
    for (const item of items) {
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file && (file.type.startsWith('audio/') || /\.(wav|wave|webm|ogg|oga|mp4|m4a|mpeg|mp3|aac|flac)$/i.test(file.name || ''))) {
          const renamedFile = new File([file], file.name || 'pasted-audio.wav', { type: file.type || 'audio/wav' });
          onFileSelect(renamedFile);
          return true;
        }
      }
    }

    if (dataTransfer.files && dataTransfer.files.length > 0) {
      const file = dataTransfer.files[0];
      if (file.type.startsWith('audio/') || /\.(wav|wave|webm|ogg|oga|mp4|m4a|mpeg|mp3|aac|flac)$/i.test(file.name || '')) {
        onFileSelect(file);
        return true;
      }
    }

    // 2. Check for text URL or Base64 Data URI
    const text = dataTransfer.getData ? dataTransfer.getData('text')?.trim() : '';
    if (text) {
      if (text.startsWith('data:audio/')) {
        try {
          const res = await fetch(text);
          const blob = await res.blob();
          const file = new File([blob], 'pasted-audio-data.wav', { type: blob.type || 'audio/wav' });
          onFileSelect(file);
          return true;
        } catch (err) {
          console.warn('Failed to parse pasted data URI:', err);
        }
      } else if (text.startsWith('http://') || text.startsWith('https://')) {
        try {
          const res = await fetch(text);
          const blob = await res.blob();
          const filename = text.split('/').pop().split('?')[0] || 'pasted-online-audio.wav';
          const file = new File([blob], filename, { type: blob.type || 'audio/wav' });
          onFileSelect(file);
          return true;
        } catch (err) {
          console.warn('Failed to fetch pasted audio URL:', err);
        }
      }
    }
    return false;
  };

  const handlePasteButtonClick = async () => {
    try {
      if (navigator.clipboard && navigator.clipboard.read) {
        const clipboardItems = await navigator.clipboard.read();
        for (const item of clipboardItems) {
          for (const type of item.types) {
            if (type.startsWith('audio/')) {
              const blob = await item.getType(type);
              const file = new File([blob], 'clipboard-audio.wav', { type });
              onFileSelect(file);
              return;
            }
          }
        }
      }
      if (navigator.clipboard && navigator.clipboard.readText) {
        const text = await navigator.clipboard.readText();
        if (text && (text.startsWith('data:audio/') || text.startsWith('http://') || text.startsWith('https://'))) {
          const res = await fetch(text);
          const blob = await res.blob();
          const file = new File([blob], 'pasted-audio.wav', { type: blob.type || 'audio/wav' });
          onFileSelect(file);
          return;
        }
      }
    } catch (err) {
      console.warn('Direct API clipboard access required fallback:', err);
    }
    setShowPasteModal(true);
  };

  const handleManualModalPaste = async () => {
    if (!pasteText.trim()) return;
    const text = pasteText.trim();
    if (text.startsWith('data:audio/') || text.startsWith('http://') || text.startsWith('https://')) {
      try {
        const res = await fetch(text);
        const blob = await res.blob();
        const file = new File([blob], 'pasted-audio.wav', { type: blob.type || 'audio/wav' });
        onFileSelect(file);
        setShowPasteModal(false);
        setPasteText('');
      } catch (err) {
        alert(`Failed to load audio: ${err.message}`);
      }
    } else {
      alert('Please paste a valid Audio Data URI (data:audio/...) or direct HTTP audio URL.');
    }
  };

  return (
    <section className="card recorder-card">
      <div className="section-header">
        <span className="eyebrow">Audio Input & Copy-Paste</span>
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

      {/* Drop & Paste zone */}
      <div
        className={`dropzone ${isDragOver ? 'drag-over' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onPaste={(e) => processPastedData(e.clipboardData)}
        tabIndex={0}
      >
        <span>🎯 Drop audio file here, or click anywhere on page & press <kbd>Ctrl</kbd> + <kbd>V</kbd> to Paste</span>
      </div>

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
        <button
          type="button"
          className="secondary"
          onClick={handlePasteButtonClick}
          disabled={isRecording}
          title="Paste copied audio file or audio URL from clipboard (Ctrl+V)"
        >
          📋 Paste Audio (Ctrl+V)
        </button>
        <button className="secondary" onClick={onReplay} disabled={!audioUrl}>Playback Audio</button>
        <button className="secondary" onClick={onReset}>Clear / Reset</button>
        <button className="primary" onClick={onSubmit} disabled={submitting || !audioUrl}>Submit for Verification</button>
      </div>

      {audioUrl && (
        <div style={{ marginTop: '1rem' }}>
          {audioFileName && (
            <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>
              Loaded Audio: <strong style={{ color: '#e2e8f0' }}>{audioFileName}</strong>
            </div>
          )}
          <audio controls src={audioUrl} className="audio-player" style={{ width: '100%' }} />
        </div>
      )}

      {showPasteModal && (
        <div className="paste-modal-backdrop" onClick={() => setShowPasteModal(false)}>
          <div className="paste-modal" onClick={(e) => e.stopPropagation()}>
            <h3>📋 Paste Audio Content</h3>
            <p>Press <kbd>Ctrl</kbd> + <kbd>V</kbd> inside the box below or paste an audio URL / Base64 Data URI:</p>
            <textarea
              autoFocus
              rows={4}
              placeholder="Press Ctrl+V here to paste copied audio file, data URI, or audio URL..."
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              onPaste={async (e) => {
                const handled = await processPastedData(e.clipboardData);
                if (handled) {
                  setShowPasteModal(false);
                  setPasteText('');
                }
              }}
            />
            <div className="modal-actions">
              <button className="secondary" onClick={() => setShowPasteModal(false)}>Cancel</button>
              <button className="primary" onClick={handleManualModalPaste}>Load Audio</button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

