import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Activity, ArrowUpRight, FileVideo, LoaderCircle, ShieldAlert, UploadCloud, X } from 'lucide-react';
import './deepfake.css';

const API_ORIGIN = (import.meta.env.VITE_SWR_API_ORIGIN || 'http://localhost:8000').replace(/\/$/, '');
const API_BASE = `${API_ORIGIN}/api`;
const ACCEPTED_VIDEO = /\.(mp4|mov|avi|mkv|webm)$/i;

function makeId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatTime(seconds) {
  const value = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`;
}

function getOutcome(item) {
  if (item?.error) return {
    tone: 'error',
    label: 'SCAN UNAVAILABLE',
    title: 'The recording could not be examined',
    detail: item.error,
  };

  const result = item?.result;
  const analysis = result?.ai_analysis;
  const analyzed = Number(analysis?.frames_analyzed) || 0;
  const flagged = Number(analysis?.frames_flagged) || 0;

  if (!result || analyzed === 0) return {
    tone: 'inconclusive',
    label: 'INCONCLUSIVE',
    title: 'No video frames received an AI score',
    detail: 'Swaraksha only runs its image classifier on frames matching a registered identity. This recording was not scored, so the result says nothing about whether it is synthetic.',
  };

  if (result.final_status === 'POTENTIAL_AI_MANIPULATION' || flagged > 0 && result.final_status === 'REVIEW_REQUIRED') return {
    tone: 'flagged',
    label: result.final_status === 'POTENTIAL_AI_MANIPULATION' ? 'SIGNAL DETECTED' : 'REVIEW',
    title: result.final_status === 'POTENTIAL_AI_MANIPULATION' ? 'Potential synthetic-media signal' : 'Some frames need review',
    detail: `${flagged} of ${analyzed} analyzed frames were flagged by the image classifier. Treat this as screening evidence, not a definitive finding.`,
  };

  if (analysis?.status === 'NO_STRONG_AI_EVIDENCE') return {
    tone: 'clear',
    label: 'NO STRONG SIGNAL',
    title: 'The sampled frames were not flagged',
    detail: `The classifier did not flag ${analyzed} analyzed frame${analyzed === 1 ? '' : 's'}. Sampling can miss brief artifacts; this does not establish that the recording is authentic.`,
  };

  return {
    tone: 'inconclusive',
    label: 'INCONCLUSIVE',
    title: 'The backend did not return a complete analysis',
    detail: result.summary || 'No definitive video-level conclusion is available.',
  };
}

function App() {
  const [backend, setBackend] = useState('checking');
  const [queue, setQueue] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [working, setWorking] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [transfer, setTransfer] = useState({ percent: 0, phase: '' });
  const [notice, setNotice] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);

  const selected = queue.find((item) => item.id === selectedId) || queue[0] || null;
  const outcome = selected?.result || selected?.error ? getOutcome(selected) : null;

  useEffect(() => {
    const controller = new AbortController();
    axios.get(API_ORIGIN, { timeout: 4000, signal: controller.signal })
      .then(() => setBackend('online'))
      .catch(() => { if (!controller.signal.aborted) setBackend('offline'); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selected?.file) {
      setPreviewUrl('');
      return undefined;
    }
    const url = URL.createObjectURL(selected.file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selected?.file]);

  const addFiles = (fileList) => {
    const valid = [...fileList].filter((file) => file.type.startsWith('video/') || ACCEPTED_VIDEO.test(file.name));
    const invalidCount = fileList.length - valid.length;
    if (invalidCount) setNotice(`${invalidCount} unsupported file${invalidCount === 1 ? '' : 's'} skipped. Use MP4, MOV, AVI, MKV, or WebM.`);
    else setNotice('');
    if (!valid.length) return;

    const next = valid.map((file) => ({ id: makeId(), file, status: 'queued', result: null, error: '' }));
    setQueue((current) => [...current, ...next]);
    setSelectedId((current) => current || next[0].id);
  };

  const removeFile = (id) => {
    if (working) return;
    const next = queue.filter((item) => item.id !== id);
    setQueue(next);
    if (selectedId === id) setSelectedId(next[0]?.id || null);
  };

  const clearFinished = () => {
    if (working) return;
    const next = queue.filter((item) => item.status === 'queued');
    setQueue(next);
    setSelectedId(next[0]?.id || null);
  };

  const examineQueue = async () => {
    if (working) return;
    const pending = queue.filter((item) => item.status === 'queued');
    if (!pending.length) return;

    setWorking(true);
    setNotice('');
    for (const item of pending) {
      setSelectedId(item.id);
      setTransfer({ percent: 0, phase: 'Preparing upload' });
      setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, status: 'uploading' } : entry));
      const body = new FormData();
      body.append('file', item.file);

      try {
        const response = await axios.post(`${API_BASE}/scan-video`, body, {
          timeout: 0,
          onUploadProgress: (event) => {
            const percent = event.total ? Math.round((event.loaded / event.total) * 100) : 0;
            setTransfer({ percent, phase: percent >= 100 ? 'Examining sampled frames' : 'Sending recording' });
            setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, status: percent >= 100 ? 'analyzing' : 'uploading' } : entry));
          },
        });
        setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, status: 'complete', result: response.data } : entry));
      } catch (error) {
        const message = error.response?.data?.detail || (error.code === 'ERR_NETWORK'
          ? 'Swaraksha is not reachable. Start the backend at localhost:8000 and try again.'
          : 'The backend could not examine this recording.');
        setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, status: 'error', error: message } : entry));
      }
    }
    setWorking(false);
    setTransfer({ percent: 0, phase: '' });
  };

  const seekToFrame = (timestamp) => {
    if (videoRef.current && Number.isFinite(Number(timestamp))) {
      videoRef.current.currentTime = Number(timestamp);
      videoRef.current.pause();
    }
  };

  const statusCopy = backend === 'online' ? 'SWARAKSHA CONNECTED' : backend === 'checking' ? 'CHECKING SERVICE' : 'SERVICE OFFLINE';

  return (
    <div className="forensic-app">
      <header className="masthead">
        <a className="wordmark" href="#top" aria-label="Frame / home">
          <span className="wordmark-symbol" aria-hidden="true"><i /><i /><i /></span>
          <span>FRAME<span className="wordmark-slash">/</span>CHECK</span>
        </a>
        <div className="masthead-center">INDEPENDENT VIDEO SCREENING <span>·</span> FIELD DESK 01</div>
        <div className={`service-status ${backend}`}><span className="service-dot" />{statusCopy}<small>LOCAL API · 8000</small></div>
      </header>

      <main id="top" className="page-frame">
        <section className="intro-row">
          <div className="intro-copy">
            <p className="eyebrow"><span>01</span> VIDEO FORENSICS / SWARAKSHA ENGINE</p>
            <h1>Look closer.<br /><em>Keep the evidence.</em></h1>
            <p className="intro-deck">Examine a recording for frame-level synthetic-media signals. Every score is tied to a sampled moment; missing analysis stays visible.</p>
          </div>
          <aside className="scope-note">
            <span className="scope-index">SCOPE NOTE / 01</span>
            <p>The connected model checks image crops from registered identities. If no frame is scored, this desk reports <strong>inconclusive</strong>, not authentic.</p>
            <a href="https://github.com/AakashTribhuvan/Swaraksha/blob/main/VIDEO_PIPELINE.md" target="_blank" rel="noreferrer">PIPELINE NOTES <ArrowUpRight size={13} /></a>
          </aside>
        </section>

        <section className="exam-layout" aria-label="Video examination">
          <aside className="intake-column">
            <div className="section-heading"><span>02 / SOURCE</span><span>01</span></div>
            <button
              className={`dropzone ${dragging ? 'is-dragging' : ''}`}
              type="button"
              onClick={() => fileInputRef.current?.click()}
              onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }}
              onDrop={(event) => { event.preventDefault(); setDragging(false); addFiles(event.dataTransfer.files); }}
              aria-label="Choose or drop video recordings"
            >
              <span className="drop-mark"><UploadCloud size={22} strokeWidth={1.4} /></span>
              <strong>Bring a recording</strong>
              <span>Drop it here or browse files</span>
              <small>MP4 · MOV · AVI · MKV · WEBM</small>
            </button>
            <input ref={fileInputRef} className="visually-hidden" type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm" multiple onChange={(event) => { addFiles(event.target.files); event.target.value = ''; }} />

            <div className="queue-heading"><span>EXAM QUEUE</span><span>{String(queue.length).padStart(2, '0')}</span></div>
            {queue.length ? (
              <ul className="file-queue">
                {queue.map((item, index) => (
                  <li key={item.id} className={selected?.id === item.id ? 'selected' : ''}>
                    <button className="queue-select" type="button" onClick={() => setSelectedId(item.id)} aria-current={selected?.id === item.id ? 'true' : undefined}>
                      <span className="queue-number">{String(index + 1).padStart(2, '0')}</span>
                      <span className="queue-name">{item.file.name}<small>{formatBytes(item.file.size)} <i>·</i> {item.status === 'queued' ? 'QUEUED' : item.status === 'uploading' ? 'UPLOADING' : item.status === 'analyzing' ? 'ANALYZING' : item.status === 'complete' ? 'EXAMINED' : 'ERROR'}</small></span>
                    </button>
                    <button className="remove-file" type="button" onClick={() => removeFile(item.id)} disabled={working} aria-label={`Remove ${item.file.name}`}><X size={15} /></button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="queue-empty">No recordings queued.<br />Your files stay local until you start an examination.</p>
            )}

            <div className="queue-actions">
              <button className="examine-button" type="button" onClick={examineQueue} disabled={!queue.some((item) => item.status === 'queued') || working || backend === 'offline'}>
                {working ? <LoaderCircle className="spin" size={16} /> : <Activity size={16} />}
                {working ? 'EXAMINATION RUNNING' : `EXAMINE ${queue.filter((item) => item.status === 'queued').length || ''} RECORDING${queue.filter((item) => item.status === 'queued').length === 1 ? '' : 'S'}`}
              </button>
              {queue.some((item) => item.status !== 'queued') && <button className="clear-button" type="button" onClick={clearFinished} disabled={working}>CLEAR FINISHED</button>}
            </div>
            {working && (
              <div className="transfer-state" role="status" aria-live="polite">
                <div><span>{transfer.phase || 'Preparing upload'}</span><b>{transfer.phase === 'Sending recording' ? `${transfer.percent}%` : '· · ·'}</b></div>
                <div className="transfer-track"><i style={{ width: transfer.phase === 'Sending recording' ? `${transfer.percent}%` : '100%' }} /></div>
              </div>
            )}
            {notice && <p className="inline-notice" role="status">{notice}</p>}
          </aside>

          <section className="evidence-column" aria-live="polite">
            <div className="section-heading"><span>03 / EXAMINATION</span><span>{selected?.result ? 'REPORT' : 'LIVE FILE'}</span></div>
            <div className="viewer">
              {previewUrl ? (
                <video ref={videoRef} key={selected?.id} src={previewUrl} controls playsInline preload="metadata" aria-label={`Preview of ${selected?.file.name}`} />
              ) : (
                <div className="viewer-empty">
                  <span className="viewer-crosshair" aria-hidden="true"><i /><i /><i /><i /></span>
                  <FileVideo size={26} strokeWidth={1.2} />
                  <strong>THE VIEWER IS EMPTY</strong>
                  <span>Select a recording to inspect its sampled frames.</span>
                  <span className="viewer-code">NO SOURCE / NO INFERENCE</span>
                </div>
              )}
              {selected && <div className="viewer-caption"><span>{selected.file.name}</span><span>{formatBytes(selected.file.size)}</span></div>}
            </div>

            {selected ? (
              <>
                {outcome && <OutcomePanel outcome={outcome} result={selected.result} />}
                {selected.result && <AnalysisReport result={selected.result} onSeek={seekToFrame} />}
                {!selected.result && !selected.error && (
                  <div className="awaiting-report"><span>{selected.status === 'queued' ? 'READY WHEN YOU ARE' : selected.status === 'uploading' ? 'TRANSFER IN PROGRESS' : 'MODEL ANALYSIS IN PROGRESS'}</span><p>{selected.status === 'queued' ? 'Choose Examine to send this recording to the local Swaraksha API.' : 'The backend samples frames and runs its image classifier on eligible crops.'}</p></div>
                )}
              </>
            ) : (
              <div className="awaiting-report"><span>EXAMINATION RECORD</span><p>Frame results, model scores, and file metadata will appear here after a recording is examined.</p></div>
            )}
          </section>
        </section>

        <footer className="page-footer">
          <span>FRAME/CHECK <i>·</i> LOCAL FORENSIC WORKSPACE</span>
          <span>MODEL OUTPUT IS A SCREENING SIGNAL, NOT PROOF OF IDENTITY OR FRAUD.</span>
          <a href="https://github.com/AakashTribhuvan/Swaraksha" target="_blank" rel="noreferrer">SWARAKSHA API <ArrowUpRight size={12} /></a>
        </footer>
      </main>
    </div>
  );
}

function OutcomePanel({ outcome, result }) {
  const analysis = result?.ai_analysis;
  const hasScore = Number(analysis?.frames_analyzed) > 0 && Number.isFinite(Number(analysis?.aggregate_score));
  const score = hasScore ? Math.round(Number(analysis.aggregate_score) * 100) : null;
  return (
    <section className={`outcome-panel ${outcome.tone}`}>
      <div className="outcome-main">
        <p className="outcome-label"><span />{outcome.label}</p>
        <h2>{outcome.title}</h2>
        <p>{outcome.detail}</p>
      </div>
      <div className="score-block">
        {hasScore ? <><span className="score-value">{score}<small>%</small></span><span className="score-caption">AGGREGATE MODEL SCORE<br />NOT A CALIBRATED PROBABILITY</span></> : <><ShieldAlert size={24} strokeWidth={1.4} /><span className="score-caption">{outcome.tone === 'inconclusive' ? 'NO SCORE' : 'REVIEW REQUIRED'}</span></>}
      </div>
    </section>
  );
}

function AnalysisReport({ result, onSeek }) {
  const analysis = result.ai_analysis || {};
  const frames = Array.isArray(result.frames) ? result.frames : [];
  const analyzed = Number(analysis.frames_analyzed) || 0;
  const flagged = Number(analysis.frames_flagged) || 0;
  const layers = [
    { name: 'Sampled-frame classifier', state: analyzed ? `${analyzed} FRAMES SCORED` : 'NOT RUN', detail: 'Image model on eligible face crops' },
    { name: 'File metadata', state: result.metadata_forensics?.confidence ? result.metadata_forensics.confidence.toUpperCase() : 'UNAVAILABLE', detail: 'Container metadata markers only' },
    { name: 'Depth / occlusion consistency', state: 'NOT MEASURED', detail: 'No depth-mask output in this API' },
    { name: 'Object persistence / temporal cues', state: 'NOT MEASURED', detail: 'No object tracking output in this API' },
  ];

  return (
    <div className="report-body">
      <div className="metric-strip">
        <Metric label="SAMPLED" value={result.video?.sampled_frames ?? '—'} suffix="frames" />
        <Metric label="MODEL SCORED" value={analyzed} suffix="frames" />
        <Metric label="FLAGGED" value={flagged} suffix={`of ${analyzed || '—'}`} />
        <Metric label="DURATION" value={formatTime(result.video?.duration)} suffix="min:sec" />
      </div>

      <section className="evidence-section">
        <div className="subsection-heading"><span>04 / SIGNAL REGISTER</span><span>AVAILABLE ≠ VERIFIED</span></div>
        <div className="signal-register">
          {layers.map((layer, index) => (
            <div className={`signal-row ${layer.state === 'NOT MEASURED' || layer.state === 'UNAVAILABLE' || layer.state === 'NOT RUN' ? 'unavailable' : ''}`} key={layer.name}>
              <span className="signal-index">0{index + 1}</span>
              <span className="signal-copy"><strong>{layer.name}</strong><small>{layer.detail}</small></span>
              <span className="signal-state">{layer.state}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="evidence-section frame-section">
        <div className="subsection-heading"><span>05 / FRAME TRACE</span><span>{frames.length ? `${frames.length} SAMPLES` : 'NO SAMPLES'}</span></div>
        {frames.length ? (
          <>
            <div className="frame-rail" aria-label="Sampled frame results">
              {frames.map((frame, index) => {
                const frameAnalysis = frame.ai_analysis || {};
                const frameState = !frameAnalysis.performed ? 'not-scored' : frameAnalysis.result === 'AI_GENERATED' ? 'flagged' : 'scored';
                return (
                  <button className={`frame-tick ${frameState}`} key={`${frame.frame_number}-${index}`} type="button" onClick={() => onSeek(frame.timestamp)} title={`Frame ${frame.frame_number}, ${Number(frame.timestamp).toFixed(1)} seconds: ${frameAnalysis.performed ? `${frameAnalysis.result || 'scored'}, model score ${Number(frameAnalysis.score || 0).toFixed(2)}` : 'not scored by backend'}`} aria-label={`Seek to frame ${frame.frame_number} at ${Number(frame.timestamp).toFixed(1)} seconds`}>
                    <span className="tick-mark" /><small>{formatTime(frame.timestamp)}</small>
                  </button>
                );
              })}
            </div>
            <div className="frame-key"><span><i className="key-scored" /> SCORED</span><span><i className="key-flagged" /> FLAGGED</span><span><i className="key-missed" /> NOT SCORED</span></div>
            <div className="frame-ledger">
              {frames.map((frame, index) => {
                const frameAnalysis = frame.ai_analysis || {};
                const performed = Boolean(frameAnalysis.performed);
                const isFlagged = frameAnalysis.result === 'AI_GENERATED';
                const skippedByBackend = frameAnalysis.reason === 'NO_PROTECTED_IDENTITY';
                const scoreDetail = performed && Number.isFinite(Number(frameAnalysis.score))
                  ? `SCORE ${Number(frameAnalysis.score).toFixed(3)}`
                  : skippedByBackend
                    ? 'Classifier skipped this sample'
                    : frameAnalysis.error || frameAnalysis.reason || 'No classifier output';
                return (
                  <button className="ledger-row" type="button" key={`${frame.frame_number}-ledger-${index}`} onClick={() => onSeek(frame.timestamp)}>
                    <span className="ledger-time">{formatTime(frame.timestamp)}</span>
                    <span className={`ledger-result ${!performed ? 'muted' : isFlagged ? 'flagged' : 'scored'}`}>{!performed ? 'NOT SCORED' : isFlagged ? 'FLAGGED BY MODEL' : 'NOT FLAGGED'}</span>
                    <span className="ledger-score">{scoreDetail}</span>
                    <span className="ledger-arrow">↗</span>
                  </button>
                );
              })}
            </div>
          </>
        ) : <p className="empty-evidence">The API returned no sampled-frame records.</p>}
      </section>

      {result.metadata_forensics && <MetadataEvidence metadata={result.metadata_forensics} />}
      <p className="model-caveat">Swaraksha samples video at intervals and applies an image classifier to eligible crops. Short-lived artifacts between samples may be missed. A model score is not a calibrated probability.</p>
    </div>
  );
}

function Metric({ label, value, suffix }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong><small>{suffix}</small></div>;
}

function MetadataEvidence({ metadata }) {
  const flags = Array.isArray(metadata.flags) ? metadata.flags : [];
  const flagged = ['high', 'medium'].includes(String(metadata.confidence).toLowerCase());
  return (
    <section className={`metadata-evidence ${flagged ? 'has-flags' : ''}`}>
      <div className="subsection-heading"><span>06 / FILE METADATA</span><span>{String(metadata.confidence || 'UNKNOWN').toUpperCase()} CONFIDENCE</span></div>
      {flags.length ? <ul>{flags.map((flag, index) => <li key={`${flag}-${index}`}>{flag}</li>)}</ul> : <p>No AI metadata markers were reported. This is not evidence that the video is genuine.</p>}
      <small>Metadata forensics is separate from visual frame classification.</small>
    </section>
  );
}

export default App;