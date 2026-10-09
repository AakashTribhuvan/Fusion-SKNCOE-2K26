import { useEffect, useRef, useState } from 'react';
import PhraseChallenge from './components/PhraseChallenge';
import AudioRecorder from './components/AudioRecorder';
import VoiceVerdictBlock from './components/VoiceVerdictBlock';
import AudioQualityPanel from './components/AudioQualityPanel';
import TranscriptPanel from './components/TranscriptPanel';
import SpectrogramPanel from './components/SpectrogramPanel';
import SpoofAnalysisPanel from './components/SpoofAnalysisPanel';
import SynchronizationPanel from './components/SynchronizationPanel';
import VerificationSummary from './components/VerificationSummary';
import { createSession, refreshChallenge, uploadAudio, verifySession, getHealth } from './services/api';

const initialStatus = 'Ready to record';
const TARGET_SAMPLE_RATE = 16000;
const MAX_RECORDING_SECONDS = 10;
const SUPPORTED_AUDIO_TYPES = new Set([
  'audio/wav',
  'audio/x-wav',
  'audio/webm',
  'audio/ogg',
  'audio/mp4',
  'audio/mpeg',
  'audio/mp3',
  'audio/aac',
  'audio/flac',
  'audio/x-m4a',
]);

function getSupportedMimeType() {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || '';
}

function writeString(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
}

function createWavBlob(samples, sampleRate = TARGET_SAMPLE_RATE) {
  const dataLength = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataLength, true);

  floatTo16BitPCM(view, 44, samples);
  return new Blob([buffer], { type: 'audio/wav' });
}

function mixToMono(audioBuffer) {
  const channelData = [];
  for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
    channelData.push(audioBuffer.getChannelData(channel));
  }

  if (channelData.length === 1) {
    return channelData[0];
  }

  const mono = new Float32Array(audioBuffer.length);
  for (let i = 0; i < audioBuffer.length; i += 1) {
    let total = 0;
    for (let channel = 0; channel < channelData.length; channel += 1) {
      total += channelData[channel][i] || 0;
    }
    mono[i] = total / channelData.length;
  }
  return mono;
}

export default function App() {
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const timerRef = useRef(null);

  const [session, setSession] = useState(null);
  const [phrase, setPhrase] = useState('');
  const [permissionState, setPermissionState] = useState('prompt');
  const [status, setStatus] = useState(initialStatus);
  const [duration, setDuration] = useState(0);
  const [audioUrl, setAudioUrl] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [audioFileName, setAudioFileName] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);

  const startTimer = () => {
    clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      setDuration((current) => {
        const next = current + 1;
        if (next >= MAX_RECORDING_SECONDS) {
          window.setTimeout(() => {
            if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
              stopRecording();
            }
          }, 0);
        }
        return next;
      });
    }, 1000);
  };

  const stopTimer = () => {
    clearInterval(timerRef.current);
  };

  const resetRecording = () => {
    stopTimer();
    setDuration(0);
    setAudioUrl('');
    setAudioFile(null);
    setAudioFileName('');
    recordedChunksRef.current = [];
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsRecording(false);
  };

  const initializeSession = async () => {
    setLoading(true);
    try {
      const payload = await createSession();
      setSession(payload);
      setPhrase(payload.challenge_phrase);
      setResult(null);
      setStatus('Session created. Ready to record.');
    } catch (error) {
      setStatus(error.message || 'Unable to create session');
    } finally {
      setLoading(false);
    }
  };

  const refreshPhrase = async () => {
    if (!session?.session_id) {
      return;
    }
    setLoading(true);
    try {
      const payload = await refreshChallenge(session.session_id);
      setPhrase(payload.challenge_phrase);
      setStatus('Challenge refreshed.');
      setResult(null);
    } catch (error) {
      setStatus(error.message || 'Unable to refresh phrase');
    } finally {
      setLoading(false);
    }
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('This browser does not support microphone recording.');
      setPermissionState('unsupported');
      return;
    }
    if (typeof MediaRecorder === 'undefined') {
      setStatus('This browser does not support audio recording.');
      setPermissionState('unsupported');
      return;
    }

    try {
      // Request raw audio: disable noise suppression and auto-gain so the
      // browser does not silence quiet-but-real speech via its noise gate.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      streamRef.current = stream;
      setPermissionState('granted');

      const mimeType = getSupportedMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recordedChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = async () => {
        const recordingMimeType = recorder.mimeType || 'audio/webm';
        const blob = new Blob(recordedChunksRef.current, { type: recordingMimeType });
        if (!blob.size) {
          setStatus('The browser returned an empty recording. Check microphone access and try again.');
          setAudioUrl('');
          setAudioFile(null);
          stopStream();
          return;
        }

        let decoded;
        let audioContext;
        try {
          const AudioCtor = window.AudioContext || window.webkitAudioContext;
          if (!AudioCtor) {
            throw new Error('This browser cannot convert the recorded audio to WAV.');
          }

          audioContext = new AudioCtor();
          decoded = await audioContext.decodeAudioData(await blob.arrayBuffer());
        } catch (error) {
          console.warn('Browser WAV conversion failed; retaining the original recording:', error);
          const extension = recordingMimeType.includes('webm')
            ? 'webm'
            : recordingMimeType.includes('ogg')
              ? 'ogg'
              : recordingMimeType.includes('mp4')
                ? 'm4a'
                : 'audio';
          const originalFile = new File([blob], `recording.${extension}`, { type: recordingMimeType });
          setAudioFile(originalFile);
          setAudioUrl(URL.createObjectURL(blob));
          setStatus('Browser WAV conversion was unavailable; keeping the original recording format for backend decoding.');
          stopStream();
          return;
        } finally {
          if (audioContext && audioContext.state !== 'closed') {
            try {
            await audioContext.close();
            } catch (error) {
              console.warn('Unable to close the temporary audio context:', error);
            }
          }
        }

        const mono = mixToMono(decoded);

        // Log actual peak amplitude to console for debugging purposes.
        // Silence detection is handled authoritatively by the backend service.
        const peak = mono.reduce((maximum, sample) => Math.max(maximum, Math.abs(sample)), 0);
        console.debug(`[FUSION] Decoded audio — peak: ${peak.toFixed(6)}, duration: ${decoded.duration.toFixed(2)}s, sampleRate: ${decoded.sampleRate}`);

        const wavBlob = createWavBlob(mono, decoded.sampleRate);
        const file = new File([wavBlob], 'recording.wav', { type: 'audio/wav' });
        setAudioFile(file);
        setAudioFileName('Microphone Recording (recording.wav)');
        setAudioUrl(URL.createObjectURL(wavBlob));
        setStatus(`Recording ready (${(wavBlob.size / 1024).toFixed(1)} KB, ${decoded.duration.toFixed(1)}s). Play it back before submitting.`);
        stopStream();
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
      setAudioFileName('');
      setStatus('Recording in progress…');
      setDuration(0);
      startTimer();
    } catch (error) {
      console.error(error);
      setPermissionState('denied');
      setStatus('Microphone permission was denied or is unavailable.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      stopTimer();
      setIsRecording(false);
      setStatus('Finishing audio recording…');
      return;
    }
    stopStream();
    stopTimer();
    setIsRecording(false);
  };

  const stopStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const handleFileSelect = (file) => {
    if (!file) return;
    stopTimer();
    stopStream();
    setIsRecording(false);
    setAudioFile(file);
    setAudioFileName(file.name);
    setAudioUrl(URL.createObjectURL(file));
    setStatus(`File selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB). Ready to verify.`);
  };

  const handleSubmit = async () => {
    if (!audioFile) {
      setStatus('Please record or upload an audio file before submitting.');
      return;
    }
    setSubmitting(true);
    setStatus('Uploading audio and running verification…');
    try {
      const audioType = (audioFile.type || '').split(';', 1)[0].toLowerCase();
      const hasAudioExtension = /\.(wav|wave|webm|ogg|oga|mp4|m4a|mpeg|mp3|aac|flac)$/i.test(audioFile.name || '');
      const isKnownAudioType = SUPPORTED_AUDIO_TYPES.has(audioType) || audioType.startsWith('audio/') || audioType.includes('mpeg') || audioType.includes('mp4');

      if ((!isKnownAudioType && !hasAudioExtension) || !audioFile.size) {
        throw new Error('Recording is empty or has an unsupported audio format. Please provide a WAV, MP3, MPEG, WebM, or OGG file.');
      }
      let currentSessionId = session?.session_id;
      if (!currentSessionId) {
        const newSession = await createSession();
        setSession(newSession);
        setPhrase(newSession.challenge_phrase);
        currentSessionId = newSession.session_id;
      }
      try {
        await uploadAudio(currentSessionId, audioFile);
      } catch (uploadErr) {
        if (uploadErr.message && uploadErr.message.toLowerCase().includes('session not found')) {
          // Auto-recreate session if the server reloaded or previous session expired
          const newSession = await createSession();
          setSession(newSession);
          setPhrase(newSession.challenge_phrase);
          currentSessionId = newSession.session_id;
          await uploadAudio(currentSessionId, audioFile);
        } else {
          throw uploadErr;
        }
      }
      const payload = await verifySession(currentSessionId);
      setResult(payload);
      setStatus('Verification complete.');
    } catch (error) {
      setStatus(error.message || 'Verification failed');
    } finally {
      setSubmitting(false);
    }
  };

  useEffect(() => {
    initializeSession();
    getHealth().then(setHealth).catch(() => setHealth({ status: 'unavailable' }));
    return () => {
      stopTimer();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="brand">FUSION</p>
          <h1>Voice and Phrase Verification</h1>
        </div>
        <div className="health-pill">{health?.status || 'checking…'}</div>
      </header>

      <main className="dashboard">
        <div className="main-column">
          <PhraseChallenge phrase={phrase} onRefresh={refreshPhrase} loading={loading} />
          <AudioRecorder
            status={status}
            permissionState={permissionState}
            duration={duration}
            isRecording={isRecording}
            audioUrl={audioUrl}
            audioFileName={audioFileName}
            onStart={startRecording}
            onStop={stopRecording}
            onReplay={() => audioUrl && new Audio(audioUrl).play()}
            onReset={resetRecording}
            onSubmit={handleSubmit}
            onFileSelect={handleFileSelect}
            submitting={submitting}
          />
        </div>

        <aside className="side-column">
          <VoiceVerdictBlock result={result} />
          <VerificationSummary result={result} />
          <AudioQualityPanel result={result} />
          <TranscriptPanel result={result} />
          <SpectrogramPanel result={result} />
          <SpoofAnalysisPanel result={result} />
          <SynchronizationPanel result={result} />
        </aside>
      </main>
    </div>
  );
}
