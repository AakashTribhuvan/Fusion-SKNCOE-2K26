# FUSION — Module B

This repository implements FUSION's voice and phrase verification module. It includes the recording and verification workflow, pretrained audio anti-spoof inference, offline dataset tooling, model evaluation/calibration, version metadata, and an audio/video synchronization contract.

## Architecture

- Backend: `backend/`
- Frontend: `frontend/`
- Shared Python environment: Python 3.11.9

## Quick start

### 1) Python setup

Use Python 3.11.9.

```powershell
"C:\Users\Sanidhya\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Backend

```powershell
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### 3) Frontend

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5174
```

## Backend endpoints

- `GET /health`
- `GET /api/v1/models/status`
- `POST /api/v1/sessions`
- `POST /api/v1/sessions/{session_id}/challenge`
- `POST /api/v1/sessions/{session_id}/audio`
- `POST /api/v1/sessions/{session_id}/verify`
- `GET /api/v1/sessions/{session_id}/result`

## Notes

- The app starts with reduced capability when optional ML models are unavailable.
- Training is an explicit offline command. The application never trains or downloads model weights at startup.
- The included MFCC-statistics logistic regression is a research baseline, not AASIST and not a production-grade deepfake detector.
- No dataset is bundled or downloaded. Use only data you are licensed to process.
- The official pretrained CLova AASIST ASVspoof2019-LA checkpoint is included under `backend/app/models/weights/` (about 1.28 MB); implementation source and MIT license notice are in `backend/app/models/`.
- AASIST expects mono 16 kHz audio and uses a 64,600-sample window. Its output is class-0 spoof / class-1 bona-fide logits; the adapter applies softmax for display but marks scores uncalibrated. Its reported benchmark is ASVspoof 2019 Logical Access; it is not an identity model or a production guarantee.
- The detector can be disabled with `FUSION_SPOOF_DETECTOR_ENABLED=false`; set `FUSION_AASIST_MODEL_PATH` to use another compatible AASIST checkpoint. The project baseline may instead be selected using `FUSION_SPOOF_MODEL_PATH`.
- Silero VAD and faster-whisper are optional-runtime pretrained components; install dependencies with `pip install -r requirements.txt`. VAD weights and the configured Whisper `tiny` model are fetched by their libraries on first use; no model training occurs.
- Phrase matching is strict and based on normalized transcript comparison; similarity is supplemental only.
- Audio/video synchronization requires timestamps and mouth-motion values from a future Module A integration. A timing mismatch is not proof of manipulation.

## Phase 2 dataset and training workflow

Create a CSV manifest with these columns:

```csv
audio_path,label,group_id
audio/clip-001.wav,bona_fide,speaker-source-001,genuine-source
audio/clip-002.wav,spoof,speaker-source-002,neural-tts
```

`audio_path` is relative to the manifest directory unless `--audio-root` is specified. Supported labels are `genuine`, `bona_fide`, `bonafide`, `real`, `spoof`, and `fake`. `group_id` should identify the source that must not leak across train, validation, and test partitions (for example speaker/session/source). Optional `attack_type` records attack families for per-attack generalization reporting. The split is deterministic for a given seed and requires both classes in every split.

From the `backend` directory, train and evaluate the research baseline:

```powershell
python -m app.training.train --manifest data\manifest.csv --output artifacts\fusion-baseline.json --seed 42
```

For an independent evaluation set, use entirely held-out group IDs:

```powershell
python -m app.training.evaluate --checkpoint artifacts\fusion-baseline.json --manifest data\heldout.csv --unseen-attack-types novel-neural-tts replay
```

Evaluation reports equal error rate, false-accept and false-reject rates, accuracy, per-attack-type metrics, and inference latency. `--unseen-attack-types` verifies that declared spoof attack types were not present in training. Do not use the final test set to tune thresholds. The trainer calibrates only on its group-separated validation partition and reports final metrics on its held-out test partition.

Set `FUSION_SPOOF_MODEL_PATH` in the environment to the produced checkpoint path to enable the runtime adapter. The API verifies the checkpoint format before reporting it as available. Checkpoint metadata records version, feature contract, sample rate, manifest hash, split groups, threshold, metrics, and a research-use warning.

### Pretrained inference

The included AASIST checkpoint is enabled by default. The first request loads it into CPU memory once. VAD and transcription will return explicit unavailable statuses until their Python dependencies are installed and the associated pretrained weights are downloaded by the libraries. Whisper defaults to the multilingual `tiny` model; configure `FUSION_WHISPER_MODEL_SIZE` and `FUSION_WHISPER_LANGUAGE` as required.

The browser converts successfully decoded microphone recordings to WAV before upload. If the browser cannot convert its MediaRecorder format, it preserves the original MIME type and extension; PyAV decodes supported WebM/Opus, MP4, or Ogg audio server-side. The API independently decodes uploaded content, reports the measured duration, and rejects malformed or over-limit recordings.

## Module A synchronization contract

The verification endpoint accepts optional timestamped mouth-motion evidence:

```json
{
  "video_frames": [
    {"timestamp_seconds": 0.10, "mouth_motion": 0.03},
    {"timestamp_seconds": 0.20, "mouth_motion": 0.41}
  ]
}
```

The backend compares these measurements with VAD speech segments and returns an experimental timing offset/correlation. If either evidence source is absent, synchronization is `unavailable`; the backend does not invent measurements.

## Testing

```powershell
"C:\Users\Sanidhya\AppData\Local\Programs\Python\Python311\python.exe" -m pytest backend/tests -q
```
