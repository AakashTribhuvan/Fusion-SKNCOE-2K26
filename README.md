# SWARAKSHA v2

SWARAKSHA includes a local-first identity API and a focused video-authenticity screening interface. The default launcher runs the video-only API for Frame/Check; the original identity registration and matching API remains available as a separate entrypoint.

This `trial_v2` folder is the cleaned runtime project. It contains the active backend and frontend only; the original `trial` folder remains the prototype/history workspace.

## What We Have Built

### Legacy identity registration (`api.main`)

- Register a person with a stable person ID and display name.
- Upload five or more reference images in one enrollment flow.
- Accept additional reference images for an existing person ID.
- Detect faces in each image with DeepFace and RetinaFace.
- Generate ArcFace embeddings and store them in a FAISS index.
- Persist person records and reference counts in SQLite.

### Legacy face and image scanning (`api.main`)

- Request webcam permission in the browser.
- Capture a face image from the live camera.
- Upload an image as an alternative to the camera flow.
- Detect all faces in the image.
- Match detected faces against registered identities using cosine similarity.
- Run the AI-generated image detector on matched face crops.
- Return an `ALLOW` or `BLOCK` result with per-face reasons, identity similarity, and AI detector data.

### Video testing

- Queue and submit multiple videos from the Frame/Check frontend.
- Upload videos one at a time to the backend queue endpoint.
- Sample video frames approximately every two seconds.
- Detect face regions and run the existing image classifier on every sampled face crop without identity enrollment or matching.
- Measure relative depth inside an elliptical face-region mask and associate face boxes across adjacent samples.
- Report sampled-frame scores, face-track continuity, relative-depth measurements, and file-metadata findings independently.
- Show an individual result card for each submitted video.

### Frontend experience

- Video-first Frame/Check examination workspace; face enrollment and identity workflows are not part of the active UI.
- Queue multiple local video files, preview the selected recording, and seek to sampled frame timestamps.
- Show image-model scores, relative face-mask depth measurements, sampled face-track continuity, and metadata findings independently.
- Mark depth/face tracking unavailable when those measurements could not be produced; never infer authenticity from an unanalyzed video.
- Responsive layout for desktop and mobile.

## Runtime Architecture

```text
Browser / React + Vite
        |
        | HTTP multipart requests
        v
FastAPI video-only API
        |
        +-- DeepFace + RetinaFace: face-region detection (no embedding lookup)
        +-- Transformers/PyTorch: AI-generated image detection
        +-- Transformers/PyTorch: Depth Anything V2 relative depth
        +-- OpenCV: image decoding and video frame sampling
        +-- Legacy api.main: optional FAISS/SQLite identity registration and matching
```

## Project Structure

```text
trial_v2/
├── api/
│   ├── main.py                 Legacy identity registration, matching, and scans
│   └── video_app.py            Video-only API used by the current frontend
├── core/
│   ├── ai_detector.py          AI-generated image detector
│   ├── encoder.py              DeepFace face detection and ArcFace embeddings
│   └── face_index.py           FAISS index persistence and matching
├── db/
│   ├── database.py             SQLite person and embedding records
│   └── swaraksha.db            Local database created at runtime
├── frontend/
│   ├── src/DeepfakeWorkbench.jsx Video screening interface
│   ├── src/deepfake.css        Frame/Check visual system
│   ├── src/main.jsx            Active React entrypoint
│   ├── public/icon2.png        Active brand asset
│   └── package.json             Frontend dependencies and scripts
├── models/                     Reserved for model assets
├── storage/
│   ├── faiss_index.bin         Persisted FAISS vectors
│   ├── faiss_id_map.json       FAISS index-to-person mapping
│   ├── registered_faces/       Reserved registered media storage
│   ├── embeddings/              Reserved embedding storage
│   ├── reports/                 Reserved scan reports
│   └── temp/                    Temporary processing files
├── config.py                   Central paths and detector thresholds
├── requirements.txt            Python dependencies
├── start_swaraksha.bat         Windows launcher for backend and frontend
└── README.md                   This document
```

## Video API Routes (Default)

### `GET /`

Reports video API status, active mode, and whether the classifier is initialized.

### `POST /api/scan-video`

Samples an uploaded video, detects face regions without identity matching, and returns classifier, relative-depth, face-track, and metadata evidence. Multipart field: `file`.

Videos are sampled at `VIDEO_SAMPLE_INTERVAL` (two seconds by default), limited to 250 MB and 180 seconds by default. Override the limits with `SWARAKSHA_MAX_VIDEO_BYTES` and `SWARAKSHA_MAX_VIDEO_SECONDS`.

## Legacy Identity API Routes (`api.main`)

The identity routes below are available only when launching the preserved `api.main:app` entrypoint.

### `GET /`

Health check. Returns API status, version, number of registered people, and total embeddings.

### `POST /api/register`

Registers one or more reference images.

Multipart fields:

- `person_id`: stable identifier for the person
- `name`: display name
- `files`: one or more image files

Returns the number of faces successfully registered and the total FAISS embedding count.

### `POST /api/recognize`

Detects and matches faces in a single image without running the AI authenticity check.

### `POST /api/scan`

Runs the complete image protection flow: face detection, identity matching, and AI-generated image analysis.

Multipart field:

- `file`: one image file

### `GET /api/persons`

Returns all registered people, including their IDs, names, creation times, and stored reference counts.

### `DELETE /api/persons/{person_id}`

Removes a person, their stored embedding records, and their vectors from the FAISS index.

## Setup

### Prerequisites

- Windows
- CPython 3.11 and the Python launcher (`py`)
- Node.js and npm
- A browser with webcam support if using live capture

### First-time setup

Create and install the backend environment from this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the frontend dependencies:

```powershell
cd frontend
npm.cmd install
```

### Start the application

Double-click:

```text
start_swaraksha.bat
```

The launcher uses `.venv` and:

1. Checks that the project Python 3.11 environment exists.
2. Checks the required backend imports.
3. Installs `requirements.txt` if dependencies are missing.
4. Starts `api.video_app:app` with Uvicorn on port `8000` (no FAISS/SQLite identity matching).
5. Starts Vite on port `5173`.

The legacy identity API remains available with `\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000`.

The depth model weights download from Hugging Face on first use. The depth field is relative monocular depth, not metric 3D. Face-track continuity associates bounding boxes across sampled frames; it is not general object permanence or identity recognition. The legacy React identity UI is not loaded by the default frontend.

Open:

- Frontend: http://localhost:5173
- API health check: http://localhost:8000
- FastAPI docs: http://localhost:8000/docs

Keep both terminal windows open while developing. The frontend does not start the backend by itself.

## Important Data Behavior

Reference counts are cumulative. If a person already has 10 stored references and five more are enrolled, the directory may show 15 total stored references. The frontend separately reports how many images were added in the latest enrollment.

The FAISS index and ID map live under `storage/`. The SQLite database lives under `db/`. Deleting files from either location manually can desynchronize the runtime state; use the Protected people delete action when possible.

## Current Limitations

- This is a local development system, not a production deployment.
- There is no authentication or user account system yet.
- CORS is intentionally permissive for local development.
- The video endpoint samples frames instead of analyzing every frame.
- Video processing is sequential and can be slow because each sampled face may invoke DeepFace and the AI detector.
- Camera access requires browser permission and generally works best from `localhost` or a secure context.
- Model downloads and first-run model initialization can take several minutes.
- Detection and similarity thresholds are configured in `config.py` and should be calibrated with representative data before real-world use.
- A match is identity evidence, not proof of consent, authenticity, or legal ownership.

## Validation Performed

The current v2 project has been validated with:

```text
Frontend production build
Frontend lint
Python backend syntax compilation
Editor diagnostics for active frontend and backend files
```

## Development Direction

The next sensible engineering steps are to add structured server-side job status for long video scans, persist scan reports, add authentication and access control, improve face/reference quality validation, and add automated API tests around registration, deletion, image scans, and video scans.
