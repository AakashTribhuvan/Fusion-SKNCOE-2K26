# Fusion — Deepfake-Resistant Identity Verification

Hackathon prototype for a multi-signal digital onboarding flow. The current demo implements short-lived sessions, second-device pairing, a fresh QR sequence, browser camera/microphone capture, and a transparent evidence report.

## Current status

Implemented:

- FastAPI health and session endpoints with five-minute session expiry.
- Phone pairing through a session-specific QR token.
- Six random, session-bound QR signals with a 30-second challenge lifetime.
- Phone page cycles the QR signals while the laptop camera scans them.
- The API checks that the browser-reported codes belong to the current session/challenge, arrived in increasing order, and moved across the camera view.
- The result separates the QR challenge from face deepfake analysis, speaker verification, and audio spoof detection.
- Camera/microphone recording remains in the browser and is discarded; only QR observations and capture durations are sent to the API.

Not implemented yet:

- A native mobile biometric prompt or device-held signing key. Current phone pairing is a one-time bearer token, not biometric identity proof.
- Independent server-side decoding of the video. QR observations currently come from the browser, so a modified client could forge them.
- Validated face/deepfake, speaker matching, and voice anti-spoofing models. The report marks these checks unavailable and leaves the decision at **Review** even when the QR challenge passes.
- Database persistence, authentication, production deployment, rate limiting, and production privacy/security controls. Session state is in memory and resets on restart.

This is a demonstration scaffold, not a KYC decision system. Do not use it to approve or reject real applicants.

## Requirements

- CPython 3.11.x, 64-bit.
- A current Chromium browser (Chrome or Edge) for the browser `BarcodeDetector` API.
- For a separate phone, the app must be reachable from both devices. Browser camera/microphone access requires a secure context: use `localhost` for laptop-only testing or deploy behind HTTPS for cross-device testing. A phone cannot reach the laptop through a `localhost` QR link.
- PyTorch defaults to the CPU build in the shared requirements. Use the [official PyTorch selector](https://pytorch.org/get-started/locally/) to install a matching CUDA build if needed.

## Run locally on Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the environment:

```powershell
.\CheckRequirements.bat
```

Start the API and web demo:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). API docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Demo flow

1. Start a session and pair a phone by scanning the QR.
2. Start a randomized challenge from the laptop.
3. Read the phrase shown on both screens; move the phone through a slow figure eight while the laptop scans the changing codes.
4. Review the report. QR sequence/movement may pass, while unavailable AI checks keep the overall result at **Review**.

## Project files

- `backend/app/main.py` — API, in-memory session/challenge state, QR checks, report.
- `backend/app/static/index.html` — laptop onboarding and capture experience.
- `backend/app/static/phone.html` — phone pairing and rotating challenge display.
- `requirements.txt` — shared Python dependencies.
- `CheckRequirements.bat` — interpreter, package consistency, and import check.
- `PLAN_OF_ACTION.md` — editable build plan.
- `PITCH_DECK.md` — editable pitch outline.
