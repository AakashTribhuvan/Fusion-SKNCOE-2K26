# Fusion — Deepfake-Resistant Identity Verification

Hackathon prototype for Mastercard CSB-02: deepfake-resistant KYC onboarding. The demo implements short-lived sessions, second-device pairing, phone platform-authenticator verification, a fresh QR sequence, browser camera QR scanning, and a transparent evidence report.

## Selected challenge direction

The Mastercard CSB-02 brief asks for a lightweight real-versus-synthetic face classifier on a short KYC clip, without an enrolled real-person baseline, with an explanation of the evidence behind its score. This classifier is the primary deliverable. The selected supporting checks are a phone-displayed figure-eight QR challenge and a platform credential that signs a fresh challenge only after local user verification.

Phone local user verification authorizes use of a platform credential; it does not identify the person in the laptop camera or prove that the clip is genuine. The OS may accept a biometric or device PIN/passcode, and the server does not learn which method was used. Biometric data remains on the phone. The current prototype has the QR and WebAuthn proof flows; the classifier is not implemented yet.

## Current status

Implemented:

- FastAPI health and session endpoints with five-minute session expiry.
- Phone WebAuthn registration and authentication with user verification required; the backend verifies a signed assertion and stores only the public credential in memory.
- Phone pairing through a session-specific QR token.
- Six random, session-bound QR signals with a 30-second challenge lifetime.
- Phone page cycles the QR signals while the laptop camera scans them.
- A figure-eight instruction and browser-reported camera path spread check.
- The API checks that the browser-reported codes belong to the current session/challenge, arrived in a complete cyclic order, and moved across both image axes.
- The result separates the QR challenge from face deepfake analysis, speaker verification, and audio spoof detection.
- Webcam video remains in the browser and is used only to scan QR codes; no video or audio is uploaded, and the deepfake classifier is not integrated yet.
- The phone's local authenticator verifies the user (biometric or device credential, depending on platform policy) before the credential signs a fresh server challenge. The biometric itself is never sent to the API.
- The software path has been syntax- and endpoint-checked; the authenticator ceremony still needs a physical-phone run over HTTPS.

Not implemented yet:

- The primary real-versus-synthetic face classifier, short-clip scoring, and inspectable artifact explanation required by the challenge.
- Independent server-side decoding of the video. QR observations currently come from the browser, so a modified client could forge them.
- Validated face/deepfake, speaker matching, and voice anti-spoofing models. The report marks these checks unavailable and leaves the decision at **Review** even when the QR challenge passes.
- Production credential storage/revocation, session persistence, account authentication, deployment, rate limiting, and production privacy/security controls. Session and public-key state is in memory and resets on restart; a previously created phone passkey may remain on the device.

This is a demonstration scaffold, not a KYC decision system. Do not use it to approve or reject real applicants.

## Requirements

- CPython 3.11.x, 64-bit.
- A current Chromium browser (Chrome or Edge) for the browser `BarcodeDetector` API.
- For a separate phone, the app must be reachable from both devices over HTTPS. WebAuthn and camera access require a secure context; `localhost` works for laptop-only testing, but a phone cannot reach the laptop through a `localhost` QR link. For a reverse-proxy deployment, set `FUSION_PUBLIC_ORIGIN` and `FUSION_WEBAUTHN_ORIGIN` to the public HTTPS origin, and `FUSION_WEBAUTHN_RP_ID` to its hostname.
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
2. On first use, register a phone platform credential; then complete local user verification to sign the session challenge. The same credential can be reused for later sessions while the API process retains its public key record.
3. Start the randomized challenge from the laptop and move the phone through a slow figure eight while the laptop scans the changing codes.
4. Review the report. QR sequence/movement and phone proof are separate; the unavailable face classifier keeps the overall result at **Review**.

## Project files

- `backend/app/main.py` — API, in-memory session/challenge state, QR checks, report.
- `backend/app/static/index.html` — laptop onboarding and capture experience.
- `backend/app/static/phone.html` — phone pairing and rotating challenge display.
- `requirements.txt` — shared Python dependencies, including the server-side WebAuthn verifier.
- `CheckRequirements.bat` — interpreter, package consistency, and import check.
- `PLAN_OF_ACTION.md` — editable build plan.
- `PITCH_DECK.md` — editable pitch outline.
