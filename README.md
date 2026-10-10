# Frame / Check — Deepfake-Resistant Identity Verification

Hackathon prototype for Mastercard CSB-02: deepfake-resistant KYC onboarding. The demo implements short-lived sessions, second-device pairing, phone platform-authenticator verification, a fresh QR sequence, live face-presence checks, a demo of the deepfake-AI video screening layer, a required voice phrase check, and a transparent evidence report.

## Selected challenge direction

The Mastercard CSB-02 brief asks for real-versus-synthetic screening without an enrolled real-person baseline. The live flow independently decodes fresh phone QR codes and checks face presence. It demonstrates the video deepfake-AI screening layer without actually running the video classifier; the report keeps that demo status, QR evidence, required phrase-recognition voice check, optional AI-voice signal, and phone proof separate.

Phone local user verification authorizes use of a platform credential; it does not identify the person in the laptop camera or prove that the clip is genuine. The OS may accept a biometric or device PIN/passcode, and the server does not learn which method was used. Biometric data remains on the phone. No enrolled-face or speaker identity matching is performed.

## Current status

Implemented:

- FastAPI health and session endpoints with five-minute session expiry.
- Phone WebAuthn registration and authentication with user verification required; the backend verifies a signed assertion and stores only the public credential in memory.
- Phone pairing through a session-specific QR token.
- Six compact, random QR signals; each remains on the phone until the server detects both a face in frame and the current code, within a four-minute challenge lifetime. Session/challenge routing binds each short payload.
- The phone QR expands to a near-fullscreen view with persistent motion/glare instructions. The live camera preview remains normal; QR contrast enhancement is applied only to an internal scan-frame copy on the API.
- The laptop workflow is split into ordered Phone, QR + video, Voice, and Report tabs. Later steps unlock only when their prerequisites are complete.
- The HDFC- and ICICI-inspired bank-portal tabs wrap the actual Frame verification UI in same-origin embedded views. Each portal runs its own session against the existing Frame backend, including phone pairing and WebAuthn, randomized QR checks, recorded video deepfake screening, required voice phrase check, and evidence report; consent gates are unchanged. These are unofficial, unaffiliated presentation demos: there is no bank login, account opening, bank API, or SMS integration.
- Bank artwork: HDFC and ICICI logo SVGs from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:HDFC-Bank-Logo.svg) and [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ICICI_Bank_Logo.svg); locally served hero photos from [Unsplash photo 1](https://images.unsplash.com/photo-1494790108377-be9c29b29330) and [Unsplash photo 2](https://images.unsplash.com/photo-1527980965255-d3b416303d12).
- Six numbered square QR targets and server-measured ordered path validation replace the figure-eight gesture. Targets run around the frame edges to leave the typical centered face area unobstructed.
- The laptop camera waits for a fresh QR lock inside box 1 and live face presence before recording. Users hold the phone steady for autofocus/auto-exposure, then move it slowly through boxes 2–6 in order while keeping a face in frame. A locally hosted, downloadable target guide is shown alongside the live viewer.
- With explicit consent, the browser records the entire box-to-box movement (up to 1 minute, without audio) and uploads the clip for AI screening. OpenCV decodes separate downscaled live preview frames for QR/path evidence and face-presence gating. PyAV samples frames from the recorded clip; MediaPipe detects face regions, and the pinned deepfake classifier scores face crops in memory.
- The completed evidence report includes the classifier consensus, available per-frame scores, basic cross-frame comparison, and a downloadable PDF. If too few face-bearing samples can be scored, the video result is marked unavailable/inconclusive rather than assumed successful. Model outputs are uncalibrated research signals, not proof of identity or authenticity and not an onboarding decision.
- The live viewer draws a QR tracking box from the server's detected corner bounds. The browser requests continuous autofocus/exposure where supported, offers a one-shot autofocus refresh, and exposes available manual focus, exposure, zoom, and camera-selection controls. Hardware support varies; fixed-focus cameras cannot be refocused by software.
- The required Voice tab has its own directions and explicit microphone consent. It shows a live waveform while recording at most eight seconds on user action, checks a fresh random three-word phrase with Whisper tiny.en (five-beam decoding, phrase-word bias, and a 90% normalized similarity threshold), screens audio quality, and uses an Apache-2.0 Wav2Vec2 classifier for an AI-like voice signal. A phrase or audio-quality failure can be retried; submitting without a successful phrase and quality check makes the overall challenge fail. The classifier weights are about 1.2 GB and download only after voice consent. The model card describes 2.5–13 second clips as its optimal range; shorter recordings are still analyzed but flagged for review. No speaker enrollment or identity match is used; audio is decoded and processed in memory and not retained.
- A report hash is appended to an in-memory, hash-only audit chain. The simulator stores no challenge codes, transcript, face/audio samples, or session identifier; it is not a real blockchain and resets when the API restarts.
- The phone's local authenticator verifies the user (biometric or device credential, depending on platform policy) before the credential signs a fresh server challenge. The biometric itself is never sent to the API.
- A completed QR challenge with the required voice phrase check produces **Review**; a failed randomized QR sequence or uncompleted required voice check produces **Challenge failed**. Research-model confidence does not make an identity decision and must not be used for real onboarding decisions.
- The physical phone/camera/microphone flow still needs an end-to-end run over HTTPS.

Not implemented yet:

- Independent validation across broad, current and unseen generators and a production-quality temporal video model. Swaraksha's published video checks use a small curated set; their results are not independent validation. The detector remains a research signal only.
- An artifact-localization model. The current report exposes scored face-frame references, not explanations of which physical artifact caused a score.
- Speaker identity matching. It is intentionally excluded because there is no enrolled speaker baseline.
- Production credential storage/revocation, session persistence, account authentication, deployment, rate limiting, and production privacy/security controls. Session and public-key state is in memory and resets on restart; a previously created phone passkey may remain on the device.

This is a demonstration scaffold, not a KYC decision system. Do not use it to approve or reject real applicants.

## Requirements

- CPython 3.11.x, 64-bit.
- A current browser with camera access support. Chrome or Edge is recommended for the demo.
- First QR scan downloads the small MediaPipe BlazeFace face-presence detector if it is not cached. The required phrase-recognition and optional voice classifier models download only after voice consent. The video classifier loads lazily when the recorded clip is submitted; model caches stay outside Git in ignored locations.
- Internet access on first run so the launcher can download the official Windows `cloudflared` binary into the ignored `.tools/` folder and verify it against Cloudflare's published SHA256 checksum. A preinstalled `cloudflared` on `PATH` is also supported.
- The launcher creates a temporary Cloudflare Quick Tunnel (`trycloudflare.com`) so both devices can reach the same HTTPS URL without sharing a Wi-Fi network. WebAuthn and camera access require a secure context. Quick Tunnels are for testing and create a public URL; anyone who obtains the URL can reach the demo while it is running. Do not use real customer data.
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

Start the API and cross-network phone demo (recommended):

```powershell
.\StartPrototype.bat
```

The launcher checks the environment, prepares `cloudflared` if needed, starts the tunnel and API, verifies local API health and a registered Cloudflare connection, and opens the random URL. A blocked public self-check from the laptop is shown as a warning rather than stopping the demo. The laptop and phone can use different networks. Keep the launcher window open; press Ctrl+C or close it to stop the API and tunnel. For laptop-only use without a tunnel, the manual development-server command is:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). API docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Demo flow

1. Start a session and pair a phone by scanning the QR.
2. On first use, tap **Verify this phone** to register a phone platform credential and complete local user verification. The same credential can be reused for later sessions while the API process retains its public key record.
3. Continue to **QR + video**, agree to camera use and upload for video screening, and start the camera. Place the bright phone QR in box 1 about 15–20 cm from the lens and hold it still until the viewer shows **QR LOCK**. Then move slowly through numbered boxes 2–6 in order. Recording starts at the lock and stops when all six QRs are scanned or after 1 minute. The recorded clip is uploaded to the API, which samples and scores face-bearing frames with the video classifier; the UI shows upload and processing progress.
4. On the distinct **Voice** tab, consent and start the required voice check. Allow microphone access, then say the displayed fresh phrase once; a live waveform appears while recording for up to eight seconds. The first consented run downloads the pinned voice-classifier weights (about 1.2 GB). Audio shorter than 2.5 seconds is still screened but marked for review; silent audio can be retried. Skipping the check or failing its phrase/quality gate makes the report fail.
5. Complete the required voice check and open the **Report** tab. Skipping voice or failing phrase/quality validation makes the report **Challenge failed**. QR sequence/path, phone proof, video-classifier signals, phrase match, audio quality, and audio anti-spoof signal are separate. Phrase comparison allows at most 10% normalized transcription difference. The report includes an evidence hash in an in-memory simulated ledger.

For a local UI-only walkthrough, `/hidden/control` is disabled by default. Enable it only on loopback during development with `ENABLE_QA_CONTROLS=true` and `APP_ENV=development`; never enable it for a tunnel or production deployment. Its next-step button simulates UI progression only and does not skip server-side verification or create evidence.

### Operator recovery panel

`/admin` is a separate, passwordless operator page. It lists in-memory session metadata and provides supervised demo controls: advance the paired phone past the current QR step, deliberately mark the QR test failed, or skip a video-processing step that the server currently reports as still processing. Each action requires an operator reason and is recorded in the report. QR skips do not create scan evidence and make verification inconclusive; the deliberate QR-failure control makes the QR result and overall challenge fail. Video skips discard video-screening results and make the final report inconclusive. The page does not expose pairing tokens, QR payloads, or media evidence.

Admin controls are disabled unless explicitly enabled in a non-production environment. The `StartPrototype.bat` launcher asks whether to enable the public panel; answer **Y** for a supervised demo. Alternatively, set the variables in the same PowerShell window before launching:

```powershell
$env:APP_ENV = "development"
$env:ENABLE_ADMIN_CONTROLS = "true"
.\StartPrototype.bat
```

Restart the running launcher after enabling admin; an already-started API cannot pick up the setting. Use the newly printed Cloudflare URL and append `/admin`. The launcher prints that full address when the panel is enabled. On the same computer, `http://127.0.0.1:8000/admin` also works. There is deliberately no link to this page in the participant UI. The passwordless controls are reachable by anyone who has the public tunnel URL, so only enable them during a supervised development demo and disable them afterward. They remain unavailable when `APP_ENV` is `production`.

While the participant workflow waits for model loading, video upload/screening, voice analysis, or report generation, it shows an accessible progress indicator. Video upload progress reflects bytes sent; analysis and model loading use an indeterminate bar because the backend does not expose fractional work progress.

## Project files

- `backend/app/main.py` — API, in-memory session/challenge state, server-side QR frame decoding, checks, and report.
- `backend/app/screening.py` — live face-presence detection, in-memory recorded-video decoding, and pinned image-classifier adapter with frame evidence/consistency summary.
- `backend/app/voice_screening.py` — in-memory audio decoding, phrase transcription, and lazy audio anti-spoof classifier adapter.
- `backend/app/audit_ledger.py` — hash-only in-memory audit-chain simulator; not a blockchain.
- `backend/app/static/index.html` — laptop onboarding and capture experience.
- `backend/app/static/admin.html` — development-only operator session monitor and video-step recovery page.
- `backend/app/static/phone.html` — phone pairing and scan-acknowledged challenge display, sized for mobile screens.
- `backend/app/static/qr-zones.svg` — original numbered square target guide.
- `backend/app/static/qa-control.html` — local-only UI simulation control page, served only when explicitly enabled for development.
- `WORKFLOW_UPDATES.md` — detailed notes on the staged tabs, QR lock/video recording, voice flow, and API changes.
- `requirements.txt` — shared Python dependencies, including the server-side WebAuthn verifier and model runtimes.
- `CheckRequirements.bat` — interpreter, package consistency, and import check.
- `StartPrototype.bat` — checks setup and starts the cross-network HTTPS demo.
- `backend/run_tunnel.py` — manages the ephemeral Cloudflare URL, WebAuthn origin settings, API process, and guaranteed Windows process-job cleanup.
- `PROJECT_ITINERARY.md` — step-by-step walkthrough, technology map, limitations, and next development steps.
- `PLAN_OF_ACTION.md` — editable build plan.
- `PITCH_DECK.md` — editable pitch outline.

## Frontend and detector reuse

The current Frame / Check workbench UI is retained. The linked Swaraksha, voice, and blockchain application repositories are not copied into this project because their application-code licensing is unclear or their behavior is not appropriate for the selected scope. The adapters below use pinned, clearly licensed model artifacts instead.

### Pinned model sources and limits

- Live face-frame classifier: [`prithivMLmods/Deep-Fake-Detector-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-Model), Apache-2.0, pinned to revision `c5cb24c6a159dd2b57ca15c6a1065bd0ce8fa380`. This SigLIP2 image classifier is run on context-padded face crops; the configured 0.38 frame threshold and multi-frame consensus follow the Swaraksha feature-branch update. Its model-card accuracy is reported on image samples, while Swaraksha's video validation uses a small curated set; neither result is an independent estimate of accuracy on this demo's deployment population.
- Face localization: Google's MediaPipe BlazeFace short-range detector, loaded through the installed Apache-2.0 MediaPipe task runtime. The small hosted model is SHA-256 checked before use and cached outside Git.
- Optional audio classifier: [`garystafford/wav2vec2-deepfake-voice-detector`](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector), Apache-2.0, pinned to revision `c66306024a7ede0be291e9c4558b37634782dc4e`. It is a roughly 315M-parameter Wav2Vec2 XLS-R model; its model card describes a 1,866-sample English dataset and 2.5–13 second optimal input range. Those model-card results have not been independently reproduced or validated for this demo. Scores are uncalibrated screening signals, not proof of authenticity, liveness, identity, or fraud.
- Optional fresh-phrase transcription: [`Systran/faster-whisper-tiny.en`](https://huggingface.co/Systran/faster-whisper-tiny.en), MIT, pinned to revision `0d3d19a32d3338f10357c0889762bd8d64bbdeba`. This is phrase matching, not speaker identity verification. The model is downloaded only when voice consent is enabled.
- Audit trail: a clean-room, in-memory hash-chain simulator. It is neither a deployed blockchain nor persistent tamper-proof storage. It records only a digest and chain metadata, not the source report or raw samples.

The supplied detector repositories remain research references, not drop-in runtime dependencies:

- [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) is licensed **CC BY-NC 4.0**. Its Xception example expects 32 frames at 256 px and is configured around FaceForensics++ training. The benchmark's documented setup pins an older Python/PyTorch/CUDA stack, while this API targets Python 3.11; it should be evaluated in an isolated research environment, not installed into the API environment. Any detector checkpoint and its training-data terms must also be checked separately.
- [FaceForensics](https://github.com/ondyari/FaceForensics) releases its code under MIT, but its dataset has separate terms and access requires approval. Its published Xception classifier targets Python 3.6 / PyTorch 1.0.1 and is explicitly not fine-tuned for general compressed videos. Neither its dataset nor model assets are bundled here.

No code, checkpoint, or dataset from DeepfakeBench or FaceForensics is integrated. The current face/audio models have compatible licenses but are not independently validated for this application; runtime/download errors are reported as unavailable, not as successful checks. QR frames and recorded video are still client-submitted, so a modified client may fabricate them.
