# Frame / Check — Deepfake-Resistant Identity Verification

Hackathon prototype for Mastercard CSB-02: deepfake-resistant KYC onboarding. The demo implements short-lived sessions, second-device pairing, phone platform-authenticator verification, a fresh QR sequence, live face-frame screening, optional voice screening, and a transparent evidence report.

## Selected challenge direction

The Mastercard CSB-02 brief asks for real-versus-synthetic screening without an enrolled real-person baseline. During the same live camera window, the API independently decodes fresh phone QR codes and periodically face-detects/crops frames for an image classifier. The report keeps QR evidence, per-frame classifier scores, basic cross-frame score consistency, optional voice signals, and phone proof separate.

Phone local user verification authorizes use of a platform credential; it does not identify the person in the laptop camera or prove that the clip is genuine. The OS may accept a biometric or device PIN/passcode, and the server does not learn which method was used. Biometric data remains on the phone. No enrolled-face or speaker identity matching is performed.

## Current status

Implemented:

- FastAPI health and session endpoints with five-minute session expiry.
- Phone WebAuthn registration and authentication with user verification required; the backend verifies a signed assertion and stores only the public credential in memory.
- Phone pairing through a session-specific QR token.
- Six compact, random QR signals; each remains on the phone until the server detects both a face in frame and the current code, within a four-minute challenge lifetime. Session/challenge routing binds each short payload.
- The phone QR expands to a near-fullscreen view with persistent motion/glare instructions. The live camera preview remains normal; QR contrast enhancement is applied only to an internal scan-frame copy on the API.
- The laptop workflow is split into ordered Phone, QR + video, Voice, and Report tabs. Later steps unlock only when their prerequisites are complete.
- The HDFC- and ICICI-inspired bank-portal tabs wrap the actual Frame verification UI in same-origin embedded views. Each portal runs its own session against the existing Frame backend, including phone pairing and WebAuthn, randomized QR/video screening, optional voice screening, and the evidence report; consent gates are unchanged. The larger, responsive hero copy and locally served Unsplash portraits make the portal mastheads easier to read. These are unofficial, unaffiliated presentation demos: there is no bank login, account opening, bank API, or SMS integration.
- Bank artwork: HDFC and ICICI logo SVGs from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:HDFC-Bank-Logo.svg) and [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ICICI_Bank_Logo.svg); locally served hero photos from [Unsplash photo 1](https://images.unsplash.com/photo-1494790108377-be9c29b29330) and [Unsplash photo 2](https://images.unsplash.com/photo-1527980965255-d3b416303d12).
- Six numbered square QR targets and server-measured ordered path validation replace the figure-eight gesture. Targets run around the frame edges to leave the typical centered face area unobstructed.
- The laptop camera waits for a fresh QR lock inside box 1 and live face presence before recording. Users hold the phone steady for autofocus/auto-exposure, then move it slowly through boxes 2–6 in order while keeping a face in frame. A locally hosted, downloadable target guide is shown alongside the live viewer.
- With explicit consent, the browser records the entire box-to-box movement (up to 1 minute, without audio) and sends it to an in-memory API decoder. OpenCV decodes separate downscaled live preview frames for QR/path evidence. PyAV decodes the full clip and samples frames every half-second, including the final frame; MediaPipe detects faces and an Apache-2.0 image classifier scores detected face crops. The uploaded clip and decoded images are discarded after analysis; only derived score/evidence metadata remains for the report.
- The report records per-frame classifier score, relative face region, and timestamp, plus median/range and a basic score/face-count consistency signal. The temporal comparison is not a trained video model and the image classifier does not explain a physical artifact.
- The live viewer draws a QR tracking box from the server's detected corner bounds. The browser requests continuous autofocus/exposure where supported, offers a one-shot autofocus refresh, and exposes available manual focus, exposure, zoom, and camera-selection controls. Hardware support varies; fixed-focus cameras cannot be refocused by software.
- The optional Voice tab has its own directions and explicit microphone consent. It shows a live waveform while recording at most eight seconds on user action, checks a fresh random three-word phrase with Whisper tiny.en, and screens for AI-like voice with an Apache-2.0 Wav2Vec2 classifier. No speaker enrollment or identity match is used; audio is decoded and processed in memory and not retained.
- A report hash is appended to an in-memory, hash-only audit chain. The simulator stores no challenge codes, transcript, face/audio samples, or session identifier; it is not a real blockchain and resets when the API restarts.
- The phone's local authenticator verifies the user (biometric or device credential, depending on platform policy) before the credential signs a fresh server challenge. The biometric itself is never sent to the API.
- A completed QR challenge produces **Review**; a failed randomized QR sequence produces **Challenge failed**. Research-model confidence does not make an identity decision and must not be used for real onboarding decisions.
- The physical phone/camera/microphone flow still needs an end-to-end run over HTTPS.

Not implemented yet:

- A calibrated detector trained/evaluated on current unseen generators and a production-quality temporal video model. The configured face model's card warns of concept drift; it is a research signal only.
- An artifact-localization model. The current report exposes scored face-frame references, not explanations of which physical artifact caused a score.
- Speaker identity matching. It is intentionally excluded because there is no enrolled speaker baseline.
- Production credential storage/revocation, session persistence, account authentication, deployment, rate limiting, and production privacy/security controls. Session and public-key state is in memory and resets on restart; a previously created phone passkey may remain on the device.

This is a demonstration scaffold, not a KYC decision system. Do not use it to approve or reject real applicants.

## Requirements

- CPython 3.11.x, 64-bit.
- A current browser with camera access support. Chrome or Edge is recommended for the demo.
- First model warm-up downloads the small MediaPipe BlazeFace detector and several hundred MB from Hugging Face (the live face classifier; the optional voice classifier and tiny English transcription model download only after voice consent). Model caches stay outside Git in ignored locations. Keep internet access available for first use.
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
3. Continue to **QR + video**, agree to camera use, and start the camera. Place the bright phone QR in box 1 about 15–20 cm from the lens and hold it still until the viewer shows **QR LOCK**. Then move slowly through numbered boxes 2–6 in order. Recording starts at the lock and stops when all six QRs are scanned or after 1 minute; the complete clip is sampled through its final frame for AI face screening.
4. On the distinct **Voice** tab, optionally consent and start the voice check. Allow microphone access, then say the displayed fresh phrase once; a live waveform appears while recording for up to eight seconds. Or skip voice without enabling the microphone.
5. Open the **Report** tab after completing or skipping voice. QR sequence/path, phone proof, recorded-video face scores, temporal signal, phrase match, and audio anti-spoof signal are separate. A completed QR challenge remains **Review**; an incomplete required QR sequence is **Challenge failed**. The report includes an evidence hash in an in-memory simulated ledger.

For a local UI-only walkthrough, `/hidden/control` is disabled by default. Enable it only on loopback during development with `ENABLE_QA_CONTROLS=true` and `APP_ENV=development`; never enable it for a tunnel or production deployment. Its next-step button simulates UI progression only and does not skip server-side verification or create evidence.

### Operator recovery panel

`/admin` is a separate, passwordless operator page. It lists in-memory session metadata and offers one limited recovery action: skip a video-processing step that the server currently reports as still processing. The action requires an operator reason, is recorded with a timestamp, discards video-screening results, and advances the participant page to optional voice/report. The resulting report marks video and capture quality unavailable, records the override reason, and is **Inconclusive**—never passed. The page does not expose pairing tokens, QR payloads, or evidence.

Admin controls are disabled unless explicitly enabled in a non-production environment. Enable them in the same PowerShell window used to start the server:

```powershell
$env:APP_ENV = "development"
$env:ENABLE_ADMIN_CONTROLS = "true"
.\StartPrototype.bat
```

On the server computer, manually enter `http://127.0.0.1:8000/admin` in the browser. There is deliberately no link to this page in the participant UI. The page and every admin API reject non-loopback clients and requests bearing common reverse-proxy/tunnel headers, so a public tunnel cannot access the passwordless controls. Do not enable admin controls in production.

## Project files

- `backend/app/main.py` — API, in-memory session/challenge state, server-side QR frame decoding, checks, and report.
- `backend/app/screening.py` — live face-presence detection, in-memory recorded-video decoding, and pinned image-classifier adapter with frame evidence/consistency summary.
- `backend/app/voice_screening.py` — in-memory audio decoding, phrase transcription, and lazy audio anti-spoof classifier adapter.
- `backend/app/audit_ledger.py` — hash-only in-memory audit-chain simulator; not a blockchain.
- `backend/app/static/index.html` — laptop onboarding and capture experience.
- `backend/app/static/admin.html` — loopback-only operator session monitor and video-step recovery page.
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

- Live face-frame classifier: [`dima806/deepfake_vs_real_image_detection`](https://huggingface.co/dima806/deepfake_vs_real_image_detection), Apache-2.0, pinned to revision `29e4cf9efc543845610045f6ba7e88e5cf9d9301`. It classifies face crops as real/fake; its model card warns of concept drift due to older training data. Scores are not calibrated for this demo.
- Face localization: Google's MediaPipe BlazeFace short-range detector, loaded through the installed Apache-2.0 MediaPipe task runtime. The small hosted model is SHA-256 checked before use and cached outside Git.
- Optional audio classifier: [`Hemgg/Deepfake-audio-detection`](https://huggingface.co/Hemgg/Deepfake-audio-detection), Apache-2.0, pinned to revision `0d75271368ef2c7efd14831dc503c431f6aab0eb`. The card reports 95.45% accuracy on its stated evaluation set, but that metric is unverified and the card says more data is needed. Treat the output as an uncalibrated screening signal.
- Optional fresh-phrase transcription: [`Systran/faster-whisper-tiny.en`](https://huggingface.co/Systran/faster-whisper-tiny.en), MIT, pinned to revision `0d3d19a32d3338f10357c0889762bd8d64bbdeba`. This is phrase matching, not speaker identity verification. The model is downloaded only when voice consent is enabled.
- Audit trail: a clean-room, in-memory hash-chain simulator. It is neither a deployed blockchain nor persistent tamper-proof storage. It records only a digest and chain metadata, not the source report or raw samples.

The supplied detector repositories remain research references, not drop-in runtime dependencies:

- [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) is licensed **CC BY-NC 4.0**. Its Xception example expects 32 frames at 256 px and is configured around FaceForensics++ training. The benchmark's documented setup pins an older Python/PyTorch/CUDA stack, while this API targets Python 3.11; it should be evaluated in an isolated research environment, not installed into the API environment. Any detector checkpoint and its training-data terms must also be checked separately.
- [FaceForensics](https://github.com/ondyari/FaceForensics) releases its code under MIT, but its dataset has separate terms and access requires approval. Its published Xception classifier targets Python 3.6 / PyTorch 1.0.1 and is explicitly not fine-tuned for general compressed videos. Neither its dataset nor model assets are bundled here.

No code, checkpoint, or dataset from DeepfakeBench or FaceForensics is integrated. The current face/audio models have compatible licenses but are not independently validated for this application; runtime/download errors are reported as unavailable, not as successful checks. QR frames and recorded video are still client-submitted, so a modified client may fabricate them.
