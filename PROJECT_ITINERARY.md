# Fusion Prototype — Build Itinerary

This is an editable walkthrough of what is in the project, how the pieces fit together, and what remains to build for the Mastercard CSB-02 challenge.

## Interface direction

The refreshed pages use a restrained editorial hierarchy, consistent spacing, clear state labels, and a quiet green/ivory palette. The visual direction takes broad inspiration from polished product sites such as [Stripe](https://stripe.com/) and [Linear](https://linear.app/) without copying their brand assets or page layouts.

## What the prototype is intended to demonstrate

The brief asks for a short-clip real-versus-synthetic face check that can work without a previously enrolled face, explain evidence behind its result, and fit into a mock onboarding flow. The selected supporting checks are a phone platform-authenticator proof and a changing QR challenge that the laptop camera scans while the phone moves in a figure eight.

The supporting checks are not the deepfake detector. Phone biometric or device-PIN verification only authorizes a phone credential locally; it does not prove who appears in the laptop video. The prototype records a short camera clip during the QR challenge, samples detected face crops from that clip, and runs a pinned image classifier. It is not a trained temporal model, is not calibrated, and does not localize the physical artifact behind a score.

## Current build, step by step

1. **Project setup:** Python 3.11, a shared `requirements.txt`, and `.venv` make the backend reproducible across teammates. `CheckRequirements.bat` checks the interpreter version, package consistency, and imports.
2. **Cross-network HTTPS:** `backend/run_tunnel.py` starts a temporary Cloudflare Quick Tunnel and receives a random `trycloudflare.com` hostname. It configures that exact origin and RP ID for WebAuthn, then starts the API on loopback only. The laptop and phone can be on unrelated networks.
3. **Local web app:** FastAPI serves the API and the laptop and phone web pages. Uvicorn runs the server. Session state is currently held in memory and expires after five minutes.
4. **Pair the phone:** The laptop creates a session-specific pairing QR. Scanning it opens the phone page on the same random HTTPS hostname and associates that phone browser with the current session.
5. **Verify the phone locally:** WebAuthn creates or uses a platform credential. The phone OS requests local user verification (biometric or device passcode, depending on device policy); the API validates the signed challenge and retains the public credential in memory. The biometric itself is not sent to the API.
6. **Run the live QR + video check:** The phone cycles six random, session-bound QR signals. The user first holds the phone QR close and still in the laptop camera until it recognizes a valid signal and lets focus/exposure settle; only then does the ten-second video recording begin. The user follows the on-screen figure-eight guide while keeping the QR facing the camera. OpenCV decodes separate downscaled preview frames for QR identity/path. With explicit consent, the recorded clip is sent after capture and sampled about once per second for face detection and classification. The clip, extracted frames, and face crops are processed in memory and discarded after analysis; derived score evidence remains in session memory.
7. **Run optional voice screening:** A separate Voice tab gives its own directions and consent. After the user explicitly starts, the browser records up to eight seconds of a fresh random phrase. Whisper tiny.en checks the phrase and the audio classifier gives a research anti-spoof signal. Audio is processed in memory and discarded. Voice may be skipped.
8. **Show the evidence report:** The Report tab displays the overall **Review** or **Challenge failed** result and keeps phone verification, QR sequence/movement, recorded-video face screening, and optional audio checks separate. A QR pass is not a verified identity decision.

## Technology map

| Technology | Used for | Current status |
|---|---|---|
| Python 3.11 | Backend runtime | In use |
| FastAPI | Health, session, pairing, WebAuthn, QR, and result endpoints; serves static pages | In use |
| Uvicorn | Runs the local ASGI web server | In use |
| HTML, CSS, JavaScript | Gated Phone, QR + video, Voice, and Report tabs; phone QR/authenticator page | In use |
| OpenCV QRCodeDetector | Decodes QR codes and positions from downscaled camera preview frames | In use; preview frames are processed in memory by the API and not saved |
| PyAV | Decodes the consented ten-second browser recording into in-memory samples for face analysis | In use; the clip is discarded after analysis |
| WebAuthn (`webauthn` Python package + browser credentials API) | Registers a phone platform credential and verifies fresh signed challenges after local user verification | In use; cross-device use needs HTTPS |
| `qrcode` + Pillow | Produces pairing and rotating challenge QR images | In use |
| `requirements.txt` + `.venv` | Pins compatible dependency ranges and isolates the team's Python packages | In use |
| `CheckRequirements.bat` | Checks Python 3.11, dependency consistency, and required imports | In use |
| Cloudflare `cloudflared` Quick Tunnel | Gives the phone and laptop a shared random HTTPS URL without requiring the same Wi-Fi | Started for each demo run; closes with launcher |
| `backend/run_tunnel.py` | Starts the tunnel and API, sets the WebAuthn origin, waits for HTTPS health, and supervises process cleanup | In use |
| Windows Job Object | Terminates both child processes if the launcher exits, including when its console is closed | Required by launcher before any tunnel is opened |
| `StartPrototype.bat` | Checks setup and starts the managed public demo | In use |
| MediaPipe Tasks | Detects faces in sampled live frames | In use; detector regions feed the face-crop classifier |
| PyTorch + Transformers | Runs the pinned face-crop classifier and optional audio anti-spoof classifier | Wired; model evaluation and calibration remain pending |
| faster-whisper | Transcribes the fresh phrase in the optional voice flow | Wired; does not identify or match a speaker |
| Hash-chain simulator | Records a digest of the report and chain metadata in memory | In use; not a real or persistent blockchain |
| Persistence | Session, credential, evidence, and audit state | In memory only; hosted storage/auth is not integrated |
| Git | Shared source history and teammate collaboration | Project is in a Git repository |

## Run the current demo on Windows

1. Install 64-bit Python 3.11 and a current Chrome or Edge browser.
2. If this is a fresh checkout, create the environment and install dependencies:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Double-click `StartPrototype.bat` (or run it from PowerShell). On first run it downloads Cloudflare's Windows tunnel binary to the ignored `.tools` folder and verifies its SHA256 against the official release metadata. It then launches a temporary HTTPS address and the API, verifies local API health and a registered Cloudflare connection, and opens the browser. If this computer cannot reach the public URL for its own health check, the launcher warns and leaves the user able to test the URL from the browser or phone.
4. Open the generated address on the laptop. Scan the session QR with the phone, even if the devices use different networks.
5. Complete local phone verification, open **QR + video**, consent, and hold the phone QR 15–20 cm from the laptop camera until the first valid QR locks. Only then does the ten-second recording start; follow the downloadable on-screen figure-eight guide slowly. The clip is sampled in memory for face screening and discarded.
6. Open **Voice** for the separately consented optional phrase recording, or skip it. Review the **Report** tab as a prototype signal only; it does not make a KYC decision.
7. Press Ctrl+C or close the launcher console to stop both the API and tunnel. The random address is temporary and stops serving this project after the connector exits.

## Device and security notes

- The API binds only to `127.0.0.1`; Cloudflare's connector makes the outbound connection and relays requests through HTTPS. No router port-forwarding is needed.
- WebAuthn uses the per-run tunnel hostname as its exact origin and RP ID. Each new random hostname is a different WebAuthn origin; a phone may ask to register its platform credential again on the next run.
- A Quick Tunnel is public while active. The hostname is random, but it is not an access-control policy. Keep the launcher open only for the demo and do not submit real identity data. The launcher requires a Windows Job Object and refuses to start if it cannot guarantee child-process cleanup.
- Cloudflare Quick Tunnels are temporary testing tunnels without a production availability guarantee. Use a named tunnel and appropriate access controls for a persistent deployment.
- Session and credential records are in memory and reset when the API stops. QR codes are decoded server-side from submitted frames, but a modified client could still submit fabricated frames or video.
- With explicit camera consent, the browser uploads the ten-second recording after QR lock; the API samples it in memory for face screening and discards the clip. Separate downscaled camera stills are sent for QR decoding and position measurement. Optional microphone audio is captured only after its own consent and explicit user action, processed in memory, and discarded. Derived report evidence remains in session memory until expiry.

## Next development itinerary

1. **Evaluate the live face baseline:** define permitted held-out real and synthetic data, measure false-accept/false-reject behavior and performance under relevant capture conditions, and document provenance/limits. DFDC is a reference dataset, not proof of real-world performance.
2. **Improve temporal evidence:** evaluate a licensed temporal model and artifact-localization methods separately; do not infer artifact explanations from image-classifier scores.
3. **Evaluate the optional audio flow:** test phrase transcription and anti-spoof performance independently. Speaker identity matching remains out of scope.
4. **Exercise the QR flow:** assess replay, ordering, expiry, dropped frames, usability, and the limitation that a modified client can submit fabricated images.
5. **Complete physical-device checks:** run pairing, WebAuthn, camera, and optional microphone flows on supported devices over HTTPS; document platform differences and recovery behavior.
6. **Decide persistence and deployment:** define credential revocation, secure storage, privacy retention, deployment, and operational controls. Supabase is an available option, not an existing implementation.

## References to add as the implementation is validated

- Challenge brief / Mastercard CSB-02: deepfake-resistant identity verification for digital onboarding (see the supplied problem statement).
- [Deepfake Detection Challenge (DFDC) dataset](https://ai.meta.com/datasets/dfdc/).
- [Web Authentication: An API for accessing Public Key Credentials](https://www.w3.org/TR/webauthn-3/).
- [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).
- [FastAPI documentation](https://fastapi.tiangolo.com/).
- [PyTorch installation selector](https://pytorch.org/get-started/locally/).
