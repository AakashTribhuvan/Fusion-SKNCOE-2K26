# Plan of Action — Deepfake-Resistant Identity Verification

> **Working draft:** edit freely. The challenge statement below is from the Mastercard CSB-02 brief supplied by the team. Owners, dates, and model choice remain open.

> **Implementation update:** The project uses Git and a shared Python environment rather than requiring Docker. The current FastAPI demo includes session creation/expiry, phone QR pairing, WebAuthn platform user verification, a fresh six-code QR sequence, server-side path-coherence checks, periodic live face-frame screening, optional consented phrase/audio screening, and a hash-only in-memory audit simulator. Model output remains a research signal; see `README.md` for sources and limitations.

## Challenge statement and selected direction

**Problem (Mastercard CSB-02):** Video/voice-based KYC onboarding is increasingly targeted by real-time deepfake injection attacks that bypass liveness checks. Detection must work on short clips without a prior baseline of the real person.

**Expected core deliverable:** A lightweight classifier that estimates real-versus-synthetic face likelihood from a short webcam clip or a permitted sample dataset, records per-frame scores and supported evidence, explains uncertainty without inventing artifact explanations, and is demonstrated inside a mock onboarding flow.

**Chosen system direction: laptop + phone.** The laptop/browser captures the short clip and presents the result. The phone displays a fresh, rotating QR challenge and authorizes a session-bound cryptographic proof after local platform user verification. The QR motion/sequence is an anti-replay liveness signal; the authenticator credential is a device-authorization signal. The classifier remains the primary deepfake check.

**Important boundary:** Phone biometrics do not tell the server who the person is, do not provide a biometric template, and do not prove that the face in the laptop video is genuine or belongs to that phone user. They locally authorize a platform credential. Depending on phone policy, the authenticator may accept a device PIN/passcode as well as biometrics; the server verifies the signed user-verification flag but does not learn which method was used. The optional voice flow checks a challenge phrase and an uncalibrated anti-spoof score; it does not identify the speaker.

## 1. Project goal

Build a hackathon-ready mock KYC flow whose primary output is a calibrated real-versus-synthetic face likelihood from a short clip, with inspectable evidence and uncertainty. Add two supporting signals: a fresh QR challenge that guides the phone through six numbered square targets and phone biometric user verification authorizing a signed session challenge. Return **Review** or **Inconclusive** whenever model confidence, capture quality, or security evidence is insufficient; do not claim certainty from one signal.

### Proposed verification signals

1. **Primary face detector:** analyze short, sampled webcam video frames with a lightweight real-versus-synthetic baseline; report clip-level score, quality/uncertainty, and evidence viewers can inspect. No enrolled face baseline or identity matching is assumed.
2. **Phone local user verification:** a phone platform authenticator locally verifies its user and permits its protected credential key to sign a server nonce. Store only the public credential and verification metadata; never request biometric templates. Depending on device policy, the local verifier may accept biometrics or a device PIN/passcode; the server receives a signed user-verification result, not the biometric modality.
3. **Randomized QR liveness:** display a fresh, expiring QR sequence on the phone and ask the user to move it through six square targets in order. Validate sequence, timing, and server-observed target entries; treat this as replay resistance, not proof of genuineness.
4. **Decision and explanation:** keep classifier, capture-quality, phone-proof, and QR results separate. Invalid signatures or replayed challenges fail those checks; uncertain model outputs request recapture or human review.
5. **Optional audio:** after separate explicit consent, request a fresh three-word phrase, compare its transcription, and report an uncalibrated AI-voice screening score. Do not record without microphone permission, retain audio, or enroll/match a speaker.
6. **Audit:** append only a digest and chain metadata to the in-memory hash-chain simulator. Do not describe it as a real blockchain or put raw media, transcripts, biometrics, or PII in it.

## 2. Scope and guardrails

### First prototype (must-have)

- Teammates can clone the Git repository, create a Python 3.11 virtual environment, install `requirements.txt`, and run the API using the README instructions. Docker is optional and is not required for the current prototype.
- The current browser client and FastAPI service support session creation/expiry, phone QR pairing, a randomized six-code challenge, capture UI, and a transparent evidence report.
- Run the pinned face classifier on periodic face crops from the live QR capture; retain timestamped score/region evidence and compare score/face-count variation across frames. Keep model failures explicit and avoid claiming calibrated predictions.
- Define the API report contract and privacy boundaries so work can be split without implying that submitted camera frames establish identity.
- Keep the short-clip classifier under evaluation on current unseen generators and capture conditions. Produce inspectable frame/region evidence; do not invent artifact explanations from a raw score.
- Verify the WebAuthn phone proof on an HTTPS deployment/device. Keep the credential public key backend-side and confirm that user verification is required; never transmit a biometric sample.
- Keep the QR target sequence as a supporting anti-replay signal and evaluate whether the six-box path is usable and measurable on ordinary devices.
- Keep voice capture optional and separately consented; report phrase match and anti-spoof screening separately. Speaker identity matching remains out of scope.
- Label the simulated audit log accurately and keep it hash-only/in-memory.
- Demo genuine and controlled replay/inconclusive cases, report measured results only, and show limitations.

### Stretch goals (only after the core flow works)

- Native Android biometric-authorized signing with Android Keystore.
- Face/background segmentation comparison if it improves held-out results.
- More robust cross-device, lighting, compression, and unseen-attack evaluation.

### Out of scope for the first demo

- Training multiple large models from scratch.
- Storing raw biometric templates or keeping recordings longer than needed for consented evaluation.
- Treating a deepfake score, voice match, phone biometric, or QR motion alone as proof of civil identity.

## 3. Phased execution

| Phase | Work | Exit criteria / deliverable | Owner |
|---|---|---|---|
| 0. Align | Confirm challenge framing, demo audience, team size, time budget, device/OS, data policy, and minimum demo | Written scope, responsibilities, and success criteria | TBD |
| 1. Environment | Clone the Git repository; use Python 3.11 and a project virtual environment; install the pinned/ranged requirements; keep Docker optional | `CheckRequirements.bat` passes; documented Uvicorn command starts the API and `/health` responds | TBD |
| 2. Contracts and session flow | Define session lifecycle, report schema, one-time server nonce, upload limits, and error states | Session create/read/expire works; report separates classifier, QR, phone proof, and capture quality | TBD |
| 3. Onboarding UI | Complete consent, phone-connect, short-video capture, progress, and evidence-report screens | Full user journey is understandable and mock checks are labeled | TBD |
| 4. Face video baseline (prototype wired; evaluation pending) | Run a pinned licensed classifier on periodic face crops; aggregate scores; retain frame/region/time references; evaluate on current unseen generators with identity/source-separated splits | Repeatable output and honest limitations; calibration and held-out metrics remain open | TBD |
| 5. QR challenge and phone proof (implemented; device test pending) | Validate fresh rotating codes, server-measured ordered square targets, expiry/replay rules; require local WebAuthn user verification | Invalid, expired, and reused proofs fail; device flow is confirmed over HTTPS; biometric data stays on-device | TBD |
| 6. Optional audio (prototype wired; evaluation pending) | After explicit consent, record a bounded phrase, transcribe it, and report an Apache-2.0 anti-spoof model score without speaker enrollment | Phrase and anti-spoof checks stay separate; limitations and unavailable failures are visible | TBD |
| 6a. Simulated audit (implemented; production design pending) | Hash a report summary and append it to an in-memory chain without raw media/PII | Clear not-a-blockchain label; hash-chain integrity endpoint | TBD |
| 7. Decision policy (prototype wired; validation pending) | Keep per-check outcomes separate; show the overall Review or Challenge failed result with evidence, quality flags, and limitations | QR failure cannot be masked by a model score; completed challenges remain Review | TBD |
| 8. Evaluation and hardening | Run attack, quality, security, and integration checks; fix critical issues; rehearse demo | Results table, known limitations, stable demo runbook | TBD |

## 4. Current architecture and choices

- **Current frontend:** static HTML/CSS/JavaScript served by FastAPI. Keep this until a framework migration clearly improves team development or testing.
- **Current API:** Python 3.11 + FastAPI, with in-memory session state and polling for phone pairing/challenge state.
- **Primary video check:** OpenCV/MediaPipe sample/locate faces; a pinned Apache-2.0 image classifier scores live face crops. Preserve timestamped per-frame scores and boxes; the temporal summary is a basic consistency signal, not a trained video detector. The model is not yet calibrated or validated for deployment.
- **Phone local user verification:** the phone page uses a WebAuthn platform credential with user verification required, and the server verifies the assertion. The operating system may accept a biometric or a device credential; the biometric modality is not reported to the server. A physical phone run still requires HTTPS and device validation.
- **Device strategy:** hybrid. Laptop/browser handles onboarding, webcam capture, and results; phone displays the QR challenge and signs a fresh server challenge after local user verification.
- **Voice:** optional separate-consent flow with a fresh phrase, tiny English transcription, and Apache-2.0 audio anti-spoof screening. No speaker identity matching or voiceprint.
- **Audit:** clean-room in-memory hash-chain simulator; not a real blockchain and not durable tamper-proof storage.
- **Data/auth:** Supabase PostgreSQL/Auth if suitable; enable Row Level Security on exposed tables. Keep service secrets server-side.
- **Development/reproducibility:** Git is the shared workflow; Python 3.11 virtual environments and `requirements.txt` are the baseline. Docker remains optional. Keep model weights and datasets out of Git.

### Initial API surface (draft)

- `POST /api/sessions` — create a short-lived onboarding session.
- `GET /api/sessions/{id}` — retrieve session state.
- `POST /api/sessions/{id}/challenge` — issue a fresh nonce/QR challenge.
- `POST /api/sessions/{id}/phone/webauthn/...` — register/verify a signed platform credential.
- `POST /api/models/warmup` and `/api/models/status` — warm/report licensed lazy-loaded models.
- `POST /api/sessions/{id}/challenge/{challenge_id}/scan` — submit bounded live JPEGs for server QR decoding and periodic face-frame screening.
- `GET /api/sessions/{id}/challenge/{challenge_id}/audio-prompt` and `POST .../audio` — return a fresh phrase and receive a bounded, consented audio recording.
- `POST /api/sessions/{id}/evidence` — aggregate per-signal results.
- `GET /api/sessions/{id}/result` — return per-signal evidence and decision.
- `GET /api/audit/ledger` — inspect and verify the in-memory hash-chain simulator.

## 5. Accuracy and testing plan

### Evaluation principles

- Split by source video, identity, and recording session where possible; never put near-duplicate frames from one source across train and test.
- Keep a held-out set with recording conditions and attack methods not used for tuning.
- Report per-signal metrics and sample counts, not only a combined score.
- Calibrate thresholds on validation data for the costs of false acceptance and false rejection; publish thresholds and uncertainty.
- Compare face-only against face + background/temporal features. Remove background features if they do not help held-out performance.
- Treat low-quality capture as **Inconclusive / recapture**, not as detected fraud.

### Metrics to report

- Video/audio spoof detection: ROC-AUC plus thresholded precision, recall, false-accept rate (FAR), and false-reject rate (FRR).
- Optional audio spoof detection: report held-out metrics and threshold separately from fresh-phrase transcription; speaker identity matching is out of scope.
- Challenge/security checks: valid acceptance rate; expired/reused/invalid proof rejection rate.
- Whole flow: completion rate, recapture rate, latency, and review rate.

### Minimum scenario matrix

| Scenario | Expected handling |
|---|---|
| Genuine user, ordinary lighting/noise | Pass or low risk when all required checks pass |
| Face-swap/manipulated clip | Flag only when detector evidence supports it; otherwise record a miss |
| Prerecorded clip or replay | Fresh challenge/temporal checks should reject or raise review; measure misses |
| Genuine face with portrait blur/compression | Do not infer manipulation from background/static artifacts alone |
| Replayed, synthesized, or converted speech | Evaluate the optional anti-spoof signal; do not perform speaker matching |
| Poor lighting, short clip, noisy audio | Request recapture or mark inconclusive |
| Invalid signature, expired/reused nonce or QR | Reject that security check |
| Disconnect, duplicate sessions, oversized upload | Expire/resume safely; isolate sessions; enforce limits |

## 6. Security, privacy, and demo readiness

- Obtain informed consent for camera, microphone, and any evaluation recordings; state retention and deletion policy.
- Minimize and time-limit raw media storage; protect transport and access; redact unnecessary data from logs.
- Keep private keys on device and service-role keys on the backend. Bind signatures and challenges to session, expiry, and nonce; reject replay.
- Add rate limits, upload size/type limits, timeouts, session expiry, access controls, and audit logging.
- Use only consented/licensed evaluation data; review dataset terms before downloading or presenting results.
- Demo one complete journey, then show attack/inconclusive cases and a concise limitations slide.

## 7. Team split (adjust to actual team)

- **Video/ML:** video preprocessing, pretrained baseline, background ablation, metrics.
- **Voice:** speaker similarity, anti-spoof baseline, test set and metrics.
- **Mobile/security:** QR pairing, biometric prompt, key registration, signature verification.
- **Integration/product:** Git workflow, API/session contracts, frontend, decision policy, Supabase controls, demo.

If the team is smaller, prioritize integration and the complete flow; mock unfinished signals clearly instead of presenting mock values as measured results.

## 8. Decisions for the team to fill in

- Challenge/track wording and judging rubric: **[fill in]**
- Team members and owners: **[fill in]**
- Deadline and available build hours: **[fill in]**
- Target phone(s), browser(s), and GPU availability: **[fill in]**
- Data collection, consent, retention, and deletion plan: **[fill in]**
- Must-have demo path and fallback path: **[fill in]**
- Numeric success targets and threshold-selection costs: **[fill in]**

## 9. Selected device strategy: laptop + phone

### Selected prototype: hybrid, with clear device roles

Use the **laptop/browser as the onboarding and live-capture station** (webcam, classifier, instructions, and results). Use the **phone as a separate challenge/authenticator device** (scan a session QR, complete local platform user verification, and authorize a fresh challenge with a WebAuthn credential). The backend verifies the signed assertion and session binding. Keep the face classifier and phone proof independent; local phone verification is not a face match against the laptop camera. Optional microphone use requires separate consent and permission.

The current demo implements the hybrid pairing, phone proof, live frame screening, optional voice check, and simulated audit flow. Cross-device use requires HTTPS. A laptop-only mode remains a development fallback, but it omits phone signals. A phone-only design is outside the chosen scope.

### Critical distinction: OS biometrics vs. our own face matching

- **Phone platform user verification:** WebAuthn allows the browser's authenticator to require local user verification before using a scoped credential. Depending on OS policy, the local method may be biometric or device PIN/passcode; the server verifies a signed user-verification flag, not the biometric or exact method.
- **Custom face enrollment and comparison (separate optional feature):** If the project enrolls a reference face image/template, a backend model can compare a fresh laptop frame against it. Supabase can store and return authorized records/files, but it does not perform face recognition. We would need to build or integrate the face detector/embedding matcher, define enrollment quality and matching thresholds, and protect the reference data. A face embedding is still sensitive biometric data.
- **Do not claim equivalence:** A phone's local biometric check does not tell us whether the person in the laptop video is the same person who enrolled the phone. A server-side custom face match could provide a separate signal, but it can be spoofed and must be validated with liveness/attack testing and human review for uncertain/high-impact decisions.

### Recommendation for the first version

Do **not** put custom face templates in Supabase as a default. The Mastercard task specifically says there is no prior real-person baseline, so the primary classifier should estimate manipulation likelihood without enrolled-face matching. Use consented/lawfully accessed real and synthetic clips for training/evaluation. If identity matching later becomes a separate requirement, make it an explicit experiment with consent, minimal protected templates, backend-only access, deletion controls, and measured false-match/false-non-match rates. Supabase access controls do not substitute for application-level privacy or biometric validation.

**Decision status:** hybrid selected; WebAuthn phone user verification and the six-box QR target sequence are supporting signals; short-clip real-versus-synthetic classification is the primary challenge deliverable. Optional phrase/audio screening is implemented as a separate consented signal. Custom enrolled-face and speaker matching remain out of scope. The current detector outputs are not validated or calibrated.
