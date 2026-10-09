# Plan of Action — Deepfake-Resistant Identity Verification

> **Working draft:** edit freely. Scope, owners, dates, and final technology choices are open decisions.

> **Implementation update (2026-10-09):** A local FastAPI demo now has session creation/expiry, phone QR pairing, a randomized six-code motion challenge, browser capture, and an evidence report. Face deepfake, speaker, and audio anti-spoof models remain unavailable; the demo does not automatically approve a session. See `README.md` for run instructions and current limitations.

## 1. Project goal

Build a hackathon-ready onboarding prototype that combines independent evidence to make identity verification harder to spoof. The prototype should explain what each check did and should return **Pass**, **Review**, or **Inconclusive** rather than claiming certainty from a single model.

### Proposed verification signals

1. **Phone authorization:** a registered phone uses its native biometric prompt to authorize a cryptographic signature. Store a public key/credential metadata on the server; do not collect or store fingerprint or face templates.
2. **Live video checks:** analyze face-region artifacts and temporal consistency; evaluate background/boundary evidence as an experiment against a face-only baseline.
3. **Voice checks:** keep speaker similarity and audio spoof likelihood as separate results. RVC or other voice-conversion tools may generate consented test samples; they are not detectors.
4. **Randomized challenge:** issue a fresh, expiring QR/symbol sequence and motion instruction; check decoded sequence, order, timing, and observed movement. Treat it as replay resistance, not proof of genuineness.
5. **Decision and explanation:** combine policy rules and calibrated evidence. Invalid signatures or replayed challenges fail those checks; uncertain model outputs request another sample or human review.

## 2. Scope and guardrails

### First prototype (must-have)

- Docker Compose starts a web client and API reproducibly.
- A user can create an onboarding session, grant camera/microphone access, and see a clear progress/results screen.
- QR pairing and challenge expiry/replay prevention work end to end; mock the phone proof until the mobile component is ready.
- API contracts and a structured verification report are agreed early so components can be developed independently.
- At least one pretrained video signal and one voice signal are integrated only after the end-to-end flow works.
- Demo includes genuine and controlled attack/inconclusive cases, with limitations shown.

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
| 1. Environment | Install/verify Docker Desktop + WSL 2; scaffold React/Vite/TypeScript and FastAPI; add Compose and health checks | One documented command starts both services; health endpoint responds | TBD |
| 2. Contracts and session flow | Define session lifecycle, API response schema, expiring server nonce, upload limits, and error states | Session create/read/expire works; frontend can show a mock verification report | TBD |
| 3. Onboarding UI | Build welcome/consent, phone-connect QR, live-check, and result screens; implement capture permissions and clear progress | Full user journey works with mocked signals | TBD |
| 4. Challenge and phone proof | Implement one-time QR challenge and replay/expiry checks; then integrate mobile key registration and signed challenge if time permits | Valid proof succeeds; expired, reused, and invalid signatures fail | TBD |
| 5. Video baseline | Sample short clips, detect/align face, run a pretrained detector; record frame/video evidence; compare face-only with optional background/temporal features | Repeatable baseline and held-out metrics; no unsupported depth claims | TBD |
| 6. Voice baseline | Capture randomized phrase; measure speaker similarity and spoof score independently; test genuine, replay, synthetic/converted audio | Separate calibrated outputs and documented failure cases | TBD |
| 7. Decision policy | Define per-check outcomes and escalation rules; build readable report with evidence, quality flags, and limitations | Pass/Review/Inconclusive behavior is consistent and explainable | TBD |
| 8. Evaluation and hardening | Run attack, quality, security, and integration checks; fix critical issues; rehearse demo | Results table, known limitations, stable demo runbook | TBD |

## 4. Suggested architecture (to validate)

- **Frontend:** React + Vite + TypeScript.
- **API:** Python + FastAPI; WebSocket or polling for session progress.
- **Video:** OpenCV/MediaPipe for preprocessing and landmarks; PyTorch/pretrained detector for inference.
- **Voice:** pretrained speaker embeddings plus a separate spoof countermeasure.
- **Mobile:** React Native with native Android integration as required; Android BiometricPrompt + authentication-gated Keystore key.
- **Data/auth:** Supabase PostgreSQL/Auth if suitable; enable Row Level Security on exposed tables. Keep service secrets server-side.
- **Local orchestration:** Docker Compose. Verify Windows/WSL2 GPU support separately; keep model weights and datasets out of container images and Git.

### Initial API surface (draft)

- `POST /api/sessions` — create a short-lived onboarding session.
- `GET /api/sessions/{id}` — retrieve session state.
- `POST /api/sessions/{id}/challenge` — issue a fresh nonce/QR challenge.
- `POST /api/sessions/{id}/phone-proof` — verify a signed challenge.
- `POST /api/sessions/{id}/video` and `/audio` — submit bounded samples.
- `GET /api/sessions/{id}/result` — return per-signal evidence and decision.

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
- Speaker verification: false-match/false-non-match rates and threshold used.
- Challenge/security checks: valid acceptance rate; expired/reused/invalid proof rejection rate.
- Whole flow: completion rate, recapture rate, latency, and review rate.

### Minimum scenario matrix

| Scenario | Expected handling |
|---|---|
| Genuine user, ordinary lighting/noise | Pass or low risk when all required checks pass |
| Face-swap/manipulated clip | Flag only when detector evidence supports it; otherwise record a miss |
| Prerecorded clip or replay | Fresh challenge/temporal checks should reject or raise review; measure misses |
| Genuine face with portrait blur/compression | Do not infer manipulation from background/static artifacts alone |
| Replayed, synthesized, or converted speech | Evaluate speaker match and spoof detector separately |
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
- **Integration/product:** Docker, API/session contracts, frontend, decision policy, Supabase controls, demo.

If the team is smaller, prioritize integration and the complete flow; mock unfinished signals clearly instead of presenting mock values as measured results.

## 8. Decisions for the team to fill in

- Challenge/track wording and judging rubric: **[fill in]**
- Team members and owners: **[fill in]**
- Deadline and available build hours: **[fill in]**
- Target phone(s), browser(s), and GPU availability: **[fill in]**
- Data collection, consent, retention, and deletion plan: **[fill in]**
- Must-have demo path and fallback path: **[fill in]**
- Numeric success targets and threshold-selection costs: **[fill in]**

## 9. Architecture decision to resolve: laptop, phone, or hybrid

### Recommended prototype: hybrid, with clear device roles

Use the **laptop/browser as the onboarding and live-capture station** (camera, microphone, instructions, and results). Use the **phone as a separate possession/authenticator device** (scan a session QR, run its native biometric prompt locally, and authorize a fresh challenge with a device-held key). The backend verifies the signature and session binding. Keep the video/voice checks independent; the phone biometric prompt is not a face match against the laptop camera.

This gives the demo a meaningful second device and lets the laptop handle media capture. A laptop-only mode can be the fallback if mobile integration is not ready. A phone-only design is a different product scope: it simplifies capture to one device but loses the cross-device pairing/independent-device signal and would require redesigning the journey.

### Critical distinction: OS biometrics vs. our own face matching

- **Phone OS biometric (recommended for the phone-auth layer):** Android/iOS expose a local success/failure result or allow a protected key operation. The app does not receive the fingerprint/Face ID template to upload. The backend receives and verifies a public-key signature, not biometric data.
- **Custom face enrollment and comparison (separate optional feature):** If the project enrolls a reference face image/template, a backend model can compare a fresh laptop frame against it. Supabase can store and return authorized records/files, but it does not perform face recognition. We would need to build or integrate the face detector/embedding matcher, define enrollment quality and matching thresholds, and protect the reference data. A face embedding is still sensitive biometric data.
- **Do not claim equivalence:** A phone's local biometric check does not tell us whether the person in the laptop video is the same person who enrolled the phone. A server-side custom face match could provide a separate signal, but it can be spoofed and must be validated with liveness/attack testing and human review for uncertain/high-impact decisions.

### Recommendation for the first version

Do **not** put custom face templates in Supabase as a default. First prove the flow with phone-key proof + laptop live video analysis + randomized challenge; use consented test data and ephemeral processing. If the team confirms that matching against an enrolled face is central to the problem, make it an explicit later experiment: enroll with clear consent, store the minimum needed (prefer a protected template over indefinitely retained raw images), restrict reads to a backend service, encrypt and set a short retention/deletion policy, and test false-match/false-non-match rates across users and capture conditions. Supabase RLS and private Storage policies can restrict access, but they do not substitute for application-level encryption, access design, retention, or biometric validation. Keep service-role credentials backend-only.

**Decision status:** recommended hybrid direction; team to confirm. Custom enrolled-face matching: defer unless required by the official challenge statement.
