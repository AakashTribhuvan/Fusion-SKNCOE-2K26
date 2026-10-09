# Pitch Deck — Deepfake-Resistant Identity Verification

> **Editable pitch draft.** Replace bracketed text with team-specific details. No performance numbers are claimed until measured.

> **Prototype update (2026-10-09):** The current build demonstrates a session-bound phone QR challenge and transparent review report. It does not yet include native biometric proof or validated face/voice models. Do not present those capabilities as implemented.

## Slide 1 — Title

**Deepfake-Resistant Identity Verification for Digital Onboarding**
Team: **[team name]** · Challenge/track: **[confirm official wording]** · Institution: **[fill in]**

**One-line pitch:** A layered onboarding flow that checks device authorization, live video, voice, and a fresh physical challenge, then explains what it could and could not verify.

## Slide 2 — Problem statement

- Remote onboarding depends on camera and microphone evidence that can be replayed, manipulated, or synthetically generated.
- A single deepfake classifier can be uncertain and may fail on unfamiliar generators, compression, devices, or lighting.
- A phone biometric prompt, by itself, only reports local user verification; it does not prove that the person on camera is the account holder.
- **Problem to solve:** How might we make remote identity checks more resilient to replay and synthetic-media attacks while keeping legitimate users informed and giving uncertain cases a safe review path?

**Evidence to add:** [target workflow, threat examples, stakeholder/user research, and properly sourced figures].

## Slide 3 — Proposed solution

An explainable, multi-signal verification workflow:

1. Pair a registered phone with the onboarding session.
2. Ask the device to authorize a fresh server challenge using its native biometric prompt and a device-held cryptographic key.
3. Analyze a short webcam sample for face/video manipulation indicators and temporal consistency.
4. Compare a short randomized spoken phrase with an enrolled speaker reference and separately check for replay/synthetic-audio signals.
5. Present an expiring randomized QR/symbol challenge and verify its sequence, timing, and observed motion.
6. Return **Pass**, **Review**, or **Inconclusive** with per-check evidence and next steps.

**Key distinction:** These checks provide risk signals; they do not individually prove civil identity or guarantee that media is genuine.

## Slide 4 — User journey and system flow

`Consent → Create session → Pair phone → Complete randomized video/voice challenges → Verify evidence → Pass / Review / Recapture`

**Diagram placeholder:** [Add flow diagram with browser, mobile app, API, model services, database, and trust boundaries.]

**Device choice (current recommendation):** hybrid. Laptop/browser handles onboarding, webcam/microphone capture, and results. Phone handles QR pairing and local biometric-authorized signing of a one-time challenge. Phone biometrics are not uploaded or compared with the laptop face. If mobile integration blocks the demo, use a laptop-only fallback and label phone proof as unavailable/mock.

## Slide 5 — Technology approach

| Layer | Approach | Output |
|---|---|---|
| Web | Current prototype: static HTML/CSS/JavaScript served by FastAPI | Pairing, capture UI, progress, evidence report |
| API/orchestration | FastAPI, session-bound nonces, bounded uploads | Session state and verification contract |
| Phone | React Native plus native Android BiometricPrompt/Keystore where needed | Signed, expiring challenge proof; biometric data stays local |
| Video | OpenCV/MediaPipe preprocessing; benchmark a pretrained model | Face/video evidence; optional background/temporal ablation |
| Voice | Pretrained speaker embedding plus separate anti-spoof model | Similarity and spoof likelihood as distinct signals |
| Data/security | PostgreSQL/Supabase if selected; backend-only secrets; access policies | Minimal session metadata and protected results |
| Runtime | Git + Python 3.11 virtual environment + `requirements.txt`; Docker optional | Teammates can install and run the same API locally |

**Enrollment decision:** A custom reference-face comparison is distinct from phone OS biometrics. Supabase can store an authorized reference image or embedding, but it does not compare faces; a backend matcher is required. Keep this feature out of the first build unless the challenge requires it. If included, obtain explicit consent, minimize/expire stored data, restrict it to backend-only access, and evaluate false matches and false non-matches. Never imply that the phone biometric proves the laptop-camera subject is the registered person.

## Slide 6 — Decision logic and explainability

- Use explicit rules for cryptographic proof, expiry, and replay; do not average away critical failures.
- Keep signal outputs separate: phone proof, face/video, speaker similarity, audio spoof, challenge validity, and capture quality.
- **Pass:** required checks satisfy validated thresholds.
- **Review:** meaningful risk signal or conflicting evidence; request human review or a fresh challenge.
- **Inconclusive:** insufficient/poor-quality evidence; ask for recapture.
- Show evidence and model limitations; never present illustrative scores as measured results.

## Slide 7 — Feasibility

- A working vertical slice is feasible with the current FastAPI/browser prototype and a shared Python 3.11 setup; first complete and validate the flow before integrating models.
- Integrate one baseline at a time, then evaluate whether additional signals improve held-out performance.
- Dataset options include the DFDC video dataset and ASVspoof speech-spoofing datasets, subject to access and license/terms review.
- Keep model training modest: use a small permitted subset and inference-first development; available laptop memory/GPU capacity may constrain training.
- **Prototype feasibility evidence to add:** [team skills, machine specs, package/model choice, latency target, integration demo].

## Slide 8 — Viability and adoption path

- Potential fit: organizations that need remote onboarding and face elevated replay or synthetic-media risk.
- Initial value proposition: add independent signals and explainable escalation to an existing onboarding flow.
- Deployment path: prototype → controlled usability/security evaluation → limited pilot → integration with a real KYC provider and compliance review.
- Open questions: [buyer/user], [integration cost], [latency], [accessibility], [privacy/legal requirements by market], [operational review cost].
- Do not claim production readiness or regulatory compliance until validated with qualified stakeholders.

## Slide 9 — Research and references

### Research questions

- Do face/background/temporal signals improve results over a face-only baseline on held-out recordings?
- How does performance change across unseen generators, devices, lighting, compression, and demographic groups?
- Does the randomized challenge measurably reduce replay success without making legitimate onboarding too difficult?
- How do speaker matching and spoof detection behave independently under replay, synthesis, voice conversion, and channel variation?
- What capture, retention, consent, accessibility, and regulatory constraints apply to the intended deployment market?

### Starting references (verify licensing and cite in final deck)

- Meta AI, [Deepfake Detection Challenge Dataset (DFDC)](https://ai.meta.com/datasets/dfdc/) — dataset overview and associated papers; Meta reports performance ranking changes between public and black-box evaluation, underscoring generalization risk.
- Dolhansky et al., [The DeepFake Detection Challenge (DFDC) Preview Dataset](https://arxiv.org/abs/1910.08854) and [The DeepFake Detection Challenge Dataset](https://arxiv.org/abs/2006.07397).
- W3C, [Web Authentication: An API for accessing Public Key Credentials, Level 3](https://www.w3.org/TR/webauthn-3/) — biometric data is used locally for user verification and is not revealed to the relying party in the WebAuthn model.
- Apple, [Logging a User into Your App with Face ID or Touch ID](https://developer.apple.com/documentation/LocalAuthentication/logging-a-user-into-your-app-with-face-id-or-touch-id) — LocalAuthentication is an app authentication mechanism, not an API for exporting biometric templates.
- Android Developers, [Show a biometric authentication dialog](https://developer.android.com/identity/sign-in/biometric-auth) — native biometric integration guidance.
- NIST, [SP 800-63A: Identity Proofing and Enrollment](https://pages.nist.gov/800-63-4/sp800-63a.html) — biometric privacy assessment, explicit consent, retention/deletion, and demographic performance considerations.
- Supabase, [Row Level Security](https://supabase.com/docs/guides/database/postgres/row-level-security), [Storage Access Control](https://supabase.com/docs/guides/storage/security/access-control), and [Securing your data](https://supabase.com/docs/guides/database/secure-data) — access controls, RLS, and server-only secret handling; these provide storage/access controls, not face matching.
- ASVspoof, [ASVspoof 2021](https://www.asvspoof.org/index2021.html) — datasets for bona fide and spoofed speech and challenge baselines.
- Wang et al., [ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech](https://arxiv.org/abs/1911.01601).
- [Supabase Row Level Security documentation](https://supabase.com/docs/guides/database/postgres/row-level-security) — if Supabase is used, review access policies and backend secret handling.

## Slide 10 — Accuracy and testing

**Evaluation design**

- Split by source identity/video and recording session to reduce leakage; reserve unseen attack methods and capture conditions for final evaluation.
- Report per-check metrics with sample counts: ROC-AUC, precision/recall, FAR/FRR; speaker verification false-match/false-non-match; challenge replay rejection; end-to-end recapture and review rates.
- Tune thresholds on validation data and state the false-accept versus false-reject trade-off.
- Compare a face-only baseline with optional background/temporal features; retain the added feature only if held-out results support it.

**Test cases**

- Genuine users under ordinary and difficult lighting/audio conditions.
- Manipulated face video, prerecorded webcam video, and poor-quality capture.
- Genuine, replayed, synthesized, and converted voice samples.
- Valid, invalid, expired, and reused phone signatures/challenges.
- Network drop, duplicate sessions, oversized media, and retry behavior.

**Results to fill after running evaluation:** [dataset and split] · [N] · [metrics and confidence intervals] · [latency/device] · [known failure modes].

## Slide 11 — Impact and benefits

- **Users:** clearer consent, fewer unsupported accusations, and a recapture/review route when evidence is weak.
- **Operators:** structured evidence and an auditable decision trail instead of one opaque score.
- **Security:** independent signals make simple replay attacks harder and expose invalid/reused challenges.
- **Trust and privacy:** local biometric verification can authorize a device-held key without sending raw biometric templates to the server.
- **Measure impact:** [onboarding completion], [false rejection], [review burden], [replay success rate], [time to complete], [user comprehension].

## Slide 12 — Demo, roadmap, and ask

**Demo:** one genuine journey, one controlled replay/spoof case, one inconclusive capture, and the evidence report.

**Roadmap:** vertical slice → pretrained baselines → phone proof → controlled evaluation → limited pilot.

**Ask:** [feedback, mentorship, access to test users/devices, or pilot partner].

## Presenter notes / claims checklist

- Replace generic context with evidence from user/stakeholder interviews and cited sources.
- Label prototype, mock, and measured outputs accurately.
- State that spoof/deepfake detection is probabilistic and vulnerable to distribution shift.
- Do not claim QR motion makes synthetic video impossible, phone biometrics identify the account holder, or a detector proves identity.
- Fill in team details, demo scope, metrics, and constraints before presenting.
