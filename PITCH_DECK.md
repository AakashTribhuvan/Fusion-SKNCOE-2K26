# Pitch Deck — Deepfake-Resistant Identity Verification

> **Editable pitch draft.** Replace bracketed text with team-specific details. No performance numbers are claimed until measured.

> **Prototype update (2026-10-09):** The current build wires a live face-crop classifier into the laptop's QR capture window, plus server-measured QR sequence/path checks, server-verified WebAuthn phone proof, optional consented phrase/audio screening, and a hash-only in-memory audit simulator. Model evaluation/calibration and physical-device testing remain open. A completed challenge remains Review; an incomplete required QR sequence fails.

> **Challenge alignment:** Mastercard CSB-02 asks for lightweight real-versus-synthetic detection on a short KYC clip without a prior real-person baseline, an explanation of the artifacts behind the score, and a mock onboarding demo. The selected direction is a laptop + phone flow: classifier first; phone biometric-authorized signing and ordered square-target QR as supporting signals.

## Slide 1 — Title

**Deepfake-Resistant Identity Verification for Digital Onboarding**
Team: **[team name]** · Challenge/track: **[confirm official wording]** · Institution: **[fill in]**

**One-line pitch:** Screen sampled face crops during a live onboarding challenge, show timestamped frame/region evidence, and pair it with phone-authenticator verification and a fresh, ordered square-target QR challenge.

## Slide 2 — Problem statement

- Mastercard CSB-02: video/voice-based KYC is targeted by real-time deepfake injection attacks that bypass liveness checks.
- The check must work on short clips without a prior baseline of the real person.
- The requested deliverable is a lightweight real-versus-synthetic face classifier that explains which visual/audio artifacts drove its score and appears in a mock onboarding flow.
- **Problem to solve:** How might a short-clip classifier estimate synthetic-face likelihood, expose evidence behind its output, and safely handle uncertain cases in a mock KYC flow?

**Evidence to add:** [target workflow, threat examples, stakeholder/user research, and properly sourced figures].

## Slide 3 — Proposed solution

An explainable onboarding workflow with one primary detector and two supporting phone signals:

1. During the complete QR movement recording (up to 30 seconds), score detected face crops sampled from the first through final frames with a pinned real-versus-synthetic image classifier.
2. Show timestamped face-region scores and a summary; the current image model does not localize artifacts or provide a trained temporal-video judgment.
3. Pair a phone and display a fresh, expiring QR sequence while the user moves through six square targets in order for replay-resistant motion evidence.
4. Ask the phone's platform authenticator to perform local user verification and authorize a signature over a fresh server challenge.
5. Show separate outputs for classifier, QR motion, phone signature, optional voice checks, and capture quality. The current prototype returns **Review** when the required QR sequence completes and **Challenge failed** otherwise.

**Key distinction:** Phone local user verification authorizes use of a platform credential; biometric data stays local. The platform may allow a device PIN/passcode as a fallback, so the server does not claim it knows which local method was used. This does not identify the camera subject or prove the video is genuine. QR motion is anti-replay evidence, not a deepfake detector. The face classifier is the core challenge deliverable.

## Slide 4 — User journey and system flow

`Consent → Create session → Pair phone → Verify locally → Capture live face samples + scan moving QR → Optional consented voice phrase → Show separate evidence → Review / Challenge failed`

**Diagram placeholder:** [Add flow diagram with browser, mobile app, API, model services, database, and trust boundaries.]

**Device choice:** hybrid. Laptop/browser handles onboarding, webcam capture, screening, and results. Phone displays the changing QR and locally authorizes signing a one-time challenge. The QR, WebAuthn, live image-classifier, optional audio, and hash-simulator paths are wired; physical-device testing and model evaluation remain open.

## Slide 5 — Technology approach

| Layer | Approach | Output |
|---|---|---|
| Web | Current prototype: static HTML/CSS/JavaScript served by FastAPI | Pairing, short-clip capture UI, progress, evidence report |
| API/orchestration | FastAPI + server-side WebAuthn assertion verification | Session state and separated verification contract |
| Phone | WebAuthn platform credential with `userVerification: required` | Challenge-bound signature; local biometric/device unlock stays on phone |
| QR motion | Rotating six-code QR plus six ordered square camera targets | Sequence, expiry, replay, and observed target-entry evidence |
| Face video | MediaPipe face detection plus a pinned pretrained image classifier, sampled during QR capture | Timestamped face-crop scores; not a trained temporal model |
| Voice | Optional, separate consent; fresh phrase transcription plus a pinned audio anti-spoof classifier | Phrase match and uncalibrated screening signal; no speaker identity match |
| Data/security | In-memory session state, public credential data, and hash-only simulated audit chain | No persistent database or real blockchain is integrated |
| Runtime | Git + Python 3.11 virtual environment + `requirements.txt`; Docker optional | Teammates can install and run the same API locally |

**Enrollment decision:** A custom reference-face comparison is distinct from phone OS biometrics. Supabase can store an authorized reference image or embedding, but it does not compare faces; a backend matcher is required. Keep this feature out of the first build unless the challenge requires it. If included, obtain explicit consent, minimize/expire stored data, restrict it to backend-only access, and evaluate false matches and false non-matches. Never imply that the phone biometric proves the laptop-camera subject is the registered person.

## Slide 6 — Decision logic and explainability

- Use explicit rules for cryptographic proof, expiry, and replay; do not average away critical failures.
- Keep signal outputs separate: face classifier, phone user-verification proof, QR sequence/motion, capture quality, optional phrase match, and optional audio anti-spoof screening. Speaker identity matching is intentionally out of scope.
- **Current prototype:** **Review** when the randomized QR sequence completes; **Challenge failed** when it does not. It has no automatic Pass outcome.
- **Future policy:** define Review/Inconclusive handling and recapture criteria only after validation on permitted held-out data.
- Show evidence and model limitations; never present illustrative scores as measured results.

## Slide 7 — Feasibility

- The current FastAPI/browser prototype wires the onboarding shell, QR challenge, live face-crop classifier, optional phrase/audio check, and simulated audit trail. Next feasibility gate: evaluate latency and accuracy on permitted held-out data and complete physical-device testing.
- Phone signing is implemented with WebAuthn platform credentials and `userVerification: required`; a client-reported prompt-success boolean is not secure proof. A real cross-device run needs HTTPS and a server-verified assertion over a fresh nonce.
- Evaluate the classifier, phone signature, and QR motion as separate components; successful phone/QR checks must not mask a weak or unavailable classifier.
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

- Which compact pretrained face-manipulation models fit the team's hardware and produce evidence that can be meaningfully inspected?
- Do frame-level, frequency/texture, or temporal cues improve results on held-out recordings without making explanations misleading?
- Does the six-box QR motion challenge reduce replay success while remaining usable on ordinary devices?
- Can the phone key be used only after platform biometric verification, with every server assertion bound to a fresh session nonce?
- How does performance change across unseen generators, devices, lighting, compression, and demographic groups?
- Does the randomized challenge measurably reduce replay success without making legitimate onboarding too difficult?
- What capture, retention, consent, accessibility, and regulatory constraints apply to the intended deployment market?

### Starting references (verify licensing and cite in final deck)

- Meta AI, [Deepfake Detection Challenge Dataset (DFDC)](https://ai.meta.com/datasets/dfdc/) — dataset overview and associated papers; Meta reports performance ranking changes between public and black-box evaluation, underscoring generalization risk.
- Dolhansky et al., [The DeepFake Detection Challenge (DFDC) Preview Dataset](https://arxiv.org/abs/1910.08854) and [The DeepFake Detection Challenge Dataset](https://arxiv.org/abs/2006.07397).
- W3C, [Web Authentication: An API for accessing Public Key Credentials](https://www.w3.org/TR/webauthn/) — biometric data is used locally for user verification and is not revealed to the relying party; the credential proves possession of a scoped public-key credential.
- Apple, [Logging a User into Your App with Face ID or Touch ID](https://developer.apple.com/documentation/LocalAuthentication/logging-a-user-into-your-app-with-face-id-or-touch-id) — LocalAuthentication is an app authentication mechanism, not an API for exporting biometric templates.
- Android Developers, [Show a biometric authentication dialog](https://developer.android.com/identity/sign-in/biometric-auth) and [Android Keystore system](https://developer.android.com/privacy-and-security/keystore) — local prompt integration and authentication-gated cryptographic keys.
- NIST, [SP 800-63A: Identity Proofing and Enrollment](https://pages.nist.gov/800-63-4/sp800-63a.html) — biometric privacy assessment, explicit consent, retention/deletion, and demographic performance considerations.
- Supabase, [Row Level Security](https://supabase.com/docs/guides/database/postgres/row-level-security), [Storage Access Control](https://supabase.com/docs/guides/storage/security/access-control), and [Securing your data](https://supabase.com/docs/guides/database/secure-data) — access controls, RLS, and server-only secret handling; these provide storage/access controls, not face matching.
- ASVspoof, [ASVspoof 2021](https://www.asvspoof.org/index2021.html) — datasets for bona fide and spoofed speech and challenge baselines.
- Wang et al., [ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech](https://arxiv.org/abs/1911.01601).
- [Supabase Row Level Security documentation](https://supabase.com/docs/guides/database/postgres/row-level-security) — if Supabase is used, review access policies and backend secret handling.

## Slide 10 — Accuracy and testing

**Evaluation design**

- Split by source identity/video and recording session to reduce leakage; reserve unseen attack methods and capture conditions for final evaluation.
- Report classifier metrics and sample counts: ROC-AUC, precision/recall, false-accept/false-reject; phone signature invalid/expired/replay rejection; QR challenge replay rejection; end-to-end recapture and review rates.
- Tune thresholds on validation data and state the false-accept versus false-reject trade-off.
- Compare a face-only baseline with optional background/temporal features; retain the added feature only if held-out results support it.

**Test cases**

- Genuine and manipulated short clips, including a held-out generator/source, prerecorded webcam/replay, and poor-quality capture.
- Valid, invalid, expired, and reused phone signatures; QR sequence replay, out-of-order, expiry, and insufficient-motion cases.
- Ordinary and difficult lighting, camera, compression, and movement conditions.
- Network drop, duplicate sessions, oversized media, and retry behavior.

**Results to fill after running evaluation:** [dataset and split] · [N] · [metrics and confidence intervals] · [latency/device] · [known failure modes].

## Slide 11 — Impact and benefits

- **Users:** clearer consent, fewer unsupported accusations, and a recapture/review route when evidence is weak.
- **Operators:** structured evidence and an auditable decision trail instead of one opaque score.
- **Security:** independent signals make simple replay attacks harder and expose invalid/reused challenges.
- **Trust and privacy:** local authenticator verification can authorize a scoped key without sending biometric templates to the server; the app does not learn whether the device used biometrics or its PIN fallback.
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
