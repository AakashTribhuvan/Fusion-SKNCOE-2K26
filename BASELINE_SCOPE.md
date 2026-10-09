# Baseline Scope — Deepfake-Resistant Onboarding Demo

This document defines the minimum agreed work for the first end-to-end demo. It is a scope baseline, not a claim that every item is already implemented.

## Goal

Demonstrate deepfake/liveness screening during a mock digital-onboarding session, using a randomized phone QR motion challenge and an AI detector that analyzes the laptop's live face video at the same time. Detection must not depend on enrolling the person or having a prior sample of their face.

## Core demo flow

1. Create an onboarding session and pair the phone with the laptop.
2. Start a fresh, short-lived QR challenge. The phone holds a compact random code until the laptop detects it together with a face in frame, then asks the user to move it through six perimeter targets that avoid the centered face area.
3. During the same capture window, analyze a short live webcam clip with a real-versus-synthetic face model. The AI analysis is a separate, concurrent check; it must not wait for the QR check to finish.
4. Show a report that keeps the QR result and AI result distinct, records AI-flagged abnormalities with evidence, and communicates uncertainty and unavailable checks.
5. Offer an optional, separately consented voice challenge using a fresh spoken phrase and an audio anti-spoof signal; do not enroll or identify a speaker.
6. Append a hash-only evidence reference to a clearly labelled in-memory audit simulator; do not send biometric samples or personal data to a ledger.

## Required behavior

### QR motion challenge — primary interactive liveness check

- Generate unpredictable per-step codes with an expiry and reject invalid, expired, reused, or out-of-order observations. Keep QR payloads compact; the active session/challenge endpoint scopes each random code.
- Require a face to be present before accepting a scanned QR step; this is only a presence gate, not face identity matching. Ask the user to move the QR through six numbered edge targets in order and assess the observed target sequence. Decode the QR and its position from submitted frames on the API rather than trusting browser-reported code contents and coordinates.
- Report the challenge outcome and relevant reason (for example, passed, incomplete, expired, or invalid).
- Treat this as a replay-resistance signal, not proof that the face is genuine or that the user is a particular person.

### Live AI analysis — concurrent deepfake check

- Analyze a short clip from the laptop webcam during the QR challenge, without requiring a personal enrollment image, face template, or historical baseline.
- Use a real-versus-synthetic model whose source, license, input requirements, and limitations are documented.
- Record per-frame face-crop scores, timestamp, and detected face region, plus an aggregate summary. Do not claim an artifact explanation: the current image classifier does not localize physical abnormalities.
- Distinguish observed evidence from model interpretation. Do not invent human-readable artifact explanations from a model score that does not provide them.
- Mark the result unavailable or inconclusive when capture quality, model output, or runtime is insufficient. Do not silently treat missing analysis as a pass.

## Evidence and user-facing result

The report must keep these outcomes separate:

- QR sequence and ordered square-target movement.
- AI real-versus-synthetic likelihood, uncertainty, and recorded abnormalities.
- Capture quality and any unavailable checks.

The mock onboarding flow may combine these signals into a review state, but must not represent either a QR pass or an AI score as certain proof of identity or fraud. Ask permission before opening the camera; explain that downscaled still frames are sent to the demo API for QR decoding and periodic face screening, processed in memory, and not saved. Do not request microphone access for the QR check. The optional voice challenge has its own explicit consent and browser permission step; bound it to a short recording and do not retain the raw audio or a voiceprint.

## Minimum acceptance criteria

- A user can complete the six-box QR challenge while the complete movement clip is analyzed from first to last in the same capture window.
- The browser explicitly obtains user consent and browser camera permission before capture, requests no microphone access for the QR check, and discloses any camera frames sent to the API.
- The optional audio feature requires its own user consent and microphone permission, uses only a fresh random phrase, and does not match speaker identity.
- The application produces both a QR result and an AI result for that session, or clearly identifies which result is unavailable and why.
- A run can show the AI score and its supporting per-frame evidence, including timestamp and detected face region. Artifact localization remains a future model/evaluation task.
- Invalid, expired, or replayed QR codes do not pass the challenge.
- The report does not require or claim comparison to an enrolled face or the phone owner's identity.
- The audit path contains only hashes/chain metadata, is labelled as an in-memory simulator, and is not represented as a real blockchain.
- The demo documentation explains that model output is a screening signal, not a definitive decision.

## Out of scope for this baseline

- Collecting a person's face or biometric data in advance to build an identity baseline.
- Using the phone's OS biometric prompt as a 3D-face/deepfake detector. A platform prompt can authorize a device user or key operation; it does not expose a portable face-depth verdict or prove that the laptop video shows that user.
- Treating phone authentication, QR movement, or one model score alone as proof of identity.
- Speaker matching/enrollment, production KYC decisions, and claims of production-grade accuracy.

## Implementation note

The QR flow and periodic face-crop screening now run in the same capture window. The QR, face, optional audio, and phone-authenticator checks remain independent in the API and report. The image classifier is wired but not calibrated or evaluated for this use; its frame scores are screening evidence, not artifact explanations or proof of authenticity.
