# Baseline Scope — Deepfake-Resistant Onboarding Demo

This document defines the minimum agreed work for the first end-to-end demo. It is a scope baseline, not a claim that every item is already implemented.

## Goal

Demonstrate deepfake/liveness screening during a mock digital-onboarding session, using a randomized phone QR motion challenge and an AI detector that analyzes the laptop's live face video at the same time. Detection must not depend on enrolling the person or having a prior sample of their face.

## Core demo flow

1. Create an onboarding session and pair the phone with the laptop.
2. Start a fresh, short-lived QR challenge. The phone displays the changing codes and asks the user to move it through a slow figure eight while the laptop camera observes the phone.
3. During the same capture window, analyze a short live webcam clip with a real-versus-synthetic face model. The AI analysis is a separate, concurrent check; it must not wait for the QR check to finish.
4. Show a report that keeps the QR result and AI result distinct, records AI-flagged abnormalities with evidence, and communicates uncertainty and unavailable checks.

## Required behavior

### QR motion challenge — primary interactive liveness check

- Generate unpredictable, session-bound codes with an expiry and reject invalid, expired, reused, or out-of-order observations.
- Ask for the figure-eight phone movement and assess whether the observed QR path has sufficient movement and spatial spread.
- Report the challenge outcome and relevant reason (for example, passed, incomplete, expired, or invalid).
- Treat this as a replay-resistance signal, not proof that the face is genuine or that the user is a particular person.

### Live AI analysis — concurrent deepfake check

- Analyze a short clip from the laptop webcam during the QR challenge, without requiring a personal enrollment image, face template, or historical baseline.
- Use a real-versus-synthetic model whose source, license, input requirements, and limitations are documented.
- Record the clip-level score and uncertainty, plus any abnormalities the model or supporting analysis can substantiate. For each reported abnormality, retain the relevant timestamp/frame reference, region or artifact category when available, and its score/confidence.
- Distinguish observed evidence from model interpretation. Do not invent human-readable artifact explanations from a model score that does not provide them.
- Mark the result unavailable or inconclusive when capture quality, model output, or runtime is insufficient. Do not silently treat missing analysis as a pass.

## Evidence and user-facing result

The report must keep these outcomes separate:

- QR sequence and figure-eight movement.
- AI real-versus-synthetic likelihood, uncertainty, and recorded abnormalities.
- Capture quality and any unavailable checks.

The mock onboarding flow may combine these signals into a review state, but must not represent either a QR pass or an AI score as certain proof of identity or fraud. Preserve enough evidence to explain a flag without retaining raw biometric media longer than the demo requires; define consent, retention, and deletion behavior before collecting or saving recordings.

## Minimum acceptance criteria

- A user can complete the QR figure-eight challenge while the webcam clip is being analyzed in the same capture window.
- The application produces both a QR result and an AI result for that session, or clearly identifies which result is unavailable and why.
- A test run can show the AI score and its supporting abnormality evidence, including timestamp/frame references where supported by the model.
- Invalid, expired, or replayed QR codes do not pass the challenge.
- The report does not require or claim comparison to an enrolled face or the phone owner's identity.
- The demo documentation explains that model output is a screening signal, not a definitive decision.

## Out of scope for this baseline

- Collecting a person's face or biometric data in advance to build an identity baseline.
- Using the phone's OS biometric prompt as a 3D-face/deepfake detector. A platform prompt can authorize a device user or key operation; it does not expose a portable face-depth verdict or prove that the laptop video shows that user.
- Treating phone authentication, QR movement, or one model score alone as proof of identity.
- Voice anti-spoofing, speaker matching, production KYC decisions, and claims of production-grade accuracy.

## Implementation note

The QR flow is the initial interactive challenge. The AI model is a parallel workstream and must run over the live capture window rather than be presented as already integrated. Keep the two checks independent in the API and report so either can be tested and evaluated separately.
