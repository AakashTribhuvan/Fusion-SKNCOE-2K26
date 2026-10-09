# Fusion QR / Video / Voice Verification Handover

This document captures the current state of the project and the exact work done so far so another coding agent can continue from this point without re-discovering the context.

## Project intent

This project is a browser-based verification flow for a phone + laptop setup:

- Phone verifies locally and displays a live QR challenge.
- Laptop camera checks for a visible face and validates QR movement through numbered square targets.
- Video is recorded and screened in memory for face / motion evidence.
- Optional voice step records a short phrase and analyzes it separately.
- Final report aggregates the evidence and presents a decision-ready summary.

The app is built around a FastAPI backend and static HTML/JS front end.

## Current repository state

Repository root:
- `D:\College\Fusion(SKNCOE)2K26`

Key app code:
- `backend/app/main.py`
- `backend/app/screening.py`
- `backend/app/static/index.html`
- `backend/app/static/phone.html`
- `backend/app/static/qr-zones.svg`
- `backend/app/static/qa-control.html`

Tests:
- `tests/test_screening_logic.py`

Docs:
- `README.md`
- `WORKFLOW_UPDATES.md`

## Current behavior implemented

### 1) QR challenge flow

- The phone companion is intentionally minimal and keeps the same branding as the main UI.
- It displays a fullscreen QR challenge with persistent instructions.
- Each QR remains on-screen until the laptop server scans it. The flow is scan-driven rather than timer-driven.
- QR payload format is compact and challenge-scoped, roughly of the form:
  - `F26|step|random_secret`
- QR step progression is tied to the active session and challenge.
- QR target boxes are arranged around the frame edges, away from the center face area.

### 2) Face presence gate

- Face detection is a presence gate only.
- The system checks that a face is visible, but does not match identity.
- A face must be present before the first QR can lock and before recording begins.
- Face detection failures pause QR progress and stop recording from advancing.

### 3) Square target path

- The old figure-eight flow was replaced with six numbered square targets.
- The server validates the ordered path and measures QR movement using server-decoded positions.
- Target zones remain largely on the left/right and lower frame edges to leave the center area clear for the user’s face.
- Target overlays are translucent green so the QR remains visible while the underlying scene stays readable.

### 4) Video screening

- After the first QR lock, browser recording begins.
- The whole movement is recorded until the six square targets are reached or the one-minute limit ends.
- Video is analyzed in memory and then discarded after scoring.
- The API accepts browser recordings and samples frames through the final point.
- MediaPipe face detection is used on sampled frames.
- The screen is designed as research screening, not as a calibrated identity/authenticity proof.

### 5) Voice step

- The Voice tab is distinct and separated from the QR/video flow.
- It shows a waveform while the user records a short phrase.
- Audio processing is optional and can be skipped.
- It uses a separate microphone consent and does not run as part of the QR step.

### 6) Hidden QA control

- `/hidden/control` exists as a local-only development control.
- It is disabled by default.
- It is only available under a local development environment and rejects proxied/tunnel requests.
- It advances the UI state for testing and does not bypass real verification checks.

## Important implementation details

### Backend

`backend/app/main.py`
- Defines the session + challenge lifecycle.
- Creates/serves QR images and challenge state.
- Validates phone verification, challenge expiry, and evidence submission.
- Handles:
  - challenge creation
  - challenge QR fetch
  - QR frame scanning
  - face + QR gating
  - evidence submission
  - report generation
- Added scan-time QR preprocessing fallback using grayscale enhancement and adaptive thresholding when normal decoding fails.
- Face detection still operates on the original frame, not the processed decode copy.

`backend/app/screening.py`
- Handles live face detection and in-memory video analysis.
- Contains the decoded video sampling logic, frame limits, and model checks.
- Uses a bounded sampling strategy to keep processing safe and predictable.

### Front end

`backend/app/static/index.html`
- Main laptop UI and live camera preview.
- Mirrored live preview is used for user guidance.
- QR tracking and target overlays are layered on the mirrored view.
- Recording starts only after the first valid QR lock and a visible face.
- The visible preview remains natural; scan enhancement is server-side only.

`backend/app/static/phone.html`
- Minimal companion UI for the phone.
- Active QR challenge fills the phone viewport, while instructions remain visible.
- Phone screen updates to the next QR only after the current one is scanned.

`backend/app/static/qr-zones.svg`
- Local visual map of the numbered square path and frame layout.

`backend/app/static/qa-control.html`
- Local-only QA panel used for controlled workflow advancement in development.

## Timing and limits currently in force

As of the latest patch:

- QR challenge capture window: 60 seconds max
- Video screening limit: about 60 seconds max
- QR path continues until either:
  - all six target boxes are reached in order, or
  - the time limit is reached
- The app no longer depends on a strict 30-second timer; it uses the one-minute ceiling and stops when the six QRs are complete.

## Validation status

The most recent test run completed successfully:

- `python -m unittest discover -s tests -v`
- Result: 25 tests passed
- `git diff --check`: passed

This was the last known clean validation state before final handoff.

## Known warnings / non-blocking items

- A Pylance warning remains in `backend/app/screening.py` for `_square_target_progress` being unused.
- This does not break runtime behavior or tests.
- The project still relies on browser/device verification in a real local run for end-to-end camera + microphone + QR validation.

## Key files to inspect first

If another agent is continuing forward, these are the most relevant files:

1. `backend/app/main.py`
   - challenge lifecycle and scan logic
2. `backend/app/screening.py`
   - detection and video analysis
3. `backend/app/static/index.html`
   - laptop-side QR tracker, recording, and gating
4. `backend/app/static/phone.html`
   - phone QR flow and instructions
5. `tests/test_screening_logic.py`
   - regression coverage for challenge, QR path, and AI screening logic
6. `README.md`
   - broad product behavior and workflow overview
7. `WORKFLOW_UPDATES.md`
   - detailed incremental change log

## Suggested next-step tasks

Recommended follow-up work if continuing the project:

1. Browser/device smoke test over localhost with actual camera and phone QR interaction.
2. Confirm the QR flow remains stable on a real phone and laptop under different lighting conditions.
3. Validate the voice step end-to-end with microphone permission and waveform behavior.
4. Review any remaining backend assumptions around challenge expiry and submission locking.
5. Remove or clean up older unused files or design artifacts if they are no longer part of the final workflow.

## Critical continuity note

Do not treat this as a final production-ready state. It is a working verification prototype with validation coverage and a real browser flow design, but actual physical camera + phone behavior still needs live browser testing to confirm device-specific reliability.

## Handoff summary

The project has already implemented:

- a full smartphone verification flow
- a QR-based challenge sequence
- per-frame face detection gating
- square-target motion validation
- complete movement video screening
- optional voice recording and waveform UI
- hidden local-only QA controls
- compact documentation and test coverage

The most important remaining work is real-device verification and any UI/flow cleanup that emerges from a live browser test session.
