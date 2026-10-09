# Verification Workflow Updates

## Ordered tabs

The laptop experience is now a four-step, gated workflow:

1. **Phone** — create a short-lived session, pair the phone, and complete its platform-authenticator check.
2. **QR + video** — consent to camera use, scan the changing phone QR, record a ten-second clip after the first valid QR lock, and screen sampled face crops.
3. **Voice** — optionally consent to microphone use, read a fresh phrase, and run phrase/AI-voice checks; this step can be skipped.
4. **Report** — review the QR, phone, video, and optional audio evidence together without treating model scores as a KYC decision.

Tabs unlock as prerequisites are met. The Voice tab is no longer automatically launched after QR capture, and microphone permission is never requested by the QR/video step.

The phone companion uses the same Frame / Check branding, color palette, and typography as the laptop experience, but intentionally has no tabs or secondary workflow controls. After local phone verification, it focuses on the large rotating QR and the brief hold-still / wait-for-lock / then-move instruction.

## QR lock and motion guidance

Before starting the camera, the QR + video tab instructs the user to:

- Bring the bright phone QR about 15–20 cm from the laptop camera and hold it still so focus and auto-exposure can settle.
- Keep it in the center guide until the server recognizes the first fresh challenge QR and displays **QR LOCK**.
- Begin a slow, continuous figure eight only after that lock, keeping the QR pointed at the camera and within view through the ten-second capture.

The recording timer and MediaRecorder start only after the first valid QR payload returns from the server. Before the lock, the camera is used only to send small preview frames for QR detection. The original figure-eight diagram is hosted locally at `/motion-guide.svg`, shown in the directions and over the viewer after lock, and available as a download. It does not depend on a remote image host.

## Recorded video screening

With explicit camera consent, the browser records the live camera stream for ten seconds (camera video only, no audio) and submits the clip after the scan. The API requires prior phone verification and at least one server-observed challenge QR. It accepts WebM, MP4, and QuickTime recordings up to 25 MB. PyAV decodes them in memory, limits duration to 15 seconds and decoded frame count, and samples at about one frame per second. Frames larger than eight megapixels are reduced before face detection and classification.

MediaPipe detects face regions in the sampled clip frames; the pinned image classifier scores detected face crops. The clip and decoded images are discarded after analysis. Derived per-frame scores and references are retained only in the in-memory session report. A missing/unavailable classifier remains an explicit unavailable signal; it is not reported as a pass. QR preview frames are submitted separately for server-side QR decode and path measurement.

This is frame-based research screening, not a trained temporal deepfake model. Its scores are uncalibrated and do not prove authenticity, liveness, identity, or fraud.

## Separate voice step

The Voice tab gives microphone placement and phrase-reading directions. Voice models are warmed only after the user checks the distinct microphone-consent box and presses **Start voice check**. The browser then requests microphone permission, shows the fresh three-word phrase, and records up to eight seconds. The API checks phrase transcription and an AI-voice signal, then discards the audio. No speaker identity, enrollment, or voiceprint is created. **Skip voice and view report** remains available when the user does not want to use a microphone.

## API and implementation changes

- Added `POST /api/sessions/{session_id}/challenge/{challenge_id}/video` for consented in-memory video decoding/screening.
- The video endpoint validates active session/challenge, phone verification, camera consent, an observed QR, media type, body size, duration, and decoded frame limits.
- The evidence-report endpoint now requires video screening to have been submitted for the challenge.
- Increased the challenge lifetime from 90 seconds to four minutes to support the distinct, optional Voice step while still being bounded by the five-minute session.
- Added `/motion-guide.svg`, an original local SVG with a download link and live-view overlay.
- Removed the old local-only uploaded-recording preview controls from the laptop screen; the live QR recording is now the video-analysis input.
- Updated README and project itinerary instructions and data-handling descriptions.

## Verification

The focused regression suite covers in-memory video decoding and the existing QR, audio, and evidence logic. Run it from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The physical camera, phone, and microphone sequence still requires a browser/device run over localhost or HTTPS. Camera support, video codec availability, model downloads, and autofocus behavior vary by device.
