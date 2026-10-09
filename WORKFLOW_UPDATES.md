# Verification Workflow Updates

## Ordered tabs

The laptop experience is now a four-step, gated workflow:

1. **Phone** — create a short-lived session, pair the phone, and complete its platform-authenticator check.
2. **QR + video** — consent to camera use, lock the first changing phone QR in target box 1, record the complete movement through six boxes, and AI-screen frames sampled across the clip.
3. **Voice** — optionally consent to microphone use, read a fresh phrase, and run phrase/AI-voice checks; this step can be skipped.
4. **Report** — review the QR, phone, video, and optional audio evidence together without treating model scores as a KYC decision.

Tabs unlock as prerequisites are met. The Voice tab is no longer automatically launched after QR capture, and microphone permission is never requested by the QR/video step.

The phone companion uses the same Frame / Check branding, color palette, and typography as the laptop experience, but intentionally has no tabs or secondary workflow controls. After local phone verification, it switches to a viewport-filling QR screen: the code uses nearly the full phone width, stays visible until the laptop server scans it, then displays the next QR. The short instructions remain on-screen with the code. Challenge payloads are compact, random, and scoped by the active session/challenge endpoint to make the code easier to resolve without weakening its per-step secret.

## QR lock and square target path

Before starting the camera, the QR + video tab instructs the user to:

- Keep your face centered in the laptop camera view, then place the bright QR inside square **box 1** at the right edge, about 15–20 cm from the camera. The face-presence check must detect a face before a QR step advances or video recording begins; this is not face identity matching.
- Hold the QR still so focus and auto-exposure can settle. Keep the phone screen bright and clean; wipe the laptop camera lens, avoid direct reflections, and tilt the phone only slightly if glare persists while keeping the QR mostly square to the lens. Each phone QR remains on screen until that exact code is decoded by the laptop API; the next code is not timer-driven.
- Wait for the server to recognize a fresh challenge QR and display **QR LOCK**. No video recording starts until both a face and the first QR in box 1 are detected.
- Move slowly through boxes **2–6** around the side and bottom edges in the order shown on the mirrored preview. The targets intentionally avoid the usual centered face area. Keep your face in frame, keep the QR facing the camera, and pause in each target until it is confirmed. The recording continues until all six QRs are scanned or the 1-minute capture limit is reached.

The MediaRecorder starts only after the first valid QR payload is decoded in box 1 while a face is detected. Before the lock, the camera is used only to send small preview frames for face/QR detection. The server decodes QR positions and validates the ordered box path; browser-reported coordinates are not trusted. The original target guide is hosted locally at `/qr-zones.svg`, shown in the directions and over the live viewer, and available for download.

Target boxes use translucent green fill so the underlying camera image and QR remain visible. For challenging lighting, the server first tries normal QR decoding and then applies grayscale CLAHE contrast enhancement and adaptive thresholding to an internal copy of the submitted scan frame. Face detection still runs on the unmodified frame, and the video shown to the user is the normal camera stream; the enhanced scan copy is never rendered to the viewer or recorded.

## Complete-movement video screening

With explicit camera consent, the browser records the live camera stream from the first QR lock until all six QRs are scanned or 1 minute elapses (camera video only, no audio), then submits the clip. The API requires prior phone verification and at least one server-observed fresh QR. It accepts WebM, MP4, and QuickTime recordings up to 25 MB. PyAV decodes the entire clip in memory, limits duration to about 1 minute and decoded frame count, and samples frames every half-second including the final frame. Frames larger than eight megapixels are reduced before face detection and classification.

MediaPipe detects face regions in sampled clip frames; the pinned image classifier scores detected face crops across the complete movement. The clip and decoded images are discarded after analysis. Derived per-frame scores and references are retained only in the in-memory session report. A missing/unavailable classifier remains an explicit unavailable signal; it is not reported as a pass. QR preview frames are submitted separately for server-side QR decode and ordered target measurement.

This is frame-based research screening, not a trained temporal deepfake model. Its scores are uncalibrated and do not prove authenticity, liveness, identity, or fraud.

## Separate voice step

The Voice tab gives microphone placement and phrase-reading directions. Voice models are warmed only after the user checks the distinct microphone-consent box and presses **Start voice check**. The browser then requests microphone permission, shows the fresh three-word phrase, displays a live waveform while recording up to eight seconds, and sends audio for phrase transcription and AI-voice screening. Audio is discarded after processing. No speaker identity, enrollment, or voiceprint is created. **Skip voice and view report** remains available when the user does not want to use a microphone.

## Local-only QA control

`/hidden/control` is disabled by default and is available only when `ENABLE_QA_CONTROLS=true` is set for a non-production local server. It rejects requests with forwarded or Cloudflare tunnel headers. Its “next step” control advances a UI-only simulation; it does not bypass server checks, create evidence, or produce a real report. Keep this flag unset for all shared or tunneled demos.

For local PowerShell testing only:

```powershell
$env:APP_ENV = "development"
$env:ENABLE_QA_CONTROLS = "true"
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/hidden/control`. The feature should remain disabled in production.

## API and implementation changes

- Added `POST /api/sessions/{session_id}/challenge/{challenge_id}/video` for consented in-memory video decoding/screening.
- The video endpoint validates active session/challenge, phone verification, camera consent, an observed QR, media type, body size, duration, and decoded frame limits.
- The evidence-report endpoint now requires video screening to have been submitted for the challenge.
- Increased the challenge lifetime from 90 seconds to four minutes to support the distinct, optional Voice step while still being bounded by the five-minute session.
- Replaced the figure-eight path with six numbered square targets measured from server-decoded QR positions.
- Changed challenge QR progression from a timer to scan acknowledgement: each code remains on the phone until the laptop server decodes it, then the phone polls the current challenge step and fetches the next code.
- Added live server-side face-presence gating to QR-step acknowledgement and recording start; this is a presence signal only and does not identify the person.
- Moved six QR targets to the left/right frame edges and bottom perimeter, leaving the typical centered face area unobstructed.
- Shortened QR payloads to a compact prefix, step index, and per-step 72-bit random secret; the active session/challenge URL scopes validation. Increased the QR quiet zone to four modules for reliable camera decoding.
- Made the phone's active QR view viewport-filling while keeping movement, face-position, and glare directions visible on-screen.
- Made camera target boxes translucent green for better QR visibility, and added server-side contrast/threshold preprocessing only to the scan copy (never the displayed camera footage).
- Enlarged the phone QR to use the available mobile viewport and mirrored the laptop's displayed live camera preview for easier following. QR scan frames are still drawn from the unmirrored video source.
- The full path recording now ends when the last target is reached or after 1 minute; face screening samples the complete clip every half-second and includes the final frame.
- Added a live microphone waveform visualization during the consented voice recording.
- Added `/hidden/control` as a local, development-only, UI simulation for advancing one workflow stage; it cannot submit or bypass verification evidence.
- Removed the old local-only uploaded-recording preview controls from the laptop screen; the live QR recording is now the video-analysis input.
- Updated README and project itinerary instructions and data-handling descriptions.

## Verification

The focused regression suite covers in-memory video decoding and the existing QR, audio, and evidence logic. Run it from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The physical camera, phone, and microphone sequence still requires a browser/device run over localhost or HTTPS. Camera support, video codec availability, model downloads, and autofocus behavior vary by device.
