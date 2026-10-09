"""Local-first API for the Fusion identity verification prototype.

Session state is intentionally in memory for the demo. No camera, audio, or
biometric samples are persisted by this service.
"""

from __future__ import annotations

import hmac
import hashlib
import io
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4
from urllib.parse import urlsplit

import qrcode
import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from webauthn import (
    base64url_to_bytes,
    generate_authentication_options,
    generate_registration_options,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers.exceptions import WebAuthnException
from webauthn.helpers.structs import (
    AuthenticatorAttachment,
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)
from webauthn.helpers import options_to_json

from .audit_ledger import SimulatedAuditLedger
from .screening import (
    analyze_video_frame,
    decode_video_frames,
    summarize_video_frames,
    video_model_status,
    warm_video_model,
)
from .voice_screening import analyze_audio, audio_model_status, warm_audio_model, warm_transcriber


ROOT = Path(__file__).resolve().parent
SESSION_TTL = timedelta(minutes=5)
CHALLENGE_TTL = timedelta(minutes=4)
CHALLENGE_STEPS = 6
CHALLENGE_STEP_SECONDS = 1.5
MAX_OBSERVATIONS = 120
MAX_QR_FRAME_BYTES = 1_000_000
MAX_VIDEO_BYTES = 25_000_000
MAX_AUDIO_BYTES = 5_000_000
QR_PATH_SAMPLE_INTERVAL_MS = 200
logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ChallengeState:
    id: UUID
    created_at: datetime
    expires_at: datetime
    codes: list[str]
    audio_phrase: str
    submitted: bool = False
    audio_submitted: bool = False
    video_submitted: bool = False


@dataclass
class PhoneCredential:
    id: bytes
    public_key: bytes
    sign_count: int


@dataclass
class SessionState:
    id: UUID
    pair_token: str
    created_at: datetime
    expires_at: datetime
    paired: bool = False
    phone_credential_id: str | None = None
    phone_verified: bool = False
    pending_webauthn_challenge: bytes | None = None
    pending_webauthn_kind: Literal["registration", "authentication"] | None = None
    pending_webauthn_expires_at: datetime | None = None
    pending_phone_device_id: str | None = None
    challenge: ChallengeState | None = None
    video_frames: list[dict[str, Any]] | None = None
    server_observations: list[Observation] | None = None
    last_qr_path_sample_ms: int = -QR_PATH_SAMPLE_INTERVAL_MS
    audio_result: dict[str, Any] | None = None
    result: dict | None = None


class SessionView(BaseModel):
    id: UUID
    status: Literal["active", "expired"]
    paired: bool
    phone_key_registered: bool
    phone_verified: bool
    created_at: datetime
    expires_at: datetime


class ChallengeView(BaseModel):
    id: UUID
    status: Literal["active", "expired", "submitted"]
    created_at: datetime
    expires_at: datetime
    step_count: int = CHALLENGE_STEPS
    step_seconds: float = CHALLENGE_STEP_SECONDS
    motion_instruction: str = "Move the phone through a slow figure eight while showing its screen to the laptop camera."


class PairRequest(BaseModel):
    token: str = Field(min_length=16, max_length=128)


class WebAuthnCredentialSubmission(BaseModel):
    credential: dict[str, Any]


class PhoneDeviceRequest(BaseModel):
    device_id: UUID


class Observation(BaseModel):
    payload: str = Field(max_length=256)
    elapsed_ms: int = Field(ge=0, le=30_000)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class FrameScanResult(BaseModel):
    payload: str | None = None
    elapsed_ms: int
    x: float | None = None
    y: float | None = None
    box_x: float | None = None
    box_y: float | None = None
    box_width: float | None = None
    box_height: float | None = None


class WarmupRequest(BaseModel):
    audio_consent: bool = False


class WarmupResult(BaseModel):
    video: dict[str, str]
    audio: dict[str, str]


class AudioChallengeView(BaseModel):
    challenge_id: UUID
    phrase: str
    expires_at: datetime


class EvidenceSubmission(BaseModel):
    challenge_id: UUID


class CheckResult(BaseModel):
    status: Literal["passed", "failed", "review", "unavailable"]
    detail: str
    evidence: dict[str, Any] | None = None


class AudioSubmissionResult(BaseModel):
    phrase_check: CheckResult
    anti_spoof_check: CheckResult
    duration_seconds: float | None = None


class VideoSubmissionResult(BaseModel):
    frames_sampled: int
    frames_with_faces: int
    median_fake_score: float | None
    score_range: float | None
    detail: str


class VerificationReport(BaseModel):
    session_id: UUID
    decision: Literal["review", "inconclusive", "challenge_failed"]
    checks: dict[str, CheckResult]
    limitations: list[str]
    audit: dict[str, Any]
    generated_at: datetime


_sessions: dict[UUID, SessionState] = {}
_phone_credentials: dict[str, PhoneCredential] = {}
_audit_ledger = SimulatedAuditLedger()

app = FastAPI(
    title="Fusion Identity Verification API",
    version="0.3.0",
    description="Demo: phone WebAuthn, a randomized QR motion challenge, optional video/audio screening, and a simulated hash audit.",
)


def _session(session_id: UUID) -> SessionState:
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def _ensure_active(state: SessionState) -> None:
    if utc_now() >= state.expires_at:
        _clear_expired_evidence(state)
        raise HTTPException(status_code=410, detail="Session expired; start a new session")


def _clear_expired_evidence(state: SessionState) -> None:
    state.challenge = None
    state.video_frames = []
    state.server_observations = []
    state.audio_result = None
    state.result = None


def _require_phone_token(state: SessionState, token: str | None) -> None:
    if not state.paired or token is None or not hmac.compare_digest(state.pair_token, token):
        raise HTTPException(status_code=403, detail="Phone is not paired with this session")


def _public_session(state: SessionState) -> SessionView:
    status = "expired" if utc_now() >= state.expires_at else "active"
    return SessionView(
        id=state.id,
        status=status,
        paired=state.paired,
        phone_key_registered=state.phone_credential_id is not None,
        phone_verified=state.phone_verified,
        created_at=state.created_at,
        expires_at=state.expires_at,
    )


def _webauthn_context(request: Request) -> tuple[str, str]:
    configured_origin = os.getenv("FUSION_WEBAUTHN_ORIGIN") or os.getenv("FUSION_PUBLIC_ORIGIN")
    configured_rp_id = os.getenv("FUSION_WEBAUTHN_RP_ID")
    origin = (configured_origin or str(request.base_url).rstrip("/")).rstrip("/")
    rp_id = configured_rp_id or request.url.hostname or "localhost"
    parsed_origin = urlsplit(origin)
    if parsed_origin.scheme not in {"http", "https"} or not parsed_origin.hostname:
        raise HTTPException(status_code=500, detail="WebAuthn origin configuration is invalid")
    if parsed_origin.path not in {"", "/"} or parsed_origin.query or parsed_origin.fragment:
        raise HTTPException(status_code=500, detail="WebAuthn origin must not include a path, query, or fragment")
    if parsed_origin.hostname != rp_id and not parsed_origin.hostname.endswith(f".{rp_id}"):
        raise HTTPException(status_code=500, detail="WebAuthn RP ID must match the configured origin host")
    if parsed_origin.scheme != "https" and parsed_origin.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise HTTPException(status_code=400, detail="Phone verification requires HTTPS outside localhost")
    return rp_id, origin


def _public_origin(request: Request) -> str:
    return (os.getenv("FUSION_PUBLIC_ORIGIN") or str(request.base_url)).rstrip("/")


def _set_pending_webauthn(state: SessionState, challenge: bytes, kind: Literal["registration", "authentication"]) -> None:
    state.pending_webauthn_challenge = challenge
    state.pending_webauthn_kind = kind
    state.pending_webauthn_expires_at = min(utc_now() + timedelta(minutes=2), state.expires_at)


def _take_pending_webauthn(state: SessionState, kind: Literal["registration", "authentication"]) -> bytes:
    challenge = state.pending_webauthn_challenge
    expires_at = state.pending_webauthn_expires_at
    valid = challenge is not None and state.pending_webauthn_kind == kind and expires_at is not None and utc_now() < expires_at
    state.pending_webauthn_challenge = None
    state.pending_webauthn_kind = None
    state.pending_webauthn_expires_at = None
    if not valid or challenge is None:
        raise HTTPException(status_code=410, detail="Phone verification challenge expired; request a fresh one")
    return challenge


def _public_challenge(challenge: ChallengeState) -> ChallengeView:
    if challenge.submitted:
        status = "submitted"
    else:
        status = "expired" if utc_now() >= challenge.expires_at else "active"
    return ChallengeView(
        id=challenge.id,
        status=status,
        created_at=challenge.created_at,
        expires_at=challenge.expires_at,
    )


def _qr_png(payload: str) -> bytes:
    image = qrcode.make(payload, box_size=8, border=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _challenge_code_step(challenge: ChallengeState, payload: str, session_id: UUID) -> int | None:
    parts = payload.split("|")
    if len(parts) != 5 or parts[0] != "FUSION26":
        return None
    _, observed_session, observed_challenge, raw_step, code = parts
    try:
        step = int(raw_step)
    except ValueError:
        return None
    if (
        observed_session != str(session_id)
        or observed_challenge != str(challenge.id)
        or not 0 <= step < len(challenge.codes)
        or not hmac.compare_digest(challenge.codes[step], code)
    ):
        return None
    return step


def _new_audio_phrase() -> str:
    words = ("amber", "copper", "garden", "harbor", "jacket", "lantern", "meadow", "orbit", "pencil", "velvet")
    return " ".join(secrets.choice(words) for _ in range(3))


def _figure_eight_coherence(observations: list[Observation]) -> tuple[bool, str]:
    path: list[tuple[float, float]] = []
    for observation in observations:
        point = (observation.x, observation.y)
        if not path or ((point[0] - path[-1][0]) ** 2 + (point[1] - path[-1][1]) ** 2) ** 0.5 >= 0.012:
            path.append(point)
    if len(path) < 8:
        return False, "Too few distinct QR positions were captured to compare the figure-eight path."

    xs = [point[0] for point in path]
    ys = [point[1] for point in path]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    if span_x < 0.12 or span_y < 0.10:
        return False, "The QR path did not cover enough of both image axes for a figure-eight check."

    center_x = (max(xs) + min(xs)) / 2
    center_y = (max(ys) + min(ys)) / 2
    left = right = upper = lower = False
    quadrants: set[tuple[int, int]] = set()
    center_crossings = 0
    was_near_center = False
    for x, y in path:
        dx = (x - center_x) / (span_x / 2)
        dy = (y - center_y) / (span_y / 2)
        if abs(dx) <= 0.28 and abs(dy) <= 0.32:
            if not was_near_center:
                center_crossings += 1
            was_near_center = True
            continue
        was_near_center = False
        if abs(dx) > 0.18 and abs(dy) > 0.18:
            quadrants.add((1 if dx > 0 else -1, 1 if dy > 0 else -1))
        left = left or dx < -0.25
        right = right or dx > 0.25
        upper = upper or dy < -0.25
        lower = lower or dy > 0.25

    coherent = left and right and upper and lower and len(quadrants) >= 3 and center_crossings >= 2
    if coherent:
        return True, "The server-observed QR path crossed its center and visited both lobes across both image axes."
    return False, "The observed QR path did not provide enough ordered crossings and lobe coverage to support a figure-eight check."


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/motion-guide.svg", include_in_schema=False)
def motion_guide() -> FileResponse:
    return FileResponse(ROOT / "static" / "figure-eight.svg", media_type="image/svg+xml")


@app.get("/phone/{session_id}", include_in_schema=False)
def phone_page(session_id: UUID) -> FileResponse:
    _session(session_id)
    return FileResponse(
        ROOT / "static" / "phone.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models/status", response_model=WarmupResult, tags=["screening"])
def get_model_status() -> WarmupResult:
    return WarmupResult(video=video_model_status(), audio=audio_model_status())


@app.post("/api/models/warmup", response_model=WarmupResult, tags=["screening"])
async def warmup_models(body: WarmupRequest) -> WarmupResult:
    video_status, audio_status = await run_in_threadpool(_warm_requested_models, body.audio_consent)
    return WarmupResult(video=video_status, audio=audio_status)


def _warm_requested_models(audio_consent: bool) -> tuple[dict[str, str], dict[str, str]]:
    video_status = warm_video_model()
    audio_status = warm_audio_model() if audio_consent else audio_model_status()
    if audio_consent:
        audio_status["transcription"] = warm_transcriber()["status"]
    return video_status, audio_status


@app.post("/api/sessions", response_model=SessionView, status_code=201, tags=["sessions"])
def create_session() -> SessionView:
    now = utc_now()
    for existing in _sessions.values():
        if now >= existing.expires_at:
            _clear_expired_evidence(existing)
    state = SessionState(
        id=uuid4(),
        pair_token=secrets.token_urlsafe(32),
        created_at=now,
        expires_at=now + SESSION_TTL,
    )
    _sessions[state.id] = state
    return _public_session(state)


@app.get("/api/sessions/{session_id}", response_model=SessionView, tags=["sessions"])
def get_session(session_id: UUID) -> SessionView:
    return _public_session(_session(session_id))


@app.get("/api/sessions/{session_id}/pairing-qr", response_class=Response, tags=["phone pairing"])
def pairing_qr(session_id: UUID, request: Request) -> Response:
    state = _session(session_id)
    _ensure_active(state)
    phone_url = _public_origin(request) + f"/phone/{session_id}?token={state.pair_token}"
    return Response(_qr_png(phone_url), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/api/sessions/{session_id}/pair", response_model=SessionView, tags=["phone pairing"])
def pair_phone(session_id: UUID, body: PairRequest) -> SessionView:
    state = _session(session_id)
    _ensure_active(state)
    if state.paired:
        raise HTTPException(status_code=409, detail="A phone is already paired")
    if not hmac.compare_digest(state.pair_token, body.token):
        raise HTTPException(status_code=403, detail="Invalid pairing token")
    state.paired = True
    return _public_session(state)


@app.post("/api/sessions/{session_id}/phone/webauthn/registration/options", tags=["phone verification"])
def phone_registration_options(
    session_id: UUID,
    body: PhoneDeviceRequest,
    request: Request,
    x_pair_token: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    state = _session(session_id)
    _ensure_active(state)
    _require_phone_token(state, x_pair_token)
    device_id = str(body.device_id)
    if device_id in _phone_credentials:
        raise HTTPException(status_code=409, detail="A phone credential is already registered")
    rp_id, _ = _webauthn_context(request)
    options = generate_registration_options(
        rp_id=rp_id,
        rp_name="Fusion Identity Check",
        user_id=body.device_id.bytes,
        user_name=f"fusion-phone-{body.device_id}",
        user_display_name="Fusion phone authenticator",
        authenticator_selection=AuthenticatorSelectionCriteria(
            authenticator_attachment=AuthenticatorAttachment.PLATFORM,
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.REQUIRED,
        ),
    )
    _set_pending_webauthn(state, options.challenge, "registration")
    state.pending_phone_device_id = device_id
    return json.loads(options_to_json(options))


@app.post("/api/sessions/{session_id}/phone/webauthn/registration/verify", response_model=SessionView, tags=["phone verification"])
def verify_phone_registration(
    session_id: UUID,
    body: WebAuthnCredentialSubmission,
    request: Request,
    x_pair_token: Annotated[str | None, Header()] = None,
) -> SessionView:
    state = _session(session_id)
    _ensure_active(state)
    _require_phone_token(state, x_pair_token)
    if body.credential.get("authenticatorAttachment") != "platform":
        raise HTTPException(status_code=400, detail="Use the phone's built-in platform authenticator")
    device_id = state.pending_phone_device_id
    if device_id is None:
        raise HTTPException(status_code=409, detail="Request phone registration options first")
    expected_challenge = _take_pending_webauthn(state, "registration")
    state.pending_phone_device_id = None
    rp_id, origin = _webauthn_context(request)
    try:
        verification = verify_registration_response(
            credential=body.credential,
            expected_challenge=expected_challenge,
            expected_rp_id=rp_id,
            expected_origin=origin,
            require_user_verification=True,
        )
    except WebAuthnException as exc:
        raise HTTPException(status_code=400, detail="Phone credential registration could not be verified") from exc

    if any(hmac.compare_digest(existing.id, verification.credential_id) for existing in _phone_credentials.values()):
        raise HTTPException(status_code=409, detail="This phone credential is already registered")
    _phone_credentials[device_id] = PhoneCredential(
        id=verification.credential_id,
        public_key=verification.credential_public_key,
        sign_count=verification.sign_count,
    )
    state.phone_credential_id = device_id
    state.phone_verified = False
    return _public_session(state)


@app.post("/api/sessions/{session_id}/phone/webauthn/authentication/options", tags=["phone verification"])
def phone_authentication_options(
    session_id: UUID,
    body: PhoneDeviceRequest,
    request: Request,
    x_pair_token: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    state = _session(session_id)
    _ensure_active(state)
    _require_phone_token(state, x_pair_token)
    device_id = str(body.device_id)
    credential = _phone_credentials.get(device_id)
    if credential is None:
        raise HTTPException(status_code=409, detail="Register the phone's platform credential first")
    state.phone_credential_id = device_id
    rp_id, _ = _webauthn_context(request)
    options = generate_authentication_options(
        rp_id=rp_id,
        allow_credentials=[PublicKeyCredentialDescriptor(id=credential.id)],
        user_verification=UserVerificationRequirement.REQUIRED,
    )
    _set_pending_webauthn(state, options.challenge, "authentication")
    state.phone_verified = False
    return json.loads(options_to_json(options))


@app.post("/api/sessions/{session_id}/phone/webauthn/authentication/verify", response_model=SessionView, tags=["phone verification"])
def verify_phone_authentication(
    session_id: UUID,
    body: WebAuthnCredentialSubmission,
    request: Request,
    x_pair_token: Annotated[str | None, Header()] = None,
) -> SessionView:
    state = _session(session_id)
    _ensure_active(state)
    _require_phone_token(state, x_pair_token)
    device_id = state.phone_credential_id
    credential = _phone_credentials.get(device_id or "")
    if credential is None:
        raise HTTPException(status_code=409, detail="Register the phone's platform credential first")
    supplied_id = body.credential.get("rawId")
    supplied_text_id = body.credential.get("id")
    if not isinstance(supplied_id, str) or not isinstance(supplied_text_id, str):
        raise HTTPException(status_code=403, detail="Phone credential does not match this session")
    try:
        supplied_id_bytes = base64url_to_bytes(supplied_id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=403, detail="Phone credential does not match this session") from exc
    if not hmac.compare_digest(supplied_id_bytes, credential.id) or not hmac.compare_digest(supplied_text_id, supplied_id):
        raise HTTPException(status_code=403, detail="Phone credential does not match this session")
    expected_challenge = _take_pending_webauthn(state, "authentication")
    rp_id, origin = _webauthn_context(request)
    try:
        verification = verify_authentication_response(
            credential=body.credential,
            expected_challenge=expected_challenge,
            expected_rp_id=rp_id,
            expected_origin=origin,
            credential_public_key=credential.public_key,
            credential_current_sign_count=credential.sign_count,
            require_user_verification=True,
        )
    except WebAuthnException as exc:
        raise HTTPException(status_code=400, detail="Phone user-verification signature could not be verified") from exc

    credential.sign_count = verification.new_sign_count
    state.phone_verified = True
    return _public_session(state)


@app.post("/api/sessions/{session_id}/challenge", response_model=ChallengeView, status_code=201, tags=["challenge"])
def create_challenge(session_id: UUID, response: Response) -> ChallengeView:
    state = _session(session_id)
    _ensure_active(state)
    if not state.phone_verified:
        raise HTTPException(status_code=409, detail="Complete phone user verification before starting the challenge")
    if state.challenge is not None and not state.challenge.submitted and utc_now() < state.challenge.expires_at:
        response.status_code = 200
        return _public_challenge(state.challenge)

    now = utc_now()
    state.challenge = ChallengeState(
        id=uuid4(),
        created_at=now,
        expires_at=min(now + CHALLENGE_TTL, state.expires_at),
        codes=[secrets.token_urlsafe(9) for _ in range(CHALLENGE_STEPS)],
        audio_phrase=_new_audio_phrase(),
    )
    state.video_frames = []
    state.server_observations = []
    state.last_qr_path_sample_ms = -QR_PATH_SAMPLE_INTERVAL_MS
    state.audio_result = None
    state.result = None
    return _public_challenge(state.challenge)


@app.get("/api/sessions/{session_id}/challenge", response_model=ChallengeView, tags=["challenge"])
def get_challenge(
    session_id: UUID,
    x_pair_token: Annotated[str | None, Header()] = None,
) -> ChallengeView:
    state = _session(session_id)
    _ensure_active(state)
    _require_phone_token(state, x_pair_token)
    if state.challenge is None:
        raise HTTPException(status_code=404, detail="Challenge has not started")
    return _public_challenge(state.challenge)


@app.get("/api/sessions/{session_id}/challenge/{challenge_id}/qr/{step}", response_class=Response, tags=["challenge"])
def challenge_qr(
    session_id: UUID,
    challenge_id: UUID,
    step: int,
    x_pair_token: Annotated[str | None, Header()] = None,
) -> Response:
    state = _session(session_id)
    _ensure_active(state)
    _require_phone_token(state, x_pair_token)
    challenge = state.challenge
    if challenge is None or challenge.id != challenge_id or utc_now() >= challenge.expires_at:
        raise HTTPException(status_code=410, detail="Challenge expired or unavailable")
    if not 0 <= step < len(challenge.codes):
        raise HTTPException(status_code=404, detail="Challenge step not found")
    payload = f"FUSION26|{session_id}|{challenge_id}|{step}|{challenge.codes[step]}"
    return Response(_qr_png(payload), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get(
    "/api/sessions/{session_id}/challenge/{challenge_id}/audio-prompt",
    response_model=AudioChallengeView,
    tags=["audio screening"],
)
def get_audio_prompt(session_id: UUID, challenge_id: UUID) -> AudioChallengeView:
    state = _session(session_id)
    _ensure_active(state)
    challenge = state.challenge
    if challenge is None or challenge.id != challenge_id or utc_now() >= challenge.expires_at:
        raise HTTPException(status_code=410, detail="Challenge expired or unavailable")
    return AudioChallengeView(
        challenge_id=challenge.id,
        phrase=challenge.audio_phrase,
        expires_at=challenge.expires_at,
    )


@app.post(
    "/api/sessions/{session_id}/challenge/{challenge_id}/audio",
    response_model=AudioSubmissionResult,
    tags=["audio screening"],
)
async def submit_audio(
    session_id: UUID,
    challenge_id: UUID,
    request: Request,
    x_audio_consent: Annotated[str | None, Header()] = None,
) -> AudioSubmissionResult:
    state = _session(session_id)
    _ensure_active(state)
    challenge = state.challenge
    if challenge is None or challenge.id != challenge_id or utc_now() >= challenge.expires_at:
        raise HTTPException(status_code=410, detail="Challenge expired or unavailable")
    if not state.phone_verified:
        raise HTTPException(status_code=409, detail="Complete phone user verification before submitting audio")
    if x_audio_consent != "true":
        raise HTTPException(status_code=403, detail="Explicit microphone consent is required for audio screening")
    if challenge.audio_submitted:
        raise HTTPException(status_code=409, detail="Audio evidence has already been submitted for this challenge")
    accepted_types = {"audio/webm", "audio/ogg", "audio/mp4", "audio/wav", "audio/x-wav"}
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
    if content_type not in accepted_types:
        raise HTTPException(status_code=415, detail="Audio must be submitted as WebM, Ogg, MP4, or WAV")

    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio recording exceeds the 5 MB limit")
        chunks.append(chunk)
    audio_bytes = b"".join(chunks)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio recording is empty")

    try:
        result = await run_in_threadpool(analyze_audio, audio_bytes, challenge.audio_phrase)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Audio screening failed for session %s", session_id)
        result = {
            "duration_seconds": None,
            "phrase_status": "unavailable",
            "phrase_detail": f"Audio screening failed: {type(error).__name__}: {error}",
            "spoof_status": "unavailable",
            "spoof": {"detail": f"Audio screening failed: {type(error).__name__}: {error}"},
        }

    challenge.audio_submitted = True
    state.audio_result = result
    spoof = result.get("spoof") or {}
    return AudioSubmissionResult(
        phrase_check=CheckResult(
            status=result["phrase_status"],
            detail=result["phrase_detail"] or "Phrase verification did not return a result.",
            evidence={"phrase_match": result["phrase_status"] == "passed"},
        ),
        anti_spoof_check=CheckResult(
            status=result["spoof_status"],
            detail=spoof.get("detail")
            or (
                f"AI-voice score {spoof['ai_voice_score']}; human-voice score {spoof['human_voice_score']}. "
                f"{spoof['detail']}"
            ),
            evidence=spoof,
        ),
        duration_seconds=result["duration_seconds"],
    )


@app.post(
    "/api/sessions/{session_id}/challenge/{challenge_id}/video",
    response_model=VideoSubmissionResult,
    tags=["challenge"],
)
async def submit_video(
    session_id: UUID,
    challenge_id: UUID,
    request: Request,
    x_video_consent: Annotated[str | None, Header()] = None,
) -> VideoSubmissionResult:
    state = _session(session_id)
    _ensure_active(state)
    challenge = state.challenge
    if challenge is None or challenge.id != challenge_id or utc_now() >= challenge.expires_at:
        raise HTTPException(status_code=410, detail="Challenge expired or unavailable")
    if challenge.submitted:
        raise HTTPException(status_code=409, detail="Challenge evidence has already been submitted")
    if not state.phone_verified:
        raise HTTPException(status_code=409, detail="Complete phone user verification before submitting video")
    if x_video_consent != "true":
        raise HTTPException(status_code=403, detail="Explicit camera consent is required for video screening")
    if not state.server_observations:
        raise HTTPException(status_code=409, detail="Scan the first live QR signal before submitting video")
    if challenge.video_submitted:
        raise HTTPException(status_code=409, detail="Video evidence has already been submitted for this challenge")
    accepted_types = {"video/webm", "video/mp4", "video/quicktime"}
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
    if content_type not in accepted_types:
        raise HTTPException(status_code=415, detail="Video must be submitted as WebM, MP4, or QuickTime")

    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_VIDEO_BYTES:
            raise HTTPException(status_code=413, detail="Video recording exceeds the 25 MB limit")
        chunks.append(chunk)
    video_bytes = b"".join(chunks)
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Video recording is empty")

    try:
        sampled_frames = await run_in_threadpool(decode_video_frames, video_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    video_frames: list[dict[str, Any]] = []
    for elapsed_ms, image in sampled_frames:
        try:
            analysis = await run_in_threadpool(analyze_video_frame, image)
        except Exception as error:
            logger.exception("Recorded video screening failed for session %s", session_id)
            analysis = {
                "status": "unavailable",
                "detail": f"Recorded video screening failed: {type(error).__name__}: {error}",
            }
        video_frames.append({"elapsed_ms": elapsed_ms, **analysis})

    state.video_frames = video_frames
    challenge.video_submitted = True
    summary = summarize_video_frames(video_frames)
    return VideoSubmissionResult(
        frames_sampled=summary["frames_sampled"],
        frames_with_faces=summary["frames_with_faces"],
        median_fake_score=summary["median_fake_score"],
        score_range=summary["score_range"],
        detail=summary["detail"],
    )


@app.post(
    "/api/sessions/{session_id}/challenge/{challenge_id}/scan",
    response_model=FrameScanResult,
    tags=["challenge"],
)
async def scan_challenge_frame(
    session_id: UUID,
    challenge_id: UUID,
    request: Request,
    elapsed_ms: Annotated[int, Query(ge=0, le=30_000)],
) -> FrameScanResult:
    state = _session(session_id)
    _ensure_active(state)
    challenge = state.challenge
    if challenge is None or challenge.id != challenge_id or utc_now() >= challenge.expires_at:
        raise HTTPException(status_code=410, detail="Challenge expired or unavailable")
    if challenge.submitted:
        raise HTTPException(status_code=409, detail="Challenge evidence has already been submitted")
    if request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip() != "image/jpeg":
        raise HTTPException(status_code=415, detail="QR scan frames must be JPEG images")

    frame_chunks: list[bytes] = []
    frame_size = 0
    async for chunk in request.stream():
        frame_size += len(chunk)
        if frame_size > MAX_QR_FRAME_BYTES:
            raise HTTPException(status_code=413, detail="QR scan frame exceeds the 1 MB limit")
        frame_chunks.append(chunk)
    image_bytes = b"".join(frame_chunks)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="QR scan frame is not a valid image")

    server_elapsed_ms = max(0, int((utc_now() - challenge.created_at).total_seconds() * 1000))
    payload, points, _ = cv2.QRCodeDetector().detectAndDecode(image)
    if points is None:
        return FrameScanResult(elapsed_ms=elapsed_ms)

    height, width = image.shape[:2]
    corners = points[0]
    left, top = corners.min(axis=0)
    right, bottom = corners.max(axis=0)
    center_x, center_y = corners.mean(axis=0)
    step = _challenge_code_step(challenge, payload, session_id) if payload else None
    if step is not None and server_elapsed_ms - state.last_qr_path_sample_ms >= QR_PATH_SAMPLE_INTERVAL_MS:
        state.last_qr_path_sample_ms = server_elapsed_ms
        if state.server_observations is None:
            state.server_observations = []
        state.server_observations.append(Observation(
            payload=payload,
            elapsed_ms=min(server_elapsed_ms, 30_000),
            x=float(np.clip(center_x / width, 0, 1)),
            y=float(np.clip(center_y / height, 0, 1)),
        ))
        if len(state.server_observations) > MAX_OBSERVATIONS:
            state.server_observations = state.server_observations[-MAX_OBSERVATIONS:]
    return FrameScanResult(
        payload=payload if step is not None else None,
        elapsed_ms=elapsed_ms,
        x=float(np.clip(center_x / width, 0, 1)),
        y=float(np.clip(center_y / height, 0, 1)),
        box_x=float(np.clip(left / width, 0, 1)),
        box_y=float(np.clip(top / height, 0, 1)),
        box_width=float(np.clip((right - left) / width, 0, 1)),
        box_height=float(np.clip((bottom - top) / height, 0, 1)),
    )


@app.post("/api/sessions/{session_id}/evidence", response_model=VerificationReport, tags=["verification"])
def submit_evidence(session_id: UUID, body: EvidenceSubmission) -> VerificationReport:
    state = _session(session_id)
    _ensure_active(state)
    challenge = state.challenge
    if not state.phone_verified:
        raise HTTPException(status_code=409, detail="Complete phone user verification before submitting evidence")
    if challenge is None or challenge.id != body.challenge_id:
        raise HTTPException(status_code=404, detail="Challenge not found for this session")
    if challenge.submitted:
        raise HTTPException(status_code=409, detail="Challenge evidence has already been submitted")
    if utc_now() >= challenge.expires_at:
        raise HTTPException(status_code=410, detail="Challenge expired; request a new one")
    if not challenge.video_submitted:
        raise HTTPException(status_code=409, detail="Complete the recorded QR video screening before viewing the report")

    observations = state.server_observations or []
    matched_steps: list[int] = []
    step_positions: dict[int, Observation] = {}
    for observation in observations:
        step = _challenge_code_step(challenge, observation.payload, session_id)
        if step is None:
            continue
        if not matched_steps:
            matched_steps.append(step)
            step_positions[step] = observation
        elif step == matched_steps[-1]:
            continue
        elif len(matched_steps) < CHALLENGE_STEPS and step == (matched_steps[-1] + 1) % CHALLENGE_STEPS:
            matched_steps.append(step)
            step_positions[step] = observation

    server_capture_duration_ms = (
        max(item.elapsed_ms for item in observations) - min(item.elapsed_ms for item in observations)
        if len(observations) > 1
        else 0
    )
    qr_passed = len(matched_steps) == CHALLENGE_STEPS and server_capture_duration_ms >= 4_000
    path_coherent, path_detail = _figure_eight_coherence(observations)
    path_moved = (
        bool(observations)
        and max(item.x for item in observations) - min(item.x for item in observations) >= 0.12
        and max(item.y for item in observations) - min(item.y for item in observations) >= 0.10
    )
    challenge.submitted = True

    video_summary = summarize_video_frames(state.video_frames or [])
    if video_summary["frames_with_faces"] >= 3:
        frame_refs = ", ".join(
            "{}ms={}".format(item["elapsed_ms"], item["fake_score"])
            for item in video_summary["frame_evidence"][:4]
        )
        face_check = CheckResult(
            status="review",
            detail=(
                f"Research image classifier scored {video_summary['frames_with_faces']} face-bearing frames; "
                f"median AI-generated score {video_summary['median_fake_score']}, range {video_summary['score_range']}. "
                f"Top frame references: {frame_refs}. "
                f"{video_summary['detail']}"
            ),
            evidence={
                "model": "dima806/deepfake_vs_real_image_detection",
                "model_revision": "29e4cf9efc543845610045f6ba7e88e5cf9d9301",
                "frames": video_summary["frame_evidence"],
                "median_fake_score": video_summary["median_fake_score"],
                "score_range": video_summary["score_range"],
            },
        )
        temporal_check = CheckResult(
            status="review",
            detail=(
                f"Cross-frame comparison: {video_summary['temporal_consistency']}. "
                "This is a basic score/face-count consistency signal, not a trained temporal deepfake detector."
            ),
            evidence={
                "frames_sampled": video_summary["frames_sampled"],
                "frames_with_faces": video_summary["frames_with_faces"],
                "score_range": video_summary["score_range"],
                "consistency": video_summary["temporal_consistency"],
            },
        )
    else:
        face_check = CheckResult(status="unavailable", detail=video_summary["detail"])
        temporal_check = CheckResult(
            status="unavailable",
            detail="Too few successfully scored face-bearing frames were available for temporal comparison.",
        )

    audio_result = state.audio_result
    if audio_result is None:
        phrase_check = CheckResult(
            status="unavailable",
            detail="No consented phrase recording was submitted for this challenge.",
        )
        audio_spoof_check = CheckResult(
            status="unavailable",
            detail="No consented audio was submitted for screening.",
        )
    else:
        phrase_check = CheckResult(
            status=audio_result["phrase_status"],
            detail=audio_result["phrase_detail"] or "Phrase verification did not return a result.",
            evidence={"phrase_match": audio_result["phrase_status"] == "passed"},
        )
        spoof_result = audio_result.get("spoof") or {}
        audio_spoof_check = CheckResult(
            status=audio_result["spoof_status"],
            detail=spoof_result.get("detail")
            or (
                f"Uncalibrated scores: AI voice {spoof_result['ai_voice_score']}; "
                f"human voice {spoof_result['human_voice_score']}."
            ),
            evidence=spoof_result,
        )

    checks = {
        "phone_pairing": CheckResult(status="passed", detail="The session-specific pairing token was accepted."),
        "phone_user_verification": CheckResult(status="passed", detail="The phone platform authenticator verified the user and signed a fresh server challenge."),
        "randomized_qr_sequence": CheckResult(
            status="passed" if qr_passed else "failed",
            detail=(
                f"Server-decoded {len(matched_steps)} of {CHALLENGE_STEPS} fresh codes in cyclic order "
                f"over {server_capture_duration_ms} ms."
            ),
        ),
        "phone_motion": CheckResult(
            status="passed" if qr_passed and path_moved else "review",
            detail=(
                "The server-observed QR path varied across both image axes."
                if path_moved
                else "The server did not observe enough QR movement across both image axes."
            ),
        ),
        "challenge_path_coherence": CheckResult(
            status="passed" if qr_passed and path_coherent else "review",
            detail=path_detail,
        ),
        "face_deepfake_analysis": face_check,
        "video_temporal_consistency": temporal_check,
        "speaker_verification": CheckResult(
            status="unavailable",
            detail="Not performed by design; speaker enrollment and identity matching are out of scope.",
        ),
        "random_phrase_verification": phrase_check,
        "audio_spoof_detection": audio_spoof_check,
        "capture_quality": CheckResult(
            status="review" if server_capture_duration_ms < 4_000 else "passed",
            detail=(
                f"AI-screened {len(state.video_frames or [])} sampled frames from the recorded QR video and "
                f"{len(observations)} valid QR path observations over {server_capture_duration_ms} ms."
            ),
        ),
    }
    decision = "review" if qr_passed else "challenge_failed"
    generated_at = utc_now()
    audit_payload = {
        "decision": decision,
        "checks": {key: value.model_dump() for key, value in sorted(checks.items())},
    }
    evidence_hash = hashlib.sha256(
        json.dumps(audit_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit_entry = _audit_ledger.append(evidence_hash, generated_at)
    report = VerificationReport(
        session_id=session_id,
        decision=decision,
        checks=checks,
        limitations=[
            "The server decodes QR payloads and measures QR positions from client-submitted frames; a modified client can still fabricate frames.",
            "Phone user verification proves use of the paired platform credential; it does not identify the camera subject or establish civil identity.",
            "Image and audio classifier scores are research signals, can be wrong or drift, and are never an identity match or an automatic decision.",
            "Temporal video analysis compares sampled face presence and image-classifier score variation; it is not a trained temporal authenticity model.",
            "The audit log is an in-memory hash-chain simulator, not a real blockchain; only evidence hashes and chain metadata are recorded.",
            "Audio is processed in memory only after the explicit microphone-consent step; no raw audio is retained.",
            "Session state is in memory and resets when the API restarts.",
        ],
        audit={
            "system": "in-memory hash-chain simulator (not a blockchain)",
            "sequence": audit_entry["sequence"],
            "previous_hash": audit_entry["previous_hash"],
            "evidence_hash": audit_entry["evidence_hash"],
            "record_hash": audit_entry["record_hash"],
        },
        generated_at=generated_at,
    )
    state.result = report.model_dump(mode="json")
    return report


@app.get("/api/audit/ledger", tags=["audit"])
def get_audit_ledger() -> dict[str, Any]:
    return {
        "system": "in-memory hash-chain simulator (not a blockchain)",
        "entries": _audit_ledger.entries(),
        "integrity_valid": _audit_ledger.verify(),
    }


@app.get("/api/sessions/{session_id}/result", response_model=VerificationReport, tags=["verification"])
def get_result(session_id: UUID) -> VerificationReport:
    state = _session(session_id)
    _ensure_active(state)
    if state.result is None:
        raise HTTPException(status_code=404, detail="No verification report is available yet")
    return VerificationReport.model_validate(state.result)
