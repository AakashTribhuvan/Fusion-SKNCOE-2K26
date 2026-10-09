"""Local-first API for the Fusion identity verification prototype.

Session state is intentionally in memory for the demo. No camera, audio, or
biometric samples are persisted by this service.
"""

from __future__ import annotations

import hmac
import io
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4
from urllib.parse import urlsplit

import qrcode
from fastapi import FastAPI, Header, HTTPException, Request
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


ROOT = Path(__file__).resolve().parent
SESSION_TTL = timedelta(minutes=5)
CHALLENGE_TTL = timedelta(seconds=30)
CHALLENGE_STEPS = 6
CHALLENGE_STEP_SECONDS = 1.5
MAX_OBSERVATIONS = 120


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ChallengeState:
    id: UUID
    created_at: datetime
    expires_at: datetime
    codes: list[str]
    submitted: bool = False


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


class EvidenceSubmission(BaseModel):
    challenge_id: UUID
    observations: list[Observation] = Field(max_length=MAX_OBSERVATIONS)
    captured_duration_ms: int = Field(ge=0, le=30_000)
    audio_duration_ms: int = Field(ge=0, le=30_000)


class CheckResult(BaseModel):
    status: Literal["passed", "failed", "review", "unavailable"]
    detail: str


class VerificationReport(BaseModel):
    session_id: UUID
    decision: Literal["review", "inconclusive", "challenge_failed"]
    checks: dict[str, CheckResult]
    limitations: list[str]
    generated_at: datetime


_sessions: dict[UUID, SessionState] = {}
_phone_credentials: dict[str, PhoneCredential] = {}

app = FastAPI(
    title="Fusion Identity Verification API",
    version="0.3.0",
    description="Demo: phone WebAuthn user verification, session-bound pairing, and a randomized QR motion challenge.",
)


def _session(session_id: UUID) -> SessionState:
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def _ensure_active(state: SessionState) -> None:
    if utc_now() >= state.expires_at:
        raise HTTPException(status_code=410, detail="Session expired; start a new session")


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


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/phone/{session_id}", include_in_schema=False)
def phone_page(session_id: UUID) -> FileResponse:
    _session(session_id)
    return FileResponse(ROOT / "static" / "phone.html")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/sessions", response_model=SessionView, status_code=201, tags=["sessions"])
def create_session() -> SessionView:
    now = utc_now()
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
def create_challenge(session_id: UUID) -> ChallengeView:
    state = _session(session_id)
    _ensure_active(state)
    if not state.phone_verified:
        raise HTTPException(status_code=409, detail="Complete phone user verification before starting the challenge")
    if state.challenge is not None and not state.challenge.submitted and utc_now() < state.challenge.expires_at:
        raise HTTPException(status_code=409, detail="A challenge is already active")

    now = utc_now()
    state.challenge = ChallengeState(
        id=uuid4(),
        created_at=now,
        expires_at=min(now + CHALLENGE_TTL, state.expires_at),
        codes=[secrets.token_urlsafe(9) for _ in range(CHALLENGE_STEPS)],
    )
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

    matched_steps: list[int] = []
    step_positions: dict[int, Observation] = {}
    for observation in body.observations:
        parts = observation.payload.split("|")
        if len(parts) != 5 or parts[0] != "FUSION26":
            continue
        _, observed_session, observed_challenge, raw_step, code = parts
        try:
            step = int(raw_step)
        except ValueError:
            continue
        valid_code = (
            observed_session == str(session_id)
            and observed_challenge == str(challenge.id)
            and 0 <= step < len(challenge.codes)
            and hmac.compare_digest(challenge.codes[step], code)
        )
        if not valid_code:
            continue
        if not matched_steps:
            matched_steps.append(step)
            step_positions[step] = observation
        elif step == matched_steps[-1]:
            continue
        elif len(matched_steps) < CHALLENGE_STEPS and step == (matched_steps[-1] + 1) % CHALLENGE_STEPS:
            matched_steps.append(step)
            step_positions[step] = observation

    qr_passed = len(matched_steps) == CHALLENGE_STEPS and body.captured_duration_ms >= 4_000
    positions = [step_positions[step] for step in matched_steps]
    moved = False
    if len(positions) >= 3:
        moved = (
            max(item.x for item in positions) - min(item.x for item in positions) >= 0.12
            and max(item.y for item in positions) - min(item.y for item in positions) >= 0.12
        )
    motion_passed = qr_passed and moved
    challenge.submitted = True

    checks = {
        "phone_pairing": CheckResult(status="passed", detail="The session-specific pairing token was accepted."),
        "phone_user_verification": CheckResult(status="passed", detail="The phone platform authenticator verified the user and signed a fresh server challenge."),
        "randomized_qr_sequence": CheckResult(
            status="passed" if qr_passed else "failed",
            detail=f"Observed {len(matched_steps)} of {CHALLENGE_STEPS} fresh codes in cyclic order.",
        ),
        "phone_motion": CheckResult(
            status="passed" if motion_passed else ("failed" if qr_passed else "review"),
            detail="The browser-reported QR path varied across both image axes." if motion_passed else "The challenge did not provide enough varied QR positions.",
        ),
        "face_deepfake_analysis": CheckResult(status="unavailable", detail="A validated deepfake detector is not integrated yet."),
        "speaker_verification": CheckResult(status="unavailable", detail="A speaker enrollment and matching model are not integrated yet."),
        "audio_spoof_detection": CheckResult(status="unavailable", detail="An audio anti-spoof model is not integrated yet."),
        "capture_quality": CheckResult(
            status="review" if body.captured_duration_ms < 4_000 else "passed",
            detail=f"Browser reported {body.captured_duration_ms} ms video capture; audio analysis is outside the selected first scope.",
        ),
    }
    decision = "review" if motion_passed else "challenge_failed"
    report = VerificationReport(
        session_id=session_id,
        decision=decision,
        checks=checks,
        limitations=[
            "QR observations and capture durations are reported by the browser and are not independently decoded from uploaded media.",
            "Phone user verification proves use of the paired platform credential; it does not identify the camera subject or establish civil identity.",
            "A validated face deepfake classifier is not integrated yet; the final outcome remains Review.",
            "Session state is in memory and resets when the API restarts.",
        ],
        generated_at=utc_now(),
    )
    state.result = report.model_dump(mode="json")
    return report


@app.get("/api/sessions/{session_id}/result", response_model=VerificationReport, tags=["verification"])
def get_result(session_id: UUID) -> VerificationReport:
    state = _session(session_id)
    _ensure_active(state)
    if state.result is None:
        raise HTTPException(status_code=404, detail="No verification report is available yet")
    return VerificationReport.model_validate(state.result)
