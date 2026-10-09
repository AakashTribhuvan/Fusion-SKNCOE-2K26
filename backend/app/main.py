"""Local-first API for the Fusion identity verification prototype.

Session state is intentionally in memory for the demo. No camera, audio, or
biometric samples are persisted by this service.
"""

from __future__ import annotations

import hmac
import io
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

import qrcode
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
SESSION_TTL = timedelta(minutes=5)
CHALLENGE_TTL = timedelta(seconds=30)
CHALLENGE_STEPS = 6
CHALLENGE_STEP_SECONDS = 1.5
MAX_OBSERVATIONS = 120
PHRASES = (
    "blue river seven",
    "silver maple twenty",
    "quiet amber lighthouse",
    "copper meadow thirty",
    "violet harbor sunrise",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ChallengeState:
    id: UUID
    phrase: str
    created_at: datetime
    expires_at: datetime
    codes: list[str]
    submitted: bool = False


@dataclass
class SessionState:
    id: UUID
    pair_token: str
    created_at: datetime
    expires_at: datetime
    paired: bool = False
    challenge: ChallengeState | None = None
    result: dict | None = None


class SessionView(BaseModel):
    id: UUID
    status: Literal["active", "expired"]
    paired: bool
    created_at: datetime
    expires_at: datetime


class ChallengeView(BaseModel):
    id: UUID
    phrase: str
    status: Literal["active", "expired", "submitted"]
    created_at: datetime
    expires_at: datetime
    step_count: int = CHALLENGE_STEPS
    step_seconds: float = CHALLENGE_STEP_SECONDS
    motion_instruction: str = "Move the phone through a slow figure eight while showing its screen to the laptop camera."


class PairRequest(BaseModel):
    token: str = Field(min_length=16, max_length=128)


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

app = FastAPI(
    title="Fusion Identity Verification API",
    version="0.2.0",
    description="Local demo: session-bound phone pairing and randomized QR motion challenge.",
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
        created_at=state.created_at,
        expires_at=state.expires_at,
    )


def _public_challenge(challenge: ChallengeState) -> ChallengeView:
    if challenge.submitted:
        status = "submitted"
    else:
        status = "expired" if utc_now() >= challenge.expires_at else "active"
    return ChallengeView(
        id=challenge.id,
        phrase=challenge.phrase,
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
    phone_url = str(request.base_url).rstrip("/") + f"/phone/{session_id}?token={state.pair_token}"
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


@app.post("/api/sessions/{session_id}/challenge", response_model=ChallengeView, status_code=201, tags=["challenge"])
def create_challenge(session_id: UUID) -> ChallengeView:
    state = _session(session_id)
    _ensure_active(state)
    if not state.paired:
        raise HTTPException(status_code=409, detail="Pair a phone before starting the challenge")
    if state.challenge is not None and not state.challenge.submitted and utc_now() < state.challenge.expires_at:
        raise HTTPException(status_code=409, detail="A challenge is already active")

    now = utc_now()
    state.challenge = ChallengeState(
        id=uuid4(),
        phrase=secrets.choice(PHRASES),
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
    if not state.paired:
        raise HTTPException(status_code=409, detail="Pair a phone before submitting evidence")
    if challenge is None or challenge.id != body.challenge_id:
        raise HTTPException(status_code=404, detail="Challenge not found for this session")
    if challenge.submitted:
        raise HTTPException(status_code=409, detail="Challenge evidence has already been submitted")
    if utc_now() >= challenge.expires_at:
        raise HTTPException(status_code=410, detail="Challenge expired; request a new one")

    matched_steps: list[int] = []
    for observation in body.observations:
        parts = observation.payload.split("|")
        if len(parts) != 5 or parts[0] != "FUSION26":
            continue
        _, observed_session, observed_challenge, raw_step, code = parts
        try:
            step = int(raw_step)
        except ValueError:
            continue
        if (
            observed_session == str(session_id)
            and observed_challenge == str(challenge.id)
            and 0 <= step < len(challenge.codes)
            and hmac.compare_digest(challenge.codes[step], code)
            and (not matched_steps or step > matched_steps[-1])
        ):
            matched_steps.append(step)

    qr_passed = len(matched_steps) == CHALLENGE_STEPS and body.captured_duration_ms >= 4_000
    positions = [observation for observation in body.observations if observation.payload in {
        f"FUSION26|{session_id}|{challenge.id}|{step}|{challenge.codes[step]}"
        for step in matched_steps
    }]
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
        "randomized_qr_sequence": CheckResult(
            status="passed" if qr_passed else "failed",
            detail=f"Observed {len(matched_steps)} of {CHALLENGE_STEPS} fresh codes in order.",
        ),
        "phone_motion": CheckResult(
            status="passed" if motion_passed else ("failed" if qr_passed else "review"),
            detail="The browser-reported QR path varied across both image axes." if motion_passed else "The challenge did not provide enough varied QR positions.",
        ),
        "face_deepfake_analysis": CheckResult(status="unavailable", detail="A validated deepfake detector is not integrated yet."),
        "speaker_verification": CheckResult(status="unavailable", detail="A speaker enrollment and matching model are not integrated yet."),
        "audio_spoof_detection": CheckResult(status="unavailable", detail="An audio anti-spoof model is not integrated yet."),
        "capture_quality": CheckResult(
            status="review" if body.audio_duration_ms < 500 or body.captured_duration_ms < 4_000 else "passed",
            detail=f"Browser reported {body.captured_duration_ms} ms video and {body.audio_duration_ms} ms audio capture.",
        ),
    }
    decision = "review" if motion_passed else "challenge_failed"
    report = VerificationReport(
        session_id=session_id,
        decision=decision,
        checks=checks,
        limitations=[
            "QR observations and capture durations are reported by the browser and are not independently decoded from uploaded media.",
            "This demo does not establish civil identity or detect deepfakes; missing model checks keep the outcome at Review.",
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
