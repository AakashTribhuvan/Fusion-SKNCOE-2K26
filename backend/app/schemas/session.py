from __future__ import annotations

from pydantic import BaseModel, Field


class SessionCreateResponse(BaseModel):
    session_id: str
    challenge_phrase: str
    created_at: str
    expires_at: str


class SessionChallengeRequest(BaseModel):
    session_id: str


class SessionChallengeResponse(BaseModel):
    session_id: str
    challenge_phrase: str
    created_at: str
    expires_at: str
    invalidated_previous: bool = True
