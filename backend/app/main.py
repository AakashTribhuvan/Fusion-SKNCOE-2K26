from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.sessions import router as sessions_router
from app.api.verification import router as verification_router
from app.core.config import settings

app = FastAPI(title="FUSION Module B", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(sessions_router, prefix=settings.api_prefix)
app.include_router(verification_router)


@app.get("/")
def root() -> dict:
    return {"app": "FUSION Module B", "status": "ok"}
