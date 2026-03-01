"""FastAPI application entrypoint.

Initializes the app, registers CORS middleware, mounts routers,
and exposes the /health endpoint.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes.realtime import router as realtime_router
from app.routes.clinical import router as clinical_router

settings = get_settings()

app = FastAPI(
    title="Behavioral Health Revenue Cycle Copilot",
    description=(
        "Voice-first AI copilot that converts clinician audio/text "
        "into validated prior-auth packets and FHIR-aligned payloads."
    ),
    version="0.1.0",
)

# ── CORS (permissive for hackathon) ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────
app.include_router(realtime_router)
app.include_router(clinical_router)


# ── Health check ────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health_check():
    """Return a simple health status."""
    return {"status": "ok"}
