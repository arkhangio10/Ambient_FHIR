"""REST route for clinical data processing.

Endpoints:
- POST /process-clinical-data — typed text or clarification response
- POST /upload-audio — audio file upload
- GET /session/{session_id} — retrieve session state
- GET /sessions — list all sessions
- GET /audio/{session_id} — serve clarification audio

Delegates all logic to the ClinicalOrchestrator.
Zero business logic lives in this module.
"""

import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import get_settings
from app.schemas.input import ClinicalInput, InputMode
from app.services.orchestrator import orchestrator
from app.services import voice_service, storage_service

router = APIRouter(tags=["clinical"])
logger = logging.getLogger(__name__)


@router.post("/process-clinical-data")
async def process_clinical_data(input: ClinicalInput):
    """Process a clinical input payload through the orchestrator.

    Routes to the correct orchestrator method based on input_mode.
    Returns the full SessionState as JSON.
    """
    if input.input_mode == InputMode.clarification_response:
        # Clarification uses the EXISTING session — don't create a new one
        state = await orchestrator.handle_clarification_response(
            input.session_id, input.clarification_answer
        )
    else:
        # New input — create a fresh session
        state = await orchestrator.create_session(input)

        if input.input_mode == InputMode.typed_text:
            state = await orchestrator.handle_typed_input(
                state.session_id, input.transcript
            )
        elif input.input_mode == InputMode.uploaded_audio:
            state = await orchestrator.handle_uploaded_audio(
                state.session_id, input.audio_reference
            )

    return state.model_dump(mode="json")


@router.post("/upload-audio")
async def upload_audio(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload an audio file for transcription and processing.

    Validates file type and size, saves to temp location,
    and hands off to the orchestrator.
    """
    settings = get_settings()

    # Validate file type
    if file.content_type and file.content_type not in settings.supported_audio_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file.content_type}. "
                   f"Supported: {', '.join(settings.supported_audio_formats)}",
        )

    # Read and validate file size
    max_bytes = settings.max_audio_size_mb * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds {settings.max_audio_size_mb}MB limit.",
        )

    # Save to temp file
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False, prefix=f"{session_id}_"
    ) as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    try:
        # Create session if it doesn't exist
        existing = await orchestrator.get_session(session_id)
        if existing is None:
            input_data = ClinicalInput(
                session_id=session_id,
                input_mode=InputMode.uploaded_audio,
                audio_reference=temp_path,
            )
            await orchestrator.create_session(input_data)

        state = await orchestrator.handle_uploaded_audio(session_id, temp_path)
        return state.model_dump(mode="json")

    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@router.get("/audio/{session_id}")
async def get_clarification_audio(session_id: str):
    """Serve the synthesized clarification audio file.

    Called by frontend audio player.
    Returns audio/mpeg binary or 404.
    """
    audio_bytes = await voice_service.serve_audio_file(session_id)

    if not audio_bytes:
        raise HTTPException(
            status_code=404,
            detail="Audio not found for this session",
        )

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"inline; filename={session_id}_clarification.mp3",
            "Cache-Control": "no-cache",
        },
    )


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve the current state of a session."""
    state = await orchestrator.get_session(session_id)
    if state is None:
        return {"error": f"Session {session_id} not found"}
    return state.model_dump(mode="json")


@router.get("/sessions")
async def list_sessions(limit: int = 50):
    """List all sessions with basic metadata.

    Used by the frontend dashboard to show recent sessions.
    Returns list of {session_id, status, updated_at}.
    """
    return await storage_service.list_sessions(limit=limit)
