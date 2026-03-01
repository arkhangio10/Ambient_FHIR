import logging
import tempfile
import os

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.services import triage_service
from app.services import transcription_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/triage", tags=["triage"])

class TriageMessage(BaseModel):
    role: str
    content: str

class TriageRequest(BaseModel):
    messages: List[TriageMessage]

class TriageResponse(BaseModel):
    reply: str
    audio_url: Optional[str] = None
    is_complete: bool

@router.post("/chat", response_model=TriageResponse)
async def triage_chat(request: TriageRequest):
    """Generate the next triage question or a final summary."""
    try:
        reply_dict = await triage_service.generate_triage_reply(request.messages)
        return TriageResponse(**reply_dict)
    except Exception as e:
        logger.error(f"Error in triage chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe user audio specifically for the triage flow."""
    try:
        content = await file.read()
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".webm"
        if not suffix:
            suffix = ".webm"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Re-use the existing transcription logic
        result = await transcription_service.transcribe_uploaded_audio(
            audio_reference=tmp_path,
            session_id="triage-session",
        )
        # Manually cleanup since the method doesn't take a cleanup arg
        try:
            os.remove(tmp_path)
        except OSError:
            pass
            
        return {"transcript": result.transcript if result.success else ""}
    except Exception as e:
        logger.error(f"Error in triage transcribe: {e}")
        raise HTTPException(status_code=500, detail=str(e))
