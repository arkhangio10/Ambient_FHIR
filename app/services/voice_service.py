"""Voice clarification service — ElevenLabs TTS.

Converts missing-field clarification prompts into short
synthesized audio clips. ONE prompt per session, ONE field at a time.

This service NEVER raises to the orchestrator.
text_fallback is ALWAYS populated in every VoiceResult.
"""

import asyncio
import base64
import logging
import os
import tempfile

from app.config import get_settings
from app.schemas.entities import VoiceResult

logger = logging.getLogger(__name__)

# In-memory audio cache — one entry per session
# Prevents duplicate ElevenLabs API calls
_audio_cache: dict[str, VoiceResult] = {}


# ── 1. synthesize_clarification ─────────────────────────────────────

async def synthesize_clarification(
    session_id: str,
    clarification_prompt: str,
    voice_id: str | None = None,
) -> VoiceResult:
    """Convert clarification prompt to audio via ElevenLabs TTS.

    Returns VoiceResult with both audio_url and base64 bytes.
    text_fallback is ALWAYS populated regardless of success.
    Never raises.
    """
    settings = get_settings()

    # Guard: check cache first
    cached = _audio_cache.get(session_id)
    if cached:
        return cached

    # Guard: empty prompt
    if not clarification_prompt or not clarification_prompt.strip():
        return VoiceResult(
            session_id=session_id,
            success=False,
            text_fallback="Please provide missing information.",
            error_message="Empty clarification prompt",
        )

    # Truncate if needed
    if len(clarification_prompt) > 150:
        logger.warning("Prompt truncated for session %s", session_id)
        clarification_prompt = clarification_prompt[:147] + "..."

    # Guard: no API key
    if not settings.elevenlabs_api_key:
        return VoiceResult(
            session_id=session_id,
            success=False,
            text_fallback=clarification_prompt,
            error_message=(
                "ElevenLabs API key not configured — "
                "displaying text prompt instead"
            ),
            character_count=len(clarification_prompt),
        )

    try:
        from elevenlabs.client import AsyncElevenLabs

        client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
        selected_voice_id = voice_id or settings.elevenlabs_voice_id

        audio_generator = client.text_to_speech.convert(
            text=clarification_prompt,
            voice_id=selected_voice_id,
            model_id=settings.elevenlabs_model_id,
            output_format="mp3_44100_128",
        )

        # Collect audio bytes from async generator
        audio_bytes = b""
        async for chunk in audio_generator:
            audio_bytes += chunk

        if not audio_bytes:
            return VoiceResult(
                session_id=session_id,
                success=False,
                text_fallback=clarification_prompt,
                error_message="ElevenLabs returned empty audio",
                character_count=len(clarification_prompt),
            )

        # Save to temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{session_id}_clarification.mp3")
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Schedule cleanup after 10 minutes
        asyncio.create_task(_cleanup_audio_file(temp_path, delay_seconds=600))

        result = VoiceResult(
            session_id=session_id,
            success=True,
            audio_url=f"/audio/{session_id}",
            audio_bytes_b64=audio_b64,
            text_fallback=clarification_prompt,
            voice_id_used=selected_voice_id,
            character_count=len(clarification_prompt),
        )

        _audio_cache[session_id] = result
        return result

    except ImportError:
        return VoiceResult(
            session_id=session_id,
            success=False,
            text_fallback=clarification_prompt,
            error_message="ElevenLabs SDK not available.",
            character_count=len(clarification_prompt),
        )
    except Exception as e:
        logger.error("Voice synthesis failed for %s: %s", session_id, e)
        return VoiceResult(
            session_id=session_id,
            success=False,
            text_fallback=clarification_prompt,
            error_message=f"Voice synthesis failed: {e}",
            character_count=len(clarification_prompt),
        )


# ── 2. serve_audio_file ────────────────────────────────────────────

async def serve_audio_file(session_id: str) -> bytes | None:
    """Read and return audio bytes for a session.

    Returns None if the file doesn't exist.
    """
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{session_id}_clarification.mp3")

    if not os.path.exists(temp_path):
        return None

    with open(temp_path, "rb") as f:
        return f.read()


# ── 3. clear_session_audio ─────────────────────────────────────────

async def clear_session_audio(session_id: str) -> None:
    """Remove cached audio and temp file for a session.

    Called by orchestrator after clarification is resolved.
    """
    _audio_cache.pop(session_id, None)

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{session_id}_clarification.mp3")
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            logger.info("Audio cleaned up for session %s", session_id)
        except OSError as e:
            logger.warning("Failed to clean audio for %s: %s", session_id, e)


# ── 4. _cleanup_audio_file (PRIVATE) ───────────────────────────────

async def _cleanup_audio_file(path: str, delay_seconds: int = 600) -> None:
    """Delete temp audio file after delay. Runs as background task."""
    await asyncio.sleep(delay_seconds)
    if os.path.exists(path):
        try:
            os.remove(path)
            logger.debug("Auto-cleaned audio file: %s", path)
        except OSError:
            pass
