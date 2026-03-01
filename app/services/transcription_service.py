"""Transcription service.

Primary: Voxtral Mini (batch transcription API)
Fallback: Manual text input

Optimized for speed:
- Audio chunks are accumulated per session
- Only NEW audio (since last transcription) is sent to the API
- Transcription runs in background tasks so WebSocket stays responsive
- On finalize, one final pass of the full audio ensures accuracy

This service NEVER raises exceptions to the orchestrator.
All failures return TranscriptionResult with success=False.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from collections import defaultdict

import httpx

from app.config import get_settings
from app.schemas.transcription import TranscriptionResult

logger = logging.getLogger(__name__)

# Supported audio extensions
_SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mp4", ".flac"}

# ── Audio buffer management ─────────────────────────────────────────

_audio_buffers: dict[str, bytearray] = defaultdict(bytearray)
_transcribed_offset: dict[str, int] = defaultdict(int)  # bytes already transcribed
_partial_transcripts: dict[str, str] = defaultdict(str)  # accumulated text
_bg_tasks: dict[str, asyncio.Task] = {}  # background transcription tasks
_bg_results: dict[str, str] = {}  # latest background transcription result


def append_audio_chunk(session_id: str, chunk: bytes) -> int:
    """Append a raw audio chunk to the session buffer. Returns buffer size."""
    _audio_buffers[session_id].extend(chunk)
    return len(_audio_buffers[session_id])


def get_audio_buffer(session_id: str) -> bytes:
    """Get the full accumulated audio buffer for a session."""
    return bytes(_audio_buffers[session_id])


def get_new_audio(session_id: str) -> bytes:
    """Get only the audio that hasn't been transcribed yet."""
    offset = _transcribed_offset.get(session_id, 0)
    return bytes(_audio_buffers[session_id][offset:])


def mark_transcribed(session_id: str) -> None:
    """Mark the current buffer position as transcribed."""
    _transcribed_offset[session_id] = len(_audio_buffers[session_id])


def get_latest_transcript(session_id: str) -> str | None:
    """Get the latest transcript text (from background task or partial)."""
    # Check if background task has a new result
    if session_id in _bg_results:
        result = _bg_results.pop(session_id)
        _partial_transcripts[session_id] = result
        return result
    return _partial_transcripts.get(session_id) or None


def clear_audio_buffer(session_id: str) -> None:
    """Clear all buffers for a session."""
    _audio_buffers.pop(session_id, None)
    _transcribed_offset.pop(session_id, None)
    _partial_transcripts.pop(session_id, None)
    _bg_results.pop(session_id, None)
    task = _bg_tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()


def get_buffer_size(session_id: str) -> int:
    """Get the current buffer size in bytes."""
    return len(_audio_buffers.get(session_id, b""))


def is_transcription_running(session_id: str) -> bool:
    """Check if a background transcription is currently running."""
    task = _bg_tasks.get(session_id)
    return task is not None and not task.done()


# ── Core transcription function ─────────────────────────────────────

async def _call_mistral_transcribe(
    session_id: str,
    audio_data: bytes,
    suffix: str = ".webm",
) -> TranscriptionResult:
    """Internal: Call Mistral API to transcribe audio bytes. Never raises."""
    settings = get_settings()

    if not settings.mistral_api_key:
        return TranscriptionResult(
            session_id=session_id,
            success=False,
            source="voxtral_mini",
            error_message="Transcription unavailable (no API key).",
        )

    if len(audio_data) < 500:
        return TranscriptionResult(
            session_id=session_id,
            success=False,
            source="voxtral_mini",
            error_message="Audio too short to transcribe.",
        )

    try:
        from mistralai import Mistral
        client = Mistral(api_key=settings.mistral_api_key)

        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix=f"live_{session_id}_"
        ) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            response = await client.audio.transcriptions.complete_async(
                model=settings.voxtral_mini_model,
                file={
                    "file_name": os.path.basename(tmp_path),
                    "content": open(tmp_path, "rb"),
                },
                language="en",
                timeout_ms=settings.transcription_timeout_seconds * 1000,
            )
            transcript_text = response.text if hasattr(response, "text") else str(response)

            return TranscriptionResult(
                session_id=session_id,
                transcript=transcript_text,
                success=True,
                source="voxtral_mini",
                language="en",
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except ImportError:
        return TranscriptionResult(
            session_id=session_id,
            success=False,
            source="voxtral_mini",
            error_message="Mistral SDK not available.",
        )
    except Exception as e:
        logger.error("Transcription API call failed for %s: %s", session_id, e)
        return TranscriptionResult(
            session_id=session_id,
            success=False,
            source="voxtral_mini",
            error_message=f"Transcription failed: {e}.",
        )


# ── Background transcription ────────────────────────────────────────

async def _bg_transcribe(session_id: str) -> None:
    """Background task: transcribe the full buffer and store result."""
    try:
        audio_data = get_audio_buffer(session_id)
        if len(audio_data) < 500:
            return

        result = await _call_mistral_transcribe(session_id, audio_data)
        if result.success and result.transcript:
            _bg_results[session_id] = result.transcript
            mark_transcribed(session_id)
            logger.info(
                "Background transcription done for %s: %d chars",
                session_id, len(result.transcript),
            )
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("Background transcription error for %s: %s", session_id, e)
    finally:
        _bg_tasks.pop(session_id, None)


def _maybe_start_bg_transcription(session_id: str) -> None:
    """Start a background transcription if buffer has enough new audio."""
    # Don't start if one is already running
    if is_transcription_running(session_id):
        return

    new_bytes = len(get_new_audio(session_id))
    # Transcribe every ~2 seconds of new audio (~32KB webm at ~128kbps)
    THRESHOLD = 32_000

    if new_bytes >= THRESHOLD:
        task = asyncio.create_task(_bg_transcribe(session_id))
        _bg_tasks[session_id] = task


# ── Public API ───────────────────────────────────────────────────────

async def transcribe_realtime_chunk(
    session_id: str,
    audio_chunk: bytes,
) -> TranscriptionResult:
    """Accumulate a live audio chunk and trigger background transcription.

    Returns immediately with the latest available transcript.
    Does NOT block waiting for Mistral API — transcription runs in the
    background and updates are picked up on subsequent calls.
    """
    append_audio_chunk(session_id, audio_chunk)

    # Kick off background transcription if enough new audio
    _maybe_start_bg_transcription(session_id)

    # Return whatever transcript we have so far (non-blocking)
    latest = get_latest_transcript(session_id)
    return TranscriptionResult(
        session_id=session_id,
        success=True,
        source="voxtral_mini",
        transcript=latest,  # None if still buffering, text if ready
    )


async def transcribe_audio_buffer(
    session_id: str,
    audio_data: bytes | None = None,
) -> TranscriptionResult:
    """Transcribe accumulated audio buffer (blocking — used for finalize).

    If audio_data is provided, transcribe that directly.
    Otherwise, transcribe the session's full accumulated buffer.
    """
    data = audio_data or get_audio_buffer(session_id)

    if not data or len(data) < 500:
        return TranscriptionResult(
            session_id=session_id,
            success=False,
            source="voxtral_mini",
            error_message="Audio buffer too small to transcribe.",
        )

    logger.info(
        "Final transcription for %s (%d bytes)",
        session_id, len(data),
    )

    return await _call_mistral_transcribe(session_id, data)


# ── File-based transcription ────────────────────────────────────────

async def transcribe_uploaded_audio(
    session_id: str,
    audio_reference: str,
) -> TranscriptionResult:
    """Transcribe a complete audio file via Voxtral Mini.

    Returns TranscriptionResult — never raises.
    """
    settings = get_settings()

    audio_path = Path(audio_reference)
    if not audio_path.exists():
        return TranscriptionResult(
            session_id=session_id,
            success=False,
            source="voxtral_mini",
            error_message=f"Audio file not found: {audio_reference}",
        )

    if audio_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        return TranscriptionResult(
            session_id=session_id,
            success=False,
            source="voxtral_mini",
            error_message=(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            ),
        )

    max_bytes = settings.max_audio_size_mb * 1024 * 1024
    file_size = audio_path.stat().st_size
    if file_size > max_bytes:
        return TranscriptionResult(
            session_id=session_id,
            success=False,
            source="voxtral_mini",
            error_message=(
                f"Audio file exceeds {settings.max_audio_size_mb}MB limit "
                f"({file_size / (1024*1024):.1f}MB)."
            ),
        )

    audio_data = audio_path.read_bytes()
    return await _call_mistral_transcribe(
        session_id, audio_data, suffix=audio_path.suffix
    )


async def transcribe_from_url(
    session_id: str,
    audio_url: str,
) -> TranscriptionResult:
    """Fetch audio from a URL and transcribe. Never raises."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(audio_url)
            response.raise_for_status()

        suffix = Path(audio_url).suffix or ".wav"
        return await _call_mistral_transcribe(
            session_id, response.content, suffix=suffix
        )
    except Exception as e:
        logger.error(
            "URL audio fetch failed for %s (%s): %s", session_id, audio_url, e
        )
        return TranscriptionResult(
            session_id=session_id,
            success=False,
            source="voxtral_mini",
            error_message=f"Failed to fetch audio from URL: {e}.",
        )


def get_manual_fallback_result(
    session_id: str,
    typed_text: str,
) -> TranscriptionResult:
    """Wrap typed text as a TranscriptionResult. No API call."""
    return TranscriptionResult(
        session_id=session_id,
        transcript=typed_text,
        success=True,
        source="manual_fallback",
    )
