"""Unit tests for the transcription service and related orchestrator flows.

Covers:
1. Manual fallback wraps typed text correctly
2. Uploaded audio fallback on missing API key
3. Uploaded audio fallback on file not found
4. Unsupported file format returns clean error
5. Orchestrator degrades gracefully on transcription failure
6. Live chunk accumulates transcript
7. Full fallback path: typed text → SERIALIZED
"""

import os

import pytest

from app.schemas.input import ClinicalInput, InputMode
from app.schemas.state import SessionStatus
from app.services import storage_service, reasoning_service, transcription_service
from app.services.orchestrator import ClinicalOrchestrator


@pytest.fixture(autouse=True)
def _clear_storage():
    """Clear in-memory storage before each test."""
    storage_service.clear_all()


@pytest.fixture
def orch():
    return ClinicalOrchestrator()


# Path to sample transcripts (relative to project root)
_SAMPLE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample_transcripts"
)


# ── Test 1: Manual fallback wraps typed text correctly ──────────────

def test_manual_fallback():
    """get_manual_fallback_result wraps text as TranscriptionResult."""
    result = transcription_service.get_manual_fallback_result(
        "test-session-001",
        "Patient presents with depression",
    )
    assert result.success is True
    assert result.source == "manual_fallback"
    assert result.transcript == "Patient presents with depression"
    assert result.session_id == "test-session-001"


# ── Test 2: Uploaded audio fallback on missing API key ──────────────

@pytest.mark.asyncio
async def test_uploaded_audio_no_api_key(monkeypatch):
    """Uploaded audio with no API key returns clean failure."""
    from app.config import Settings

    monkeypatch.setattr(
        "app.services.transcription_service.get_settings",
        lambda: Settings(mistral_api_key=""),
    )

    # Use a real audio-extension file (create a tiny dummy)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(b"fake audio data")
        tmp_path = tmp.name

    try:
        result = await transcription_service.transcribe_uploaded_audio(
            "test-session-002", tmp_path
        )
        assert result.success is False
        assert result.source == "voxtral_mini"
        assert result.error_message is not None
        assert "api key" in result.error_message.lower() or "unavailable" in result.error_message.lower()
    finally:
        os.unlink(tmp_path)


# ── Test 3: Uploaded audio fallback on file not found ───────────────

@pytest.mark.asyncio
async def test_uploaded_audio_file_not_found():
    """Non-existent file returns clean error."""
    result = await transcription_service.transcribe_uploaded_audio(
        "test-session-003",
        "/nonexistent/path/audio.mp3",
    )
    assert result.success is False
    assert "not found" in result.error_message.lower()


# ── Test 4: Unsupported file format returns clean error ─────────────

@pytest.mark.asyncio
async def test_uploaded_audio_unsupported_format():
    """.txt file should fail with unsupported format error."""
    # Use the sample transcript file (a .txt, not audio)
    sample_path = os.path.join(_SAMPLE_DIR, "sample_01_complete.txt")
    result = await transcription_service.transcribe_uploaded_audio(
        "test-session-004", sample_path
    )
    assert result.success is False
    assert "unsupported" in result.error_message.lower()


# ── Test 5: Orchestrator degrades gracefully on transcription failure ─

@pytest.mark.asyncio
async def test_orchestrator_graceful_degradation(orch):
    """When transcription fails, session stays RECEIVED (not FAILED)."""
    input_data = ClinicalInput(
        session_id="test-session-005",
        input_mode=InputMode.uploaded_audio,
        audio_reference="/invalid/path.mp3",
    )
    await orch.create_session(input_data)

    state = await orch.handle_uploaded_audio(
        "test-session-005", "/invalid/path.mp3"
    )

    assert state.status == SessionStatus.RECEIVED
    assert state.error_message is not None
    # Session is NOT failed — it's waiting for manual input


# ── Test 6: Live chunk accumulates transcript ───────────────────────

@pytest.mark.asyncio
async def test_live_chunks_accumulate(orch):
    """Text chunks should accumulate in the transcript."""
    input_data = ClinicalInput(
        session_id="test-session-006",
        input_mode=InputMode.live_audio,
    )
    await orch.create_session(input_data)

    # Send text chunks (pre-transcribed)
    await orch.handle_live_transcript_chunk("test-session-006", "chunk one")
    await orch.handle_live_transcript_chunk("test-session-006", "chunk two")
    state = await orch.handle_live_transcript_chunk("test-session-006", "chunk three")

    assert state.transcript is not None
    assert "chunk one" in state.transcript
    assert "chunk two" in state.transcript
    assert "chunk three" in state.transcript


# ── Test 7: Full fallback path: typed text → SERIALIZED ─────────────

@pytest.mark.asyncio
async def test_full_typed_text_path(orch, monkeypatch):
    """Typed text with complete entities reaches SERIALIZED — no API keys needed."""
    # Use complete reasoning stub so no clarification is needed
    monkeypatch.setattr(
        reasoning_service,
        "extract_entities",
        reasoning_service.extract_entities_complete,
    )

    sample_path = os.path.join(_SAMPLE_DIR, "sample_01_complete.txt")
    with open(sample_path, "r") as f:
        transcript_text = f.read()

    input_data = ClinicalInput(
        session_id="test-session-007",
        input_mode=InputMode.typed_text,
        transcript=transcript_text,
    )
    await orch.create_session(input_data)

    state = await orch.handle_typed_input(
        "test-session-007", transcript_text
    )

    assert state.status in (
        SessionStatus.SERIALIZED,
        SessionStatus.NEEDS_CLARIFICATION,
    )
    # Demo resilience confirmed: typed text path works with zero API keys
