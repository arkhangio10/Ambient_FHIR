"""Unit tests for the ClinicalOrchestrator state machine.

Covers:
1. Full happy path (typed text, all fields present)
2. Clarification loop (missing dosage)
3. Clarification resolution
4. Uploaded audio path
5. Live transcript chunking + finalize
6. Epic export fail-safe
"""

import pytest

from app.schemas.input import ClinicalInput, InputMode
from app.schemas.entities import ExtractedEntities, ReasoningResult
from app.schemas.transcription import TranscriptionResult
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


# ── Stub helpers ────────────────────────────────────────────────────

async def _stub_extract_missing_dosage(session_id, transcript):
    """Stub returning entities with dosage intentionally missing."""
    return ReasoningResult(
        session_id=session_id,
        entities=ExtractedEntities(
            session_id=session_id,
            chief_complaint="Depression and anxiety",
            diagnosis="Major depressive disorder, recurrent",
            diagnosis_code="F33.1",
            procedure_or_intervention="Individual psychotherapy, 45 min",
            procedure_code="90834",
            medication="Sertraline",
            dosage=None,  # intentionally missing
            frequency="Daily",
            clinical_rationale=(
                "Patient presents with persistent low mood."
            ),
            modifier_flags=["GT"],
            confidence=0.87,
        ),
        success=True,
        model_used="stub_missing_dosage",
    )


# ── Test 1: Full happy path (typed text, no missing fields) ─────────

@pytest.mark.asyncio
async def test_happy_path_typed_text(orch, monkeypatch):
    """Typed input with all fields present should reach SERIALIZED."""
    monkeypatch.setattr(
        reasoning_service,
        "extract_entities",
        reasoning_service.extract_entities_complete,
    )

    input_data = ClinicalInput(
        session_id="happy-001",
        input_mode=InputMode.typed_text,
        transcript="Patient presents with depression, prescribed Sertraline 50mg daily.",
    )
    await orch.create_session(input_data)

    state = await orch.handle_typed_input(
        "happy-001", input_data.transcript
    )

    assert state.status == SessionStatus.SERIALIZED
    assert state.final_prior_auth is not None
    assert state.final_fhir is not None
    assert state.extracted_entities is not None
    assert state.extracted_entities.dosage == "50mg"


# ── Test 2: Clarification loop (missing dosage) ────────────────────

@pytest.mark.asyncio
async def test_clarification_triggered(orch, monkeypatch):
    """Missing dosage should trigger NEEDS_CLARIFICATION."""
    monkeypatch.setattr(
        reasoning_service,
        "extract_entities",
        _stub_extract_missing_dosage,
    )

    input_data = ClinicalInput(
        session_id="clarify-001",
        input_mode=InputMode.typed_text,
        transcript="Patient has depression, on Sertraline, frequency daily.",
    )
    await orch.create_session(input_data)

    state = await orch.handle_typed_input(
        "clarify-001", input_data.transcript
    )

    assert state.status == SessionStatus.NEEDS_CLARIFICATION
    assert "dosage" in state.missing_fields
    assert state.clarification_prompt is not None
    assert "dosage" in state.clarification_prompt.lower() or "Sertraline" in state.clarification_prompt
    # Audio URL may be None without API key — text fallback is sufficient


# ── Test 3: Clarification resolution ───────────────────────────────

@pytest.mark.asyncio
async def test_clarification_resolution(orch, monkeypatch):
    """After providing dosage, session should advance to SERIALIZED."""
    monkeypatch.setattr(
        reasoning_service,
        "extract_entities",
        _stub_extract_missing_dosage,
    )

    input_data = ClinicalInput(
        session_id="resolve-001",
        input_mode=InputMode.typed_text,
        transcript="Patient has depression, on Sertraline, frequency daily.",
    )
    await orch.create_session(input_data)
    await orch.handle_typed_input("resolve-001", input_data.transcript)

    # Provide the missing dosage
    state = await orch.handle_clarification_response("resolve-001", "50mg")

    assert state.status == SessionStatus.SERIALIZED
    assert state.extracted_entities.dosage == "50mg"
    assert state.final_prior_auth is not None
    assert state.final_fhir is not None


# ── Test 4: Uploaded audio path ────────────────────────────────────

@pytest.mark.asyncio
async def test_uploaded_audio_path(orch, monkeypatch):
    """Uploaded audio should produce a transcript and advance the pipeline."""
    # Mock transcription
    async def _mock_transcribe(session_id, audio_ref):
        return TranscriptionResult(
            session_id=session_id,
            transcript="Patient is a 34-year-old with depression. "
                       "On Sertraline. Individual psychotherapy 45 min.",
            success=True,
            source="voxtral_mini",
        )

    monkeypatch.setattr(
        transcription_service, "transcribe_uploaded_audio", _mock_transcribe,
    )
    # Use missing-dosage stub for reasoning
    monkeypatch.setattr(
        reasoning_service, "extract_entities", _stub_extract_missing_dosage,
    )

    input_data = ClinicalInput(
        session_id="audio-001",
        input_mode=InputMode.uploaded_audio,
        audio_reference="/recordings/session_audio.wav",
    )
    await orch.create_session(input_data)

    state = await orch.handle_uploaded_audio(
        "audio-001", "/recordings/session_audio.wav"
    )

    assert state.transcript is not None
    assert len(state.transcript) > 0
    assert state.status in (
        SessionStatus.NEEDS_CLARIFICATION,
        SessionStatus.SERIALIZED,
    )


# ── Test 5: Live transcript chunking + finalize ───────────────────

@pytest.mark.asyncio
async def test_live_chunks_and_finalize(orch, monkeypatch):
    """Multiple chunks should accumulate; finalize should trigger extraction."""
    monkeypatch.setattr(
        reasoning_service, "extract_entities", _stub_extract_missing_dosage,
    )

    input_data = ClinicalInput(
        session_id="live-001",
        input_mode=InputMode.live_audio,
    )
    await orch.create_session(input_data)

    chunks = [
        "Patient reports feeling depressed",
        "for the past six months",
        "Currently on Sertraline daily",
    ]

    for chunk in chunks:
        state = await orch.handle_live_transcript_chunk("live-001", chunk)

    assert state.transcript is not None
    assert "depressed" in state.transcript
    assert "six months" in state.transcript
    assert "Sertraline" in state.transcript

    state = await orch.finalize_live_transcript("live-001")
    assert state.status not in (SessionStatus.RECEIVED, SessionStatus.TRANSCRIBED)


# ── Test 6: Epic export fail-safe ──────────────────────────────────

@pytest.mark.asyncio
async def test_epic_export_failsafe(orch, monkeypatch):
    """When Epic is not configured, status should stay SERIALIZED (not FAILED)."""
    from app.config import Settings

    monkeypatch.setattr(
        "app.services.orchestrator.get_settings",
        lambda: Settings(epic_fhir_base_url="", epic_fhir_token=""),
    )
    monkeypatch.setattr(
        reasoning_service,
        "extract_entities",
        reasoning_service.extract_entities_complete,
    )

    input_data = ClinicalInput(
        session_id="epic-001",
        input_mode=InputMode.typed_text,
        transcript="Patient presents with depression, prescribed Sertraline 50mg daily.",
    )
    await orch.create_session(input_data)

    state = await orch.handle_typed_input("epic-001", input_data.transcript)

    assert state.status == SessionStatus.SERIALIZED
    assert state.error_message is None
