"""Unit tests for the reasoning service.

Covers:
1. Empty transcript returns clean failure
2. Missing API key returns clean failure
3. Quality gate blocks low confidence
4. Quality gate blocks missing diagnosis
5. Direct mapping fallback on clarification
6. Full pipeline with sample transcript (skips if no API key)
7. Orchestrator extract_entities() advances state correctly
"""

import os

import pytest

from app.schemas.entities import ExtractedEntities, ReasoningResult
from app.schemas.input import ClinicalInput, InputMode
from app.schemas.state import SessionState, SessionStatus
from app.services import reasoning_service, storage_service
from app.services.orchestrator import ClinicalOrchestrator


@pytest.fixture(autouse=True)
def _clear_storage():
    """Clear in-memory storage before each test."""
    storage_service.clear_all()


@pytest.fixture
def orch():
    return ClinicalOrchestrator()


_SAMPLE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample_transcripts"
)


# ── Test 1: Empty transcript returns clean failure ──────────────────

@pytest.mark.asyncio
async def test_empty_transcript():
    """Empty transcript should return success=False cleanly."""
    result = await reasoning_service.extract_entities("test-001", "")
    assert result.success is False
    assert result.entities is None
    assert result.error_message is not None
    assert "empty" in result.error_message.lower()


# ── Test 2: Missing API key returns clean failure ───────────────────

@pytest.mark.asyncio
async def test_missing_api_key(monkeypatch):
    """No API key should return clean failure without exception."""
    from app.config import Settings

    monkeypatch.setattr(
        "app.services.reasoning_service.get_settings",
        lambda: Settings(mistral_api_key=""),
    )

    result = await reasoning_service.extract_entities(
        "test-002",
        "Patient presents with depression F33.1",
    )
    assert result.success is False
    assert result.error_message is not None


# ── Test 3: Quality gate blocks low confidence ──────────────────────

def test_quality_gate_low_confidence():
    """Confidence below threshold should be rejected."""
    low_conf_result = ReasoningResult(
        session_id="test-003",
        success=True,
        entities=ExtractedEntities(
            session_id="test-003",
            confidence=0.20,  # Below "low" threshold of 0.40
        ),
    )
    is_valid, error = reasoning_service.validate_extraction_quality(low_conf_result)
    assert is_valid is False
    assert error is not None
    assert "confidence" in error.lower()


# ── Test 4: Quality gate blocks missing diagnosis ───────────────────

def test_quality_gate_no_diagnosis():
    """Missing both diagnosis and chief_complaint should be rejected."""
    no_diag_result = ReasoningResult(
        session_id="test-004",
        success=True,
        entities=ExtractedEntities(
            session_id="test-004",
            confidence=0.90,
            diagnosis=None,
            chief_complaint=None,
        ),
    )
    is_valid, error = reasoning_service.validate_extraction_quality(no_diag_result)
    assert is_valid is False
    assert error is not None
    assert "diagnosis" in error.lower() or "chief complaint" in error.lower()


# ── Test 5: Direct mapping fallback on clarification ────────────────

@pytest.mark.asyncio
async def test_direct_mapping_fallback(monkeypatch):
    """When Mistral API fails, direct field mapping should work."""
    from app.config import Settings

    # Force no API key — triggers direct mapping fallback
    monkeypatch.setattr(
        "app.services.reasoning_service.get_settings",
        lambda: Settings(mistral_api_key=""),
    )

    existing = ExtractedEntities(
        session_id="test-005",
        diagnosis="Major depressive disorder",
        chief_complaint="Depression",
        medication="Sertraline",
        dosage=None,
        missing_fields=["dosage"],
        confidence=0.80,
    )

    result = await reasoning_service.extract_with_clarification(
        session_id="test-005",
        transcript="Patient on Sertraline...",
        existing_entities=existing,
        missing_field="dosage",
        clarification_answer="50mg",
    )

    assert result.success is True
    assert result.entities.dosage == "50mg"
    assert "dosage" not in result.entities.missing_fields
    assert result.model_used == "direct_mapping_fallback"


# ── Test 6: Full pipeline with sample transcript ────────────────────

@pytest.mark.asyncio
async def test_full_pipeline_with_sample():
    """If MISTRAL_API_KEY is set, full extraction should work.

    Skips gracefully if no API key — this is NOT a test failure.
    """
    from app.config import get_settings

    settings = get_settings()

    sample_path = os.path.join(_SAMPLE_DIR, "sample_01_complete.txt")
    with open(sample_path, "r") as f:
        transcript = f.read()

    result = await reasoning_service.extract_entities("test-006", transcript)

    if settings.mistral_api_key:
        assert result.success is True
        assert result.entities is not None
        assert result.entities.diagnosis is not None
        assert result.entities.confidence > 0.5
    else:
        # Graceful — no API key is fine for tests
        assert result.success is False
        assert result.error_message is not None


# ── Test 7: Orchestrator advances state after extraction ────────────

@pytest.mark.asyncio
async def test_orchestrator_extract_entities_advance(orch, monkeypatch):
    """Orchestrator.extract_entities() should advance state correctly."""
    monkeypatch.setattr(
        reasoning_service,
        "extract_entities",
        reasoning_service.extract_entities_complete,
    )

    input_data = ClinicalInput(
        session_id="test-007",
        input_mode=InputMode.typed_text,
        transcript="Patient presents with F33.1, Sertraline 50mg daily.",
    )
    await orch.create_session(input_data)

    # Manually bring state to TRANSCRIBED
    state = await storage_service.load_session("test-007")
    state.transcript = input_data.transcript
    state.advance_status(SessionStatus.TRANSCRIBED)
    await storage_service.save_session(state)

    state = await orch.extract_entities("test-007")

    assert state.status in (
        SessionStatus.VALIDATED,
        SessionStatus.NEEDS_CLARIFICATION,
        SessionStatus.SERIALIZED,
    )
    assert state.extracted_entities is not None
