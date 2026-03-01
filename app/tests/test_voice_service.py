"""Unit tests for the voice clarification service.

Covers:
1. Text fallback when API key is missing
2. Prompt truncation at 150 chars
3. Empty prompt returns clean failure
4. Audio cache prevents duplicate calls
5. build_clarification_prompt field mapping
6. build_clarification_prompt uses first field only
7. clear_session_audio removes cache
8. Orchestrator clarification loop text-only path
"""

import pytest

from app.schemas.entities import ExtractedEntities, VoiceResult
from app.schemas.input import ClinicalInput, InputMode
from app.schemas.state import SessionStatus
from app.services import voice_service, storage_service, reasoning_service
from app.services.orchestrator import ClinicalOrchestrator
from app.validators.entity_checks import build_clarification_prompt


@pytest.fixture(autouse=True)
def _clear_state():
    """Clear in-memory storage and voice cache before each test."""
    storage_service.clear_all()
    voice_service._audio_cache.clear()


@pytest.fixture
def orch():
    return ClinicalOrchestrator()


# ── Stub helpers ────────────────────────────────────────────────────

async def _stub_extract_missing_dosage(session_id, transcript):
    """Stub returning entities with dosage missing."""
    from app.schemas.entities import ReasoningResult
    return ReasoningResult(
        session_id=session_id,
        entities=ExtractedEntities(
            session_id=session_id,
            chief_complaint="Depression",
            diagnosis="Major depressive disorder",
            diagnosis_code="F33.1",
            procedure_or_intervention="Individual psychotherapy, 45 min",
            procedure_code="90834",
            medication="Sertraline",
            dosage=None,
            frequency="Daily",
            clinical_rationale="Patient presents with persistent low mood.",
            modifier_flags=["GT"],
            confidence=0.87,
        ),
        success=True,
        model_used="stub_missing_dosage",
    )


# ── Test 1: Text fallback when API key is missing ───────────────────

@pytest.mark.asyncio
async def test_text_fallback_no_api_key(monkeypatch):
    """No ElevenLabs API key → success=False, text_fallback populated."""
    from app.config import Settings

    monkeypatch.setattr(
        "app.services.voice_service.get_settings",
        lambda: Settings(elevenlabs_api_key=""),
    )

    result = await voice_service.synthesize_clarification(
        "test-001",
        "What is the dosage for Sertraline?",
    )
    assert result.success is False
    assert result.text_fallback == "What is the dosage for Sertraline?"
    assert result.error_message is not None


# ── Test 2: Prompt truncation at 150 chars ──────────────────────────

@pytest.mark.asyncio
async def test_prompt_truncation(monkeypatch):
    """Prompt > 150 chars is truncated."""
    from app.config import Settings

    monkeypatch.setattr(
        "app.services.voice_service.get_settings",
        lambda: Settings(elevenlabs_api_key=""),
    )

    long_prompt = "A" * 200
    result = await voice_service.synthesize_clarification(
        "test-002", long_prompt,
    )
    assert len(result.text_fallback) <= 150


# ── Test 3: Empty prompt returns clean failure ──────────────────────

@pytest.mark.asyncio
async def test_empty_prompt():
    """Empty prompt → success=False with error message."""
    result = await voice_service.synthesize_clarification("test-003", "")
    assert result.success is False
    assert result.error_message is not None
    assert "empty" in result.error_message.lower()


# ── Test 4: Audio cache prevents duplicate calls ────────────────────

@pytest.mark.asyncio
async def test_audio_cache(monkeypatch):
    """Second call returns cached result, not a new API call."""
    from app.config import Settings

    monkeypatch.setattr(
        "app.services.voice_service.get_settings",
        lambda: Settings(elevenlabs_api_key=""),
    )

    # Manually populate cache with a "successful" result
    cached_result = VoiceResult(
        session_id="test-004",
        success=True,
        text_fallback="What is the dosage?",
        audio_url="/audio/test-004",
    )
    voice_service._audio_cache["test-004"] = cached_result

    # Second call should return cached
    result = await voice_service.synthesize_clarification(
        "test-004", "What is the dosage?",
    )
    assert result.success is True
    assert result.audio_url == "/audio/test-004"
    assert result is cached_result


# ── Test 5: build_clarification_prompt field mapping ────────────────

def test_prompt_field_mapping():
    """Dosage field should produce context-aware prompt."""
    entities = ExtractedEntities(
        session_id="test-005",
        medication="Sertraline",
        confidence=0.8,
    )
    prompt = build_clarification_prompt(
        missing_fields=["dosage"],
        entities=entities,
    )
    assert "dosage" in prompt.lower() or "sertraline" in prompt.lower()
    assert len(prompt) <= 150


# ── Test 6: build_clarification_prompt uses first field only ────────

def test_prompt_first_field_only():
    """Only the first missing field is addressed."""
    entities = ExtractedEntities(session_id="test-006", confidence=0.8)
    prompt = build_clarification_prompt(
        missing_fields=["dosage", "frequency", "clinical_rationale"],
        entities=entities,
    )
    assert prompt != ""
    assert len(prompt) <= 150
    # Only dosage addressed — frequency and rationale NOT mentioned
    assert "frequency" not in prompt.lower()


# ── Test 7: clear_session_audio removes cache ──────────────────────

@pytest.mark.asyncio
async def test_clear_session_audio():
    """Cache entry should be removed after clear."""
    voice_service._audio_cache["test-007"] = VoiceResult(
        session_id="test-007",
        success=True,
        text_fallback="test",
    )
    assert "test-007" in voice_service._audio_cache

    await voice_service.clear_session_audio("test-007")
    assert "test-007" not in voice_service._audio_cache


# ── Test 8: Orchestrator clarification loop text-only ───────────────

@pytest.mark.asyncio
async def test_orchestrator_clarification_text_only(orch, monkeypatch):
    """Text-only clarification loop works without ElevenLabs."""
    monkeypatch.setattr(
        reasoning_service, "extract_entities", _stub_extract_missing_dosage,
    )

    input_data = ClinicalInput(
        session_id="test-008",
        input_mode=InputMode.typed_text,
        transcript="Patient on Sertraline for depression, daily.",
    )
    await orch.create_session(input_data)
    state = await orch.handle_typed_input("test-008", input_data.transcript)

    # Should be NEEDS_CLARIFICATION with a text prompt
    assert state.status == SessionStatus.NEEDS_CLARIFICATION
    assert state.clarification_prompt is not None
    assert "dosage" in state.clarification_prompt.lower() or \
           "sertraline" in state.clarification_prompt.lower()
    # Audio URL may be None — that is acceptable (text fallback)
    # Text prompt must always be present
