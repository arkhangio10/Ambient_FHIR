"""Unit tests for all Pydantic v2 schemas.

Covers:
1. ClinicalInput typed_text validation
2. ClinicalInput clarification_response validation
3. ExtractedEntities defaults
4. PriorAuthPacket submission guard
5. SessionState advance_status happy path
6. SessionState advance_status invalid transition
7. FHIRPayload defaults
"""

import pytest
from pydantic import ValidationError

from app.schemas.input import ClinicalInput, InputMode
from app.schemas.entities import ExtractedEntities
from app.schemas.prior_auth import PriorAuthPacket
from app.schemas.fhir import FHIRPayload
from app.schemas.state import SessionState, SessionStatus


# ── Test 1: ClinicalInput typed_text requires transcript ─────────────

def test_clinical_input_typed_text_missing_transcript():
    """typed_text mode with no transcript must raise ValidationError."""
    with pytest.raises(ValidationError, match="transcript"):
        ClinicalInput(
            session_id="test-001",
            input_mode=InputMode.typed_text,
            transcript=None,
        )


# ── Test 2: ClinicalInput clarification_response needs answer ───────

def test_clinical_input_clarification_missing_answer():
    """clarification_response mode with no answer must raise ValidationError."""
    with pytest.raises(ValidationError, match="clarification_answer"):
        ClinicalInput(
            session_id="test-002",
            input_mode=InputMode.clarification_response,
            clarification_answer=None,
        )


# ── Test 3: ExtractedEntities defaults ──────────────────────────────

def test_extracted_entities_defaults():
    """Minimal ExtractedEntities should have correct defaults."""
    entities = ExtractedEntities(session_id="test-003")
    assert entities.missing_fields == []
    assert entities.modifier_flags == []
    assert entities.confidence == 0.0


# ── Test 4: PriorAuthPacket submission guard ─────────────────────────

def test_prior_auth_packet_submission_guard():
    """ready_for_submission=True with missing_fields_resolved=False must fail."""
    with pytest.raises(ValidationError, match="ready_for_submission"):
        PriorAuthPacket(
            session_id="test-004",
            summary="Test summary",
            clinical_justification="Test justification",
            missing_fields_resolved=False,
            ready_for_submission=True,
        )


# ── Test 5: SessionState advance_status happy path ──────────────────

def test_session_state_advance_status_happy():
    """RECEIVED → TRANSCRIBED should succeed and update updated_at."""
    state = SessionState(session_id="test-005")
    assert state.status == SessionStatus.RECEIVED

    old_updated = state.updated_at
    state.advance_status(SessionStatus.TRANSCRIBED)

    assert state.status == SessionStatus.TRANSCRIBED
    assert state.updated_at >= old_updated


# ── Test 6: SessionState advance_status invalid transition ──────────

def test_session_state_advance_status_invalid():
    """VALIDATED → RECEIVED (backward) must raise ValueError."""
    state = SessionState(session_id="test-006", status=SessionStatus.VALIDATED)

    with pytest.raises(ValueError, match="Invalid transition"):
        state.advance_status(SessionStatus.RECEIVED)


# ── Test 7: FHIRPayload defaults ────────────────────────────────────

def test_fhir_payload_defaults():
    """Minimal FHIRPayload should default resource_type and supporting_info."""
    payload = FHIRPayload(session_id="test-007")
    assert payload.resource_type == "Bundle"
    assert payload.supporting_info == []
