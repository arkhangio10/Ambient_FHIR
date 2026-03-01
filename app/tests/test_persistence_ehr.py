"""Unit tests for storage and EHR export services.

Covers:
1.  Save and load session round-trip
2.  Load non-existent session returns None
3.  Save overwrites existing session
4.  clear_all removes all sessions
5.  list_sessions returns session summaries
6.  EHR export without token returns skipped
7.  EHR export builds correct FHIR bundle shape
8.  EHR export handles medication-only payload
9.  Orchestrator persist+load round-trip
10. EHR export non-fatal on failure
"""

import pytest

from app.schemas.entities import ExtractedEntities
from app.schemas.fhir import FHIRPayload
from app.schemas.input import ClinicalInput, InputMode
from app.schemas.state import SessionState, SessionStatus
from app.services import storage_service, ehr_service
from app.services.orchestrator import ClinicalOrchestrator


@pytest.fixture(autouse=True)
def _clear_storage():
    storage_service.clear_all()


@pytest.fixture
def orch():
    return ClinicalOrchestrator()


def _make_state(sid: str = "store-001", status=SessionStatus.RECEIVED) -> SessionState:
    return SessionState(session_id=sid, status=status)


def _make_fhir(sid: str = "ehr-001") -> FHIRPayload:
    return FHIRPayload(
        session_id=sid,
        patient_reference="Patient/demo-001",
        encounter_summary="Test encounter",
        condition={
            "resourceType": "Condition",
            "code": {"text": "MDD"},
            "clinicalStatus": {"text": "active"},
        },
        medication_request={
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"text": "Sertraline"},
            "dosageInstruction": [{"text": "50mg Daily"}],
        },
    )


# ── Test 1: Save and load round-trip ───────────────────────────────

@pytest.mark.asyncio
async def test_save_load_roundtrip():
    """Saved session should be loadable with matching data."""
    state = _make_state("store-001")
    await storage_service.save_session(state)

    loaded = await storage_service.load_session("store-001")
    assert loaded is not None
    assert loaded.session_id == "store-001"
    assert loaded.status == SessionStatus.RECEIVED


# ── Test 2: Load non-existent returns None ─────────────────────────

@pytest.mark.asyncio
async def test_load_nonexistent():
    """Loading a non-existent session should return None."""
    result = await storage_service.load_session("does-not-exist")
    assert result is None


# ── Test 3: Save overwrites existing ───────────────────────────────

@pytest.mark.asyncio
async def test_save_overwrites():
    """Saving same session_id again should overwrite."""
    state = _make_state("store-003")
    await storage_service.save_session(state)

    state.status = SessionStatus.TRANSCRIBED
    await storage_service.save_session(state)

    loaded = await storage_service.load_session("store-003")
    assert loaded.status == SessionStatus.TRANSCRIBED


# ── Test 4: clear_all removes all ──────────────────────────────────

@pytest.mark.asyncio
async def test_clear_all():
    """clear_all should remove all sessions."""
    await storage_service.save_session(_make_state("a"))
    await storage_service.save_session(_make_state("b"))
    storage_service.clear_all()

    assert await storage_service.load_session("a") is None
    assert await storage_service.load_session("b") is None


# ── Test 5: list_sessions returns summaries ────────────────────────

@pytest.mark.asyncio
async def test_list_sessions():
    """list_sessions should return summary dicts."""
    await storage_service.save_session(_make_state("list-1"))
    await storage_service.save_session(
        _make_state("list-2", SessionStatus.SERIALIZED)
    )

    sessions = await storage_service.list_sessions()
    assert len(sessions) >= 2

    ids = [s["session_id"] for s in sessions]
    assert "list-1" in ids
    assert "list-2" in ids


# ── Test 6: EHR export without token ──────────────────────────────

@pytest.mark.asyncio
async def test_ehr_no_token(monkeypatch):
    """No Epic token → (False, skipped)."""
    from app.config import Settings

    monkeypatch.setattr(
        "app.services.ehr_service.get_settings",
        lambda: Settings(epic_fhir_token=""),
    )

    success, response = await ehr_service.post_fhir_payload(_make_fhir())
    assert success is False
    assert response["status"] == "skipped"


# ── Test 7: FHIR bundle shape ─────────────────────────────────────

def test_fhir_bundle_shape():
    """_build_fhir_bundle should produce a Transaction Bundle."""
    payload = _make_fhir()
    bundle = ehr_service._build_fhir_bundle(payload)

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "transaction"
    assert len(bundle["entry"]) == 2  # Condition + MedicationRequest

    # Check entry structure
    for entry in bundle["entry"]:
        assert "resource" in entry
        assert "request" in entry
        assert entry["request"]["method"] == "POST"


# ── Test 8: FHIR bundle with supporting info ──────────────────────

def test_fhir_bundle_with_supporting():
    """Supporting info resources should appear as bundle entries."""
    payload = FHIRPayload(
        session_id="ehr-008",
        condition={"resourceType": "Condition", "code": {"text": "Anxiety"}},
        supporting_info=[
            {"resourceType": "Procedure", "code": {"text": "CBT"}},
            {"resourceType": "DocumentReference", "description": "Rationale"},
        ],
    )
    bundle = ehr_service._build_fhir_bundle(payload)

    # Condition + 2 supporting = 3 entries
    assert len(bundle["entry"]) == 3
    types = [e["request"]["url"] for e in bundle["entry"]]
    assert "Condition" in types
    assert "Procedure" in types
    assert "DocumentReference" in types


# ── Test 9: Orchestrator persist + load round-trip ──────────────────

@pytest.mark.asyncio
async def test_orchestrator_persist_load(orch):
    """Orchestrator should persist and load sessions correctly."""
    input_data = ClinicalInput(
        session_id="orch-009",
        input_mode=InputMode.typed_text,
        transcript="Test transcript.",
    )
    state = await orch.create_session(input_data)
    assert state.session_id == "orch-009"

    loaded = await orch.get_session("orch-009")
    assert loaded is not None
    assert loaded.session_id == "orch-009"
    assert loaded.status == SessionStatus.RECEIVED


# ── Test 10: EHR export non-fatal on failure ───────────────────────

@pytest.mark.asyncio
async def test_ehr_export_nonfatal(orch, monkeypatch):
    """EHR export failure should not crash orchestrator."""
    from app.services import reasoning_service

    monkeypatch.setattr(
        reasoning_service,
        "extract_entities",
        reasoning_service.extract_entities_complete,
    )

    input_data = ClinicalInput(
        session_id="orch-010",
        input_mode=InputMode.typed_text,
        transcript="Patient with depression.",
    )
    await orch.create_session(input_data)
    state = await orch.handle_typed_input("orch-010", input_data.transcript)

    # Should reach SERIALIZED without crashing (no Epic token = skip export)
    assert state.status == SessionStatus.SERIALIZED
