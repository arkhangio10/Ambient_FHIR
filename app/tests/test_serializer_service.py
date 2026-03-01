"""Unit tests for the serializer service.

Covers:
1. PriorAuthPacket with complete entities
2. PriorAuthPacket summary includes diagnosis
3. PriorAuthPacket without medication
4. FHIRPayload with complete entities
5. FHIRPayload Condition resource shape
6. FHIRPayload MedicationRequest when medication present
7. FHIRPayload without medication — no MedicationRequest
8. FHIRPayload supporting_info includes Procedure
9. Documentation checklist reflects entity presence
10. Orchestrator full pipeline reaches SERIALIZED with real serializer
"""

import pytest

from app.schemas.entities import ExtractedEntities
from app.schemas.input import ClinicalInput, InputMode
from app.schemas.state import SessionStatus
from app.services import serializer_service, storage_service, reasoning_service
from app.services.orchestrator import ClinicalOrchestrator


@pytest.fixture(autouse=True)
def _clear_storage():
    storage_service.clear_all()


@pytest.fixture
def orch():
    return ClinicalOrchestrator()


def _complete_entities(session_id: str = "ser-001") -> ExtractedEntities:
    """Build complete entities for testing."""
    return ExtractedEntities(
        session_id=session_id,
        chief_complaint="Depression and anxiety",
        diagnosis="Major depressive disorder, recurrent",
        diagnosis_code="F33.1",
        procedure_or_intervention="Individual psychotherapy, 45 min",
        procedure_code="90834",
        medication="Sertraline",
        dosage="50mg",
        frequency="Daily",
        clinical_rationale=(
            "Patient presents with persistent low mood, anhedonia, "
            "and difficulty concentrating for over 6 months."
        ),
        modifier_flags=["GT"],
        confidence=0.92,
    )


def _no_med_entities(session_id: str = "ser-002") -> ExtractedEntities:
    """Build entities without medication."""
    return ExtractedEntities(
        session_id=session_id,
        chief_complaint="Anxiety",
        diagnosis="Generalized anxiety disorder",
        diagnosis_code="F41.1",
        procedure_or_intervention="CBT 60 min",
        procedure_code="90837",
        clinical_rationale=(
            "Patient reports significant anxiety interfering with daily functioning."
        ),
        confidence=0.88,
    )


# ── Test 1: PriorAuthPacket with complete entities ─────────────────

@pytest.mark.asyncio
async def test_prior_auth_complete():
    """Complete entities should produce a ready-for-submission packet."""
    entities = _complete_entities()
    packet = await serializer_service.build_prior_auth(entities)

    assert packet.session_id == "ser-001"
    assert packet.ready_for_submission is True
    assert packet.missing_fields_resolved is True
    assert len(packet.summary) > 0
    assert len(packet.clinical_justification) > 0
    assert len(packet.documentation_checklist) > 0


# ── Test 2: PriorAuthPacket summary includes diagnosis ─────────────

@pytest.mark.asyncio
async def test_prior_auth_summary_content():
    """Summary should mention procedure and diagnosis."""
    entities = _complete_entities()
    packet = await serializer_service.build_prior_auth(entities)

    assert "psychotherapy" in packet.summary.lower()
    assert "major depressive" in packet.summary.lower() or "F33.1" in packet.summary


# ── Test 3: PriorAuthPacket without medication ─────────────────────

@pytest.mark.asyncio
async def test_prior_auth_no_medication():
    """Entities without medication should still produce valid packet."""
    entities = _no_med_entities()
    packet = await serializer_service.build_prior_auth(entities)

    assert packet.ready_for_submission is True
    assert "Medication" not in packet.clinical_justification or "medication" not in packet.clinical_justification.lower()


# ── Test 4: FHIRPayload with complete entities ─────────────────────

@pytest.mark.asyncio
async def test_fhir_complete():
    """Complete entities should produce FHIR bundle with all resources."""
    entities = _complete_entities()
    payload = await serializer_service.build_fhir_payload(entities)

    assert payload.session_id == "ser-001"
    assert payload.resource_type == "Bundle"
    assert payload.patient_reference == "Patient/demo-001"
    assert payload.condition is not None
    assert payload.medication_request is not None
    assert len(payload.supporting_info) >= 1


# ── Test 5: FHIRPayload Condition resource shape ──────────────────

@pytest.mark.asyncio
async def test_fhir_condition_shape():
    """Condition resource should have proper FHIR shape."""
    entities = _complete_entities()
    payload = await serializer_service.build_fhir_payload(entities)

    assert payload.condition["resourceType"] == "Condition"
    assert "code" in payload.condition
    assert "text" in payload.condition["code"]
    assert payload.condition["code"]["text"] == "Major depressive disorder, recurrent"
    # Should have coding array with ICD-10
    assert "coding" in payload.condition["code"]
    assert payload.condition["code"]["coding"][0]["code"] == "F33.1"


# ── Test 6: FHIRPayload MedicationRequest ──────────────────────────

@pytest.mark.asyncio
async def test_fhir_medication_request():
    """MedicationRequest should include medication, dosage, and frequency."""
    entities = _complete_entities()
    payload = await serializer_service.build_fhir_payload(entities)

    med_req = payload.medication_request
    assert med_req["resourceType"] == "MedicationRequest"
    assert med_req["medicationCodeableConcept"]["text"] == "Sertraline"
    assert "50mg" in med_req["dosageInstruction"][0]["text"]
    assert "Daily" in med_req["dosageInstruction"][0]["text"]


# ── Test 7: FHIRPayload without medication ─────────────────────────

@pytest.mark.asyncio
async def test_fhir_no_medication():
    """No medication → medication_request should be None."""
    entities = _no_med_entities()
    payload = await serializer_service.build_fhir_payload(entities)

    assert payload.medication_request is None
    assert payload.condition is not None  # Still has diagnosis


# ── Test 8: FHIRPayload supporting_info includes Procedure ────────

@pytest.mark.asyncio
async def test_fhir_supporting_info():
    """Supporting info should include Procedure resource."""
    entities = _complete_entities()
    payload = await serializer_service.build_fhir_payload(entities)

    proc = next(
        (r for r in payload.supporting_info if r.get("resourceType") == "Procedure"),
        None,
    )
    assert proc is not None
    assert proc["code"]["text"] == "Individual psychotherapy, 45 min"
    assert "coding" in proc["code"]
    assert proc["code"]["coding"][0]["code"] == "90834"


# ── Test 9: Documentation checklist reflects entities ──────────────

@pytest.mark.asyncio
async def test_documentation_checklist():
    """Checklist should include items based on present entities."""
    entities = _complete_entities()
    packet = await serializer_service.build_prior_auth(entities)

    checklist_str = " ".join(packet.documentation_checklist).lower()
    assert "diagnosis" in checklist_str
    assert "f33.1" in checklist_str
    assert "medication" in checklist_str
    assert "90834" in checklist_str
    assert "gt" in checklist_str


# ── Test 10: Orchestrator full pipeline SERIALIZED ─────────────────

@pytest.mark.asyncio
async def test_orchestrator_serialized(orch, monkeypatch):
    """Complete entities through orchestrator reach SERIALIZED."""
    monkeypatch.setattr(
        reasoning_service,
        "extract_entities",
        reasoning_service.extract_entities_complete,
    )

    input_data = ClinicalInput(
        session_id="ser-010",
        input_mode=InputMode.typed_text,
        transcript="Patient with MDD F33.1, Sertraline 50mg daily, 45min therapy.",
    )
    await orch.create_session(input_data)
    state = await orch.handle_typed_input("ser-010", input_data.transcript)

    assert state.status == SessionStatus.SERIALIZED
    assert state.final_prior_auth is not None
    assert state.final_fhir is not None

    # Check that real serializer output is used (not empty stubs)
    pa = state.final_prior_auth
    assert pa.ready_for_submission is True
    assert len(pa.documentation_checklist) > 2

    fhir = state.final_fhir
    assert fhir.condition is not None
    assert fhir.medication_request is not None
