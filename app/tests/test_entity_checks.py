"""Unit tests for the validation layer (entity_checks).

Covers:
1.  All required fields present → passes
2.  Missing diagnosis → ERROR → clarification
3.  Medication present, dosage missing → ERROR
4.  Medication present, both dosage+frequency missing → two ERRORs
5.  No medication → dosage/frequency NOT required
6.  Low confidence → WARNING not ERROR
7.  Max clarification rounds → force pass
8.  ICD-10 format warning
9.  Recommended field absent → WARNING only
10. Orchestrator full validation path happy case
"""

import pytest

from app.schemas.entities import ExtractedEntities, ValidationSeverity
from app.schemas.input import ClinicalInput, InputMode
from app.schemas.state import SessionState, SessionStatus
from app.services import storage_service, reasoning_service
from app.services.orchestrator import ClinicalOrchestrator
from app.validators.entity_checks import run_all_checks


@pytest.fixture(autouse=True)
def _clear_storage():
    """Clear in-memory storage before each test."""
    storage_service.clear_all()


@pytest.fixture
def orch():
    return ClinicalOrchestrator()


# ── Test 1: All required fields present → passes ───────────────────

def test_all_required_present():
    """Complete entities should pass validation."""
    entities = ExtractedEntities(
        session_id="test-001",
        chief_complaint="Depression and anxiety",
        diagnosis="Major depressive disorder, recurrent",
        procedure_or_intervention="Individual psychotherapy 45 min",
        clinical_rationale=(
            "Patient meets criteria for pharmacological "
            "support and ongoing psychotherapy"
        ),
        confidence=0.90,
    )
    report = run_all_checks(entities)
    assert report.passed is True
    assert report.error_count == 0
    assert report.clarification_needed is False


# ── Test 2: Missing diagnosis → ERROR → clarification ──────────────

def test_missing_diagnosis():
    """Missing diagnosis should produce ERROR and clarification."""
    entities = ExtractedEntities(
        session_id="test-002",
        chief_complaint="Anxiety",
        procedure_or_intervention="CBT 60 min",
        clinical_rationale="Patient reports significant work impairment",
        confidence=0.80,
    )
    report = run_all_checks(entities)
    assert report.passed is False
    assert "diagnosis" in report.missing_required
    assert report.clarification_needed is True
    assert report.first_missing_field == "diagnosis"


# ── Test 3: Medication present, dosage missing → ERROR ─────────────

def test_medication_missing_dosage():
    """Medication without dosage should produce conditional ERROR."""
    entities = ExtractedEntities(
        session_id="test-003",
        chief_complaint="ADHD",
        diagnosis="ADHD combined presentation",
        procedure_or_intervention="Individual therapy",
        clinical_rationale="Functional impairment at school and home",
        medication="Methylphenidate",
        dosage=None,
        frequency="Daily",
        confidence=0.85,
    )
    report = run_all_checks(entities)
    assert report.passed is False
    assert "dosage" in report.missing_required


# ── Test 4: Both dosage and frequency missing → two ERRORs ─────────

def test_medication_missing_both():
    """Both dosage and frequency missing → two errors, first=dosage."""
    entities = ExtractedEntities(
        session_id="test-004",
        chief_complaint="Anxiety",
        diagnosis="GAD",
        procedure_or_intervention="CBT",
        clinical_rationale="Ongoing functional impairment",
        medication="Buspirone",
        dosage=None,
        frequency=None,
        confidence=0.80,
    )
    report = run_all_checks(entities)
    assert "dosage" in report.missing_required
    assert "frequency" in report.missing_required
    assert report.first_missing_field == "dosage"


# ── Test 5: No medication → dosage/frequency NOT required ──────────

def test_no_medication_no_conditional():
    """Without medication, dosage/frequency are not required."""
    entities = ExtractedEntities(
        session_id="test-005",
        chief_complaint="Depression",
        diagnosis="Major depressive disorder",
        procedure_or_intervention="Psychotherapy 45 min",
        clinical_rationale="Patient not currently on medication",
        medication=None,
        dosage=None,
        frequency=None,
        confidence=0.88,
    )
    report = run_all_checks(entities)
    assert "dosage" not in report.missing_required
    assert "frequency" not in report.missing_required
    assert report.passed is True


# ── Test 6: Low confidence → WARNING not ERROR ─────────────────────

def test_low_confidence_warning():
    """Low confidence produces warning, not error — still passes."""
    entities = ExtractedEntities(
        session_id="test-006",
        chief_complaint="Depression",
        diagnosis="MDD",
        procedure_or_intervention="Therapy",
        clinical_rationale="Patient presents with symptoms",
        confidence=0.45,
    )
    report = run_all_checks(entities)
    assert report.passed is True
    assert report.warning_count > 0
    conf_warning = next(
        (i for i in report.issues if i.field == "confidence"), None
    )
    assert conf_warning is not None
    assert conf_warning.severity == ValidationSeverity.WARNING


# ── Test 7: Max clarification rounds → force pass ──────────────────

def test_max_rounds_force_pass():
    """After max rounds, validation forces pass with warning."""
    entities = ExtractedEntities(
        session_id="test-007",
        chief_complaint="Depression",
        diagnosis=None,  # Still missing
        procedure_or_intervention="Therapy",
        clinical_rationale="Ongoing treatment needed",
        confidence=0.75,
    )
    report = run_all_checks(entities, clarification_round=3)
    assert report.passed is True
    assert report.clarification_needed is False
    assert any("Max clarification" in w for w in report.warnings)


# ── Test 8: ICD-10 format warning ──────────────────────────────────

def test_icd10_format_warning():
    """Bad ICD-10 code produces warning, not error — still passes."""
    entities = ExtractedEntities(
        session_id="test-008",
        chief_complaint="Anxiety",
        diagnosis="Generalized anxiety disorder",
        diagnosis_code="INVALID-CODE",
        procedure_or_intervention="CBT",
        clinical_rationale="Ongoing anxiety symptoms noted",
        confidence=0.85,
    )
    report = run_all_checks(entities)
    assert report.passed is True
    code_warning = next(
        (i for i in report.issues if i.field == "diagnosis_code"), None
    )
    assert code_warning is not None
    assert code_warning.severity == ValidationSeverity.WARNING


# ── Test 9: Recommended field absent → WARNING only ────────────────

def test_recommended_absent_warning():
    """Missing recommended fields produce warnings, not errors."""
    entities = ExtractedEntities(
        session_id="test-009",
        chief_complaint="Depression",
        diagnosis="MDD recurrent moderate",
        procedure_or_intervention="Psychotherapy",
        clinical_rationale="Patient meets criteria for treatment",
        diagnosis_code=None,
        procedure_code=None,
        modifier_flags=[],
        confidence=0.88,
    )
    report = run_all_checks(entities)
    assert report.passed is True
    assert report.warning_count >= 3


# ── Test 10: Orchestrator full validation path happy case ──────────

@pytest.mark.asyncio
async def test_orchestrator_validation_happy_path(orch, monkeypatch):
    """Complete entities through orchestrator reach SERIALIZED."""
    # Use complete reasoning stub
    monkeypatch.setattr(
        reasoning_service,
        "extract_entities",
        reasoning_service.extract_entities_complete,
    )

    input_data = ClinicalInput(
        session_id="test-010",
        input_mode=InputMode.typed_text,
        transcript="Patient with MDD on Sertraline 50mg daily.",
    )
    await orch.create_session(input_data)
    state = await orch.handle_typed_input("test-010", input_data.transcript)

    assert state.status == SessionStatus.SERIALIZED
    assert state.final_prior_auth is not None
    assert state.final_fhir is not None
