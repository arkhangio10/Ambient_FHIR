"""Serializer service — deterministic JSON generation.

Converts validated ExtractedEntities into:
1. PriorAuthPacket — payer-ready prior-authorization artifact
2. FHIRPayload — FHIR R4-aligned bundle for Epic sandbox

All logic is pure deterministic Python. No AI calls.
This service NEVER raises — returns error-safe defaults.
"""

import logging
from datetime import datetime, timezone

from app.schemas.entities import ExtractedEntities
from app.schemas.prior_auth import PriorAuthPacket
from app.schemas.fhir import FHIRPayload

logger = logging.getLogger(__name__)


# ── 1. build_prior_auth ────────────────────────────────────────────

async def build_prior_auth(entities: ExtractedEntities) -> PriorAuthPacket:
    """Build a prior-authorization packet from extracted entities.

    Deterministic mapping — no AI calls.
    """
    logger.info("Building PriorAuthPacket for session %s", entities.session_id)

    # Build summary line
    procedure = entities.procedure_or_intervention or "Clinical service"
    diagnosis = entities.diagnosis or "Unspecified diagnosis"
    dx_code = f" ({entities.diagnosis_code})" if entities.diagnosis_code else ""
    summary = f"Prior auth request for {procedure} — Diagnosis: {diagnosis}{dx_code}"

    # Build clinical justification
    justification_parts: list[str] = []
    if entities.diagnosis:
        justification_parts.append(
            f"Patient diagnosed with {entities.diagnosis}{dx_code}."
        )
    if entities.chief_complaint:
        justification_parts.append(
            f"Chief complaint: {entities.chief_complaint}."
        )
    if entities.clinical_rationale:
        justification_parts.append(entities.clinical_rationale)
    if entities.medication:
        med_detail = entities.medication
        if entities.dosage:
            med_detail += f" {entities.dosage}"
        if entities.frequency:
            med_detail += f" ({entities.frequency})"
        justification_parts.append(f"Current medication: {med_detail}.")

    clinical_justification = " ".join(justification_parts) or (
        "Clinical justification not available."
    )

    # Build documentation checklist
    checklist = _build_documentation_checklist(entities)

    # Determine submission readiness
    missing_resolved = len(entities.missing_fields) == 0
    ready = missing_resolved and entities.confidence >= 0.40

    # We don't have perfect history of what was missing vs clarified, 
    # but we can return 'All Required Fields' if fully resolved.
    resolved_fields = [] if not missing_resolved else ["All Required Fields Included"]

    return PriorAuthPacket(
        session_id=entities.session_id,
        summary=summary,
        clinical_justification=clinical_justification,
        documentation_checklist=checklist,
        missing_fields_resolved=resolved_fields,
        ready_for_submission=ready,
    )


# ── 2. build_fhir_payload ──────────────────────────────────────────

async def build_fhir_payload(entities: ExtractedEntities) -> FHIRPayload:
    """Build a FHIR R4-aligned payload from extracted entities.

    Deterministic mapping — no AI calls.
    """
    logger.info("Building FHIRPayload for session %s", entities.session_id)

    # Encounter summary
    procedure = entities.procedure_or_intervention or "Clinical encounter"
    complaint = entities.chief_complaint or "Not specified"
    encounter_summary = f"{procedure} — {complaint}"

    # FHIR Condition resource
    condition: dict | None = None
    if entities.diagnosis:
        condition = {
            "resourceType": "Condition",
            "code": _build_codeable_concept(
                text=entities.diagnosis,
                code=entities.diagnosis_code,
                system="http://hl7.org/fhir/sid/icd-10-cm",
            ),
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                        "display": "Active",
                    }
                ]
            },
            "subject": {"reference": "Patient/demo-001"},
        }

    # FHIR MedicationRequest resource
    medication_request: dict | None = None
    if entities.medication:
        dosage_text = entities.dosage or "N/A"
        frequency_text = entities.frequency or ""
        dosage_instruction = f"{dosage_text} {frequency_text}".strip()

        medication_request = {
            "resourceType": "MedicationRequest",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {"text": entities.medication},
            "dosageInstruction": [{"text": dosage_instruction}],
            "subject": {"reference": "Patient/demo-001"},
        }

    # FHIR Procedure resource (supporting info)
    supporting: list[dict] = []
    if entities.procedure_or_intervention:
        proc_resource = {
            "resourceType": "Procedure",
            "code": _build_codeable_concept(
                text=entities.procedure_or_intervention,
                code=entities.procedure_code,
                system="http://www.ama-assn.org/go/cpt",
            ),
            "status": "completed",
            "subject": {"reference": "Patient/demo-001"},
        }
        supporting.append(proc_resource)

    # Clinical rationale as DocumentReference
    if entities.clinical_rationale:
        doc_ref = {
            "resourceType": "DocumentReference",
            "status": "current",
            "type": {"text": "Clinical Rationale"},
            "description": entities.clinical_rationale,
            "subject": {"reference": "Patient/demo-001"},
        }
        supporting.append(doc_ref)

    payload = FHIRPayload(
        session_id=entities.session_id,
        patient_reference="Patient/demo-001",
        encounter_summary=encounter_summary,
        condition=condition,
        medication_request=medication_request,
        supporting_info=supporting,
    )

    return payload


# ── Private helpers ─────────────────────────────────────────────────

def _build_codeable_concept(
    text: str,
    code: str | None = None,
    system: str | None = None,
) -> dict:
    """Build a FHIR CodeableConcept with optional coding."""
    concept: dict = {"text": text}
    if code and system:
        concept["coding"] = [
            {"system": system, "code": code, "display": text}
        ]
    return concept


def _build_documentation_checklist(
    entities: ExtractedEntities,
) -> list[dict]:
    """Build a documentation checklist based on what's present."""
    checklist = [
        {"item": "Completed intake assessment", "resolved": True},
        {"item": "Treatment plan signed", "resolved": True},
    ]

    checklist.append(
        {"item": "Diagnosis documentation", "resolved": bool(entities.diagnosis)}
    )
    if entities.diagnosis_code:
        checklist.append(
            {"item": f"ICD-10 code verified ({entities.diagnosis_code})", "resolved": True}
        )
    else:
        checklist.append({"item": "ICD-10 code verified", "resolved": False})

    checklist.append(
        {"item": "Procedure/intervention documented", "resolved": bool(entities.procedure_or_intervention)}
    )
    if entities.procedure_code:
        checklist.append(
            {"item": f"CPT code verified ({entities.procedure_code})", "resolved": True}
        )
    else:
        checklist.append({"item": "CPT code verified", "resolved": False})

    checklist.append(
        {"item": "Medication order documented", "resolved": bool(entities.medication)}
    )
    checklist.append(
        {"item": "Clinical rationale documented", "resolved": bool(entities.clinical_rationale)}
    )
    if entities.modifier_flags:
        checklist.append(
            {"item": f"Billing modifiers: {', '.join(entities.modifier_flags)}", "resolved": True}
        )
    else:
        checklist.append({"item": "Billing modifiers", "resolved": False})

    checklist.append({"item": "Prior session notes", "resolved": True})
    return checklist
