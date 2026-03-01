"""FHIR payload schema — FHIRPayload model.

FHIR R4-aligned structure for Epic sandbox submission.
Demo-safe: uses dict fields for resource shapes rather than
full FHIR spec compliance.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class FHIRPayload(BaseModel):
    """FHIR-aligned output bundle for EHR integration.

    Not full FHIR R4 compliance — uses dict fields for Condition and
    MedicationRequest to stay demo-safe and avoid over-engineering.
    """

    session_id: str = Field(..., description="Owning session")
    resource_type: str = Field(
        default="Bundle", description="FHIR resource type"
    )
    patient_reference: str | None = Field(
        default=None,
        description='Patient reference in "Patient/[id]" format',
    )
    encounter_summary: str | None = Field(
        default=None, description="Summary of the clinical encounter"
    )
    condition: dict | None = Field(
        default=None,
        description=(
            "FHIR Condition resource shape, e.g. "
            '{"resourceType": "Condition", "code": {"text": "..."}, '
            '"clinicalStatus": {"text": "active"}}'
        ),
    )
    medication_request: dict | None = Field(
        default=None,
        description=(
            "FHIR MedicationRequest shape, e.g. "
            '{"resourceType": "MedicationRequest", '
            '"medicationCodeableConcept": {"text": "..."}, '
            '"dosageInstruction": [{"text": "..."}]}'
        ),
    )
    supporting_info: list[dict] = Field(
        default_factory=list,
        description="Additional supporting documents",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this payload was generated (UTC)",
    )

    model_config = {"str_strip_whitespace": True}
