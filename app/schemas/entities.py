"""Entities schema — ExtractedEntities model.

Captures all required clinical/administrative fields extracted
from a transcript by the reasoning service.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ── Validation types ────────────────────────────────────────────────


class ValidationSeverity(str, Enum):
    """Severity level for a validation issue."""
    ERROR = "error"       # Blocks serialization
    WARNING = "warning"   # Logged but does not block
    INFO = "info"         # Informational only


class ValidationIssue(BaseModel):
    """A single validation finding."""
    field: str
    severity: ValidationSeverity
    message: str
    suggested_action: str | None = None


class ValidationReport(BaseModel):
    """Aggregated result of all validation checks."""
    session_id: str
    passed: bool = False
    issues: list[ValidationIssue] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    clarification_needed: bool = False
    first_missing_field: str | None = None
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)


# ── Entity models ──────────────────────────────────────────────────


class ExtractedEntities(BaseModel):
    """Structured clinical entities extracted from a transcript.

    The reasoning service (Mistral Large 3) populates these fields.
    The validator layer checks for missing required fields downstream.
    """

    session_id: str = Field(..., description="Session this extraction belongs to")

    # ── Clinical fields ─────────────────────────────────────────────
    chief_complaint: str | None = Field(
        default=None, description="Patient's primary concern"
    )
    diagnosis: str | None = Field(
        default=None, description="ICD-10 code or description"
    )
    diagnosis_code: str | None = Field(
        default=None, description="ICD-10 code only, if extractable"
    )
    procedure_or_intervention: str | None = Field(
        default=None, description="CPT code or description"
    )
    procedure_code: str | None = Field(
        default=None, description="CPT code only, if extractable"
    )

    # ── Medication fields ───────────────────────────────────────────
    medication: str | None = Field(default=None, description="Medication name")
    dosage: str | None = Field(
        default=None,
        description="Dosage — required if medication is present (validated downstream)",
    )
    frequency: str | None = Field(
        default=None,
        description="Frequency — required if medication is present (validated downstream)",
    )

    # ── Billing / auth fields ───────────────────────────────────────
    clinical_rationale: str | None = Field(
        default=None,
        description="Supports prior-auth clinical justification",
    )
    modifier_flags: list[str] = Field(
        default_factory=list,
        description="Behavioral health billing modifiers",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Populated by the validator, not the extractor",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (0.0–1.0)",
    )

    # ── Source reference ────────────────────────────────────────────
    raw_transcript_ref: str | None = Field(
        default=None, description="Reference back to source transcript"
    )

    model_config = {"str_strip_whitespace": True}

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        """Ensure confidence stays within [0.0, 1.0]."""
        return max(0.0, min(1.0, v))


class ReasoningResult(BaseModel):
    """Wrapper for reasoning service responses.

    Provides metadata (model, tokens, raw response) alongside
    the extracted entities. The orchestrator inspects `success`
    and `entities` — raw_response is for debugging only.
    """

    session_id: str = Field(..., description="Owning session")
    entities: ExtractedEntities | None = Field(
        default=None, description="Extracted entities, or None on failure"
    )
    success: bool = Field(default=False, description="Whether extraction succeeded")
    error_message: str | None = Field(
        default=None, description="Error details when success is False"
    )
    model_used: str | None = Field(
        default=None, description="Model identifier used for extraction"
    )
    prompt_tokens: int | None = Field(
        default=None, description="Prompt token count"
    )
    completion_tokens: int | None = Field(
        default=None, description="Completion token count"
    )
    raw_response: str | None = Field(
        default=None, description="Raw model output for debugging — never expose to frontend"
    )

    model_config = {"str_strip_whitespace": True}


class VoiceResult(BaseModel):
    """Wrapper for voice synthesis responses.

    text_fallback is ALWAYS populated so the frontend
    can display the question even if audio fails.
    """

    session_id: str = Field(..., description="Owning session")
    success: bool = Field(default=False, description="Whether TTS succeeded")
    audio_url: str | None = Field(
        default=None, description="URL to fetch audio from /audio/{session_id}"
    )
    audio_bytes_b64: str | None = Field(
        default=None, description="Base64-encoded audio for direct frontend embedding"
    )
    text_fallback: str = Field(
        ..., description="Always populated — frontend shows this if audio fails"
    )
    voice_id_used: str | None = Field(
        default=None, description="ElevenLabs voice ID used"
    )
    character_count: int = Field(default=0, description="Prompt character count")
    error_message: str | None = Field(
        default=None, description="Error details when success is False"
    )

    model_config = {"str_strip_whitespace": True}
