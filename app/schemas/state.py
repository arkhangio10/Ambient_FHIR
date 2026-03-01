"""Session state schema — SessionState model + SessionStatus enum.

Tracks the full lifecycle of a clinical session through the
state machine managed by the orchestrator.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.input import InputMode
from app.schemas.entities import ExtractedEntities
from app.schemas.prior_auth import PriorAuthPacket
from app.schemas.fhir import FHIRPayload


class SessionStatus(str, Enum):
    """Ordered session lifecycle states."""

    RECEIVED = "RECEIVED"
    TRANSCRIBED = "TRANSCRIBED"
    EXTRACTED = "EXTRACTED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    VALIDATED = "VALIDATED"
    SERIALIZED = "SERIALIZED"
    EXPORTED = "EXPORTED"
    FAILED = "FAILED"


# Valid forward transitions (current → allowed next states)
_FORWARD_TRANSITIONS: dict[SessionStatus, set[SessionStatus]] = {
    SessionStatus.RECEIVED: {SessionStatus.TRANSCRIBED, SessionStatus.FAILED},
    SessionStatus.TRANSCRIBED: {SessionStatus.EXTRACTED, SessionStatus.FAILED},
    SessionStatus.EXTRACTED: {
        SessionStatus.NEEDS_CLARIFICATION,
        SessionStatus.VALIDATED,
        SessionStatus.FAILED,
    },
    SessionStatus.NEEDS_CLARIFICATION: {
        SessionStatus.EXTRACTED,
        SessionStatus.VALIDATED,
        SessionStatus.FAILED,
    },
    SessionStatus.VALIDATED: {SessionStatus.SERIALIZED, SessionStatus.FAILED},
    SessionStatus.SERIALIZED: {SessionStatus.EXPORTED, SessionStatus.FAILED},
    SessionStatus.EXPORTED: {SessionStatus.FAILED},
    SessionStatus.FAILED: set(),
}


class SessionState(BaseModel):
    """Full mutable state for a single clinical session.

    Updated at every status transition by the orchestrator and
    persisted to SQLite by the storage service.
    """

    session_id: str = Field(..., description="Unique session identifier")
    status: SessionStatus = Field(
        default=SessionStatus.RECEIVED,
        description="Current lifecycle status",
    )

    # ── Input ───────────────────────────────────────────────────────
    input_mode: InputMode | None = Field(
        default=None, description="How the data was originally provided"
    )
    transcript: str | None = Field(
        default=None, description="Current transcript text"
    )

    # ── Extraction ──────────────────────────────────────────────────
    extracted_entities: ExtractedEntities | None = Field(
        default=None, description="Entities extracted by reasoning service"
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Fields still missing after extraction",
    )

    # ── Clarification ───────────────────────────────────────────────
    clarification_prompt: str | None = Field(
        default=None, description="Question asked to the clinician"
    )
    clarification_answer: str | None = Field(
        default=None, description="Clinician's response"
    )
    clarification_audio_url: str | None = Field(
        default=None, description="ElevenLabs TTS audio reference"
    )
    clarification_rounds_count: int = Field(
        default=0, description="Number of clarification rounds attempted"
    )

    # ── Outputs ─────────────────────────────────────────────────────
    final_prior_auth: PriorAuthPacket | None = Field(
        default=None, description="Generated prior-auth packet"
    )
    final_fhir: FHIRPayload | None = Field(
        default=None, description="Generated FHIR payload"
    )

    # ── Timestamps ──────────────────────────────────────────────────
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Session creation time (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time (UTC), refreshed on every state change",
    )

    # ── Error ───────────────────────────────────────────────────────
    error_message: str | None = Field(
        default=None, description="Populated if status == FAILED"
    )

    model_config = {"str_strip_whitespace": True}

    def advance_status(self, new_status: SessionStatus) -> None:
        """Move the session to a new status.

        Raises ValueError if the transition is not a valid forward move.
        Updates ``updated_at`` on success.
        """
        allowed = _FORWARD_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition: {self.status.value} → {new_status.value}. "
                f"Allowed: {sorted(s.value for s in allowed)}"
            )
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)
