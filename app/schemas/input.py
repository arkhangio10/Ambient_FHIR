"""Input schema — ClinicalInput model.

Defines the inbound payload for POST /process-clinical-data:
session_id, input_mode, transcript, audio_reference, clarification_answer.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class InputMode(str, Enum):
    """How the clinical data was provided."""

    live_audio = "live_audio"
    uploaded_audio = "uploaded_audio"
    typed_text = "typed_text"
    clarification_response = "clarification_response"


class ClinicalInput(BaseModel):
    """Inbound clinical data payload.

    Accepted by POST /process-clinical-data and the orchestrator.
    Cross-field validators ensure the right fields are present for each mode.
    """

    session_id: str = Field(..., description="UUID identifying the session")
    input_mode: InputMode = Field(..., description="How the data was provided")
    transcript: str | None = Field(
        default=None,
        description="Transcript text — required for typed_text mode",
    )
    audio_reference: str | None = Field(
        default=None,
        description="File path or URL for uploaded audio",
    )
    clarification_answer: str | None = Field(
        default=None,
        description="Clinician's answer to a clarification question",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the input was created (UTC)",
    )

    model_config = {"str_strip_whitespace": True}

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "ClinicalInput":
        """Ensure the correct fields are populated for each input mode."""
        if self.input_mode == InputMode.typed_text:
            if not self.transcript:
                raise ValueError(
                    "transcript must not be None or empty when input_mode is typed_text"
                )

        if self.input_mode == InputMode.uploaded_audio:
            if not self.audio_reference:
                raise ValueError(
                    "audio_reference must not be None when input_mode is uploaded_audio"
                )

        if self.input_mode == InputMode.clarification_response:
            if not self.clarification_answer:
                raise ValueError(
                    "clarification_answer must not be None when input_mode is clarification_response"
                )

        return self
