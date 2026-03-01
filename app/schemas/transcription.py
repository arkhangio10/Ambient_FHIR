"""Transcription result schema — TranscriptionResult model.

Provides a uniform shape for all transcription sources:
Voxtral Realtime, Voxtral Mini, and manual text fallback.
"""

from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    """Result of any transcription operation.

    All transcription paths (live, batch, manual) produce this same
    shape so the orchestrator can treat them uniformly.
    """

    session_id: str = Field(..., description="Owning session")
    transcript: str | None = Field(
        default=None, description="Transcribed text, or None on failure"
    )
    success: bool = Field(
        default=False, description="Whether transcription succeeded"
    )
    error_message: str | None = Field(
        default=None, description="Error details when success is False"
    )
    source: str = Field(
        ...,
        description='Origin: "voxtral_realtime" | "voxtral_mini" | "manual_fallback"',
    )
    duration_seconds: float | None = Field(
        default=None, description="Audio duration in seconds, if known"
    )
    language: str | None = Field(
        default="en", description="Detected or assumed language"
    )

    model_config = {"str_strip_whitespace": True}
