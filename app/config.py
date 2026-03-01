"""Application configuration.

Loads environment variables via python-dotenv and exposes
a Pydantic BaseSettings instance for type-safe access.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    # ── App ──
    app_env: str = "development"
    log_level: str = "INFO"

    # ── Mistral AI ──
    mistral_api_key: str = ""

    # ── ElevenLabs ──
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    # Default: Rachel — clear, professional, neutral
    elevenlabs_model_id: str = "eleven_turbo_v2"
    elevenlabs_max_chars: int = 150
    elevenlabs_timeout_seconds: int = 10

    # ── Epic on FHIR ──
    epic_fhir_base_url: str = "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"
    epic_fhir_client_id: str = ""
    epic_fhir_token: str = ""

    # ── SQLite ──
    sqlite_db_path: str = "./data/sessions.db"

    # ── Transcription ──
    voxtral_realtime_model: str = "voxtral-mini-latest"
    voxtral_mini_model: str = "voxtral-mini-latest"
    transcription_timeout_seconds: int = 30
    max_audio_size_mb: int = 25
    supported_audio_formats: list[str] = [
        "audio/mpeg", "audio/wav", "audio/mp4", "audio/ogg", "audio/webm"
    ]

    # ── Reasoning (Mistral Large 3) ──
    mistral_large_model: str = "mistral-large-latest"
    mistral_reasoning_temperature: float = 0.1
    mistral_reasoning_max_tokens: int = 1024
    mistral_reasoning_timeout_seconds: int = 15

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
