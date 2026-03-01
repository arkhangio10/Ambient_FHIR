"""Reasoning service — clinical entity extraction via Mistral Large 3.

This service has ONE job: receive a clinical transcript and return
structured ExtractedEntities. No narrative, no summaries, no FHIR.

All prompts live in prompts/reasoning_prompts.py.
This service NEVER raises to the orchestrator — always returns ReasoningResult.
"""

import json
import logging

from app.config import get_settings
from app.schemas.entities import ExtractedEntities, ReasoningResult
from app.prompts.reasoning_prompts import (
    SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
    CLARIFICATION_MERGE_TEMPLATE,
    CONFIDENCE_THRESHOLDS,
)

logger = logging.getLogger(__name__)


# ── 1. extract_entities ─────────────────────────────────────────────

async def extract_entities(
    session_id: str,
    transcript: str,
) -> ReasoningResult:
    """Extract structured clinical entities from a transcript.

    Uses Mistral Large 3 with structured JSON output mode.
    Returns ReasoningResult — never raises.
    """
    settings = get_settings()

    # Guard: empty transcript
    if not transcript or not transcript.strip():
        return ReasoningResult(
            session_id=session_id,
            success=False,
            error_message="Empty transcript — cannot extract entities",
        )

    # Guard: missing API key
    if not settings.mistral_api_key:
        return ReasoningResult(
            session_id=session_id,
            success=False,
            error_message=(
                "Entity extraction unavailable (no API key). "
                "Falling back to stub extraction."
            ),
            model_used=settings.mistral_large_model,
        )

    try:
        from mistralai import Mistral

        client = Mistral(api_key=settings.mistral_api_key)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": EXTRACTION_USER_TEMPLATE.format(
                    session_id=session_id,
                    transcript=transcript,
                ),
            },
        ]

        response = await client.chat.complete_async(
            model=settings.mistral_large_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=settings.mistral_reasoning_temperature,
            max_tokens=settings.mistral_reasoning_max_tokens,
        )

        raw_json = response.choices[0].message.content
        parsed = json.loads(raw_json)

        entities = _map_to_entities(session_id, parsed)

        return ReasoningResult(
            session_id=session_id,
            entities=entities,
            success=True,
            model_used=settings.mistral_large_model,
            prompt_tokens=getattr(response.usage, "prompt_tokens", None),
            completion_tokens=getattr(response.usage, "completion_tokens", None),
            raw_response=raw_json,
        )

    except ImportError:
        return ReasoningResult(
            session_id=session_id,
            success=False,
            error_message="Mistral SDK not available.",
            model_used=settings.mistral_large_model,
        )
    except json.JSONDecodeError as e:
        logger.error("JSON parse error for %s: %s", session_id, e)
        return ReasoningResult(
            session_id=session_id,
            success=False,
            error_message=f"Failed to parse extraction response: {e}",
            model_used=settings.mistral_large_model,
        )
    except Exception as e:
        logger.error("Entity extraction failed for %s: %s", session_id, e)
        return ReasoningResult(
            session_id=session_id,
            success=False,
            error_message=f"Entity extraction failed: {e}",
            model_used=settings.mistral_large_model,
        )


# ── 2. extract_with_clarification ──────────────────────────────────

async def extract_with_clarification(
    session_id: str,
    transcript: str,
    existing_entities: ExtractedEntities,
    missing_field: str,
    clarification_answer: str,
) -> ReasoningResult:
    """Re-run extraction after a clarification answer is provided.

    Attempts Mistral Large 3 first. On failure, falls back to
    direct field mapping (no second API call).
    Returns ReasoningResult — never raises.
    """
    settings = get_settings()

    # Try Mistral-powered merge first
    if settings.mistral_api_key:
        try:
            from mistralai import Mistral

            client = Mistral(api_key=settings.mistral_api_key)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": CLARIFICATION_MERGE_TEMPLATE.format(
                        transcript=transcript,
                        existing_entities_json=existing_entities.model_dump_json(indent=2),
                        missing_field=missing_field,
                        clarification_answer=clarification_answer,
                    ),
                },
            ]

            response = await client.chat.complete_async(
                model=settings.mistral_large_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=settings.mistral_reasoning_temperature,
                max_tokens=settings.mistral_reasoning_max_tokens,
            )

            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            entities = _map_to_entities(session_id, parsed)

            return ReasoningResult(
                session_id=session_id,
                entities=entities,
                success=True,
                model_used=settings.mistral_large_model,
                prompt_tokens=getattr(response.usage, "prompt_tokens", None),
                completion_tokens=getattr(response.usage, "completion_tokens", None),
                raw_response=raw_json,
            )

        except Exception as e:
            logger.warning(
                "Clarification merge via Mistral failed for %s: %s — "
                "falling back to direct mapping",
                session_id, e,
            )

    # ── Direct field mapping fallback ───────────────────────────────
    return _direct_mapping_fallback(
        session_id, existing_entities, missing_field, clarification_answer
    )


# ── 3. validate_extraction_quality ──────────────────────────────────

def validate_extraction_quality(
    result: ReasoningResult,
) -> tuple[bool, str | None]:
    """Check confidence score and entity completeness.

    Synchronous — no API call. Returns (is_valid, error_message).
    """
    if not result.success or result.entities is None:
        return False, "Extraction failed — no entities returned"

    entities = result.entities

    # Confidence gate
    if entities.confidence < CONFIDENCE_THRESHOLDS["low"]:
        return False, (
            f"Extraction confidence too low: {entities.confidence:.2f}. "
            f"Please re-state or retype the clinical notes."
        )

    # Minimum viability: diagnosis OR chief_complaint must be present
    if not entities.diagnosis and not entities.chief_complaint:
        return False, (
            "Could not identify diagnosis or chief complaint. "
            "Please include the primary diagnosis in your notes."
        )

    return True, None


# ── Stub fallback (used when no API key) ────────────────────────────

async def extract_entities_complete(
    session_id: str,
    transcript: str,
) -> ReasoningResult:
    """Alternate stub with ALL fields populated — for happy-path tests."""
    entities = ExtractedEntities(
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
        raw_transcript_ref=transcript[:200] if transcript else None,
    )
    return ReasoningResult(
        session_id=session_id,
        entities=entities,
        success=True,
        model_used="stub_complete",
    )


# ── Private helpers ─────────────────────────────────────────────────

def _map_to_entities(session_id: str, parsed: dict) -> ExtractedEntities:
    """Map a parsed JSON dict to ExtractedEntities."""
    return ExtractedEntities(
        session_id=session_id,
        chief_complaint=parsed.get("chief_complaint"),
        diagnosis=parsed.get("diagnosis"),
        diagnosis_code=parsed.get("diagnosis_code"),
        procedure_or_intervention=parsed.get("procedure_or_intervention"),
        procedure_code=parsed.get("procedure_code"),
        medication=parsed.get("medication"),
        dosage=parsed.get("dosage"),
        frequency=parsed.get("frequency"),
        clinical_rationale=parsed.get("clinical_rationale"),
        modifier_flags=parsed.get("modifier_flags", []),
        missing_fields=parsed.get("missing_fields", []),
        confidence=parsed.get("confidence", 0.0),
        raw_transcript_ref=session_id,
    )


def _direct_mapping_fallback(
    session_id: str,
    existing_entities: ExtractedEntities,
    missing_field: str,
    clarification_answer: str,
) -> ReasoningResult:
    """Direct field mapping — no API call.

    Sets the missing field value, removes it from missing_fields.
    """
    entity_dict = existing_entities.model_dump()
    entity_dict[missing_field] = clarification_answer

    if missing_field in entity_dict.get("missing_fields", []):
        entity_dict["missing_fields"].remove(missing_field)

    updated = ExtractedEntities(**entity_dict)

    return ReasoningResult(
        session_id=session_id,
        entities=updated,
        success=True,
        model_used="direct_mapping_fallback",
    )
