"""Central orchestrator — session state machine.

Manages the full lifecycle of a clinical session:
RECEIVED → TRANSCRIBED → EXTRACTED → NEEDS_CLARIFICATION → VALIDATED → SERIALIZED → EXPORTED

All workflow logic converges through this module.
No route or service may bypass the orchestrator.
"""

import logging

from pydantic import ValidationError

from app.config import get_settings
from app.schemas.input import ClinicalInput
from app.schemas.entities import ExtractedEntities
from app.schemas.state import SessionState, SessionStatus
from app.services import (
    storage_service,
    reasoning_service,
    serializer_service,
    voice_service,
    ehr_service,
    transcription_service,
)
from app.validators.entity_checks import run_all_checks, build_clarification_prompt

logger = logging.getLogger(__name__)

# Safety limit for clarification rounds
_MAX_CLARIFICATION_ROUNDS = 3


class ClinicalOrchestrator:
    """Central state machine for clinical session processing.

    Every user action flows through this orchestrator. Routes call
    orchestrator methods only — zero business logic in routes.
    """

    # ── 1. create_session ───────────────────────────────────────────

    async def create_session(self, input: ClinicalInput) -> SessionState:
        """Create a new session from clinical input."""
        try:
            state = SessionState(
                session_id=input.session_id,
                input_mode=input.input_mode,
            )
            await self.persist_state(state)
            logger.info("Created session %s (mode=%s)", state.session_id, input.input_mode.value)
            return state
        except Exception as e:
            logger.error("Failed to create session: %s", e)
            state = SessionState(
                session_id=input.session_id,
                status=SessionStatus.FAILED,
                error_message=f"Session creation failed: {e}",
            )
            await self.persist_state(state)
            return state

    # ── 2. handle_typed_input ───────────────────────────────────────

    async def handle_typed_input(
        self, session_id: str, transcript: str
    ) -> SessionState:
        """Process typed text input through the full pipeline."""
        state = await self._load_or_fail(session_id)
        if state.status == SessionStatus.FAILED:
            return state

        try:
            state.transcript = transcript
            state.advance_status(SessionStatus.TRANSCRIBED)
            await self.persist_state(state)

            # Continue to extraction
            state = await self.extract_entities(session_id)
            return state

        except ValidationError as e:
            return await self._fail(state, f"Validation error: {e}")
        except Exception as e:
            return await self._fail(state, f"Unexpected error: {e}")

    # ── 3. handle_uploaded_audio ────────────────────────────────────

    async def handle_uploaded_audio(
        self, session_id: str, audio_reference: str
    ) -> SessionState:
        """Process uploaded audio through transcription and the full pipeline."""
        state = await self._load_or_fail(session_id)
        if state.status == SessionStatus.FAILED:
            return state

        try:
            result = await transcription_service.transcribe_uploaded_audio(
                session_id, audio_reference
            )

            if not result.success:
                # Graceful degradation — do not fail the session
                state.error_message = result.error_message
                # Status stays RECEIVED — frontend shows fallback message
                await self.persist_state(state)
                return state

            state.transcript = result.transcript
            state.advance_status(SessionStatus.TRANSCRIBED)
            await self.persist_state(state)

            state = await self.extract_entities(session_id)
            return state

        except ValidationError as e:
            return await self._fail(state, f"Validation error: {e}")
        except Exception as e:
            return await self._fail(state, f"Unexpected error: {e}")

    # ── 4. handle_live_transcript_chunk ─────────────────────────────

    async def handle_live_transcript_chunk(
        self, session_id: str, chunk: str | bytes
    ) -> SessionState:
        """Append a live transcript chunk (non-blocking for audio).

        For audio bytes: accumulates in buffer, triggers background
        transcription, and returns immediately with latest text.
        For text: appends directly.
        """
        state = await self._load_or_fail(session_id)
        if state.status == SessionStatus.FAILED:
            return state

        try:
            if isinstance(chunk, bytes):
                # This returns INSTANTLY — background task does API call
                result = await transcription_service.transcribe_realtime_chunk(
                    session_id, chunk
                )
                # Pick up any transcript from completed background tasks
                if result.transcript:
                    state.transcript = result.transcript
                    await self.persist_state(state)
                return state
            else:
                # Pre-transcribed text: append directly
                current = state.transcript or ""
                state.transcript = (current + " " + chunk).strip()
                await self.persist_state(state)
                return state

        except Exception as e:
            return await self._fail(state, f"Unexpected error: {e}")

    # ── 5. finalize_live_transcript ─────────────────────────────────

    async def finalize_live_transcript(self, session_id: str) -> SessionState:
        """Finalize accumulated live transcript and trigger extraction.

        Does a final transcription of the full audio buffer for best
        accuracy, then cleans up the buffer and advances the pipeline.
        """
        state = await self._load_or_fail(session_id)
        if state.status == SessionStatus.FAILED:
            return state

        try:
            # Final transcription of the full accumulated audio buffer
            buffer_size = transcription_service.get_buffer_size(session_id)
            if buffer_size > 0:
                logger.info(
                    "Final transcription for %s (%d bytes buffered)",
                    session_id, buffer_size,
                )
                result = await transcription_service.transcribe_audio_buffer(
                    session_id
                )
                if result.success and result.transcript:
                    state.transcript = result.transcript
                    await self.persist_state(state)

                # Clean up audio buffer
                transcription_service.clear_audio_buffer(session_id)

            if not state.transcript or not state.transcript.strip():
                return await self._fail(state, "Cannot finalize: transcript is empty")

            state.advance_status(SessionStatus.TRANSCRIBED)
            await self.persist_state(state)

            state = await self.extract_entities(session_id)
            return state

        except ValidationError as e:
            return await self._fail(state, f"Validation error: {e}")
        except Exception as e:
            return await self._fail(state, f"Unexpected error: {e}")

    # ── 6. extract_entities (INTERNAL) ──────────────────────────────

    async def extract_entities(self, session_id: str) -> SessionState:
        """Run entity extraction on the transcript."""
        state = await self._load_or_fail(session_id)
        if state.status == SessionStatus.FAILED:
            return state

        try:
            result = await reasoning_service.extract_entities(
                session_id, state.transcript or ""
            )

            # Quality gate
            is_valid, quality_error = reasoning_service.validate_extraction_quality(result)
            if not is_valid:
                return await self._fail(state, quality_error)

            state.extracted_entities = result.entities
            state.advance_status(SessionStatus.EXTRACTED)
            await self.persist_state(state)

            # Continue to validation
            state = await self.validate_entities(session_id)
            return state

        except ValidationError as e:
            return await self._fail(state, f"Extraction validation error: {e}")
        except Exception as e:
            return await self._fail(state, f"Extraction error: {e}")

    # ── 7. validate_entities (INTERNAL) ─────────────────────

    async def validate_entities(self, session_id: str) -> SessionState:
        """Validate extracted entities and branch to clarification or serialization."""
        state = await self._load_or_fail(session_id)
        if state.status == SessionStatus.FAILED:
            return state

        try:
            report = run_all_checks(
                entities=state.extracted_entities,
                clarification_round=state.clarification_rounds_count,
            )

            # Always persist the latest missing fields
            state.missing_fields = report.missing_required

            # Log warnings regardless of pass/fail
            for warning in report.warnings:
                logger.warning("[%s] Validation warning: %s", session_id, warning)

            if report.passed:
                state.advance_status(SessionStatus.VALIDATED)
                await self.persist_state(state)
                state = await self.serialize_outputs(session_id)
            elif report.clarification_needed:
                state.advance_status(SessionStatus.NEEDS_CLARIFICATION)
                await self.persist_state(state)
                state = await self.request_clarification(session_id)
            else:
                state.status = SessionStatus.FAILED
                state.error_message = (
                    "Validation failed with unresolvable missing fields"
                )
                await self.persist_state(state)

            return state

        except Exception as e:
            return await self._fail(state, f"Validation error: {e}")

    # ── 8. request_clarification (INTERNAL) ─────────────────────────

    async def request_clarification(self, session_id: str) -> SessionState:
        """Generate a clarification prompt and synthesize audio."""
        state = await self._load_or_fail(session_id)
        if state.status == SessionStatus.FAILED:
            return state

        try:
            prompt = build_clarification_prompt(
                state.missing_fields, state.extracted_entities
            )

            if not prompt:
                logger.warning(
                    "Could not build clarification prompt for session %s",
                    session_id,
                )
                return state

            state.clarification_prompt = prompt
            state.clarification_rounds_count += 1

            # Synthesize audio — failure is non-fatal
            voice_result = await voice_service.synthesize_clarification(
                session_id=session_id,
                clarification_prompt=prompt,
            )

            if voice_result.success:
                state.clarification_audio_url = voice_result.audio_url
            else:
                # Text fallback handles UI — log and continue
                logger.info(
                    "Voice synthesis unavailable for %s: %s",
                    session_id, voice_result.error_message,
                )
                state.clarification_audio_url = None

            await self.persist_state(state)
            logger.info(
                "Clarification requested for session %s: %s",
                session_id, prompt,
            )
            return state

        except Exception as e:
            return await self._fail(state, f"Clarification error: {e}")

    # ── 9. handle_clarification_response ────────────────────────────

    async def handle_clarification_response(
        self, session_id: str, answer: str
    ) -> SessionState:
        """Process the clinician's clarification answer."""
        state = await self._load_or_fail(session_id)
        if state.status == SessionStatus.FAILED:
            return state

        try:
            if state.status != SessionStatus.NEEDS_CLARIFICATION:
                return await self._fail(
                    state,
                    f"Cannot process clarification: session is in {state.status.value}, "
                    f"expected NEEDS_CLARIFICATION",
                )

            state.clarification_answer = answer
            await self.persist_state(state)

            state = await self.merge_clarification_answer(session_id)
            return state

        except ValidationError as e:
            return await self._fail(state, f"Validation error: {e}")
        except Exception as e:
            return await self._fail(state, f"Clarification handling error: {e}")

    # ── 10. merge_clarification_answer (INTERNAL) ───────────────────

    async def merge_clarification_answer(
        self, session_id: str, _round: int = 0
    ) -> SessionState:
        """Merge the clarification answer into extracted entities and re-validate."""
        state = await self._load_or_fail(session_id)
        if state.status == SessionStatus.FAILED:
            return state

        try:
            if not state.missing_fields or not state.clarification_answer:
                return state

            target_field = state.missing_fields[0]

            # Use reasoning service with clarification context
            result = await reasoning_service.extract_with_clarification(
                session_id=session_id,
                transcript=state.transcript or "",
                existing_entities=state.extracted_entities,
                missing_field=target_field,
                clarification_answer=state.clarification_answer,
            )

            if result.success and result.entities:
                state.extracted_entities = result.entities
                # Refresh missing_fields from updated entities
                state.missing_fields = result.entities.missing_fields
            else:
                logger.warning(
                    "Clarification merge used fallback for session %s",
                    session_id,
                )

            # Clean up audio after resolution
            await voice_service.clear_session_audio(session_id)

            await self.persist_state(state)

            # Re-validate (advance back to EXTRACTED first)
            state.advance_status(SessionStatus.EXTRACTED)
            await self.persist_state(state)

            # Safety limit on clarification rounds
            if _round >= _MAX_CLARIFICATION_ROUNDS:
                logger.warning(
                    "Max clarification rounds reached for session %s", session_id
                )
                state.advance_status(SessionStatus.VALIDATED)
                await self.persist_state(state)
                state = await self.serialize_outputs(session_id)
                return state

            state = await self.validate_entities(session_id)
            return state

        except Exception as e:
            return await self._fail(state, f"Merge clarification error: {e}")

    # ── 11. serialize_outputs (INTERNAL) ────────────────────────────

    async def serialize_outputs(self, session_id: str) -> SessionState:
        """Build PriorAuthPacket and FHIRPayload from validated entities."""
        state = await self._load_or_fail(session_id)
        if state.status == SessionStatus.FAILED:
            return state

        try:
            prior_auth = await serializer_service.build_prior_auth(
                state.extracted_entities
            )
            fhir_payload = await serializer_service.build_fhir_payload(
                state.extracted_entities
            )

            state.final_prior_auth = prior_auth
            state.final_fhir = fhir_payload
            state.advance_status(SessionStatus.SERIALIZED)
            await self.persist_state(state)

            # Optionally export to EHR
            settings = get_settings()
            if settings.epic_fhir_base_url and settings.epic_fhir_token:
                state = await self.export_to_ehr(session_id)
            else:
                # Demo mode: automatically advance to EXPORTED so the UI completes 
                logger.info("No Epic FHIR config detected — auto-advancing to EXPORTED for demo")
                state.advance_status(SessionStatus.EXPORTED)
                await self.persist_state(state)

            return state

        except ValidationError as e:
            return await self._fail(state, f"Serialization validation error: {e}")
        except Exception as e:
            return await self._fail(state, f"Serialization error: {e}")

    # ── 12. persist_state (INTERNAL) ────────────────────────────────

    async def persist_state(self, state: SessionState) -> None:
        """Save session state. Logs errors but never blocks the main flow."""
        try:
            await storage_service.save_session(state)
        except Exception as e:
            logger.error(
                "Failed to persist session %s: %s (non-blocking)",
                state.session_id, e,
            )

    # ── 13. export_to_ehr (INTERNAL) ────────────────────────────────

    async def export_to_ehr(self, session_id: str) -> SessionState:
        """POST FHIR payload to Epic sandbox. Never fails fatally."""
        state = await self._load_or_fail(session_id)
        if state.status == SessionStatus.FAILED:
            return state

        settings = get_settings()
        if not settings.epic_fhir_base_url:
            logger.info("Epic FHIR not configured — skipping export for %s", session_id)
            return state

        try:
            success, response = await ehr_service.post_fhir_payload(state.final_fhir)

            if success:
                state.advance_status(SessionStatus.EXPORTED)
                await self.persist_state(state)
                logger.info("Exported session %s to Epic: %s", session_id, response)
            else:
                logger.warning(
                    "Epic export failed for %s (non-fatal): %s", session_id, response
                )

        except Exception as e:
            logger.warning("Epic export error for %s (non-fatal): %s", session_id, e)

        return state

    # ── 14. get_session ─────────────────────────────────────────────

    async def get_session(self, session_id: str) -> SessionState | None:
        """Load and return a session. Returns None if not found."""
        return await storage_service.load_session(session_id)

    # ── Private helpers ─────────────────────────────────────────────

    async def _load_or_fail(self, session_id: str) -> SessionState:
        """Load a session, or return a FAILED state if not found."""
        state = await storage_service.load_session(session_id)
        if state is None:
            state = SessionState(
                session_id=session_id,
                status=SessionStatus.FAILED,
                error_message=f"Session {session_id} not found",
            )
            await self.persist_state(state)
        return state

    async def _fail(self, state: SessionState, message: str) -> SessionState:
        """Set a session to FAILED status with an error message."""
        logger.error("Session %s FAILED: %s", state.session_id, message)
        state.status = SessionStatus.FAILED
        state.error_message = message
        await self.persist_state(state)
        return state


# Module-level singleton
orchestrator = ClinicalOrchestrator()
