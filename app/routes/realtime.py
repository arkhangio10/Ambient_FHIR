"""WebSocket route for real-time transcription.

Endpoint: /ws/realtime-transcribe
Accepts both binary audio chunks and text control signals.
Delegates all logic to the ClinicalOrchestrator.
Zero business logic lives in this module.
"""

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.input import ClinicalInput, InputMode
from app.services.orchestrator import orchestrator

router = APIRouter(tags=["realtime"])
logger = logging.getLogger(__name__)


@router.websocket("/ws/realtime-transcribe")
async def realtime_transcribe(websocket: WebSocket):
    """Handle a real-time transcription WebSocket session.

    Protocol:
    - Server sends session_id on connection
    - Client sends binary frames (raw audio chunks)
    - Client sends text "__FINALIZE__" to trigger extraction
    - Client sends text "__CANCEL__" to abort
    - Server responds with transcript state after each frame
    """
    await websocket.accept()
    session_id = str(uuid4())
    logger.info("WebSocket session started: %s", session_id)

    # Create session via orchestrator
    input_data = ClinicalInput(
        session_id=session_id,
        input_mode=InputMode.live_audio,
    )
    await orchestrator.create_session(input_data)

    # Send session_id back to client immediately
    await websocket.send_json({"session_id": session_id, "status": "connected"})

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            elif message["type"] == "websocket.receive":

                # Text control signals
                if "text" in message:
                    data = message["text"]

                    if data == "__FINALIZE__":
                        state = await orchestrator.finalize_live_transcript(
                            session_id
                        )
                        await websocket.send_json(
                            state.model_dump(mode="json")
                        )
                        break

                    elif data == "__CANCEL__":
                        await websocket.send_json({"status": "cancelled"})
                        break

                    else:
                        # Treat as pre-transcribed text chunk
                        state = await orchestrator.handle_live_transcript_chunk(
                            session_id, data
                        )
                        await websocket.send_json({
                            "status": state.status.value,
                            "partial_transcript": state.transcript,
                        })

                # Binary audio chunks — handle fast, don't block
                elif "bytes" in message:
                    audio_chunk = message["bytes"]
                    state = await orchestrator.handle_live_transcript_chunk(
                        session_id, audio_chunk
                    )
                    await websocket.send_json({
                        "status": state.status.value,
                        "partial_transcript": state.transcript or "",
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", session_id)
    except Exception as exc:
        logger.error("WebSocket error for %s: %s", session_id, exc)
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
