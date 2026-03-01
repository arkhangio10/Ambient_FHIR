# API Contract — Frontend ↔ Backend

> This document defines every endpoint, WebSocket event, payload shape,
> and error response for the Lovable UI integration.

## Status

**Stub** — will be fully documented in Phase 12.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Health check |
| `POST` | `/process-clinical-data` | Process clinical input |
| `WS` | `/ws/realtime-transcribe` | Real-time audio transcription |

## Payload Shapes

_To be defined in Phase 3 (Pydantic Schemas) and documented here in Phase 12._
