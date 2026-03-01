"""EHR service — Epic on FHIR sandbox integration.

Posts FHIR-aligned payloads to the Epic sandbox.
NEVER raises — returns (success, response_dict).
If no token or network fails → returns (False, error_dict).
Demo stays stable regardless of sandbox availability.
"""

import logging

import httpx

from app.config import get_settings
from app.schemas.fhir import FHIRPayload

logger = logging.getLogger(__name__)

# Timeout for Epic API calls
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


async def post_fhir_payload(payload: FHIRPayload) -> tuple[bool, dict]:
    """POST a FHIR payload to the Epic sandbox.

    Returns:
        (success: bool, response: dict)

    Never raises — all errors returned as (False, {...}).
    """
    settings = get_settings()

    # Guard: no API token
    if not settings.epic_fhir_token:
        logger.info(
            "Epic FHIR token not configured — skipping export for %s",
            payload.session_id,
        )
        return False, {
            "status": "skipped",
            "message": "Epic FHIR token not configured",
            "session_id": payload.session_id,
        }

    # Guard: no base URL
    if not settings.epic_fhir_base_url:
        return False, {
            "status": "skipped",
            "message": "Epic FHIR base URL not configured",
            "session_id": payload.session_id,
        }

    # Build FHIR Bundle for submission
    bundle = _build_fhir_bundle(payload)
    url = f"{settings.epic_fhir_base_url.rstrip('/')}"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                url,
                json=bundle,
                headers={
                    "Authorization": f"Bearer {settings.epic_fhir_token}",
                    "Content-Type": "application/fhir+json",
                    "Accept": "application/fhir+json",
                },
            )

        if response.status_code in (200, 201):
            logger.info(
                "Epic export succeeded for %s: %d",
                payload.session_id, response.status_code,
            )
            return True, {
                "status": "success",
                "http_status": response.status_code,
                "session_id": payload.session_id,
                "response": response.json() if response.text else {},
            }
        else:
            logger.warning(
                "Epic export returned %d for %s: %s",
                response.status_code, payload.session_id, response.text[:200],
            )
            return False, {
                "status": "error",
                "http_status": response.status_code,
                "session_id": payload.session_id,
                "detail": response.text[:500],
            }

    except httpx.TimeoutException:
        logger.warning("Epic export timed out for %s", payload.session_id)
        return False, {
            "status": "timeout",
            "message": "Epic FHIR sandbox request timed out",
            "session_id": payload.session_id,
        }
    except httpx.ConnectError:
        logger.warning("Epic export connection failed for %s", payload.session_id)
        return False, {
            "status": "connection_error",
            "message": "Cannot connect to Epic FHIR sandbox",
            "session_id": payload.session_id,
        }
    except Exception as e:
        logger.error("Epic export failed for %s: %s", payload.session_id, e)
        return False, {
            "status": "error",
            "message": f"Unexpected error: {e}",
            "session_id": payload.session_id,
        }


def _build_fhir_bundle(payload: FHIRPayload) -> dict:
    """Build a FHIR Transaction Bundle from the payload."""
    entries: list[dict] = []

    if payload.condition:
        entries.append({
            "resource": payload.condition,
            "request": {
                "method": "POST",
                "url": "Condition",
            },
        })

    if payload.medication_request:
        entries.append({
            "resource": payload.medication_request,
            "request": {
                "method": "POST",
                "url": "MedicationRequest",
            },
        })

    for resource in payload.supporting_info:
        resource_type = resource.get("resourceType", "Resource")
        entries.append({
            "resource": resource,
            "request": {
                "method": "POST",
                "url": resource_type,
            },
        })

    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": entries,
    }
