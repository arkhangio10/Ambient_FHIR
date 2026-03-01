"""Storage service — SQLite + in-memory persistence.

Stores session state as JSON in SQLite for traceability.
Falls back to in-memory dict if SQLite is unavailable.
Tests always use in-memory mode via clear_all().
"""

import json
import logging
import os
import sqlite3
from typing import Optional

from app.config import get_settings
from app.schemas.state import SessionState

logger = logging.getLogger(__name__)

# In-memory fallback store (also used by tests)
_sessions: dict[str, SessionState] = {}

# SQLite connection — lazily initialized
_db: Optional[sqlite3.Connection] = None
_db_initialized: bool = False


# ── SQLite initialization ──────────────────────────────────────────

def _get_db() -> Optional[sqlite3.Connection]:
    """Get or create the SQLite connection."""
    global _db, _db_initialized

    if _db_initialized:
        return _db

    _db_initialized = True
    settings = get_settings()
    db_path = settings.sqlite_db_path

    try:
        # Create parent directories
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        _db = sqlite3.connect(db_path)
        _db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        _db.commit()
        logger.info("SQLite initialized at %s", db_path)
        return _db

    except Exception as e:
        logger.warning("SQLite unavailable (%s), using in-memory fallback", e)
        _db = None
        return None


# ── Public API ─────────────────────────────────────────────────────

async def save_session(state: SessionState) -> None:
    """Persist a session state snapshot.

    Writes to SQLite first, then updates in-memory cache.
    """
    # Always update in-memory cache
    _sessions[state.session_id] = state.model_copy(deep=True)

    # Attempt SQLite persistence
    db = _get_db()
    if db is not None:
        try:
            state_json = state.model_dump_json()
            db.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, state_json, status, updated_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                (state.session_id, state_json, state.status.value),
            )
            db.commit()
        except Exception as e:
            logger.warning(
                "SQLite write failed for %s: %s (in-memory still valid)",
                state.session_id, e,
            )

    logger.info(
        "Saved session %s (status=%s)", state.session_id, state.status.value
    )


async def load_session(session_id: str) -> SessionState | None:
    """Load a session by ID.

    Tries in-memory first, then SQLite. Returns None if not found.
    """
    # Check in-memory cache first
    state = _sessions.get(session_id)
    if state is not None:
        return state.model_copy(deep=True)

    # Fall back to SQLite
    db = _get_db()
    if db is not None:
        try:
            cursor = db.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if row:
                state = SessionState.model_validate_json(row[0])
                _sessions[session_id] = state  # Warm the cache
                return state.model_copy(deep=True)
        except Exception as e:
            logger.warning("SQLite read failed for %s: %s", session_id, e)

    logger.warning("Session %s not found", session_id)
    return None


async def list_sessions(limit: int = 50) -> list[dict]:
    """List recent sessions with basic metadata.

    Returns summary dicts (not full state) for dashboard use.
    """
    db = _get_db()
    if db is not None:
        try:
            cursor = db.execute(
                """SELECT session_id, status, updated_at
                   FROM sessions
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (limit,),
            )
            return [
                {"session_id": r[0], "status": r[1], "updated_at": r[2]}
                for r in cursor.fetchall()
            ]
        except Exception as e:
            logger.warning("SQLite list failed: %s", e)

    # Fallback to in-memory
    return [
        {
            "session_id": sid,
            "status": s.status.value,
            "updated_at": s.updated_at.isoformat() if hasattr(s, "updated_at") else None,
        }
        for sid, s in list(_sessions.items())[:limit]
    ]


def clear_all() -> None:
    """Clear all sessions — in-memory and SQLite.

    Used by tests for clean state.
    """
    _sessions.clear()

    db = _get_db()
    if db is not None:
        try:
            db.execute("DELETE FROM sessions")
            db.commit()
        except Exception:
            pass
