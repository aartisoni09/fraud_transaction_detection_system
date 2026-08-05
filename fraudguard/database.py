"""
fraudguard/database.py — SQLite Persistence Layer

Replaces in-memory SCORED/CASES/FEEDBACK stores with SQLite database.
Data survives server restarts. Uses Python's built-in sqlite3 module.
"""

import json
import sqlite3
import os
from contextlib import contextmanager
from typing import Optional

from fraudguard.config import get_settings
from fraudguard.logging_config import get_logger

logger = get_logger("fraudguard.database")


def _get_db_path() -> str:
    """Get the absolute path to the database file."""
    settings = get_settings()
    # Special case: in-memory database (for testing)
    if settings.database_url in (":memory:", ""):
        return ":memory:"
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isabs(settings.database_url):
        return settings.database_url
    return os.path.join(base, settings.database_url)


# Shared in-memory connection for testing
_shared_memory_conn: sqlite3.Connection | None = None


@contextmanager
def get_connection():
    """Context manager for database connections with auto-commit."""
    global _shared_memory_conn
    db_path = _get_db_path()

    if db_path == ":memory:":
        # Use a shared in-memory connection so all calls see the same data
        # check_same_thread=False is needed because FastAPI's TestClient
        # runs endpoint handlers in a different thread
        if _shared_memory_conn is None:
            _shared_memory_conn = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
            _shared_memory_conn.row_factory = sqlite3.Row
        try:
            yield _shared_memory_conn
            _shared_memory_conn.commit()
        except Exception:
            _shared_memory_conn.rollback()
            raise
        # Don't close — keep shared connection alive
    else:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db() -> None:
    """Initialize database tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scored_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                data TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                scored_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                data TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                priority TEXT NOT NULL DEFAULT 'high',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                actual_fraud INTEGER NOT NULL,
                analyst_id TEXT NOT NULL,
                notes TEXT,
                submitted_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_scored_risk ON scored_transactions(risk_level);
            CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
            CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases(priority);
        """)
    logger.info("Database initialized at: %s", _get_db_path())


# ── Scored Transactions ────────────────────────────────────────

def save_scored(record: dict) -> None:
    """Save a scored transaction to the database."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO scored_transactions (transaction_id, data, fraud_probability, risk_level, scored_at) VALUES (?, ?, ?, ?, ?)",
            (record["transaction_id"], json.dumps(record), record["fraud_probability"], record["risk_level"], record.get("scored_at", "")),
        )


def save_scored_batch(records: list[dict]) -> None:
    """Save multiple scored transactions in a single transaction."""
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO scored_transactions (transaction_id, data, fraud_probability, risk_level, scored_at) VALUES (?, ?, ?, ?, ?)",
            [(r["transaction_id"], json.dumps(r), r["fraud_probability"], r["risk_level"], r.get("scored_at", "")) for r in records],
        )


def list_scored(skip: int = 0, limit: int = 50) -> tuple[int, list[dict]]:
    """List scored transactions with pagination."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM scored_transactions").fetchone()[0]
        rows = conn.execute(
            "SELECT data FROM scored_transactions ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, skip),
        ).fetchall()
    return total, [json.loads(r["data"]) for r in rows]


def get_recent_scored(limit: int = 100) -> list[dict]:
    """Get the most recently scored transactions."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT data FROM scored_transactions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [json.loads(r["data"]) for r in rows]


def clear_scored() -> None:
    """Clear all scored transactions."""
    with get_connection() as conn:
        conn.execute("DELETE FROM scored_transactions")


def count_scored() -> int:
    """Get total count of scored transactions."""
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM scored_transactions").fetchone()[0]


def get_all_scored() -> list[dict]:
    """Get ALL scored transactions (for analytics). Use with caution for large datasets."""
    with get_connection() as conn:
        rows = conn.execute("SELECT data FROM scored_transactions ORDER BY id DESC").fetchall()
    return [json.loads(r["data"]) for r in rows]


# ── Cases ──────────────────────────────────────────────────────

def save_case(case: dict) -> None:
    """Save a fraud case to the database."""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cases (case_id, transaction_id, data, status, priority, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (case["case_id"], case["transaction_id"], json.dumps(case), case["status"], case["priority"], case["created_at"], case["updated_at"]),
        )


def list_cases(status: Optional[str] = None, priority: Optional[str] = None) -> list[dict]:
    """List cases with optional filtering by status and priority."""
    query = "SELECT data FROM cases WHERE 1=1"
    params: list = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if priority:
        query += " AND priority = ?"
        params.append(priority)
    query += " ORDER BY created_at DESC"
    
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [json.loads(r["data"]) for r in rows]


def get_case(case_id: str) -> Optional[dict]:
    """Get a single case by ID."""
    with get_connection() as conn:
        row = conn.execute("SELECT data FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    return json.loads(row["data"]) if row else None


def update_case(case_id: str, updates: dict) -> Optional[dict]:
    """Update a case and return the updated data."""
    existing = get_case(case_id)
    if not existing:
        return None
    existing.update(updates)
    with get_connection() as conn:
        conn.execute(
            "UPDATE cases SET data = ?, status = ?, updated_at = ? WHERE case_id = ?",
            (json.dumps(existing), existing.get("status", "open"), existing.get("updated_at", ""), case_id),
        )
    return existing


def clear_cases() -> None:
    """Clear all cases."""
    with get_connection() as conn:
        conn.execute("DELETE FROM cases")


def count_cases(status: Optional[str] = None) -> int:
    """Count cases, optionally filtered by status."""
    with get_connection() as conn:
        if status:
            return conn.execute("SELECT COUNT(*) FROM cases WHERE status = ?", (status,)).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]


# ── Feedback ───────────────────────────────────────────────────

def save_feedback(entry: dict) -> None:
    """Save analyst feedback to the database."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO feedback (id, transaction_id, actual_fraud, analyst_id, notes, submitted_at) VALUES (?, ?, ?, ?, ?, ?)",
            (entry["id"], entry["transaction_id"], entry["actual_fraud"], entry["analyst_id"], entry.get("notes"), entry["submitted_at"]),
        )


def list_feedback(skip: int = 0, limit: int = 50) -> tuple[int, list[dict]]:
    """List feedback with pagination."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY submitted_at DESC LIMIT ? OFFSET ?",
            (limit, skip),
        ).fetchall()
    return total, [dict(r) for r in rows]


def count_feedback() -> int:
    """Count total feedback entries."""
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
