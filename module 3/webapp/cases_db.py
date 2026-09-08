"""
FLORA :: Module 4 extension -- treatment follow-up tracking.

Module 4 gives treatment advice and then the conversation ends -- no record
of whether it worked. This closes that loop with a tiny SQLite-backed log:
a farmer logs a case after getting advice, comes back later, and records
whether the plant got better, stayed the same, or got worse. No ML here,
just a durable record -- the follow-up judgement is the farmer's own.

Uses only the standard library (sqlite3) -- no new dependency for what is
fundamentally a small structured log.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "artifacts" / "webapp_cases.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS treatment_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT NOT NULL,
    common_name TEXT,
    species TEXT,
    treatment_answer TEXT,
    language TEXT DEFAULT 'en',
    created_at TEXT NOT NULL,
    followup_result TEXT,
    followup_notes TEXT,
    followup_at TEXT
);
"""

VALID_RESULTS = {"better", "same", "worse"}


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute(SCHEMA)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_case(disease, common_name=None, species=None, treatment_answer=None, language="en"):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO treatment_cases "
            "(disease, common_name, species, treatment_answer, language, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (disease, common_name, species, treatment_answer, language, _now()),
        )
        return cur.lastrowid


def list_cases(limit=100):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM treatment_cases ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_case(case_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM treatment_cases WHERE id = ?", (case_id,)
        ).fetchone()
        return dict(row) if row else None


def record_followup(case_id, result, notes=None):
    if result not in VALID_RESULTS:
        raise ValueError(f"result must be one of {VALID_RESULTS}, got {result!r}")
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE treatment_cases SET followup_result = ?, followup_notes = ?, "
            "followup_at = ? WHERE id = ?",
            (result, notes, _now(), case_id),
        )
        return cur.rowcount > 0
