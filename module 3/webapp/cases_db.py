"""
FLORA :: Module 3 extension -- treatment follow-up tracking.

Module 4 gives treatment advice and then the conversation ends -- no record
of whether it worked. This closes that loop with a tiny SQLite-backed log:
a farmer logs a case after getting advice, then comes back over time to
leave comments, record how the plant is doing, or ask a further question
about the same case (answered live via the Module 4 RAG pipeline). Every
one of those is stored as a timestamped entry in `case_updates` so a case
reads as a timeline, not a single write-once verdict.

Uses only the standard library (sqlite3) -- no new dependency for what is
fundamentally a small structured log.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "artifacts" / "webapp_cases.sqlite3"

SCHEMA_CASES = """
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

SCHEMA_UPDATES = """
CREATE TABLE IF NOT EXISTS case_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL,
    kind TEXT NOT NULL,       -- 'status' | 'comment' | 'question'
    text TEXT,                -- notes (status), comment body, or the question asked
    answer TEXT,              -- generated RAG answer, only set for kind='question'
    result TEXT,              -- 'better' | 'same' | 'worse', only set for kind='status'
    created_at TEXT NOT NULL,
    FOREIGN KEY(case_id) REFERENCES treatment_cases(id)
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
        conn.execute(SCHEMA_CASES)
        conn.execute(SCHEMA_UPDATES)
        _migrate_legacy_followups(conn)


def _migrate_legacy_followups(conn):
    """Cases logged before case_updates existed carry their one followup
    directly on the row. Backfill it as a timeline entry (once) so old and
    new cases render the same way."""
    legacy = conn.execute(
        "SELECT id, followup_result, followup_notes, followup_at FROM treatment_cases "
        "WHERE followup_result IS NOT NULL"
    ).fetchall()
    for row in legacy:
        exists = conn.execute(
            "SELECT 1 FROM case_updates WHERE case_id = ? AND kind = 'status' LIMIT 1",
            (row["id"],),
        ).fetchone()
        if exists:
            continue
        conn.execute(
            "INSERT INTO case_updates (case_id, kind, text, result, created_at) "
            "VALUES (?, 'status', ?, ?, ?)",
            (row["id"], row["followup_notes"], row["followup_result"], row["followup_at"]),
        )


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


def list_updates(case_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM case_updates WHERE case_id = ? ORDER BY id ASC", (case_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def _attach_updates(conn, case_row):
    case = dict(case_row)
    updates = conn.execute(
        "SELECT * FROM case_updates WHERE case_id = ? ORDER BY id ASC", (case["id"],)
    ).fetchall()
    case["updates"] = [dict(u) for u in updates]
    case["last_activity_at"] = max(
        [case["created_at"]] + [u["created_at"] for u in case["updates"]]
    )
    return case


def list_cases(limit=200):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM treatment_cases ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        cases = [_attach_updates(conn, r) for r in rows]
        cases.sort(key=lambda c: c["last_activity_at"], reverse=True)
        return cases


def get_case(case_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM treatment_cases WHERE id = ?", (case_id,)
        ).fetchone()
        return _attach_updates(conn, row) if row else None


def record_followup(case_id, result, notes=None):
    """Logs a status update ('better'/'same'/'worse') as a new timeline
    entry and refreshes the cached latest-status columns on the case row.
    Safe to call more than once per case -- each call is a new check-in."""
    if result not in VALID_RESULTS:
        raise ValueError(f"result must be one of {VALID_RESULTS}, got {result!r}")
    with _connect() as conn:
        case = conn.execute("SELECT id FROM treatment_cases WHERE id = ?", (case_id,)).fetchone()
        if not case:
            return False
        now = _now()
        conn.execute(
            "INSERT INTO case_updates (case_id, kind, text, result, created_at) "
            "VALUES (?, 'status', ?, ?, ?)",
            (case_id, notes, result, now),
        )
        conn.execute(
            "UPDATE treatment_cases SET followup_result = ?, followup_notes = ?, "
            "followup_at = ? WHERE id = ?",
            (result, notes, now, case_id),
        )
        return True


def add_comment(case_id, text):
    text = (text or "").strip()
    if not text:
        return None
    with _connect() as conn:
        case = conn.execute("SELECT id FROM treatment_cases WHERE id = ?", (case_id,)).fetchone()
        if not case:
            return None
        cur = conn.execute(
            "INSERT INTO case_updates (case_id, kind, text, created_at) VALUES (?, 'comment', ?, ?)",
            (case_id, text, _now()),
        )
        return cur.lastrowid


def add_question(case_id, question, answer):
    question = (question or "").strip()
    if not question:
        return None
    with _connect() as conn:
        case = conn.execute("SELECT id FROM treatment_cases WHERE id = ?", (case_id,)).fetchone()
        if not case:
            return None
        cur = conn.execute(
            "INSERT INTO case_updates (case_id, kind, text, answer, created_at) "
            "VALUES (?, 'question', ?, ?, ?)",
            (case_id, question, answer, _now()),
        )
        return cur.lastrowid
