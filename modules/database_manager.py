"""
Database Manager – SQLite persistence layer.
Handles all create/read/update/delete operations.
"""

import sqlite3
import pandas as pd
import json
import os
from config import DB_PATH, DATABASE_DIR


def _get_conn():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = _get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS applicants (
            applicant_id    TEXT PRIMARY KEY,
            full_name       TEXT,
            dob             TEXT,
            email           TEXT,
            phone           TEXT,
            address         TEXT,
            city            TEXT,
            state           TEXT,
            pincode         TEXT,
            id_ref          TEXT,
            device_id       TEXT,
            application_date TEXT,
            employer        TEXT,
            employment_type TEXT,
            annual_income   REAL,
            application_status TEXT DEFAULT 'New',
            ground_truth_label TEXT DEFAULT 'genuine',
            applicant_type  TEXT DEFAULT 'normal',
            risk_score      REAL DEFAULT 0,
            risk_category   TEXT DEFAULT 'Low',
            risk_flags      TEXT DEFAULT '[]',
            risk_explanation TEXT DEFAULT '',
            analyst_note    TEXT DEFAULT '',
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS investigation_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id    TEXT,
            action          TEXT,
            analyst_note    TEXT,
            previous_status TEXT,
            new_status      TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS fraud_rings (
            ring_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ring_label      TEXT,
            member_ids      TEXT,
            shared_attrs    TEXT,
            risk_level      TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ── Applicant CRUD ─────────────────────────────────────────────────────────

DB_COLS = {
    "applicant_id", "full_name", "dob", "email", "phone", "address",
    "city", "state", "pincode", "id_ref", "device_id", "application_date",
    "employer", "employment_type", "annual_income", "application_status",
    "ground_truth_label", "applicant_type", "risk_score", "risk_category",
    "risk_flags", "risk_explanation", "analyst_note",
}


def insert_applicants(df: pd.DataFrame):
    """Bulk insert applicants (replace on conflict for idempotency)."""
    conn = _get_conn()
    for _, row in df.iterrows():
        row_dict = {k: v for k, v in row.to_dict().items() if k in DB_COLS}
        for key in ("risk_flags",):
            val = row_dict.get(key, [])
            if isinstance(val, list):
                row_dict[key] = json.dumps(val)
        cols = ", ".join(row_dict.keys())
        placeholders = ", ".join(["?"] * len(row_dict))
        values = list(row_dict.values())
        conn.execute(
            f"INSERT OR REPLACE INTO applicants ({cols}) VALUES ({placeholders})",
            values
        )
    conn.commit()
    conn.close()


def get_all_applicants() -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql_query("SELECT * FROM applicants ORDER BY created_at DESC", conn)
    conn.close()
    if not df.empty and "risk_flags" in df.columns:
        df["risk_flags"] = df["risk_flags"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else (x or [])
        )
    return df


def get_applicant_by_id(applicant_id: str) -> pd.Series | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM applicants WHERE applicant_id = ?", (applicant_id,)
    ).fetchone()
    conn.close()
    if row:
        d = dict(row)
        if "risk_flags" in d and isinstance(d["risk_flags"], str):
            d["risk_flags"] = json.loads(d["risk_flags"])
        return pd.Series(d)
    return None


def update_applicant_risk(applicant_id: str, risk_score: float,
                           risk_category: str, risk_flags: list,
                           risk_explanation: str):
    conn = _get_conn()
    conn.execute("""
        UPDATE applicants
        SET risk_score=?, risk_category=?, risk_flags=?, risk_explanation=?
        WHERE applicant_id=?
    """, (risk_score, risk_category, json.dumps(risk_flags),
          risk_explanation, applicant_id))
    conn.commit()
    conn.close()


def update_applicant_status(applicant_id: str, new_status: str,
                             analyst_note: str = ""):
    conn = _get_conn()
    old_row = conn.execute(
        "SELECT application_status FROM applicants WHERE applicant_id=?",
        (applicant_id,)
    ).fetchone()
    old_status = old_row["application_status"] if old_row else "Unknown"

    conn.execute("""
        UPDATE applicants
        SET application_status=?, analyst_note=?
        WHERE applicant_id=?
    """, (new_status, analyst_note, applicant_id))

    conn.execute("""
        INSERT INTO investigation_log
            (applicant_id, action, analyst_note, previous_status, new_status)
        VALUES (?, ?, ?, ?, ?)
    """, (applicant_id, "status_update", analyst_note, old_status, new_status))

    conn.commit()
    conn.close()


def upsert_applicant(row_dict: dict):
    """Insert a single applicant row or replace on conflict."""
    df = pd.DataFrame([row_dict])
    insert_applicants(df)


def delete_all_applicants():
    conn = _get_conn()
    conn.execute("DELETE FROM applicants")
    conn.execute("DELETE FROM fraud_rings")
    conn.commit()
    conn.close()


def get_total_count() -> int:
    conn = _get_conn()
    n = conn.execute("SELECT COUNT(*) FROM applicants").fetchone()[0]
    conn.close()
    return n


# ── Investigation Log ──────────────────────────────────────────────────────

def get_investigation_log() -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM investigation_log ORDER BY created_at DESC", conn
    )
    conn.close()
    return df


# ── Fraud Rings ────────────────────────────────────────────────────────────

def save_fraud_rings(rings: list):
    """Persist detected fraud rings."""
    conn = _get_conn()
    conn.execute("DELETE FROM fraud_rings")
    for ring in rings:
        conn.execute("""
            INSERT INTO fraud_rings (ring_label, member_ids, shared_attrs, risk_level)
            VALUES (?, ?, ?, ?)
        """, (
            ring.get("ring_label", ""),
            json.dumps(ring.get("member_ids", [])),
            json.dumps(ring.get("shared_attrs", {})),
            ring.get("risk_level", "Medium"),
        ))
    conn.commit()
    conn.close()


def get_fraud_rings() -> list:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM fraud_rings ORDER BY ring_id").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["member_ids"] = json.loads(d.get("member_ids", "[]"))
        d["shared_attrs"] = json.loads(d.get("shared_attrs", "{}"))
        result.append(d)
    return result
