"""
database.py
-----------
SQLite-based prediction logging for MLOps monitoring.
Creates and manages the predictions table.
"""

from __future__ import annotations

import sqlite3
import datetime
from pathlib import Path
from typing import Optional

ROOT  = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "predictions.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    url             TEXT    NOT NULL,
    prediction      TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    risk_level      TEXT    NOT NULL,
    threat_score    INTEGER NOT NULL,
    is_phishing     INTEGER NOT NULL,
    model_version   TEXT    DEFAULT 'best_phishing_model'
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(CREATE_TABLE)
    conn.commit()
    return conn


def log_prediction(
    url:           str,
    prediction:    str,
    confidence:    float,
    risk_level:    str,
    threat_score:  int,
    is_phishing:   bool,
    model_version: str = "best_phishing_model",
) -> int:
    """
    Insert a prediction record and return its row id.
    """
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (timestamp, url, prediction, confidence, risk_level, threat_score, is_phishing, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, url, prediction, confidence, risk_level, threat_score, int(is_phishing), model_version),
        )
        return cur.lastrowid


def get_recent_predictions(limit: int = 50) -> list[dict]:
    """Return the most recent *limit* predictions as a list of dicts."""
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_stats() -> dict:
    """Return aggregate statistics for the dashboard."""
    with _get_conn() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        phishing = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_phishing=1").fetchone()[0]
        avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0
    return {
        "total_analyzed":   total,
        "threats_detected": phishing,
        "avg_confidence":   round(avg_conf * 100, 1),
        "safe_urls":        total - phishing,
    }
