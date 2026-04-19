"""
wiki_engine/db.py
=================
SQLite data-access layer for the Sentinel-LAC intelligence wiki.

Provides clean query interfaces over data/llm_wiki.db.
Used by dashboard.py for DB-backed analysis (faster than wiki-scan
for structured queries like "all boot-print reports in May").

All methods are synchronous (SQLite is fast enough; no async needed).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("data/llm_wiki.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ═══════════════════════════════════════════════════════════════════════════════
# Patrol reports
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_reports() -> list[dict[str, Any]]:
    """Return all patrol reports as list of dicts, ordered by date."""
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM patrol_reports ORDER BY patrol_date"
        ).fetchall()
    return [dict(r) for r in rows]


def get_report(report_id: str) -> dict[str, Any] | None:
    if not DB_PATH.exists():
        return None
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM patrol_reports WHERE report_id = ?",
            (report_id.upper(),)
        ).fetchone()
    return dict(row) if row else None


def get_reports_by_sector(sector_pattern: str) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM patrol_reports WHERE sector LIKE ? ORDER BY patrol_date",
            (f"%{sector_pattern}%",)
        ).fetchall()
    return [dict(r) for r in rows]


def get_reports_by_depth_cat(category: str) -> list[dict[str, Any]]:
    """category: 'Shallow' | 'Moderate' | 'Deep'"""
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM patrol_reports WHERE depth_cat = ? ORDER BY depth_m DESC",
            (category,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_deepest_reports(n: int = 3) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM patrol_reports ORDER BY depth_m DESC LIMIT ?",
            (n,)
        ).fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# TTS (Tell-Tale Signs)
# ═══════════════════════════════════════════════════════════════════════════════

def get_tts_for_report(report_id: str) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tts_items WHERE report_id = ?",
            (report_id.upper(),)
        ).fetchall()
    return [dict(r) for r in rows]


def get_tts_by_category(category: str) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            """SELECT t.*, r.sector, r.patrol_period
               FROM tts_items t
               JOIN patrol_reports r ON t.report_id = r.report_id
               WHERE t.category = ?
               ORDER BY r.patrol_date""",
            (category,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_tts_summary() -> list[dict[str, Any]]:
    """Aggregated TTS count per category across all reports."""
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            """SELECT category, COUNT(*) as count,
               GROUP_CONCAT(DISTINCT report_id) as reports
               FROM tts_items
               GROUP BY category
               ORDER BY count DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_tts() -> list[dict[str, Any]]:
    """All TTS items with report context."""
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            """SELECT t.report_id, t.item_text, t.category,
               r.sector, r.patrol_period, r.depth_cat
               FROM tts_items t
               JOIN patrol_reports r ON t.report_id = r.report_id
               ORDER BY r.patrol_date"""
        ).fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# Personnel
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_personnel() -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM personnel ORDER BY report_count DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_officer_reports(officer_name: str) -> list[dict[str, Any]]:
    """Return all patrol reports signed by this officer."""
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            """SELECT r.* FROM patrol_reports r
               WHERE r.signed_by LIKE ? OR r.officer_name LIKE ?
               ORDER BY r.patrol_date""",
            (f"%{officer_name}%", f"%{officer_name}%")
        ).fetchall()
    return [dict(r) for r in rows]


def get_personnel_by_report(report_id: str) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            """SELECT p.* FROM personnel p
               JOIN personnel_reports pr ON p.id = pr.personnel_id
               WHERE pr.report_id = ?""",
            (report_id.upper(),)
        ).fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# Grids
# ═══════════════════════════════════════════════════════════════════════════════

def get_grids_for_report(report_id: str) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM grids WHERE report_id = ?",
            (report_id.upper(),)
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_grids() -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            """SELECT g.*, r.sector, r.patrol_period
               FROM grids g
               JOIN patrol_reports r ON g.report_id = r.report_id"""
        ).fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# Issues
# ═══════════════════════════════════════════════════════════════════════════════

def get_issues_for_report(report_id: str) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM issues WHERE report_id = ?",
            (report_id.upper(),)
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_issues() -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            """SELECT i.*, r.sector, r.patrol_period, r.depth_cat
               FROM issues i
               JOIN patrol_reports r ON i.report_id = r.report_id
               ORDER BY r.patrol_date"""
        ).fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# Query log
# ═══════════════════════════════════════════════════════════════════════════════

def log_query(question: str, answer: str, mode: str,
              score: float, sources: list[str]) -> int:
    """Log a query+answer to DB. Returns row id."""
    if not DB_PATH.exists():
        DB_PATH.parent.mkdir(exist_ok=True)
    import json as _json
    from datetime import datetime, timezone
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO query_log
               (question, answer, mode, score, sources, asked_at)
               VALUES (?,?,?,?,?,?)""",
            (question, answer, mode, score,
             _json.dumps(sources),
             datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        return cur.lastrowid


def approve_query(query_id: int) -> None:
    if not DB_PATH.exists():
        return
    with _conn() as conn:
        conn.execute(
            "UPDATE query_log SET approved = 1 WHERE id = ?",
            (query_id,)
        )
        conn.commit()


def commit_analysis(question: str, answer: str,
                    sources: list[str], wiki_path: str) -> None:
    if not DB_PATH.exists():
        return
    import json as _json
    from datetime import datetime, timezone
    with _conn() as conn:
        conn.execute(
            """INSERT INTO approved_analyses
               (question, answer, sources, approved_at, wiki_path)
               VALUES (?,?,?,?,?)""",
            (question, answer, _json.dumps(sources),
             datetime.now(timezone.utc).isoformat(), wiki_path)
        )
        conn.commit()


def get_recent_queries(n: int = 20) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM query_log ORDER BY asked_at DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════════

def get_stats() -> dict[str, Any]:
    """High-level corpus statistics."""
    if not DB_PATH.exists():
        return {}
    with _conn() as conn:
        reports  = conn.execute("SELECT COUNT(*) FROM patrol_reports").fetchone()[0]
        tts      = conn.execute("SELECT COUNT(*) FROM tts_items").fetchone()[0]
        personnel= conn.execute("SELECT COUNT(*) FROM personnel").fetchone()[0]
        grids    = conn.execute("SELECT COUNT(*) FROM grids").fetchone()[0]
        issues   = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
        queries  = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        approved = conn.execute("SELECT COUNT(*) FROM query_log WHERE approved=1").fetchone()[0]
        max_depth= conn.execute("SELECT MAX(depth_m) FROM patrol_reports").fetchone()[0] or 0
        drone    = conn.execute(
            "SELECT COUNT(DISTINCT report_id) FROM tts_items WHERE category = 'Drone Activity'"
        ).fetchone()[0]
    return {
        "reports":       reports,
        "tts_indicators":tts,
        "personnel":     personnel,
        "grids":         grids,
        "issues":        issues,
        "queries_asked": queries,
        "queries_approved": approved,
        "max_depth_m":   max_depth,
        "drone_reports": drone,
    }
