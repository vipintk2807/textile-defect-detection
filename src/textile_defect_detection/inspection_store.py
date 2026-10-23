"""Local SQLite inspection history (metadata and predictions only - no image data).

All queries are parameterised. Every failure is converted to :class:`StorageError` so callers can
report "could not be saved" honestly instead of crashing or claiming success.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .errors import StorageError
from .predictions import ALL_STATUSES, STATUS_FAILED, InspectionResult
from .provenance import utc_now_iso

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1

COLUMNS = ("inspection_id", "created_at", "image_name", "image_reference", "model_id",
           "task_type", "status", "predicted_class", "top_score", "threshold",
           "processing_time_ms", "image_width", "image_height", "predictions_json",
           "error_message")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    image_name TEXT NOT NULL,
    image_reference TEXT,
    model_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    predicted_class TEXT,
    top_score REAL,
    threshold REAL,
    processing_time_ms REAL,
    image_width INTEGER,
    image_height INTEGER,
    predictions_json TEXT,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_inspections_created ON inspections(created_at);
CREATE INDEX IF NOT EXISTS idx_inspections_status ON inspections(status);
"""


@dataclass(frozen=True)
class HistoryFilter:
    date_from: str | None = None   # ISO date/time, inclusive
    date_to: str | None = None     # ISO date/time, inclusive (prefix comparison)
    predicted_class: str | None = None
    status: str | None = None
    limit: int = 500


class InspectionStore:
    """SQLite-backed history. The database file is created on first use."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self._initialise()

    # ------------------------------------------------------------------ plumbing
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5.0)

    def _initialise(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as conn, conn:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise StorageError(
                        "The inspection history database was created by a newer version of "
                        "this application.")
                conn.executescript(_CREATE_SQL)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except StorageError:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise StorageError("Inspection history is unavailable (the database could not be "
                               "opened).", detail=repr(exc)) from exc

    # ------------------------------------------------------------------ writes
    def add(self, result: InspectionResult, image_reference: str | None = None) -> None:
        row = (
            result.inspection_id, result.created_at, result.image_name, image_reference,
            result.model_id, result.task_type, result.status, result.prediction.predicted_class,
            result.prediction.score, result.threshold, result.processing_time_ms,
            result.image_width, result.image_height,
            json.dumps(result.prediction.to_dict(), ensure_ascii=False), None)
        self._insert(row)

    def add_failure(self, *, image_name: str, model_id: str, error_message: str,
                    image_reference: str | None = None) -> None:
        """Record a failed attempt; it is stored with status 'failed' and no prediction."""
        from .predictions import new_inspection_id  # noqa: PLC0415

        row = (new_inspection_id(), utc_now_iso(), image_name, image_reference, model_id,
               "classification", STATUS_FAILED, None, None, None, None, None, None, None,
               error_message[:500])
        self._insert(row)

    def _insert(self, row: tuple) -> None:
        placeholders = ",".join("?" for _ in COLUMNS)
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    f"INSERT INTO inspections ({','.join(COLUMNS)}) VALUES ({placeholders})", row)
        except sqlite3.Error as exc:
            raise StorageError("The inspection could not be saved to the history database.",
                               detail=repr(exc)) from exc

    # ------------------------------------------------------------------ reads
    def list(self, flt: HistoryFilter | None = None) -> list[dict]:
        flt = flt or HistoryFilter()
        if flt.status is not None and flt.status not in ALL_STATUSES:
            raise StorageError(f"Unknown status filter '{flt.status}'.")
        clauses, params = [], []
        if flt.date_from:
            clauses.append("created_at >= ?")
            params.append(flt.date_from)
        if flt.date_to:
            clauses.append("created_at <= ?")
            params.append(flt.date_to if "T" in flt.date_to else flt.date_to + "T23:59:59.999999Z")
        if flt.predicted_class:
            clauses.append("predicted_class = ?")
            params.append(flt.predicted_class)
        if flt.status:
            clauses.append("status = ?")
            params.append(flt.status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit = max(1, min(int(flt.limit), 100_000))
        sql = (f"SELECT {','.join(COLUMNS)} FROM inspections {where} "
               "ORDER BY created_at DESC, inspection_id LIMIT ?")
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(sql, (*params, limit)).fetchall()
        except sqlite3.Error as exc:
            raise StorageError("Inspection history could not be read.",
                               detail=repr(exc)) from exc
        return [dict(zip(COLUMNS, r)) for r in rows]

    def count(self) -> int:
        try:
            with closing(self._connect()) as conn:
                return int(conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0])
        except sqlite3.Error as exc:
            raise StorageError("Inspection history could not be read.",
                               detail=repr(exc)) from exc
