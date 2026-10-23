"""Export stored inspection records to CSV or JSON without altering their content."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from .errors import StorageError
from .inspection_store import COLUMNS

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value):
    """Neutralise spreadsheet formula injection for text cells (e.g. a file named '=cmd...')."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def records_to_csv_bytes(records: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(COLUMNS)
    for record in records:
        writer.writerow([_csv_safe(record.get(col)) for col in COLUMNS])
    return buffer.getvalue().encode("utf-8-sig")  # BOM so Excel on Windows reads UTF-8


def records_to_json_bytes(records: list[dict]) -> bytes:
    exported = []
    for record in records:
        item = {col: record.get(col) for col in COLUMNS}
        if item.get("predictions_json"):
            try:
                item["predictions"] = json.loads(item.pop("predictions_json"))
            except ValueError:
                item["predictions"] = None
        else:
            item.pop("predictions_json", None)
            item["predictions"] = None
        exported.append(item)
    return (json.dumps({"schema": "textile-defect-detection inspections v1",
                        "note": "Model predictions, not human-confirmed inspection results.",
                        "records": exported}, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def export_records(records: list[dict], directory: str | Path, fmt: str = "csv") -> Path:
    """Write records to a new timestamped file (existing files are never overwritten)."""
    if fmt not in {"csv", "json"}:
        raise StorageError("Export format must be 'csv' or 'json'.")
    payload = records_to_csv_bytes(records) if fmt == "csv" else records_to_json_bytes(records)
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = directory / f"inspections-{stamp}.{fmt}"
        counter = 1
        while path.exists():
            counter += 1
            path = directory / f"inspections-{stamp}-{counter}.{fmt}"
        path.write_bytes(payload)
    except OSError as exc:
        raise StorageError("The export was not completed (the file could not be written).",
                           detail=repr(exc)) from exc
    return path
