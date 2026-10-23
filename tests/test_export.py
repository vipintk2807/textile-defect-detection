import csv
import io
import json

import pytest

from helpers import image_bytes, make_engine, one_hot_predictor
from textile_defect_detection.errors import StorageError
from textile_defect_detection.export import (export_records, records_to_csv_bytes,
                                             records_to_json_bytes)
from textile_defect_detection.inspection_store import COLUMNS, InspectionStore


def stored_records(tmp_path, names=("a.png",)):
    store = InspectionStore(tmp_path / "h.db")
    for n in names:
        store.add(make_engine(one_hot_predictor(2)).inspect_bytes(image_bytes(), n))
    return store.list()


def test_csv_contains_exactly_the_stored_values(tmp_path):
    records = stored_records(tmp_path)
    rows = list(csv.DictReader(io.StringIO(records_to_csv_bytes(records).decode("utf-8-sig"))))
    assert list(rows[0]) == list(COLUMNS)
    assert rows[0]["inspection_id"] == records[0]["inspection_id"]
    assert rows[0]["predicted_class"] == "gamma"
    assert float(rows[0]["top_score"]) == records[0]["top_score"]


def test_json_export_nests_predictions_and_states_they_are_model_outputs(tmp_path):
    records = stored_records(tmp_path)
    data = json.loads(records_to_json_bytes(records))
    assert "not human-confirmed" in data["note"]
    assert data["records"][0]["predictions"]["predicted_class"] == "gamma"
    assert "predictions_json" not in data["records"][0]


def test_csv_formula_injection_is_neutralised(tmp_path):
    records = stored_records(tmp_path, names=("=HYPERLINK(1).png", "+cmd.png", "@x.png"))
    text = records_to_csv_bytes(records).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert all(r["image_name"][0] == "'" for r in rows)


def test_export_writes_new_files_and_never_overwrites(tmp_path):
    records = stored_records(tmp_path)
    first = export_records(records, tmp_path / "out", "csv")
    second = export_records(records, tmp_path / "out", "csv")
    assert first != second and first.exists() and second.exists()
    assert export_records(records, tmp_path / "out", "json").suffix == ".json"


def test_export_failure_is_reported_not_swallowed(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    with pytest.raises(StorageError, match="not completed"):
        export_records([], blocker / "sub", "csv")
    with pytest.raises(StorageError):
        export_records([], tmp_path / "o", "xml")


def test_empty_export_is_valid(tmp_path):
    assert records_to_csv_bytes([]).decode("utf-8-sig").strip() == ",".join(COLUMNS)
    assert json.loads(records_to_json_bytes([]))["records"] == []
