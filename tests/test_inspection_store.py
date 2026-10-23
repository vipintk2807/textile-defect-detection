import sqlite3

import pytest

from helpers import image_bytes, make_engine, one_hot_predictor
from textile_defect_detection.errors import StorageError
from textile_defect_detection.inspection_store import HistoryFilter, InspectionStore


def make_result(index=1, name="a.png", threshold=0.5):
    return make_engine(one_hot_predictor(index), threshold=threshold).inspect_bytes(
        image_bytes(), name)


def test_creates_database_and_directories(tmp_path):
    store = InspectionStore(tmp_path / "deep" / "nested" / "h.db")
    assert store.count() == 0 and (tmp_path / "deep" / "nested" / "h.db").exists()


def test_add_and_retrieve_matches_result(tmp_path):
    store = InspectionStore(tmp_path / "h.db")
    result = make_result()
    store.add(result)
    (row,) = store.list()
    assert row["inspection_id"] == result.inspection_id and row["image_name"] == "a.png"
    assert row["status"] == result.status and row["model_id"] == "test-model"
    assert row["image_width"] == 64 and row["threshold"] == 0.5


def test_filters(tmp_path):
    store = InspectionStore(tmp_path / "h.db")
    store.add(make_result(0))
    store.add(make_result(1))
    store.add(make_result(1, threshold=1.1 - 0.0))  # below threshold -> low confidence
    assert len(store.list(HistoryFilter(predicted_class="beta"))) == 2
    assert len(store.list(HistoryFilter(predicted_class="alpha"))) == 1
    assert len(store.list(HistoryFilter(status="completed_low_confidence"))) == 1
    assert len(store.list(HistoryFilter(date_from="2999-01-01"))) == 0
    assert len(store.list(HistoryFilter(date_to="2999-01-01"))) == 3
    assert len(store.list(HistoryFilter(limit=2))) == 2


def test_sql_injection_attempt_is_inert(tmp_path):
    store = InspectionStore(tmp_path / "h.db")
    store.add(make_result())
    assert store.list(HistoryFilter(predicted_class="' OR 1=1 --")) == []
    store.add(make_result(name="x'); DROP TABLE inspections;--.png"))
    assert store.count() == 2


def test_unknown_status_filter_rejected(tmp_path):
    with pytest.raises(StorageError):
        InspectionStore(tmp_path / "h.db").list(HistoryFilter(status="bogus"))


def test_failure_record_has_no_prediction(tmp_path):
    store = InspectionStore(tmp_path / "h.db")
    store.add_failure(image_name="bad.png", model_id="m", error_message="could not decode")
    (row,) = store.list()
    assert row["status"] == "failed" and row["predictions_json"] is None
    assert row["error_message"] == "could not decode"


def test_duplicate_primary_key_is_reported_as_storage_error(tmp_path):
    store = InspectionStore(tmp_path / "h.db")
    result = make_result()
    store.add(result)
    with pytest.raises(StorageError, match="could not be saved"):
        store.add(result)
    assert store.count() == 1


def test_corrupt_database_file_is_handled(tmp_path):
    path = tmp_path / "h.db"
    path.write_bytes(b"this is definitely not an sqlite database" * 50)
    with pytest.raises(StorageError):
        InspectionStore(path)


def test_unwritable_location_is_handled(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not a folder")
    with pytest.raises(StorageError):
        InspectionStore(blocker / "h.db")


def test_schema_has_documented_columns(tmp_path):
    InspectionStore(tmp_path / "h.db")
    with sqlite3.connect(tmp_path / "h.db") as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(inspections)")}
    for needed in ("inspection_id", "created_at", "image_name", "model_id", "task_type",
                   "status", "threshold", "processing_time_ms", "predictions_json",
                   "error_message"):
        assert needed in cols
