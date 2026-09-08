"""
Treatment follow-up tracking (module 3/webapp/cases_db.py). Uses a
temporary DB path per test (monkeypatched) so these never touch the real
artifacts/webapp_cases.sqlite3 the running webapp uses.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "module 3" / "webapp"))
import cases_db  # noqa: E402


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cases_db, "DB_PATH", tmp_path / "test_cases.sqlite3")
    cases_db.init_db()
    yield


def test_create_and_get_case():
    cid = cases_db.create_case("rose_black_spot", common_name="Rose black spot",
                               species="rose", treatment_answer="remove infected leaves",
                               language="en")
    assert cid is not None
    case = cases_db.get_case(cid)
    assert case["disease"] == "rose_black_spot"
    assert case["common_name"] == "Rose black spot"
    assert case["followup_result"] is None
    assert case["created_at"]


def test_get_unknown_case_returns_none():
    assert cases_db.get_case(99999) is None


def test_list_cases_most_recent_first():
    id1 = cases_db.create_case("rose_black_spot")
    id2 = cases_db.create_case("tomato_late_blight")
    cases = cases_db.list_cases()
    assert cases[0]["id"] == id2
    assert cases[1]["id"] == id1


def test_list_cases_respects_limit():
    for i in range(5):
        cases_db.create_case(f"disease_{i}")
    assert len(cases_db.list_cases(limit=3)) == 3


def test_record_followup_updates_case():
    cid = cases_db.create_case("rose_black_spot")
    ok = cases_db.record_followup(cid, "better", notes="looking much healthier")
    assert ok is True
    case = cases_db.get_case(cid)
    assert case["followup_result"] == "better"
    assert case["followup_notes"] == "looking much healthier"
    assert case["followup_at"]


def test_record_followup_rejects_invalid_result():
    cid = cases_db.create_case("rose_black_spot")
    with pytest.raises(ValueError):
        cases_db.record_followup(cid, "excellent")


def test_record_followup_unknown_case_returns_false():
    assert cases_db.record_followup(99999, "better") is False
