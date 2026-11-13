import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil

import pytest

from ci_agents.dataset import DatasetValidationError, load_dataset

PUBLIC = Path(__file__).resolve().parents[1] / "data/public"


@pytest.fixture
def corpus(tmp_path):
    for name in ["public_web_records.jsonl", "pricing_events.csv", "public_questions.csv"]:
        shutil.copyfile(PUBLIC / name, tmp_path / name)
    return tmp_path


def change_record(directory, **changes):
    path = directory / "public_web_records.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0].update(changes)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_supplied_benchmark():
    dataset = load_dataset(PUBLIC)
    assert len(dataset.records) == 310
    assert len({r["company_name"] for r in dataset.records}) == 5
    assert len(dataset.questions) == 15
    assert len(dataset.pricing_events) == 77
    record = dataset.records[0]
    assert record["published_at"] != record["available_at"]
    assert record["effective_date"] == datetime(2025, 1, 6, tzinfo=timezone.utc)
    assert dataset.pricing_events[0]["effective_date"] == record["effective_date"]
    assert dataset.pricing_events[0]["price_usd"] == Decimal("79.0")
    for row in dataset.records:
        for field in ["published_at", "available_at", "effective_date", "end_date"]:
            assert row[field] is None or row[field].tzinfo == timezone.utc


def test_duplicate_record_id(corpus):
    rows = load_dataset(corpus).records
    change_record(corpus, record_id=rows[1]["record_id"])
    with pytest.raises(DatasetValidationError, match="duplicate record_id"):
        load_dataset(corpus)


@pytest.mark.parametrize("field,value", [
    ("published_at", "2026-02-30T00:00:00Z"),
    ("available_at", "2026-11-11T00:00:00"),
    ("available_at", None),
    ("effective_date", "2026-02-30"),
    ("end_date", "not-a-date"),
])
def test_invalid_record_dates(corpus, field, value):
    change_record(corpus, **{field: value})
    with pytest.raises(DatasetValidationError, match=field):
        load_dataset(corpus)


def test_unknowns_remain_unknown_and_known_false_zero_survive(corpus):
    change_record(corpus, metric_value=None, onboarding_fee_usd="not_specified",
                  price_includes_ai="", seat_minimum=None, end_date=None)
    row = load_dataset(corpus).records[0]
    for field in ["metric_value", "onboarding_fee_usd", "price_includes_ai", "seat_minimum", "end_date"]:
        assert row[field] is None
    change_record(corpus, price_includes_ai="false", onboarding_fee_usd="0")
    row = load_dataset(corpus).records[0]
    assert row["price_includes_ai"] is False
    assert row["onboarding_fee_usd"] == Decimal(0)


def test_cutoff_inclusive_and_independent_of_other_dates(corpus):
    change_record(corpus, available_at="2026-11-11T18:59:59-05:00",
                  published_at="2026-11-12T00:00:00Z", effective_date="2026-12-01",
                  verification_status="disputed")
    dataset = load_dataset(corpus)
    first = dataset.records[0]
    assert first in dataset.get_available_records()
    assert first not in dataset.get_available_records("2026-11-11T23:59:58Z")
    assert first["verification_status"] == "disputed"
    change_record(corpus, available_at="2026-11-12T00:00:00Z")
    dataset = load_dataset(corpus)
    assert dataset.records[0] not in dataset.get_available_records()
    assert len(dataset.records) == 310


def test_bad_cutoff():
    with pytest.raises(DatasetValidationError, match="as_of cutoff"):
        load_dataset(PUBLIC).get_available_records("2026-11-11")


@pytest.mark.parametrize("content,pattern", [("{oops", "malformed JSON"), ("[]", "JSON object")])
def test_bad_json(corpus, content, pattern):
    (corpus / "public_web_records.jsonl").write_text(content, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=pattern):
        load_dataset(corpus)


@pytest.mark.parametrize("filename,column,value,pattern", [
    ("pricing_events.csv", "effective_date", "31/02/2026", "effective_date"),
    ("pricing_events.csv", "price_usd", "oops", "price_usd"),
    ("public_questions.csv", "question_id", "Q02", "duplicate question_id"),
])
def test_csv_validation(corpus, filename, column, value, pattern):
    path = corpus / filename
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        fields = reader.fieldnames
        rows = list(reader)
    rows[0][column] = value
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DatasetValidationError, match=pattern):
        load_dataset(corpus)


def test_only_public_files_are_read(corpus):
    # Invalid private files must have no effect, even if placed beside public inputs.
    (corpus / "private_fact_ledger.csv").write_bytes(b"\xff\xfeinvalid")
    assert len(load_dataset(corpus).records) == 310
