from datetime import datetime, timezone
from decimal import Decimal

import pytest

from ci_agents.dataset import Dataset, load_dataset
from ci_agents import retrieval
from ci_agents.retrieval import RetrievalError, RetrievalStore


@pytest.fixture
def store():
    with RetrievalStore(load_dataset()) as instance:
        yield instance


def test_real_search_metadata_and_filters(store):
    results = store.search_records("Growth", company="vertexops", category="pricing")
    assert results
    assert all(r["company_name"] == "VertexOps" and r["event_category"] == "Pricing" for r in results)
    assert all(r["claim_id"] and r["source_independence_group"] for r in results)
    assert all("supersedes_record_id" in r for r in results)
    assert all("details" not in r and len(r["summary"]) <= 600 for r in results)
    assert store.search_records("Growth", company="does-not-exist") == []
    assert store.search_records("Growth", category="does-not-exist") == []


def test_relevance_and_all_four_text_fields():
    base = load_dataset().records[0]
    rows = []
    for i, field in enumerate(["document_title", "summary", "details", "retrieval_text"]):
        row = dict(base, record_id=str(i), document_title="plain", summary="plain",
                   details="plain", retrieval_text="plain")
        row[field] = "uniqueterm"
        rows.append(row)
    with RetrievalStore(Dataset(rows, [], [])) as store:
        results = store.search_records("uniqueterm")
        assert {r["record_id"] for r in results} == {"0", "1", "2", "3"}
        assert results[0]["record_id"] == "0"  # Title receives the greatest BM25 weight.
        assert [r["relevance_score"] for r in results] == sorted(
            [r["relevance_score"] for r in results], reverse=True)


def test_missing_record_and_complete_details(store):
    assert store.get_record("MISSING") is None
    row = store.get_record("WEB-00001")
    assert row["details"] and row["retrieval_text"]
    assert row["onboarding_fee_usd"] is None
    row["details"] = "modified"
    assert store.get_record("WEB-00001")["details"] != "modified"


def test_cutoffs_apply_to_every_tool(store):
    row = store.get_record("WEB-00001")
    instant = row["available_at"]
    assert store.get_record(row["record_id"], instant) is not None
    assert store.get_record(row["record_id"], "2025-01-08T07:00:00-05:00") is not None
    early = "2020-01-01T00:00:00Z"
    assert store.get_record(row["record_id"], early) is None
    assert store.search_records("Growth", as_of=early) == []
    assert store.get_pricing_history(as_of=early) == []
    cutoff = datetime(2025, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
    assert all(r["available_at"] <= cutoff for r in store.search_records("pricing", as_of=cutoff))
    assert all(r["available_at"] <= cutoff and r["source_available_at"] <= cutoff
               for r in store.get_pricing_history(as_of=cutoff))


def test_pricing_preserves_observations_and_units(store):
    history = store.get_pricing_history()
    assert len(history) == 77
    subset = store.get_pricing_history("vertexops", "growth")
    assert len(subset) > 1
    assert all(r["company"] == "VertexOps" and r["plan"] == "Growth" for r in subset)
    assert all(r["currency"] == "USD" and isinstance(r["price_usd"], Decimal) for r in subset)
    assert all(r["billing_frequency"] and r["price_basis"] and r["source_record_id"] for r in subset)
    assert any(r["source_type"] == "Third-party comparison" for r in subset)
    assert [r["effective_date"] for r in subset] == sorted(r["effective_date"] for r in subset)
    assert store.get_pricing_history("missing") == []


def test_future_effective_and_unknown_price_are_preserved():
    dataset = load_dataset()
    dataset.pricing_events[0]["price_usd"] = None
    dataset.pricing_events[0]["effective_date"] = datetime(2030, 1, 1, tzinfo=timezone.utc)
    with RetrievalStore(dataset) as store:
        row = next(r for r in store.get_pricing_history() if r["source_record_id"] == "WEB-00001")
        assert row["price_usd"] is None
        assert row["effective_date"].year == 2030


@pytest.mark.parametrize("query", ["", "   ", '"()*:-'])
def test_empty_and_fts_punctuation(store, query):
    assert store.search_records(query) == []


def test_sql_and_fts_input_is_literal(store):
    assert store.search_records("Growth", company="x' OR 1=1 --") == []
    store.search_records('Growth OR "bad*: NEAR(')
    assert store.get_record("WEB-00001") is not None
    assert len(store.search_records("Growth", limit=2)) <= 2


@pytest.mark.parametrize("kwargs", [
    {"query": None}, {"query": "x" * 501}, {"query": "Growth", "limit": 0},
    {"query": "Growth", "limit": True}, {"query": "Growth", "limit": 51},
    {"query": "Growth", "company": []}, {"query": "Growth", "category": ""},
    {"query": "Growth", "as_of": "2026-11-11"},
])
def test_invalid_search_inputs(store, kwargs):
    with pytest.raises(RetrievalError):
        store.search_records(**kwargs)


@pytest.mark.parametrize("method,args", [
    ("get_record", (None,)), ("get_record", ("x", "bad")),
    ("get_pricing_history", ([],)), ("get_pricing_history", (None, "",)),
    ("get_pricing_history", (None, None, datetime(2026, 1, 1))),
])
def test_other_invalid_inputs(store, method, args):
    with pytest.raises(RetrievalError):
        getattr(store, method)(*args)


def test_public_functions_and_environment(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "data/public")
    monkeypatch.setenv("EVALUATION_CUTOFF", "2020-01-01T00:00:00Z")
    assert retrieval.search_records("Growth") == []
    assert retrieval.get_record("WEB-00001") is None
    assert retrieval.get_pricing_history() == []
    assert retrieval.get_record("WEB-00001", "2026-11-11T23:59:59Z") is not None
