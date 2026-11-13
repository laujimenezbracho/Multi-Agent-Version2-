"""Offline public-source retrieval. Relevance and verification labels are not truth."""
from datetime import datetime
from pathlib import Path
import os
import re
import sqlite3

import pandas as pd

from .config import DEFAULT_CUTOFF, PROJECT_ROOT
from .dataset import Dataset, _date, load_dataset

MAX_SEARCH_RESULTS = 50
MAX_HISTORY_RESULTS = 100
MAX_QUERY_LENGTH = 500
SUMMARY_LENGTH = 600
SEARCH_FIELDS = (
    "record_id", "company_name", "document_title", "source_url", "source_type",
    "published_at", "available_at", "effective_date", "end_date",
    "verification_status", "claim_id", "supersedes_record_id",
    "source_independence_group", "event_category", "feature_status",
)


class RetrievalError(ValueError):
    """Invalid retrieval input or unavailable local search capability."""


def _text(value, name, *, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_QUERY_LENGTH:
        raise RetrievalError(f"{name} must be non-empty text of at most {MAX_QUERY_LENGTH} characters")
    return value.strip()


def _cutoff(value):
    if value is None:
        value = os.environ.get("EVALUATION_CUTOFF", DEFAULT_CUTOFF)
    try:
        return _date(value)
    except (ValueError, TypeError) as exc:
        raise RetrievalError(f"as_of must be a timezone-aware ISO timestamp or datetime: {exc}") from exc


class RetrievalStore:
    """An in-memory FTS5 index of a validated public Dataset.

    Use as a context manager or call close(). No local database files are written.
    Dataset injection supports offline tests and explicitly configured data directories.
    """

    def __init__(self, dataset: Dataset):
        self._records = {r["record_id"]: dict(r) for r in dataset.records}
        self._pricing = pd.DataFrame(dataset.pricing_events, dtype=object)
        self._db = sqlite3.connect(":memory:")
        try:
            self._db.execute("""CREATE VIRTUAL TABLE documents USING fts5(
                record_id UNINDEXED, company UNINDEXED, category UNINDEXED,
                available_at UNINDEXED, title, summary, details, retrieval_text
            )""")
            self._db.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(r["record_id"], r["company_name"], r.get("event_category"),
                  r["available_at"].isoformat(), r["document_title"], r["summary"],
                  r["details"], r["retrieval_text"]) for r in self._records.values()],
            )
        except sqlite3.Error as exc:
            self._db.close()
            raise RetrievalError(f"Unable to initialize SQLite FTS5 index: {exc}") from exc

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def search_records(self, query, company=None, category=None, as_of=None, limit=20):
        """Search literal words (AND semantics), with weighted BM25 relevance.

        FTS operators are treated as ordinary words; punctuation cannot alter SQL/FTS.
        Empty/whitespace/punctuation-only queries return []; malformed types raise.
        Results include a bounded summary and metadata, never full document bodies.
        """
        if not isinstance(query, str) or len(query) > MAX_QUERY_LENGTH:
            raise RetrievalError(f"query must be text of at most {MAX_QUERY_LENGTH} characters")
        company = _text(company, "company", optional=True)
        category = _text(category, "category", optional=True)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SEARCH_RESULTS:
            raise RetrievalError(f"limit must be an integer between 1 and {MAX_SEARCH_RESULTS}")
        cutoff = _cutoff(as_of)
        terms = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
        if not terms:
            return []
        expression = " AND ".join('"' + term + '"' for term in terms)
        sql = """SELECT record_id, bm25(documents, 0, 0, 0, 0, 5, 2, 1, 1) AS score
                 FROM documents WHERE documents MATCH ? AND available_at <= ?"""
        params = [expression, cutoff.isoformat()]
        if company is not None:
            sql += " AND company = ? COLLATE NOCASE"
            params.append(company)
        if category is not None:
            sql += " AND category = ? COLLATE NOCASE"
            params.append(category)
        sql += " ORDER BY score, record_id LIMIT ?"
        params.append(limit)
        results = []
        for record_id, score in self._db.execute(sql, params):
            record = self._records[record_id]
            result = {key: record.get(key) for key in SEARCH_FIELDS}
            result["summary"] = (record.get("summary") or "")[:SUMMARY_LENGTH]
            result["relevance_score"] = -score  # Larger means more relevant.
            results.append(result)
        return results

    def get_record(self, record_id, as_of=None):
        """Return complete source details, or None if missing/unavailable at cutoff."""
        record_id = _text(record_id, "record_id")
        cutoff = _cutoff(as_of)
        record = self._records.get(record_id)
        return dict(record) if record is not None and record["available_at"] <= cutoff else None

    def get_pricing_history(self, company=None, plan=None, as_of=None):
        """Return at most 100 source observations, oldest effective date first.

        Both observation and referenced source must be available. Future-effective
        announcements remain visible with their effective dates; no current-price
        inference or deduplication is performed. Decimal amounts/None are preserved.
        """
        company = _text(company, "company", optional=True)
        plan = _text(plan, "plan", optional=True)
        cutoff = _cutoff(as_of)
        frame = self._pricing
        if frame.empty:
            return []
        mask = frame["available_at"].map(lambda date: date <= cutoff)
        mask &= frame["source_record_id"].map(
            lambda rid: rid in self._records and self._records[rid]["available_at"] <= cutoff
        )
        for column, value in (("company", company), ("plan", plan)):
            if value is not None:
                mask &= frame[column].map(lambda text: isinstance(text, str) and text.casefold() == value.casefold())
        selected = frame.loc[mask].sort_values(
            ["effective_date", "available_at", "source_record_id"], kind="stable"
        )
        if len(selected) > MAX_HISTORY_RESULTS:
            raise RetrievalError("Pricing history exceeds 100 observations; narrow company or plan filters")
        results = []
        for observation in selected.to_dict(orient="records"):
            source = self._records[observation["source_record_id"]]
            observation.update({key: source.get(key) for key in (
                "currency", "published_at", "end_date", "supersedes_record_id",
                "source_independence_group", "claim_id", "document_title", "source_url",
            )})
            # Preserve the source's independent availability/effective timestamps too.
            observation["source_available_at"] = source["available_at"]
            observation["source_effective_date"] = source["effective_date"]
            results.append(observation)
        return results


def _store():
    directory = Path(os.environ.get("DATA_DIR", "data/public")).expanduser()
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    return RetrievalStore(load_dataset(directory))


def search_records(query, company=None, category=None, as_of=None, limit=20):
    with _store() as store:
        return store.search_records(query, company, category, as_of, limit)


def get_record(record_id, as_of=None):
    with _store() as store:
        return store.get_record(record_id, as_of)


def get_pricing_history(company=None, plan=None, as_of=None):
    with _store() as store:
        return store.get_pricing_history(company, plan, as_of)
