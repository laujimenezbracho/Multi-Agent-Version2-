"""Validated public benchmark loading. Source assertions are not ground truth.

Date-only values represent midnight UTC; CSV slash dates are explicitly DD/MM/YYYY.
Only the three named public files are opened. No credential configuration is needed.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
import csv
import json

from .config import DEFAULT_CUTOFF, PROJECT_ROOT


class DatasetValidationError(ValueError):
    """A public input failed validation, with file/row/field context."""


UNKNOWN = {"", "not_specified", "unknown", "null", "none", "n/a"}
REQUIRED_RECORD = {
    "record_id", "company_name", "document_title", "summary", "details",
    "retrieval_text", "source_type", "source_url", "published_at", "available_at",
    "effective_date", "end_date", "verification_status", "claim_id",
    "supersedes_record_id", "feature_status", "plan", "billing_frequency",
    "price_basis", "source_independence_group",
}
REQUIRED_PRICING = {
    "company", "plan", "effective_date", "available_at", "price_usd",
    "price_basis", "billing_frequency", "source_record_id", "source_type",
    "verification_status",
}
REQUIRED_QUESTION = {"question_id", "user_question", "analysis_type"}
DATE_ONLY = {"event_date", "effective_date", "end_date"}
TIMESTAMPS = {"published_at", "available_at", "scraped_at"}
NUMBERS = {"metric_value", "old_value", "new_value", "price", "price_usd",
           "onboarding_fee_usd", "promotion_discount_pct"}
BOOLEANS = {"price_includes_ai", "synthetic_flag", "is_primary_source"}


def _unknown(value: Any) -> bool:
    return value is None or isinstance(value, str) and value.strip().lower() in UNKNOWN


def _date(value: Any, *, date_only: bool = False, slash: bool = False) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        value = value.strip()
        if slash and "/" in value:
            result = datetime.strptime(value, "%d/%m/%Y").replace(tzinfo=timezone.utc)
        elif date_only and len(value) == 10:
            result = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("expected a date/timestamp string")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def _convert(row: dict, required: set[str], context: str, *, slash: bool = False) -> dict:
    missing = required - row.keys()
    if missing:
        raise DatasetValidationError(f"{context}: missing fields: {', '.join(sorted(missing))}")
    result = {}
    for field, value in row.items():
        try:
            if _unknown(value):
                result[field] = None
            elif field in DATE_ONLY | TIMESTAMPS:
                result[field] = _date(value, date_only=field in DATE_ONLY, slash=slash)
            elif field in NUMBERS or field == "seat_minimum":
                if isinstance(value, bool):
                    raise ValueError("expected a number, not a boolean")
                number = Decimal(str(value))
                if not number.is_finite():
                    raise ValueError("number must be finite")
                if field == "seat_minimum":
                    if number != number.to_integral_value() or number < 0:
                        raise ValueError("expected a non-negative integer")
                    number = int(number)
                result[field] = number
            elif field in BOOLEANS:
                if isinstance(value, bool):
                    result[field] = value
                elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                    result[field] = value.strip().lower() == "true"
                else:
                    raise ValueError("expected true, false, or unknown")
            else:
                if not isinstance(value, str):
                    raise ValueError("expected text or null")
                result[field] = value
        except (ValueError, TypeError, InvalidOperation) as exc:
            raise DatasetValidationError(f"{context}: invalid {field}: {exc}") from exc
    nonnull = required - {"end_date", "supersedes_record_id", "feature_status", "plan",
                          "billing_frequency", "price_basis", "price_usd"}
    for field in nonnull:
        if result[field] is None:
            raise DatasetValidationError(f"{context}: {field} is required and cannot be unknown")
    return result


def _csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise DatasetValidationError(f"{path.name}: missing or duplicate CSV headers")
        if required - set(reader.fieldnames):
            raise DatasetValidationError(f"{path.name}: missing required CSV headers")
        rows = []
        for row in reader:
            context = f"{path.name}: line {reader.line_num}"
            if None in row or any(v is None for v in row.values()):
                raise DatasetValidationError(f"{context}: wrong number of CSV columns")
            rows.append(_convert(row, required, context, slash=True))
        return rows


@dataclass
class Dataset:
    records: list[dict[str, Any]]
    pricing_events: list[dict[str, Any]]
    questions: list[dict[str, Any]]

    def get_available_records(self, as_of: str | datetime = DEFAULT_CUTOFF) -> list[dict[str, Any]]:
        """Include available_at <= cutoff; effective dates remain separate evidence.

        This does not resolve conflicting claims or establish active prices/features.
        """
        try:
            cutoff = _date(as_of)
        except (ValueError, TypeError) as exc:
            raise DatasetValidationError(f"Invalid as_of cutoff: {exc}") from exc
        return [r for r in self.records if r["available_at"] <= cutoff]


def load_dataset(data_dir: str | Path = PROJECT_ROOT / "data/public") -> Dataset:
    """Load and validate the complete benchmark, retaining post-cutoff records."""
    directory = Path(data_dir)
    path = directory / "public_web_records.jsonl"
    records = []
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                context = f"{path.name}: line {line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetValidationError(f"{context}: malformed JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise DatasetValidationError(f"{context}: expected a JSON object")
                records.append(_convert(row, REQUIRED_RECORD, context))
        pricing = _csv(directory / "pricing_events.csv", REQUIRED_PRICING)
        questions = _csv(directory / "public_questions.csv", REQUIRED_QUESTION)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetValidationError(f"Unable to read public dataset in {directory}: {exc}") from exc

    def unique(rows: list[dict], field: str, label: str) -> set[str]:
        seen = set()
        for row in rows:
            value = row[field]
            if value in seen:
                raise DatasetValidationError(f"{label}: duplicate {field}: {value}")
            seen.add(value)
        return seen

    ids = unique(records, "record_id", path.name)
    unique(questions, "question_id", "public_questions.csv")
    if len(records) != 310:
        raise DatasetValidationError(f"Expected 310 records; found {len(records)}")
    companies = {r["company_name"] for r in records}
    if len(companies) != 5:
        raise DatasetValidationError(f"Expected five unique companies; found {len(companies)}")
    if len(questions) != 15:
        raise DatasetValidationError(f"Expected 15 questions; found {len(questions)}")
    for i, event in enumerate(pricing, 2):
        if event["source_record_id"] not in ids:
            raise DatasetValidationError(f"pricing_events.csv: row {i}: unknown source_record_id")
        if event["company"] not in companies:
            raise DatasetValidationError(f"pricing_events.csv: row {i}: unknown company")
    return Dataset(records, pricing, questions)
