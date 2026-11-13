"""Local context-scoped telemetry. No exporters, API calls, or token estimates."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import re

from pydantic import Field, model_validator
from .schemas import Contract, NonNegative, StageMetrics, Text, TotalMetrics
from .diagnostics import log_exception_diagnostic


class PricingConfig(Contract):
    schema_version: str = "model_pricing.v1"
    model: Text = "gpt-5-mini"
    source: Text | None = None
    source_date: Text | None = None
    input_usd_per_million: NonNegative | None = None
    output_usd_per_million: NonNegative | None = None
    assumptions: list[Text] = Field(default_factory=lambda: [
        "Estimated cost, not billed amount; all input tokens use one rate, including cached tokens"])

    @model_validator(mode="after")
    def documented_rates(self):
        if (self.input_usd_per_million is not None or self.output_usd_per_million is not None) and not (self.source and self.source_date):
            raise ValueError("Configured rates require source and source_date")
        return self


@dataclass
class StageCapture:
    usage: dict | None = None
    events: list[dict] = field(default_factory=list)


_capture: ContextVar[StageCapture | None] = ContextVar("ci_stage_capture", default=None)


@contextmanager
def capture_stage():
    capture = StageCapture()
    token = _capture.set(capture)
    try:
        yield capture
    finally:
        _capture.reset(token)


def observe_sdk_result(result):
    capture = _capture.get()
    if capture is None:
        return
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage is None:
        return
    fields = {name: getattr(usage, name, None) for name in ("input_tokens", "output_tokens", "total_tokens", "requests")}
    # SDK default Usage() zeros without requests are not proof of measured usage.
    if not isinstance(fields["requests"], int) or fields["requests"] <= 0:
        return
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0) for value in fields.values()):
        return
    fields["cached_input_tokens"] = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None)
    capture.usage = fields


def observe_retrieval_event(event):
    capture = _capture.get()
    if capture is not None:
        capture.events.append(dict(event))


def utc_now():
    return datetime.now(timezone.utc)


def sanitized_error(exc, stage, *, secrets=()):
    # Deliberately do not serialize str(exc), traceback, HTTP bodies, or headers.
    name = type(exc).__name__
    safe_type = name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,80}", name) else "Exception"
    return {"error_type": safe_type, "message": f"Execution failed during {stage}; provider details omitted",
            "diagnostic": log_exception_diagnostic(exc, stage, secrets=secrets)}


def redact_text(text, secrets=()):
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)(Bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    return text


def estimate_cost(input_tokens, output_tokens, pricing):
    def cost(tokens, rate):
        return None if tokens is None or rate is None else float(Decimal(tokens) * Decimal(str(rate)) / Decimal(1000000))
    incoming = cost(input_tokens, pricing.input_usd_per_million)
    outgoing = cost(output_tokens, pricing.output_usd_per_million)
    total = None if incoming is None or outgoing is None else incoming + outgoing
    return incoming, outgoing, total


def retrieval_summary(events, selected_ids=()):
    counts = {name: 0 for name in ("search_records", "get_record", "get_pricing_history")}
    discovered, full = set(), set()
    for event in events:
        name = event.get("tool")
        if name in counts:
            counts[name] += 1
        ids = set(event.get("record_ids", []))
        discovered.update(ids)
        if name == "get_record" and event.get("status") == "ok":
            full.update(ids)
    selected = set(selected_ids)
    return {"tool_attempts": sum(counts.values()), "counts": counts,
            "retrieved_record_ids": sorted(discovered), "full_record_ids": sorted(full),
            "selected_evidence_ids": sorted(selected), "retrieved_but_not_selected": sorted(discovered - selected),
            "budget_exhaustion_events": [e for e in events if e.get("status") == "budget_exhausted"]}


def stage_metrics(name, model, start, finish, duration, capture, status, pricing, error=None):
    usage = capture.usage or {}
    incoming, outgoing, total = estimate_cost(usage.get("input_tokens"), usage.get("output_tokens"), pricing)
    summary = retrieval_summary(capture.events)
    known_retrieval = name != "researcher" or bool(capture.events)
    return StageMetrics(stage_name=name, model=model, started_at=start, finished_at=finish,
        latency_seconds=max(0.0, duration), status=status,
        input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"), llm_requests=usage.get("requests"),
        cached_input_tokens=usage.get("cached_input_tokens"),
        usage_source="sdk.context_wrapper.usage" if capture.usage is not None else "unavailable",
        estimated_input_cost_usd=incoming, estimated_output_cost_usd=outgoing, estimated_cost_usd=total,
        retrieval_tool_calls=summary["tool_attempts"] if known_retrieval else None,
        full_record_retrievals=summary["counts"]["get_record"] if known_retrieval else None,
        search_calls=summary["counts"]["search_records"] if known_retrieval else None,
        pricing_history_calls=summary["counts"]["get_pricing_history"] if known_retrieval else None,
        error_message=error["message"] if error else None,
        error_type=error["error_type"] if error else None)


def aggregate_metrics(stages, latency):
    def total(field):
        values = [getattr(stage, field) for stage in stages]
        return sum(values) if values and all(value is not None for value in values) else None
    return TotalMetrics(input_tokens=total("input_tokens"), output_tokens=total("output_tokens"),
                        latency_seconds=latency, estimated_cost_usd=total("estimated_cost_usd"))
