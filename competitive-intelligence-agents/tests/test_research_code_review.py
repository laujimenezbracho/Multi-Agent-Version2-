"""Regression cases identified during the 16 November research-readiness review."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from agents import Usage
import pytest

from ci_agents.dataset import load_dataset
from ci_agents.evidence_context import numerical_aids
from ci_agents.schemas import EvidenceItem, EvidencePack, PricingObservation
from ci_agents.scope import normalize_price_plan
from ci_agents.telemetry import capture_stage, stage_metrics, utc_now
from ci_agents.workflow_runtime import BudgetHooks
from test_price_scope_control import plan, COMPANIES
from test_workflow_improvements import runtime

PUBLIC = Path(__file__).resolve().parents[1] / "data/public"
CUTOFF = datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "question",
    [
        "Which competitor has the lowest Growth price in Germany?",
        "Which competitor has the lowest Growth price excluding VertexOps?",
        "Which competitor has the lowest Growth price among NexaFlow?",
        "Which competitor has the lowest Growth price as of January 2025?",
    ],
)
def test_scope_control_preserves_explicit_restrictions(question):
    original = plan(question)
    assert normalize_price_plan(original, COMPANIES) == original


def price_item():
    source = next(
        row for row in load_dataset(PUBLIC).records if row["record_id"] == "WEB-00002"
    )
    fields = {
        name: source.get(name)
        for name in (
            "record_id",
            "claim_id",
            "company_name",
            "source_type",
            "source_url",
            "published_at",
            "available_at",
            "effective_date",
            "end_date",
            "source_independence_group",
            "supersedes_record_id",
            "verification_status",
            "feature_status",
        )
    }
    return EvidenceItem(
        **fields,
        source_title=source["document_title"],
        finding=source["summary"],
        quoted_or_paraphrased_evidence=source["details"],
        confidence=0.5,
        pricing_observation=PricingObservation(
            plan=source["plan"],
            amount=str(source["metric_value"]),
            currency=source["currency"],
            billing_frequency=source["billing_frequency"],
            price_basis=source["price_basis"],
            seat_minimum=source["seat_minimum"],
        ),
    )


def test_future_effective_replacement_does_not_hide_current_price():
    old = price_item()
    future = old.model_copy(deep=True)
    future.record_id = "FUTURE-PRICE"
    future.supersedes_record_id = old.record_id
    future.effective_date = datetime(2027, 1, 1, tzinfo=timezone.utc)
    future.pricing_observation.amount = "199"
    p = plan()
    p.as_of = CUTOFF
    pack = EvidencePack(
        original_question=p.original_question,
        as_of=CUTOFF,
        evidence_items=[old, future],
    )
    aids = numerical_aids(p, pack)
    included = [
        r["record_id"]
        for group in aids["comparison_groups"]
        for r in group["observations"]
    ]
    assert old.record_id in included and future.record_id not in included


@pytest.mark.parametrize(
    "usage",
    [
        Usage(requests=1),
        SimpleNamespace(input_tokens=10, output_tokens=None, total_tokens=None),
        SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=999),
    ],
)
def test_invalid_usage_never_publishes_zero_or_inconsistent_cost(usage):
    state = runtime(max_total_tokens=10000)
    hooks = BudgetHooks(state)
    with capture_stage() as capture:
        asyncio.run(hooks.on_llm_end(None, None, SimpleNamespace(usage=usage)))
    now = utc_now()
    metrics = stage_metrics(
        "reviewer", "gpt-5-mini", now, now, 0.1, capture, "success", state.pricing
    )
    assert not state.usage_complete
    assert metrics.total_tokens is None and metrics.estimated_cost_usd is None
