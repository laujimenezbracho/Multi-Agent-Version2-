from types import SimpleNamespace

from agents import Usage
import pytest

from ci_agents.telemetry import (PricingConfig, StageCapture, aggregate_metrics, capture_stage,
    estimate_cost, observe_retrieval_event, observe_sdk_result, redact_text, retrieval_summary,
    sanitized_error, stage_metrics, utc_now)
from ci_agents.config import load_config, use_settings


def sdk_result(input_tokens=100, output_tokens=20):
    return SimpleNamespace(context_wrapper=SimpleNamespace(usage=Usage(
        requests=1, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens)))


def test_sdk_usage_and_context_isolation():
    with capture_stage() as outer:
        observe_sdk_result(sdk_result())
        with capture_stage() as inner:
            assert inner.usage is None
            observe_sdk_result(sdk_result(3, 4))
        assert outer.usage["input_tokens"] == 100
    assert inner.usage["total_tokens"] == 7
    observe_sdk_result(sdk_result())  # No active capture: no export or side effects.


@pytest.mark.parametrize("result", [SimpleNamespace(), SimpleNamespace(context_wrapper=SimpleNamespace(usage=Usage()))])
def test_unknown_usage_remains_none(result):
    with capture_stage() as captured:
        observe_sdk_result(result)
    assert captured.usage is None
    now = utc_now()
    metrics = stage_metrics("planner", "gpt-5-mini", now, now, 0.1, captured, "success", PricingConfig())
    assert metrics.input_tokens is None and metrics.output_tokens is None and metrics.llm_requests is None
    assert metrics.estimated_cost_usd is None


def test_aggregation_and_explicit_test_rates():
    rates = PricingConfig(source="fixture-not-real-pricing", source_date="2026-11-11",
                          input_usd_per_million=2, output_usd_per_million=5)
    assert estimate_cost(1000000, 2000000, rates) == (2.0, 10.0, 12.0)
    now = utc_now()
    stages = []
    for name in ["planner", "researcher"]:
        with capture_stage() as captured:
            observe_sdk_result(sdk_result())
        stages.append(stage_metrics(name, rates.model, now, now, 0.2, captured, "success", rates))
    totals = aggregate_metrics(stages, 0.5)
    assert totals.input_tokens == 200 and totals.output_tokens == 40
    assert totals.estimated_cost_usd == pytest.approx(0.0006)
    assert totals.latency_seconds == 0.5
    stages[0].input_tokens = None
    stages[0].estimated_cost_usd = None
    totals = aggregate_metrics(stages, 0.6)
    assert totals.input_tokens is None and totals.estimated_cost_usd is None
    assert totals.output_tokens == 40


def test_rates_require_documentation():
    with pytest.raises(ValueError, match="source"):
        PricingConfig(input_usd_per_million=1)
    assert estimate_cost(None, 3, PricingConfig()) == (None, None, None)


def test_retrieval_counts_selection_and_budget_events():
    events = [{"tool": "search_records", "status": "ok", "record_ids": ["A", "B"]},
              {"tool": "get_record", "status": "ok", "record_ids": ["A"]},
              {"tool": "get_pricing_history", "status": "ok", "record_ids": ["C"]},
              {"tool": "get_record", "status": "budget_exhausted", "budget": "max_full_records"}]
    with capture_stage() as capture:
        for event in events:
            observe_retrieval_event(event)
    summary = retrieval_summary(capture.events, ["A"])
    assert summary["tool_attempts"] == 4
    assert summary["full_record_ids"] == ["A"]
    assert summary["retrieved_but_not_selected"] == ["B", "C"]
    assert len(summary["budget_exhaustion_events"]) == 1


def test_secret_redaction_and_error_details():
    assert "secret-value" not in redact_text("secret-value sk-abc Authorization: Bearer hidden", ["secret-value"])
    assert "hidden" not in redact_text("Authorization: Bearer hidden")
    error = sanitized_error(RuntimeError("secret-value authentication headers"), "reviewer")
    assert "secret-value" not in str(error) and error["error_type"] == "RuntimeError"


def test_frozen_settings_restored():
    settings = load_config(environ={"OPENAI_API_KEY": "test", "OPENAI_MODEL": "test"})
    with use_settings(settings):
        assert load_config() is settings
