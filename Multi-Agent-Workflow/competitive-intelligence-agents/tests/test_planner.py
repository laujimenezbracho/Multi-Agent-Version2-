import asyncio
from datetime import datetime, timezone, timedelta
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents import AgentOutputSchema
import httpx
import pytest

from ci_agents.config import load_config
from ci_agents.schemas import ResearchPlan, ResearchTask

module = importlib.import_module("ci_agents.agents.planner")
PUBLIC = Path(__file__).resolve().parents[1] / "data/public"
CUTOFF = datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)
Q01 = "Which competitor has the lowest current Growth plan price?"
Q15 = "What are the top 3 competitive threats for PulseGrid?"
COMPANIES = ["ClearPilot", "NexaFlow", "OrbitDesk", "PulseGrid", "VertexOps"]


def make_plan(question=Q01):
    pricing = question == Q01
    needs = (["Comparable Growth plan prices", "Currency", "Billing basis and price basis",
              "Effective dates and available_at cutoff", "Source reliability"] if pricing else
             ["Source-backed product capabilities and AI availability", "Pricing/value evidence",
              "Customer sentiment limitations", "Evidence supporting a ranked threat assessment"])
    return ResearchPlan(
        original_question=question, as_of=CUTOFF,
        objective="Plan evidence collection without competitive conclusions",
        tasks=[ResearchTask(task_id="T1", research_question="Collect comparable competitor evidence",
                            companies=COMPANIES, categories=["Pricing"] if pricing else
                            ["Pricing", "Competitor positioning", "Customer experience"],
                            time_scope="Available by the supplied cutoff; effective on requested date",
                            evidence_requirements=needs)],
        expected_answer_elements=["Comparable prices with sources"] if pricing else
                                 ["Evidence-supported ranked top three threats and uncertainty"],
        comparison_dimensions=["Pricing"] if pricing else ["Pricing/value", "Product capabilities", "Customer sentiment"],
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # Any accidental SDK request fails before reaching a network transport.
    send = AsyncMock(side_effect=AssertionError("Network requests forbidden in Planner tests"))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    yield
    send.assert_not_called()


@pytest.fixture
def sdk(monkeypatch):
    settings = load_config(environ={"OPENAI_API_KEY": "test-only-secret", "OPENAI_MODEL": "gpt-5-mini",
                                    "DATA_DIR": str(PUBLIC)})
    config = lambda: settings
    monkeypatch.setattr(module, "load_config", config)
    run = AsyncMock(return_value=SimpleNamespace(final_output=make_plan()))
    monkeypatch.setattr(module.Runner, "run", run)
    return run


def call(question=Q01, cutoff=CUTOFF):
    return asyncio.run(module.planner(question, cutoff))


def test_valid_plan_and_native_sdk_contract(sdk):
    result = call()
    assert isinstance(result, ResearchPlan)
    args = sdk.call_args
    agent = args.args[0]
    assert agent.output_type is ResearchPlan
    assert agent.tools == [] and agent.handoffs == []
    assert args.kwargs["max_turns"] == 1
    assert args.kwargs["run_config"].tracing_disabled
    assert agent.model_settings.temperature is None
    schema = AgentOutputSchema(ResearchPlan)
    assert schema.is_strict_json_schema()
    assert schema.validate_json(result.model_dump_json()) == result


def test_q01_evidence_contract(sdk):
    plan = call()
    requirements = " ".join(plan.tasks[0].evidence_requirements).lower()
    for term in ["growth", "currency", "billing", "effective", "reliability"]:
        assert term in requirements
    assert set(plan.tasks[0].companies) == set(COMPANIES)
    instructions = sdk.call_args.args[0].instructions.lower()
    for term in ["currency", "billing", "effective", "reliability", "unknown", "source independence"]:
        assert term in instructions


def test_q15_multiple_dimensions_and_ranking_evidence(sdk):
    sdk.return_value = SimpleNamespace(final_output=make_plan(Q15))
    plan = call(Q15)
    assert len(plan.comparison_dimensions) >= 3
    assert any("ranked" in text for text in plan.expected_answer_elements)
    assert "PulseGrid" in plan.tasks[0].companies
    assert "ranked threat" in " ".join(plan.tasks[0].evidence_requirements)
    assert "leave the actual ranking to the Analyst" in sdk.call_args.args[0].instructions


@pytest.mark.parametrize("question", [None, "", "   ", 12, "x" * 10001])
def test_invalid_question_without_sdk(sdk, question):
    with pytest.raises(module.PlannerInputError, match="question"):
        call(question)
    sdk.assert_not_called()


def test_cutoff_and_exact_question_propagation(sdk):
    local = CUTOFF.astimezone(timezone(timedelta(hours=-5)))
    result = call(cutoff=local)
    context = json.loads(sdk.call_args.kwargs["input"])
    assert context["as_of"] == CUTOFF.isoformat()
    assert context["original_question"] == Q01
    assert result.as_of == CUTOFF


@pytest.mark.parametrize("cutoff", [None, "2026-11-11", datetime(2026, 11, 11)])
def test_invalid_cutoff(sdk, cutoff):
    with pytest.raises(module.PlannerInputError, match="timezone-aware"):
        call(cutoff=cutoff)
    sdk.assert_not_called()


def test_duplicate_id_revalidation(sdk):
    plan = make_plan()
    plan.tasks.append(plan.tasks[0].model_copy())
    sdk.return_value = SimpleNamespace(final_output=plan)
    with pytest.raises(module.PlannerOutputError, match="invalid ResearchPlan"):
        call()


def test_duplicate_content_rejection(sdk):
    plan = make_plan()
    plan.tasks.append(plan.tasks[0].model_copy(update={"task_id": "T2"}))
    sdk.return_value = SimpleNamespace(final_output=plan)
    with pytest.raises(module.PlannerOutputError, match="duplicate research tasks"):
        call()


@pytest.mark.parametrize("output", [None, {}, '{"tasks": []}', "not a plan"])
def test_requires_native_structured_output(sdk, output):
    sdk.return_value = SimpleNamespace(final_output=output)
    with pytest.raises(module.PlannerOutputError, match="structured ResearchPlan"):
        call()


@pytest.mark.parametrize("change", ["question", "cutoff", "company", "category", "requirements", "dimensions"])
def test_output_invariants(sdk, change):
    plan = make_plan()
    if change == "question":
        plan.original_question = "Changed"
    elif change == "cutoff":
        plan.as_of = CUTOFF - timedelta(days=1)
    elif change == "company":
        plan.tasks[0].companies.append("Imaginary competitor")
    elif change == "category":
        plan.tasks[0].categories.append("Imaginary category")
    elif change == "requirements":
        plan.tasks[0].evidence_requirements.clear()
    else:
        plan.comparison_dimensions.clear()
    sdk.return_value = SimpleNamespace(final_output=plan)
    with pytest.raises(module.PlannerOutputError):
        call()


def test_api_failure_is_sanitized_and_not_retried(sdk):
    sdk.side_effect = RuntimeError("test-only-secret sensitive API body")
    with pytest.raises(module.PlannerError) as error:
        call()
    assert "test-only-secret" not in str(error.value)
    assert error.value.__suppress_context__
    sdk.assert_awaited_once()


def test_model_loaded_from_dotenv(sdk, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=test-only-secret\nOPENAI_MODEL=configured-test-model\nDATA_DIR="
        + PUBLIC.as_posix() + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "load_config", lambda: load_config(project_root=tmp_path, environ={}))
    call()
    assert sdk.call_args.args[0].model.model == "configured-test-model"


def test_metadata_only_context_and_no_private_file_access(sdk, monkeypatch):
    original_open = Path.open
    opened = []
    def guarded(path, *args, **kwargs):
        assert "evaluation_private" not in path.parts
        assert not path.name.startswith("private_")
        opened.append(path.name)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    call()
    context = json.loads(sdk.call_args.kwargs["input"])
    assert set(context) == {"original_question", "as_of", "companies", "evidence_categories"}
    assert context["companies"] == COMPANIES
    assert "Pricing" in context["evidence_categories"]
    assert "record_id" not in sdk.call_args.kwargs["input"]
    assert "pricing_events.csv" in opened  # Loader validates public inputs locally.
    assert not sdk.call_args.args[0].tools
