import asyncio
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from agents import AgentOutputSchema
from agents.tool_context import ToolContext
import httpx
import pytest
from pydantic import ValidationError

from ci_agents.config import load_config
from ci_agents.dataset import load_dataset
from ci_agents.retrieval import RetrievalStore
from ci_agents.schemas import EvidenceItem, EvidencePack, ResearchPlan, ResearchTask, TaskCoverage

module = importlib.import_module("ci_agents.agents.researcher")
CUTOFF = datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)
PUBLIC = Path(__file__).resolve().parents[1] / "data/public"
Q01 = "Which competitor has the lowest current Growth plan price?"
Q15 = "What are the top 3 competitive threats for PulseGrid?"


def plan(question=Q01, cutoff=CUTOFF):
    return ResearchPlan(original_question=question, as_of=cutoff, objective="Gather source evidence",
                        tasks=[ResearchTask(task_id="T1", research_question=question,
                                            companies=["VertexOps", "PulseGrid"], categories=["Pricing"],
                                            time_scope="By cutoff", evidence_requirements=["Comparable evidence"])],
                        expected_answer_elements=["Source-backed facts"], comparison_dimensions=["Value"])


def item(row):
    fields = {name: row.get(name) for name in (
        "record_id", "claim_id", "company_name", "source_type", "source_url", "published_at",
        "available_at", "effective_date", "end_date", "source_independence_group",
        "supersedes_record_id", "verification_status", "feature_status")}
    return EvidenceItem(**fields, source_title=row["document_title"], finding=row["summary"],
                        quoted_or_paraphrased_evidence=row["details"], confidence=0.7,
                        limitations=["Synthetic source assertion, not independent ground truth"])


def pack(research_plan, rows=()):
    return EvidencePack(original_question=research_plan.original_question, as_of=research_plan.as_of,
                        evidence_items=[item(row) for row in rows],
                        coverage_by_task=[TaskCoverage(task_id="T1", coverage_score=None,
                            supporting_record_ids=[row["record_id"] for row in rows],
                            notes=[] if rows else ["No relevant source found"])],
                        missing_information=[] if rows else ["Missing task evidence"])


async def tool(agent, name, **arguments):
    target = next(t for t in agent.tools if t.name == name)
    return json.loads(await target.on_invoke_tool(ToolContext(context=None, tool_name=name, tool_call_id="test-call", tool_arguments=json.dumps(arguments)), json.dumps(arguments)))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    send = AsyncMock(side_effect=AssertionError("Network forbidden"))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    yield
    send.assert_not_called()


@pytest.fixture
def store():
    with RetrievalStore(load_dataset(PUBLIC)) as service:
        yield service


@pytest.fixture
def sdk(monkeypatch):
    settings = load_config(environ={"OPENAI_API_KEY": "test-only-secret", "OPENAI_MODEL": "gpt-5-mini",
                                   "DATA_DIR": str(PUBLIC)})
    monkeypatch.setattr(module, "load_config", lambda: settings)
    runner = AsyncMock()
    monkeypatch.setattr(module.Runner, "run", runner)
    return runner


def run(research_plan, store, **kwargs):
    return asyncio.run(module.researcher(research_plan, store, **kwargs))


@pytest.mark.parametrize("question", [Q01, Q15])
def test_success_native_tools_and_plan_propagation(sdk, store, question):
    research_plan = plan(question)
    async def respond(agent, **kwargs):
        assert json.loads(kwargs["input"])["research_plan"] == research_plan.model_dump(mode="json")
        assert agent.model.model == "gpt-5-mini"
        assert agent.output_type is EvidencePack and agent.handoffs == []
        assert agent.model_settings.temperature is None
        assert agent.model_settings.parallel_tool_calls is False
        assert kwargs["run_config"].tracing_disabled
        for function in agent.tools:
            assert "as_of" not in function.params_json_schema["properties"]
            assert function.params_json_schema["additionalProperties"] is False
        snippets = await tool(agent, "search_records", query="Growth", company="VertexOps", category="Pricing", limit=2)
        assert len(snippets) == 2
        full = await tool(agent, "get_record", record_id=snippets[0]["record_id"])
        assert full["details"]
        history = await tool(agent, "get_pricing_history", company="VertexOps", plan="Growth")
        assert len(history) > 1
        return SimpleNamespace(final_output=pack(research_plan, [store.get_record(full["record_id"])]))
    sdk.side_effect = respond
    result = run(research_plan, store)
    assert len(result.evidence_items) == 1
    assert len(result.retrieval_log) == 3
    assert [json.loads(event)["tool"] for event in result.retrieval_log] == [
        "search_records", "get_record", "get_pricing_history"]
    schema = AgentOutputSchema(EvidencePack)
    assert schema.is_strict_json_schema()
    assert schema.validate_json(result.model_dump_json()) == result


def test_cutoff_on_every_service_call(sdk, store):
    research_plan = plan(cutoff=datetime(2020, 1, 1, tzinfo=timezone.utc))
    service = Mock(wraps=store)
    async def respond(agent, **kwargs):
        assert await tool(agent, "search_records", query="Growth") == []
        assert await tool(agent, "get_record", record_id="WEB-00001") is None
        assert await tool(agent, "get_pricing_history") == []
        return SimpleNamespace(final_output=pack(research_plan))
    sdk.side_effect = respond
    run(research_plan, service)
    for name in ["search_records", "get_record", "get_pricing_history"]:
        assert getattr(service, name).call_args.kwargs["as_of"] == research_plan.as_of
    service.close.assert_not_called()


@pytest.mark.parametrize("mode", ["fabricated", "unretrieved", "snippet_only", "history_only"])
def test_rejects_unretrieved_evidence(sdk, store, mode):
    research_plan = plan()
    row = store.get_record("WEB-00001")
    async def respond(agent, **kwargs):
        if mode == "snippet_only":
            await tool(agent, "search_records", query="Starter")
        if mode == "history_only":
            await tool(agent, "get_pricing_history")
        result = pack(research_plan, [row])
        if mode == "fabricated":
            result.evidence_items[0].record_id = "FAKE-ID"
        return SimpleNamespace(final_output=result)
    sdk.side_effect = respond
    with pytest.raises(module.EvidenceValidationError, match="not retrieved in full"):
        run(research_plan, store)


@pytest.mark.parametrize("field", ["claim_id", "source_title", "source_url", "company_name", "source_type",
                                    "source_independence_group", "supersedes_record_id", "verification_status",
                                    "feature_status", "published_at", "effective_date", "end_date"])
def test_metadata_mismatch(sdk, store, field):
    research_plan = plan()
    async def respond(agent, **kwargs):
        await tool(agent, "get_record", record_id="WEB-00001")
        result = pack(research_plan, [store.get_record("WEB-00001")])
        value = datetime(2025, 2, 1, tzinfo=timezone.utc) if field.endswith("_at") or field.endswith("_date") else "wrong"
        setattr(result.evidence_items[0], field, value)
        return SimpleNamespace(final_output=result)
    sdk.side_effect = respond
    with pytest.raises(module.EvidenceValidationError, match="metadata mismatch"):
        run(research_plan, store)


def test_duplicate_ids(sdk, store):
    research_plan = plan()
    result = pack(research_plan, [store.get_record("WEB-00001")])
    result.evidence_items.append(result.evidence_items[0].model_copy())
    sdk.return_value = SimpleNamespace(final_output=result)
    with pytest.raises(module.EvidenceValidationError, match="duplicate IDs"):
        run(research_plan, store)


def test_conflicting_prices_expired_promotions_preserved(sdk, store):
    research_plan = plan()
    ids = ["WEB-30008", "WEB-30009", "WEB-00038"]
    async def respond(agent, **kwargs):
        for rid in ids:
            await tool(agent, "get_record", record_id=rid)
        result = pack(research_plan, [store.get_record(rid) for rid in ids])
        result.contradictions = ["Third-party and official VertexOps Growth observations disagree; preserve both"]
        result.evidence_items[-1].limitations.append("Promotion expired before cutoff")
        return SimpleNamespace(final_output=result)
    sdk.side_effect = respond
    result = run(research_plan, store)
    assert len(result.evidence_items) == 3 and result.contradictions
    assert result.evidence_items[-1].end_date < CUTOFF
    assert result.evidence_items[1].supersedes_record_id


def test_missing_evidence_and_authoritative_log(sdk, store):
    research_plan = plan()
    async def respond(agent, **kwargs):
        assert await tool(agent, "get_record", record_id="NOT-FOUND") is None
        result = pack(research_plan)
        result.retrieval_log = ["Invented log event"]
        return SimpleNamespace(final_output=result)
    sdk.side_effect = respond
    result = run(research_plan, store)
    assert result.missing_information and not result.evidence_items
    assert "Invented" not in str(result.retrieval_log)
    assert json.loads(result.retrieval_log[0])["result_count"] == 0


@pytest.mark.parametrize("budget", ["max_tool_calls", "max_full_records"])
def test_budget_exhaustion_is_logged(sdk, store, caplog, budget):
    research_plan = plan()
    limits = module.ResearcherLimits(**{budget: 1})
    async def respond(agent, **kwargs):
        await tool(agent, "get_record", record_id="WEB-00001")
        denied = await tool(agent, "get_record", record_id="WEB-00002")
        assert denied["error"] == "budget_exhausted" and denied["budget"] == budget
        return SimpleNamespace(final_output=pack(research_plan, [store.get_record("WEB-00001")]))
    sdk.side_effect = respond
    result = run(research_plan, store, limits=limits)
    assert any(json.loads(event)["status"] == "budget_exhausted" for event in result.retrieval_log)
    assert budget in caplog.text


def test_search_limit_and_evidence_limit(sdk, store):
    research_plan = plan()
    async def respond(agent, **kwargs):
        assert len(await tool(agent, "search_records", query="Growth", limit=50)) <= 1
        for rid in ["WEB-00001", "WEB-00002"]:
            await tool(agent, "get_record", record_id=rid)
        return SimpleNamespace(final_output=pack(research_plan, [store.get_record("WEB-00001"), store.get_record("WEB-00002")]))
    sdk.side_effect = respond
    with pytest.raises(module.EvidenceValidationError, match="max_evidence_items"):
        run(research_plan, store, limits=module.ResearcherLimits(max_search_results=1, max_evidence_items=1))


def test_api_failure_and_no_retry(sdk, store):
    sdk.side_effect = RuntimeError("test-only-secret API payload")
    with pytest.raises(module.ResearcherError) as error:
        run(plan(), store)
    assert "test-only-secret" not in str(error.value)
    sdk.assert_awaited_once()


@pytest.mark.parametrize("case", ["coverage", "coverage_reference", "question", "cutoff", "missing_notes", "not_structured"])
def test_output_invariants(sdk, store, case):
    research_plan = plan()
    result = pack(research_plan)
    if case == "coverage":
        result.coverage_by_task.clear()
    elif case == "coverage_reference":
        result.coverage_by_task[0].supporting_record_ids = ["UNRETRIEVED"]
    elif case == "question":
        result.original_question = "Changed"
    elif case == "cutoff":
        result.as_of = datetime(2025, 1, 1, tzinfo=timezone.utc)
    elif case == "missing_notes":
        result.coverage_by_task[0].notes.clear()
    else:
        result = {}
    sdk.return_value = SimpleNamespace(final_output=result)
    with pytest.raises(module.EvidenceValidationError):
        run(research_plan, store)


def test_no_private_access_default_service(sdk, monkeypatch):
    opened = []
    original = Path.open
    def guard(path, *args, **kwargs):
        assert "evaluation_private" not in path.parts and not path.name.startswith("private_")
        opened.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    sdk.return_value = SimpleNamespace(final_output=pack(plan()))
    asyncio.run(module.researcher(plan()))
    assert "public_web_records.jsonl" in opened
    assert "researcher.md" in opened


def test_provenance_extension_backwards_compatible(store):
    original = item(store.get_record("WEB-00001")).model_dump()
    for field in ["source_independence_group", "supersedes_record_id", "end_date", "verification_status", "feature_status"]:
        original.pop(field)
    restored = EvidenceItem.model_validate(original)
    assert restored.source_independence_group is None
    assert restored.end_date is None
    assert EvidenceItem.model_validate_json(restored.model_dump_json()) == restored


@pytest.mark.parametrize("value", [0, -1, True, 201])
def test_invalid_limits(value):
    with pytest.raises(ValidationError):
        module.ResearcherLimits(max_tool_calls=value)


def test_llm_cannot_supply_a_future_cutoff(sdk, store):
    research_plan = plan(cutoff=datetime(2020, 1, 1, tzinfo=timezone.utc))
    service = Mock(wraps=store)
    async def respond(agent, **kwargs):
        target = next(t for t in agent.tools if t.name == "get_record")
        args = json.dumps({"record_id": "WEB-00001", "as_of": "2099-01-01T00:00:00Z"})
        context = ToolContext(context=None, tool_name="get_record", tool_call_id="test", tool_arguments=args)
        # SDK 0.24 ignores unknown JSON keys locally even though its schema forbids
        # them. The closure still binds the plan cutoff, so no future record leaks.
        assert json.loads(await target.on_invoke_tool(context, args)) is None
        return SimpleNamespace(final_output=pack(research_plan))
    sdk.side_effect = respond
    assert not run(research_plan, service).evidence_items
    assert service.get_record.call_args.kwargs["as_of"] == research_plan.as_of


@pytest.mark.parametrize("name", ["get_record", "search_records", "get_pricing_history"])
def test_defensive_cutoff_rejects_broken_service(sdk, store, name):
    research_plan = plan()
    service = Mock(wraps=store)
    row = store.get_record("WEB-00001")
    row["available_at"] = datetime(2099, 1, 1, tzinfo=timezone.utc)
    if name == "get_record":
        service.get_record.return_value = row
    elif name == "search_records":
        service.search_records.return_value = [row]
    else:
        observation = store.get_pricing_history()[0]
        observation["source_available_at"] = row["available_at"]
        service.get_pricing_history.return_value = [observation]
    async def respond(agent, **kwargs):
        arguments = {"record_id": "WEB-00001"} if name == "get_record" else {"query": "Growth"} if name == "search_records" else {}
        await tool(agent, name, **arguments)
        return SimpleNamespace(final_output=pack(research_plan))
    sdk.side_effect = respond
    with pytest.raises(module.EvidenceValidationError):
        run(research_plan, service)
