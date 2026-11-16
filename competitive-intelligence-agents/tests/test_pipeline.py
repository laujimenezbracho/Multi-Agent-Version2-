import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents import Usage
import httpx
import pytest

from ci_agents import pipeline, cli
from ci_agents.agents.reviewer import report_fingerprint
from ci_agents.config import load_config
from ci_agents.dataset import load_dataset
from ci_agents.evaluation_contracts import IndependentEvaluation, PairedComparison
from ci_agents.persistence import baseline_reference, verify_baseline_reference, verify_manifest
from ci_agents.schemas import (CompetitiveReport, EvidenceItem, EvidencePack, ReportClaim,
    ResearchPlan, ResearchTask, ReviewIssue, ReviewResult, RunResult, TaskCoverage)
from ci_agents.telemetry import observe_retrieval_event, observe_sdk_result

PUBLIC = Path(__file__).resolve().parents[1] / "data/public"
CUTOFF = datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)
QUESTION = "Compare supplied evidence"


def outputs(decision="PASS"):
    row = load_dataset(PUBLIC).records[0]
    plan = ResearchPlan(original_question=QUESTION, as_of=CUTOFF, objective="Compare evidence",
        tasks=[ResearchTask(task_id="T1", research_question=QUESTION, companies=[row["company_name"]],
            categories=["Pricing"], time_scope="At cutoff", evidence_requirements=["Comparable sources"])],
        expected_answer_elements=["Evidence and limitations"], comparison_dimensions=["Pricing"])
    item = EvidenceItem(record_id=row["record_id"], claim_id=row["claim_id"], company_name=row["company_name"],
        finding=row["summary"], quoted_or_paraphrased_evidence=row["details"], source_title=row["document_title"],
        source_type=row["source_type"], source_url=row["source_url"], published_at=row["published_at"],
        available_at=row["available_at"], effective_date=row["effective_date"], confidence=0.8,
        source_independence_group=row["source_independence_group"], verification_status=row["verification_status"])
    events = [dict(tool="search_records", status="ok", record_ids=[row["record_id"], "WEB-00002"],
                   arguments={"query": "Starter"}),
              dict(tool="get_record", status="ok", record_ids=[row["record_id"]]),
              dict(tool="get_record", status="budget_exhausted", budget="max_full_records")]
    pack = EvidencePack(original_question=QUESTION, as_of=CUTOFF, evidence_items=[item],
        coverage_by_task=[TaskCoverage(task_id="T1", coverage_score=None, supporting_record_ids=[row["record_id"]])],
        retrieval_log=[json.dumps(event) for event in events])
    report = CompetitiveReport(question=QUESTION, as_of=CUTOFF, executive_summary="Supplied observation only (C1).",
        key_findings=["C1 is one source assertion."], competitor_comparison=["Insufficient comparative coverage."],
        strategic_implications=["No supported strategic ranking."], limitations=["Synthetic single observation."],
        claims=[ReportClaim(claim_id="C1", text=row["summary"], claim_type="factual",
                            supporting_record_ids=[row["record_id"]], confidence=0.8)], sources=[row["record_id"]])
    issues = [] if decision == "PASS" else [ReviewIssue(issue_id="I1", severity="major",
        issue_type="missing_required_information", affected_claim_ids=["C1"], related_record_ids=[row["record_id"]],
        explanation="Comparison lacks competitor coverage", suggested_correction="Collect comparable evidence")]
    review = ReviewResult(decision=decision, completeness_score=1 if decision=="PASS" else 0.5,
        evidence_support_score=1, temporal_consistency_score=1, flagged_issues=issues,
        reviewed_question=QUESTION, reviewed_report_hash=report_fingerprint(report))
    return plan, pack, report, review


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    send = AsyncMock(side_effect=AssertionError("No live HTTP requests allowed"))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    yield
    send.assert_not_called()


@pytest.fixture
def settings(tmp_path):
    return load_config(environ={"OPENAI_API_KEY": "private-test-token", "OPENAI_MODEL": "gpt-5-mini",
                               "DATA_DIR": str(PUBLIC), "RUNS_DIR": str(tmp_path / "runs")})


@pytest.fixture
def flow(monkeypatch):
    state = SimpleNamespace(order=[], inputs={}, outputs=outputs(), fail=None, usage=True)
    for index, name in enumerate(["planner", "researcher", "analyst", "reviewer"]):
        async def execute(*args, _name=name, _index=index):
            state.order.append(_name)
            state.inputs[_name] = args
            if state.fail == _name:
                if _name == "researcher":
                    observe_retrieval_event(dict(tool="search_records", status="ok", record_ids=["WEB-00001"]))
                raise RuntimeError("private-test-token Authorization: Bearer secret; sk-provider-secret")
            if state.usage:
                observe_sdk_result(SimpleNamespace(context_wrapper=SimpleNamespace(usage=Usage(
                    requests=1, input_tokens=100, output_tokens=20, total_tokens=120))))
            return state.outputs[_index].model_copy(deep=True)
        monkeypatch.setattr(pipeline, name, AsyncMock(side_effect=execute))
    return state


def run(settings, **kwargs):
    return asyncio.run(pipeline.run_pipeline(QUESTION, CUTOFF, settings=settings, **kwargs))


@pytest.mark.parametrize("decision,status", [("PASS", "completed"), ("FAIL", "completed_review_failed")])
def test_fixed_order_exactly_once_pass_or_fail(flow, settings, decision, status):
    flow.outputs = outputs(decision)
    before = flow.outputs[2].model_dump_json()
    result = run(settings, run_id="initial")
    assert flow.order == ["planner", "researcher", "analyst", "reviewer"]
    for name in flow.order:
        getattr(pipeline, name).assert_awaited_once()
    assert result.execution_status == status and result.review_result.decision == decision
    assert flow.outputs[2].model_dump_json() == before
    directory = settings.runs_dir / "initial"
    expected = {"plan.json", "evidence_pack.json", "report.json", "review.json", "metrics.json", "metadata.json",
                "report.md", "manifest.json", "run_result.json", "retrieval_logs.json", "claim_traceability.json", "evaluation_pending.json"}
    assert expected.issubset({p.name for p in directory.iterdir()})
    assert RunResult.model_validate_json((directory / "run_result.json").read_text()) == result
    verify_manifest(directory)
    assert result.total_metrics.input_tokens == 400 and result.total_metrics.output_tokens == 80
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["totals"]["total_tokens"] == 480 and metrics["totals"]["llm_requests"] == 4
    assert metrics["totals"]["estimated_cost_usd"] is None
    assert metrics["retrieval"]["retrieved_but_not_selected"] == ["WEB-00002"]
    assert metrics["retrieval"]["budget_exhaustion_events"]
    for stage in result.stage_metrics:
        assert stage.finished_at >= stage.started_at and stage.latency_seconds >= 0
        assert stage.started_at.tzinfo == timezone.utc
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["pipeline_version"] == "v1_fixed" and metadata["dataset"]["sha256"]
    assert len(metadata["prompt_fingerprints"]) == 4
    assert metadata["reviewer_decision"] == decision


@pytest.mark.parametrize("failed,index", [("planner", 0), ("researcher", 1), ("analyst", 2), ("reviewer", 3)])
def test_execution_failure_partial_results_and_no_retry(flow, settings, failed, index):
    flow.fail = failed
    result = run(settings, run_id="failed")
    assert flow.order == ["planner", "researcher", "analyst", "reviewer"][:index + 1]
    assert result.execution_status == "execution_failed"
    directory = settings.runs_dir / "failed"
    for i, filename in enumerate(["plan.json", "evidence_pack.json", "report.json", "review.json"]):
        assert (directory / filename).exists() == (i < index)
    assert result.stage_metrics[-1].status == "failed"
    assert result.stage_metrics[-1].input_tokens is None
    assert result.total_metrics.input_tokens is None
    if failed == "researcher":
        assert json.loads((directory / "retrieval_logs.json").read_text())[0]["tool"] == "search_records"
    for path in directory.iterdir():
        text = path.read_text()
        assert "private-test-token" not in text and "sk-provider-secret" not in text and "Bearer secret" not in text
    assert json.loads((directory / "error.json").read_text())["error_type"] == "RuntimeError"
    verify_manifest(directory)


def test_prior_artifacts_exist_before_next_stage(flow, settings, monkeypatch):
    original = pipeline.analyst
    async def inspect(*args):
        directory = settings.runs_dir / "partial"
        assert (directory / "plan.json").exists() and (directory / "evidence_pack.json").exists()
        assert (directory / "researcher_metrics.json").exists()
        return await original(*args)
    monkeypatch.setattr(pipeline, "analyst", inspect)
    assert run(settings, run_id="partial").execution_status == "completed"


def test_unknown_usage_is_not_zero(flow, settings):
    flow.usage = False
    result = run(settings)
    assert all(stage.input_tokens is None and stage.output_tokens is None for stage in result.stage_metrics)
    assert result.total_metrics.input_tokens is None and result.total_metrics.estimated_cost_usd is None


def test_claim_issue_traceability_and_evaluation_isolation(flow, settings):
    flow.outputs = outputs("FAIL")
    result = run(settings, run_id="trace")
    directory = settings.runs_dir / "trace"
    report = CompetitiveReport.model_validate_json((directory / "report.json").read_text())
    review = ReviewResult.model_validate_json((directory / "review.json").read_text())
    assert report.claims[0] == flow.outputs[2].claims[0]
    assert review.flagged_issues[0] == flow.outputs[3].flagged_issues[0]
    trace = json.loads((directory / "claim_traceability.json").read_text())[0]
    assert trace["claim_id"] == "C1" and trace["supporting_record_ids"] == ["WEB-00001"]
    assert trace["relevant_companies"] is None and trace["relevant_time_scope"] is None
    pending = IndependentEvaluation.model_validate_json((directory / "evaluation_pending.json").read_text())
    assert pending.factual_accuracy is None and pending.independent_error_annotations is None
    for name, arguments in flow.inputs.items():
        assert all(not isinstance(arg, IndependentEvaluation) for arg in arguments)
        text = " ".join(arg.model_dump_json() for arg in arguments if hasattr(arg, "model_dump_json"))
        assert "independent_error_annotations" not in text and "evaluation_pending" not in text
        assert "private-test-token" not in text


def test_immutable_baseline_and_paired_reference(flow, settings):
    run(settings, run_id="baseline")
    ref = baseline_reference(settings.runs_dir, "baseline")
    assert verify_baseline_reference(settings.runs_dir, ref) == ref
    assert all(name in ref.artifact_fingerprints for name in ["report.json", "review.json", "plan.json", "evidence_pack.json"])
    pair = PairedComparison(baseline=ref)
    assert pair.incremental_input_tokens is None and pair.incremental_estimated_cost_usd is None
    assert PairedComparison.model_validate_json(pair.model_dump_json()) == pair
    before = (settings.runs_dir / "baseline/report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(settings, run_id="baseline")
    assert (settings.runs_dir / "baseline/report.json").read_bytes() == before
    (settings.runs_dir / "baseline/review.json").write_text("tampered")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_baseline_reference(settings.runs_dir, ref)


@pytest.mark.parametrize("question,cutoff,run_id", [("", CUTOFF, "x"), (QUESTION, datetime(2026, 11, 11), "x"),
                                                   (QUESTION, CUTOFF, "../escape")])
def test_invalid_request_before_agents(flow, settings, question, cutoff, run_id):
    with pytest.raises(ValueError):
        asyncio.run(pipeline.run_pipeline(question, cutoff, run_id, settings=settings))
    assert not flow.order


def test_initialization_failure_saved(flow, settings, monkeypatch):
    def bad(*args):
        raise ValueError("private-test-token")
    monkeypatch.setattr(pipeline, "load_dataset", bad)
    result = run(settings, run_id="bad-data")
    assert result.execution_status == "execution_failed" and not flow.order
    assert (settings.runs_dir / "bad-data/error.json").exists()
    assert not (settings.runs_dir / "bad-data/plan.json").exists()


def test_cli_dry_run_public_question_lookup(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: settings)
    runner = AsyncMock(side_effect=AssertionError("Must not execute"))
    monkeypatch.setattr(cli, "run_pipeline", runner)
    assert cli.main(["run", "--question-id", "Q01", "--dry-run"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["record_count"] == 310 and value["question_id"] == "Q01"
    assert value["as_of"] == CUTOFF.isoformat()
    runner.assert_not_called()
    assert not settings.runs_dir.exists()


@pytest.mark.parametrize("args", [["run", "--question-id", "Q01"], ["run", "--question", "Direct question"]])
def test_cli_live_requires_explicit_allow_api(args):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("arguments", [["--question-id", "BAD"], ["--question", " "],
    ["--question", "Test", "--as-of", "2026-11-11"], ["--question", "Test", "--run-id", "../bad"]])
def test_cli_invalid_inputs(settings, monkeypatch, arguments):
    monkeypatch.setattr(cli, "load_config", lambda: settings)
    with pytest.raises(SystemExit) as error:
        cli.main(["run", *arguments, "--dry-run"])
    assert error.value.code == 2


def test_cli_allow_api_uses_mocked_execution_only(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: settings)
    execution = AsyncMock(return_value=SimpleNamespace(run_id="mocked", execution_status="completed_review_failed",
                                                       review_result=SimpleNamespace(decision="FAIL")))
    monkeypatch.setattr(cli, "run_pipeline", execution)
    assert cli.main(["run", "--question", "A direct question", "--allow-api", "--run-id", "mocked"]) == 0
    assert execution.call_args.args[0] == "A direct question"
    assert json.loads(capsys.readouterr().out)["reviewer_decision"] == "FAIL"


def test_cli_exception_does_not_print_secret(settings, monkeypatch, capsys):
    def broken():
        raise RuntimeError("private-test-token")
    monkeypatch.setattr(cli, "load_config", broken)
    with pytest.raises(SystemExit):
        cli.main(["run", "--question", "Test", "--dry-run"])
    assert "private-test-token" not in capsys.readouterr().err


def test_real_agent_wrappers_with_mocked_sdk_usage(settings, monkeypatch):
    from agents import AgentOutputSchema, Runner
    from agents.tool_context import ToolContext
    expected = outputs()
    sequence = []
    async def sdk_response(agent, **kwargs):
        name = agent.name.lower()
        sequence.append(name)
        payload = json.loads(kwargs["input"])
        assert "independent_error_annotations" not in kwargs["input"]
        assert "evaluation_pending" not in kwargs["input"]
        assert "private-test-token" not in kwargs["input"]
        index = ["planner", "researcher", "analyst", "reviewer"].index(name)
        output = expected[index].model_copy(deep=True)
        if name == "researcher":
            from ci_agents.agents.researcher import ResearchOutcome
            output=ResearchOutcome(stop_reason='sufficient_for_question')
            function = next(t for t in agent.tools if t.name == "get_record")
            arguments = json.dumps({"record_id": "WEB-00001"})
            context = ToolContext(context=None, tool_name="get_record", tool_call_id="test", tool_arguments=arguments)
            assert json.loads(await function.on_invoke_tool(context, arguments))["record_id"] == "WEB-00001"
        if name == "reviewer":
            output.reviewed_report_hash = payload["report_fingerprint"]
        validated = AgentOutputSchema(agent.output_type).validate_json(output.model_dump_json())
        return SimpleNamespace(final_output=validated, context_wrapper=SimpleNamespace(usage=Usage(
            requests=1, input_tokens=10, output_tokens=5, total_tokens=15)))
    mock = AsyncMock(side_effect=sdk_response)
    monkeypatch.setattr(Runner, "run", mock)
    result = run(settings, run_id="sdk-integration")
    assert result.execution_status == "completed", (settings.runs_dir / "sdk-integration/error.json").read_text() if result.execution_status=="execution_failed" else ""
    assert sequence == ["planner", "researcher", "analyst", "reviewer"]
    assert result.total_metrics.input_tokens == 40 and result.total_metrics.output_tokens == 20
    assert result.stage_metrics[1].full_record_retrievals >= 1
    assert mock.await_count == 4
