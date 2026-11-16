import asyncio
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from ci_agents import cli, pilot, pipeline
from ci_agents.config import load_config
from ci_agents.dataset import load_dataset
from ci_agents.persistence import baseline_reference, verify_manifest
from ci_agents.schemas import CompetitiveReport, EvidenceItem, EvidencePack, ReportClaim, ResearchPlan, ResearchTask, ReviewIssue, ReviewResult

PUBLIC = Path(__file__).resolve().parents[1] / "data/public"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    send = AsyncMock(side_effect=AssertionError("No HTTP allowed"))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    yield
    send.assert_not_called()


@pytest.fixture
def settings(tmp_path):
    return load_config(environ={"OPENAI_API_KEY": "", "OPENAI_MODEL": "gpt-5-mini", "DATA_DIR": str(PUBLIC),
        "RUNS_DIR": str(tmp_path / "runs")}, require_api_key=False)


@pytest.fixture
def fake_agents(monkeypatch):
    state = {"order": [], "fail_q01": False, "review_fail": False}
    async def planner(question, cutoff):
        state["order"].append(("planner", question))
        if state["fail_q01"] and question == load_dataset(PUBLIC).questions[0]["user_question"]:
            raise RuntimeError("private-api-secret")
        return ResearchPlan(original_question=question, as_of=cutoff, objective="Gather public evidence",
            tasks=[ResearchTask(task_id="T1", research_question=question, time_scope="At cutoff")])
    async def researcher(plan, service):
        state["order"].append(("researcher", plan.original_question))
        row = service.get_record("WEB-00001", as_of=plan.as_of)
        item = EvidenceItem(record_id=row["record_id"], claim_id=row["claim_id"], company_name=row["company_name"],
            finding=row["summary"], quoted_or_paraphrased_evidence=row["details"], source_title=row["document_title"],
            source_type=row["source_type"], source_url=row["source_url"], published_at=row["published_at"],
            available_at=row["available_at"], effective_date=row["effective_date"], confidence=0.7,
            end_date=row.get('end_date'),source_independence_group=row.get('source_independence_group'),
            supersedes_record_id=row.get('supersedes_record_id'),verification_status=row.get('verification_status'),
            feature_status=row.get('feature_status'))
        return EvidencePack(original_question=plan.original_question, as_of=plan.as_of, evidence_items=[item],
                            missing_information=["Only one source in this mocked scenario"],
                            retrieval_log=[json.dumps({'tool':'get_record','status':'ok','record_ids':[row['record_id']]})])
    async def analyst(plan, evidence):
        state["order"].append(("analyst", plan.original_question))
        return CompetitiveReport(question=plan.original_question, as_of=plan.as_of,
            executive_summary="Cannot conclude from this mocked evidence.",
            key_findings=["No evidence"], competitor_comparison=["Not established"],
            strategic_implications=["Unknown"], limitations=["Incomplete mocked inputs"],
            claims=[ReportClaim(claim_id="C1", text=evidence.evidence_items[0].finding, claim_type="factual",
                                supporting_record_ids=["WEB-00001"], confidence=0.7)], sources=["WEB-00001"])
    async def reviewer(plan, evidence, report):
        state["order"].append(("reviewer", plan.original_question))
        from ci_agents.agents.reviewer import report_fingerprint
        return ReviewResult(decision="FAIL" if state["review_fail"] else "PASS", completeness_score=0.5 if state["review_fail"] else 1,
                            evidence_support_score=1, temporal_consistency_score=1,
                            reviewed_question=plan.original_question, reviewed_report_hash=report_fingerprint(report),
                            flagged_issues=[ReviewIssue(issue_id="I1", severity="major", issue_type="missing_required_information",
                                affected_claim_ids=["C1"], related_record_ids=["WEB-00001"], explanation="Incomplete comparison",
                                suggested_correction="Obtain comparable evidence")] if state["review_fail"] else [])
    for name, handler in (("planner", planner), ("researcher", researcher), ("analyst", analyst), ("reviewer", reviewer)):
        monkeypatch.setattr(pipeline, name, AsyncMock(side_effect=handler))
    return state


def prepared(settings, **kwargs):
    return pilot.preflight(replace(settings, openai_api_key="private-api-secret"), **kwargs)


def execute(preparation, **kwargs):
    return asyncio.run(pilot.run_pilot(preparation, allow_api=True, **kwargs))


def test_public_csv_selection_exact_question_text(settings):
    assert settings.openai_api_key == ""
    result = pilot.preflight(settings)
    with (PUBLIC / "public_questions.csv").open(encoding="utf-8-sig", newline="") as stream:
        expected = {row["question_id"]: row["user_question"] for row in csv.DictReader(stream, delimiter=";")}
    assert result.questions == tuple((qid, expected[qid]) for qid in ("Q01", "Q15"))
    assert pilot.preflight(settings, ["Q15"]).questions == (("Q15", expected["Q15"]),)
    assert result.as_of == settings.evaluation_cutoff
    assert not result.output_dir.exists()


@pytest.mark.parametrize("ids", [[], ["BAD"], ["Q01", "Q01"], ["Q02"]])
def test_invalid_selected_questions(settings, ids):
    with pytest.raises(pilot.PilotPreflightError, match="question IDs"):
        pilot.preflight(settings, ids)


def test_dry_run_without_credentials(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: settings)
    runner = AsyncMock(side_effect=AssertionError("Dry-run must never execute"))
    monkeypatch.setattr(pilot, "run_pilot", runner)
    assert cli.main(["pilot", "--dry-run"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["question_ids"] == ["Q01", "Q15"] and summary["preflight"] == "passed"
    assert summary["pricing_rates_configured"] is False and summary["cost_warning"]
    runner.assert_not_called()
    assert not settings.runs_dir.exists()


def test_live_requires_flag_and_present_key(settings, monkeypatch):
    with pytest.raises(SystemExit) as caught:
        cli.main(["pilot"])
    assert caught.value.code == 2
    with pytest.raises(pilot.PilotPreflightError, match="OPENAI_API_KEY"):
        pilot.preflight(settings, live=True)
    check = pilot.preflight(settings)
    with pytest.raises(pilot.PilotPreflightError, match="allow_api"):
        asyncio.run(pilot.run_pilot(check))


def test_preflight_missing_file_and_retrieval_error(settings, monkeypatch, tmp_path):
    folder = tmp_path / "incomplete"
    folder.mkdir()
    with pytest.raises(pilot.PilotPreflightError, match="Missing public dataset file"):
        pilot.preflight(replace(settings, data_dir=folder))
    def bad(*args):
        raise RuntimeError("private-api-secret")
    monkeypatch.setattr(pilot, "RetrievalStore", bad)
    with pytest.raises(pilot.PilotPreflightError, match="FTS5"):
        pilot.preflight(settings)


def test_preflight_schema_and_prompt_failures(settings, monkeypatch):
    def bad(*args):
        raise RuntimeError("secret")
    with monkeypatch.context() as patch:
        patch.setattr(pilot, "AgentOutputSchema", bad)
        with pytest.raises(pilot.PilotPreflightError, match="SDK strict"):
            pilot.preflight(settings)
    monkeypatch.setattr(pilot, "prompt_fingerprints", bad)
    with pytest.raises(pilot.PilotPreflightError, match="prompts"):
        pilot.preflight(settings)


def test_invalid_output_and_baseline_path(settings, tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("blocked")
    with pytest.raises(pilot.PilotPreflightError, match="blocked"):
        pilot.preflight(settings, output_dir=blocked / "child")
    with pytest.raises(pilot.PilotPreflightError, match="separate"):
        pilot.preflight(settings, output_dir=PUBLIC)
    baseline = tmp_path / "old-run"
    baseline.mkdir()
    (baseline / "manifest.json").write_text("existing baseline")
    with pytest.raises(pilot.PilotPreflightError, match="existing baseline"):
        pilot.preflight(settings, output_dir=baseline)
    assert (baseline / "manifest.json").read_text() == "existing baseline"


def test_output_permission_probe_failure(settings, monkeypatch):
    def denied(*args):
        raise PermissionError("secret")
    monkeypatch.setattr(pilot.os, "link", denied)
    with pytest.raises(pilot.PilotPreflightError, match="writable"):
        pilot.preflight(settings)


def test_sequential_runs_unique_ids_summaries_and_baselines(settings, fake_agents):
    before = pilot.dataset_fingerprints(PUBLIC)
    preparation = prepared(settings)
    result = execute(preparation)
    directory = Path(result["pilot_directory"])
    rows = result["rows"]
    assert [row["question_id"] for row in rows] == ["Q01", "Q15"]
    assert len({row["run_id"] for row in rows}) == 2
    assert [stage for stage, _ in fake_agents["order"]] == ["planner", "researcher", "analyst", "reviewer"] * 2
    assert all(row["execution_status"] == "completed" for row in rows)
    assert all(row["total_tokens"] is None and row["estimated_cost_usd"] is None for row in rows)
    assert all(row["evaluation_ready"] is True for row in rows)
    assert all(row["cited_evidence_records"] == 1 and row["report_claims"] == 1 for row in rows)
    with (directory / "pilot_summary.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 2 and csv_rows[0]["total_tokens"] == ""
    markdown = (directory / "pilot_summary.md").read_text()
    assert "not independent factual accuracy" in markdown and "null" in markdown
    assert "reviewer_completeness_score" in markdown
    verify_manifest(directory)
    for row in rows:
        run_dir = Path(row["artifact_directory"])
        reference = baseline_reference(run_dir.parent, run_dir.name)
        assert reference.baseline_run_id == row["run_id"]
        assert reference.artifact_fingerprints["report.json"] and reference.artifact_fingerprints["review.json"]
        assert json.loads((run_dir / "evaluation_pending.json").read_text())["factual_accuracy"] is None
    assert before == pilot.dataset_fingerprints(PUBLIC)
    assert not list(settings.runs_dir.rglob("revised_report.json"))
    with pytest.raises(FileExistsError):
        execute(preparation)
    next_prep = prepared(settings)
    assert set(next_prep.run_ids).isdisjoint(preparation.run_ids)


def test_partial_execution_failure_continues_without_retry(settings, fake_agents):
    fake_agents["fail_q01"] = True
    result = execute(prepared(settings))
    first, second = result["rows"]
    assert first["execution_status"] == "execution_failed" and first["reviewer_decision"] is None
    assert first["report_claims"] is None and first["total_tokens"] is None
    assert second["execution_status"] == "completed"
    assert len(fake_agents["order"]) == 5  # One failed Planner, then exactly four stages.
    assert (Path(first["artifact_directory"]) / "error.json").is_file()
    assert not (Path(first["artifact_directory"]) / "report.json").exists()
    for file in Path(result["pilot_directory"]).rglob("*.json"):
        assert "private-api-secret" not in file.read_text()


def test_reviewer_fail_is_normal_completion(settings, fake_agents):
    fake_agents["review_fail"] = True
    rows = execute(prepared(settings))["rows"]
    assert all(row["execution_status"] == "completed_review_failed" and row["reviewer_decision"] == "FAIL" for row in rows)
    assert all(row["reviewer_issues"] == 1 and row["evaluation_ready"] for row in rows)
    assert len(fake_agents["order"]) == 8


def test_between_question_budget_stops_on_unknown_usage(settings, fake_agents):
    result = execute(prepared(settings), max_total_tokens=1)
    assert result["rows"][0]["execution_status"] == "completed"
    assert result["rows"][1]["execution_status"] == "not_started_budget"
    assert result["rows"][1]["total_tokens"] is None
    assert len(fake_agents["order"]) == 4
    with pytest.raises(pilot.PilotPreflightError, match="pricing rates"):
        execute(prepared(settings), max_estimated_cost=1)


def test_pipeline_exception_does_not_retry(settings, monkeypatch):
    mock = AsyncMock(side_effect=RuntimeError("private-api-secret"))
    monkeypatch.setattr(pilot, "run_pipeline", mock)
    result = execute(prepared(settings))
    assert mock.await_count == 2
    assert all(row["execution_status"] == "execution_failed" for row in result["rows"])
    assert "private-api-secret" not in (Path(result["pilot_directory"]) / "pilot_errors.json").read_text()


def test_readiness_does_not_modify_baselines(settings, fake_agents):
    result = execute(prepared(settings))
    directory = Path(result["rows"][0]["artifact_directory"])
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    assert pilot.evaluation_readiness(directory)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    placeholder = directory / "evaluation_pending.json"
    content = json.loads(placeholder.read_text())
    content["factual_accuracy"] = 1
    placeholder.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="fingerprint"):
        pilot.evaluation_readiness(directory)


def test_private_files_not_opened(settings, monkeypatch):
    original = Path.open
    def guard(path, *args, **kwargs):
        assert "evaluation_private" not in path.parts and not path.name.startswith("private_")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    assert pilot.preflight(settings).questions


def test_cli_overrides_and_model_display_before_mocked_execution(settings, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: replace(settings, openai_api_key="test-key"))
    async def fake(preparation, **kwargs):
        displayed = json.loads(capsys.readouterr().out)
        assert displayed["model"] == "gpt-5-mini" and displayed["cost_warning"]
        assert [qid for qid, _ in preparation.questions] == ["Q15"]
        assert preparation.as_of.isoformat() == "2025-09-30T23:59:59+00:00"
        assert kwargs["allow_api"] is True
        return {"rows": [], "pilot_directory": "mocked"}
    monkeypatch.setattr(pilot, "run_pilot", fake)
    assert cli.main(["pilot", "--allow-api", "--question-ids", "Q15", "--as-of", "2025-09-30T23:59:59Z",
                     "--output-dir", str(tmp_path / "chosen")]) == 0


@pytest.mark.parametrize("budget", ["tokens", "cost"])
def test_known_usage_budget_stops_next_question(settings, fake_agents, monkeypatch, budget):
    from agents import Usage
    from types import SimpleNamespace
    from ci_agents.telemetry import observe_sdk_result, PricingConfig
    for name in ["planner", "researcher", "analyst", "reviewer"]:
        original = getattr(pipeline, name)
        async def measured(*args, _original=original):
            result = await _original(*args)
            observe_sdk_result(SimpleNamespace(context_wrapper=SimpleNamespace(usage=Usage(
                requests=1, input_tokens=10, output_tokens=0, total_tokens=10))))
            return result
        monkeypatch.setattr(pipeline, name, measured)
    rates = PricingConfig(model=settings.openai_model, source="fixture-only", source_date="2026-11-11",
                          input_usd_per_million=1, output_usd_per_million=1)
    check = prepared(settings, pricing=rates)
    result = execute(check, **({"max_total_tokens": 40} if budget == "tokens" else {"max_estimated_cost": 0.000001}))
    assert result["rows"][0]["total_tokens"] == 40
    assert result["rows"][1]["execution_status"] == "not_started_budget"
    assert len(fake_agents["order"]) == 4


@pytest.mark.parametrize("arguments", [dict(max_total_tokens=0), dict(max_total_tokens=True),
                                       dict(max_estimated_cost=-1), dict(max_estimated_cost=float("nan"))])
def test_invalid_budgets_rejected_before_execution(settings, fake_agents, arguments):
    with pytest.raises(pilot.PilotPreflightError):
        execute(prepared(settings), **arguments)
    assert not fake_agents["order"]


def test_pilot_directory_collision_does_not_overwrite(settings, tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(pilot, "uuid4", lambda: SimpleNamespace(hex="collision"))
    root = tmp_path / "pilots"
    existing = root / "pilot-collision"
    existing.mkdir(parents=True)
    marker = existing / "original.txt"
    marker.write_text("preserved")
    with pytest.raises(pilot.PilotPreflightError, match="already exists"):
        pilot.preflight(settings, output_dir=root)
    assert marker.read_text() == "preserved"
