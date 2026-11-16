"""Mocked review acceptance tests, not measurements of semantic detection accuracy."""
import asyncio
from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents import AgentOutputSchema
from agents.exceptions import ModelBehaviorError
import httpx
import pytest
from pydantic import ValidationError

from ci_agents.config import load_config
from ci_agents.schemas import (CompetitiveReport, EvidenceItem, EvidencePack, ReportClaim,
                               ResearchPlan, ResearchTask, ReviewIssue, ReviewResult)

module = importlib.import_module("ci_agents.agents.reviewer")
PUBLIC = Path(__file__).resolve().parents[1] / "data/public"
CUTOFF = datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)


def scenario():
    # Illustrative facts only; no benchmark answer keys.
    plan = ResearchPlan(original_question="Compare Growth pricing and explain uncertainty", as_of=CUTOFF,
                        objective="Compare supplied observations", tasks=[ResearchTask(
                            task_id="T1", research_question="Compare like-for-like Growth prices",
                            time_scope="At cutoff", evidence_requirements=["Price and unit evidence"])],
                        expected_answer_elements=["Comparable prices", "Limitations"],
                        comparison_dimensions=["Annual per-seat list price"])
    evidence = EvidencePack(original_question=plan.original_question, as_of=CUTOFF,
        evidence_items=[EvidenceItem(record_id="TEST-A", claim_id=None, company_name="NexaFlow",
            finding="Growth costs 100 USD per seat per month with annual commitment",
            quoted_or_paraphrased_evidence="100 USD/seat/month, annual commitment",
            source_title="Illustrative pricing", source_type="Official pricing page",
            source_url="https://synthetic.example/test", published_at=CUTOFF - timedelta(days=10),
            available_at=CUTOFF - timedelta(days=9), effective_date=CUTOFF - timedelta(days=5), confidence=0.8)])
    report = CompetitiveReport(question=plan.original_question, as_of=CUTOFF,
        executive_summary="Only a limited price observation is available (C1).",
        key_findings=["C1 gives an annual-commitment per-seat price."],
        competitor_comparison=["Insufficient comparable competitor coverage."],
        strategic_implications=["No supported strategic ranking."], limitations=["Only one illustrative source."],
        claims=[ReportClaim(claim_id="C1", text=evidence.evidence_items[0].finding,
                            supporting_record_ids=["TEST-A"], claim_type="factual", confidence=0.8)],
        sources=["TEST-A"])
    return plan, evidence, report


def review_for(plan, report, *, decision="PASS", issues=None, missing=None, scores=None, notes=None):
    values = dict(completeness_score=1, evidence_support_score=1, temporal_consistency_score=1)
    values.update(scores or {})
    return ReviewResult(decision=decision, **values, flagged_issues=issues or [],
                        missing_requirements=missing or [], reviewer_notes=notes or [],
                        reviewed_question=plan.original_question,
                        reviewed_report_hash=module.report_fingerprint(report))


def issue(kind="unsupported_factual_claim", severity="major", **changes):
    values = dict(issue_id="I1", severity=severity, issue_type=kind, affected_claim_ids=["C1"],
                  explanation="The cited evidence does not establish the report assertion.",
                  suggested_correction="Qualify the assertion using only the supplied evidence.",
                  related_record_ids=["TEST-A"], missing_record_ids=[])
    values.update(changes)
    return ReviewIssue(**values)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    send = AsyncMock(side_effect=AssertionError("HTTP forbidden in Reviewer tests"))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    yield
    send.assert_not_called()


@pytest.fixture
def sdk(monkeypatch):
    settings = load_config(environ={"OPENAI_API_KEY": "test-only-secret", "OPENAI_MODEL": "gpt-5-mini",
                                   "DATA_DIR": str(PUBLIC)})
    monkeypatch.setattr(module, "load_config", lambda: settings)
    runner = AsyncMock()
    monkeypatch.setattr(module.Runner, "run", runner)
    return runner


def run(plan, evidence, report):
    return asyncio.run(module.reviewer(plan, evidence, report))


def test_pass_and_strict_sdk_compatibility(sdk):
    plan, evidence, report = scenario()
    sdk.return_value = SimpleNamespace(final_output=review_for(plan, report))
    result = run(plan, evidence, report)
    assert result.decision == "PASS"
    arguments = sdk.call_args
    context = json.loads(arguments.kwargs["input"])
    assert context["research_plan"] == plan.model_dump(mode="json")
    assert context["evidence_pack"] == evidence.model_dump(mode="json")
    assert context["report"] == report.model_dump(mode="json")
    assert context["report_fingerprint"] == result.reviewed_report_hash
    assert context["review_policy"]["minimum_scores"] == module.SCORE_THRESHOLDS
    agent = arguments.args[0]
    assert agent.output_type is ReviewResult and agent.model.model == "gpt-5-mini"
    assert not agent.tools and not agent.handoffs
    assert agent.model_settings.temperature is None
    assert arguments.kwargs["max_turns"] == 1
    assert arguments.kwargs["run_config"].tracing_disabled
    schema = AgentOutputSchema(ReviewResult)
    assert schema.is_strict_json_schema()
    assert schema.validate_json(result.model_dump_json()) == result
    sdk.assert_awaited_once()


@pytest.mark.parametrize("kind", sorted(module.ISSUE_TYPES))
def test_mocked_material_semantic_failures_are_accepted(sdk, kind):
    # This checks acceptance of supplied reviewer findings, NOT detection by an LLM.
    plan, evidence, report = scenario()
    defects = {
        "unsupported_factual_claim": "NexaFlow has audited zero churn.",
        "citation_mismatch": "The pricing source proves NexaFlow leads the AI market.",
        "factual_contradiction": "Growth costs 50 USD per seat per month.",
        "temporal_inconsistency": "An expired promotion is active today.",
        "numerical_or_unit_error": "100 USD per seat equals 100 USD flat for 50 seats.",
        "missing_required_information": "The requested competitor comparison was omitted.",
        "unresolved_evidence_conflict": "Conflicting prices are settled with no uncertainty.",
        "overstated_certainty": "This observation is unquestionably complete market truth.",
        "weak_strategic_inference": "NexaFlow is the top strategic threat solely because its name comes first.",
    }
    report.claims[0].text = defects[kind]
    if kind == "temporal_inconsistency":
        evidence.evidence_items[0].end_date = CUTOFF - timedelta(days=1)
    if kind == "unresolved_evidence_conflict":
        evidence.contradictions = ["Official and third-party Growth prices disagree."]
    if kind == "missing_required_information":
        report.competitor_comparison.clear()
    if kind == "weak_strategic_inference":
        from ci_agents.schemas import InferenceClaim
        report.claims[0]=InferenceClaim(**{**report.claims[0].model_dump(),'claim_type':'inference'})
    output = review_for(plan, report, decision="FAIL", issues=[issue(kind)])
    sdk.return_value = SimpleNamespace(final_output=output)
    original_report = report.model_dump_json()
    result = run(plan, evidence, report)
    assert result.decision == "FAIL" and result.flagged_issues[0].issue_type == kind
    assert report.model_dump_json() == original_report
    sdk.assert_awaited_once()  # Valid FAIL returns normally; no feedback/regeneration.
    prompt = sdk.call_args.args[0].instructions
    assert kind in prompt
    assert "syntactically valid citations do not establish factual support" in prompt


def test_appropriate_uncertainty_and_minor_issue_can_pass(sdk):
    plan, evidence, report = scenario()
    evidence.missing_information = ["No independently corroborated competitor coverage"]
    output = review_for(plan, report, issues=[issue("missing_required_information", severity="minor",
        explanation="Nonessential heading could be clearer.")],
        notes=["Honest inability to compare is justified by supplied evidence; no unsupported winner asserted."])
    sdk.return_value = SimpleNamespace(final_output=output)
    assert run(plan, evidence, report).decision == "PASS"


@pytest.mark.parametrize("field,ids", [("affected_claim_ids", ["FAKE-CLAIM"]),
                                       ("related_record_ids", ["FAKE-RECORD"]),
                                       ("missing_record_ids", ["FAKE-RECORD"])])
def test_invalid_references_rejected(sdk, field, ids):
    plan, evidence, report = scenario()
    output = review_for(plan, report, decision="FAIL", issues=[issue(**{field: ids})])
    sdk.return_value = SimpleNamespace(final_output=output)
    with pytest.raises(module.ReviewerOutputError, match="unknown"):
        run(plan, evidence, report)


@pytest.mark.parametrize("field", list(module.SCORE_THRESHOLDS))
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_scores(sdk, field, value):
    plan, evidence, report = scenario()
    output = review_for(plan, report).model_copy(update={field: value})
    sdk.return_value = SimpleNamespace(final_output=output)
    with pytest.raises(module.ReviewerOutputError, match="Invalid ReviewResult"):
        run(plan, evidence, report)


def test_duplicate_issue_ids(sdk):
    plan, evidence, report = scenario()
    output = review_for(plan, report, decision="FAIL", issues=[issue()])
    output.flagged_issues.append(issue())
    sdk.return_value = SimpleNamespace(final_output=output)
    with pytest.raises(module.ReviewerOutputError, match="duplicate issue IDs"):
        run(plan, evidence, report)


def test_duplicate_defects_with_distinct_ids(sdk):
    plan, evidence, report = scenario()
    output = review_for(plan, report, decision="FAIL", issues=[issue(), issue(issue_id="I2")])
    sdk.return_value = SimpleNamespace(final_output=output)
    with pytest.raises(module.ReviewerOutputError, match="Duplicate review defects"):
        run(plan, evidence, report)


@pytest.mark.parametrize("severity", ["critical", "major"])
def test_pass_with_material_issue_is_invalid(sdk, severity):
    plan, evidence, report = scenario()
    sdk.return_value = SimpleNamespace(final_output=review_for(plan, report, issues=[issue(severity=severity)]))
    with pytest.raises(module.ReviewerOutputError, match="expected FAIL"):
        run(plan, evidence, report)


@pytest.mark.parametrize("minor", [False, True])
def test_fail_without_policy_reason_is_invalid(sdk, minor):
    plan, evidence, report = scenario()
    output = review_for(plan, report, decision="FAIL", issues=[issue(severity="minor")] if minor else [])
    sdk.return_value = SimpleNamespace(final_output=output)
    with pytest.raises(module.ReviewerOutputError, match="expected PASS"):
        run(plan, evidence, report)


@pytest.mark.parametrize("field,threshold", list(module.SCORE_THRESHOLDS.items()))
def test_threshold_boundaries_and_low_score_failure(sdk, field, threshold):
    plan, evidence, report = scenario()
    sdk.return_value = SimpleNamespace(final_output=review_for(plan, report, scores={field: threshold}))
    assert run(plan, evidence, report).decision == "PASS"
    sdk.return_value = SimpleNamespace(final_output=review_for(plan, report, decision="FAIL",
        scores={field: threshold - 0.001}, notes=[f"{field} below threshold for the supplied report."]))
    assert run(plan, evidence, report).decision == "FAIL"
    sdk.return_value = SimpleNamespace(final_output=review_for(plan, report, scores={field: threshold - 0.001}))
    with pytest.raises(module.ReviewerOutputError, match="expected FAIL"):
        run(plan, evidence, report)


def test_low_scores_require_explanation(sdk):
    plan, evidence, report = scenario()
    sdk.return_value = SimpleNamespace(final_output=review_for(plan, report, decision="FAIL", scores={"completeness_score": 0.5}))
    with pytest.raises(module.ReviewerOutputError, match="explicit explanation"):
        run(plan, evidence, report)


def test_missing_essential_requirement_forces_fail(sdk):
    plan, evidence, report = scenario()
    sdk.return_value = SimpleNamespace(final_output=review_for(plan, report, decision="FAIL", missing=["Essential price comparison"]))
    assert run(plan, evidence, report).decision == "FAIL"
    sdk.return_value = SimpleNamespace(final_output=review_for(plan, report, missing=["Essential price comparison"]))
    with pytest.raises(module.ReviewerOutputError, match="expected FAIL"):
        run(plan, evidence, report)


@pytest.mark.parametrize("field,value", [("reviewed_question", "Other question"),
                                         ("reviewed_report_hash", "0" * 64),
                                         ("reviewed_report_hash", None)])
def test_wrong_review_identity(sdk, field, value):
    plan, evidence, report = scenario()
    output = review_for(plan, report)
    setattr(output, field, value)
    sdk.return_value = SimpleNamespace(final_output=output)
    with pytest.raises(module.ReviewerOutputError, match="fingerprint"):
        run(plan, evidence, report)


def test_hash_changes_with_report_content():
    _, _, report = scenario()
    before = module.report_fingerprint(report)
    report.limitations.append("Another limitation")
    assert module.report_fingerprint(report) != before
    assert module.report_fingerprint(CompetitiveReport.model_validate_json(report.model_dump_json())) == module.report_fingerprint(report)


@pytest.mark.parametrize("value", [None, {}, "PASS"])
def test_invalid_structured_output(sdk, value):
    plan, evidence, report = scenario()
    sdk.return_value = SimpleNamespace(final_output=value)
    with pytest.raises(module.ReviewerOutputError, match="structured ReviewResult"):
        run(plan, evidence, report)


def test_invalid_issue_category_and_severity(sdk):
    plan, evidence, report = scenario()
    sdk.return_value = SimpleNamespace(final_output=review_for(plan, report, decision="FAIL", issues=[issue("other")]))
    with pytest.raises(module.ReviewerOutputError, match="issue_type"):
        run(plan, evidence, report)
    output = review_for(plan, report, decision="FAIL", issues=[issue()])
    output.flagged_issues[0] = output.flagged_issues[0].model_copy(update={"severity": "urgent"})
    sdk.return_value = SimpleNamespace(final_output=output)
    with pytest.raises(module.ReviewerOutputError, match="Invalid ReviewResult"):
        run(plan, evidence, report)


@pytest.mark.parametrize("error,expected", [(RuntimeError("test-only-secret payload"), module.ReviewerError),
                                           (ModelBehaviorError("bad output"), module.ReviewerOutputError)])
def test_sdk_failure_without_retry(sdk, error, expected):
    plan, evidence, report = scenario()
    sdk.side_effect = error
    with pytest.raises(expected) as captured:
        run(plan, evidence, report)
    assert "test-only-secret" not in str(captured.value)
    assert captured.value.__suppress_context__
    sdk.assert_awaited_once()


@pytest.mark.parametrize("case", ["type", "question", "report_cutoff", "pack_cutoff", "post_cutoff"])
def test_invalid_handoffs_rejected_before_api(sdk, case):
    plan, evidence, report = scenario()
    if case == "type":
        report = {}
    elif case == "question":
        report.question = "Different"
    elif case == "report_cutoff":
        report.as_of = CUTOFF - timedelta(days=1)
    elif case == "pack_cutoff":
        evidence.as_of = CUTOFF + timedelta(days=1)
    else:
        evidence.evidence_items[0].available_at = CUTOFF + timedelta(days=1)
    with pytest.raises(module.ReviewerInputError):
        run(plan, evidence, report)
    sdk.assert_not_called()


def test_no_private_retrieval_or_other_agent_access(sdk, monkeypatch):
    from ci_agents import dataset, retrieval
    def forbidden(*args, **kwargs):
        raise AssertionError("No research or other-stage calls allowed")
    monkeypatch.setattr(dataset, "load_dataset", forbidden)
    for name in ["search_records", "get_record", "get_pricing_history"]:
        monkeypatch.setattr(retrieval, name, forbidden)
    for name in ["planner", "researcher", "analyst"]:
        monkeypatch.setattr(importlib.import_module("ci_agents.agents." + name), name, forbidden)
    original = Path.open
    opened = []
    def guard(path, *args, **kwargs):
        assert path.name == "reviewer.md"
        opened.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    plan, evidence, report = scenario()
    before = [model.model_dump_json() for model in (plan, evidence, report)]
    sdk.return_value = SimpleNamespace(final_output=review_for(plan, report, decision="FAIL", issues=[issue()]))
    assert run(plan, evidence, report).decision == "FAIL"
    assert before == [model.model_dump_json() for model in (plan, evidence, report)]
    assert opened == ["reviewer.md"]
    sdk.assert_awaited_once()


def test_schema_extensions_are_backwards_compatible():
    old_issue = ReviewIssue(issue_id="I1", severity="minor", issue_type="completeness",
                            explanation="Small omission", suggested_correction="Clarify")
    assert old_issue.related_record_ids == []
    legacy = ReviewResult(decision="PASS", completeness_score=1, evidence_support_score=1, temporal_consistency_score=1)
    assert legacy.reviewed_question is None and legacy.reviewed_report_hash is None
    assert ReviewResult.model_validate_json(legacy.model_dump_json()) == legacy
    with pytest.raises(ValidationError):
        legacy.reviewed_report_hash = "not-a-sha256"
