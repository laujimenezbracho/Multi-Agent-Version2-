import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ci_agents.schemas import (
    ResearchTask, ResearchPlan, EvidenceItem, EvidencePack, TaskCoverage,
    ReportClaim, CompetitiveReport, ReviewIssue, ReviewResult,
    StageMetrics, TotalMetrics, RunResult,
)

CUTOFF = "2026-11-11T23:59:59Z"


@pytest.fixture
def examples():
    task = ResearchTask(task_id="T1", research_question="What is the price?",
                        companies=["NexaFlow"], categories=["Pricing"],
                        time_scope="Available by November 11, 2026", evidence_requirements=["Official price"])
    plan = ResearchPlan(original_question="Compare prices", as_of=CUTOFF,
                        objective="Compare like-for-like prices", tasks=[task],
                        expected_answer_elements=["Currency"], comparison_dimensions=["Annual price"])
    item = EvidenceItem(record_id="WEB-00001", claim_id=None, company_name="NexaFlow",
                        finding="Price is 79 USD", quoted_or_paraphrased_evidence="79 USD per seat",
                        source_title="Pricing", source_type="Official pricing page",
                        source_url="https://synthetic.example/pricing", published_at=CUTOFF,
                        available_at=CUTOFF, effective_date=None, confidence=0.9)
    coverage = TaskCoverage(task_id="T1", coverage_score=None, supporting_record_ids=[item.record_id])
    pack = EvidencePack(original_question=plan.original_question, as_of=CUTOFF,
                        evidence_items=[item], coverage_by_task=[coverage])
    claim = ReportClaim(claim_id="C1", text="The price is 79 USD", supporting_record_ids=[item.record_id],
                        claim_type="factual", confidence=0.9)
    report = CompetitiveReport(question=plan.original_question, as_of=CUTOFF,
                               executive_summary="Price comparison", claims=[claim], sources=[item.record_id])
    issue = ReviewIssue(issue_id="I1", severity="minor", issue_type="completeness",
                        affected_claim_ids=["C1"], explanation="Billing period omitted",
                        suggested_correction="State annual commitment")
    review = ReviewResult(decision="FAIL", completeness_score=0.8, evidence_support_score=1,
                          temporal_consistency_score=1, flagged_issues=[issue])
    metrics = StageMetrics(stage_name="reviewer", model="test-model", status="success")
    totals = TotalMetrics()
    run = RunResult(run_id="R1", experiment_id="E1", question_id="Q01", configuration="v1",
                    model="test-model", as_of=CUTOFF, stage_metrics=[metrics],
                    final_report=report, review_result=review, total_metrics=totals)
    return [task, plan, item, coverage, pack, claim, report, issue, review, metrics, totals, run]


def test_every_contract_and_json_round_trip(examples):
    for example in examples:
        encoded = example.model_dump_json()
        assert type(example).model_validate_json(encoded) == example
        assert type(example).model_validate(example.model_dump()) == example
        assert isinstance(json.loads(encoded), dict)


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf")])
def test_score_bounds(examples, score):
    for example in examples:
        fields = set(type(example).model_fields) & {"confidence", "coverage_score", "completeness_score",
                                            "evidence_support_score", "temporal_consistency_score"}
        for field in fields:
            payload = example.model_dump()
            payload[field] = score
            with pytest.raises(ValidationError):
                type(example).model_validate(payload)


@pytest.mark.parametrize("identifier", ["", "   ", "\n\t"])
def test_empty_identifiers(examples, identifier):
    for example in examples:
        for field in ["task_id", "record_id", "claim_id", "issue_id", "run_id", "experiment_id", "question_id"]:
            if field in type(example).model_fields:
                payload = example.model_dump()
                payload[field] = identifier
                with pytest.raises(ValidationError):
                    type(example).model_validate(payload)


@pytest.mark.parametrize("index,field", [(1, "tasks"), (4, "evidence_items"), (6, "claims")])
def test_duplicate_ids(examples, index, field):
    example = examples[index]
    payload = example.model_dump()
    payload[field] *= 2
    with pytest.raises(ValidationError, match="Duplicate"):
        type(example).model_validate(payload)


def test_factual_claim_needs_support_and_inference_is_distinct():
    payload = dict(claim_id="C1", text="Potential strategic threat", confidence=0.5)
    with pytest.raises(ValidationError, match="Factual claims require"):
        ReportClaim(**payload, claim_type="factual")
    inference = ReportClaim(**payload, claim_type="inference")
    assert inference.supporting_record_ids == []
    with pytest.raises(ValidationError):
        ReportClaim(**payload, claim_type="opinion")


def test_report_sources_must_include_claim_citations(examples):
    payload = examples[6].model_dump()
    payload["sources"] = []
    with pytest.raises(ValidationError, match="report sources"):
        CompetitiveReport.model_validate(payload)


def test_timezone_normalization_and_naive_rejection(examples):
    for example in examples:
        for field in ["as_of", "published_at", "available_at", "effective_date"]:
            if field not in type(example).model_fields:
                continue
            payload = example.model_dump()
            payload[field] = "2026-11-11T18:59:59-05:00"
            # Evidence timestamp equality and RunResult report equality are preserved.
            result = type(example).model_validate(payload)
            value = getattr(result, field)
            assert value == datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)
            assert value.tzinfo == timezone.utc
            payload[field] = datetime(2026, 11, 11)
            with pytest.raises(ValidationError):
                type(example).model_validate(payload)


def test_unknown_values_and_independent_defaults(examples):
    assert examples[2].claim_id is None
    assert examples[2].effective_date is None
    assert examples[3].coverage_score is None
    metrics = StageMetrics(stage_name="analyst", status="failed", error_message="Not run")
    assert metrics.input_tokens is None and metrics.estimated_cost_usd is None
    assert TotalMetrics().model_dump() == dict(input_tokens=None, output_tokens=None,
                                             latency_seconds=None, estimated_cost_usd=None)
    first = ResearchTask(task_id="A", research_question="Q", time_scope="Now")
    second = ResearchTask(task_id="B", research_question="Q", time_scope="Now")
    first.companies.append("NexaFlow")
    assert second.companies == []
    run = RunResult(run_id="R", experiment_id="E", question_id="Q", configuration="single",
                    model="test", as_of=CUTOFF)
    assert run.final_report is None and run.review_result is None


@pytest.mark.parametrize("index,field,value", [
    (8, "decision", "MAYBE"), (7, "severity", "urgent"), (11, "configuration", "parallel"),
    (9, "input_tokens", -1), (9, "input_tokens", True), (9, "output_tokens", 1.5),
    (9, "latency_seconds", -1), (9, "estimated_cost_usd", float("inf")),
    (9, "status", "whatever"),
])
def test_invalid_categories_and_metrics(examples, index, field, value):
    payload = examples[index].model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        type(examples[index]).model_validate(payload)


def test_nested_validation(examples):
    payload = examples[-1].model_dump()
    payload["final_report"]["claims"][0]["confidence"] = 2
    with pytest.raises(ValidationError, match="final_report.claims.0.confidence"):
        RunResult.model_validate(payload)
    payload = examples[4].model_dump()
    payload["evidence_items"][0]["available_at"] = "2026-11-12T00:00:00Z"
    with pytest.raises(ValidationError, match="available_at"):
        EvidencePack.model_validate(payload)


def test_unexpected_fields_rejected(examples):
    for example in examples:
        payload = example.model_dump()
        payload["unexpected"] = "value"
        with pytest.raises(ValidationError, match="Extra inputs"):
            type(example).model_validate(payload)


def test_generated_schemas_have_closed_typed_objects(examples):
    # SDK strict-schema conversion requires closed objects without arbitrary maps.
    def inspect(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert "properties" in node
            for value in node.values():
                inspect(value)
        elif isinstance(node, list):
            for value in node:
                inspect(value)
    for example in examples:
        inspect(type(example).model_json_schema())
