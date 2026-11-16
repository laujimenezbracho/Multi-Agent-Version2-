"""Mocked flow/validation tests; these do not measure LLM reasoning accuracy."""
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

from ci_agents.config import load_config
from ci_agents.schemas import (
    CompetitiveReport, EvidenceItem, EvidencePack, ReportClaim, ResearchPlan,
    ResearchTask, TaskCoverage,
)

module = importlib.import_module("ci_agents.agents.analyst")
PUBLIC = Path(__file__).resolve().parents[1] / "data/public"
CUTOFF = datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)
Q01 = "Which competitor has the lowest current Growth plan price?"
Q15 = "What are the top 3 competitive threats for PulseGrid?"


def scenario(question=Q01):
    # Deliberately illustrative fixture observations, not benchmark answer keys.
    plan = ResearchPlan(original_question=question, as_of=CUTOFF, objective="Compare supplied evidence",
                        tasks=[ResearchTask(task_id="T1", research_question=question,
                                            companies=["NexaFlow", "OrbitDesk"], categories=["Pricing"],
                                            time_scope="At the cutoff", evidence_requirements=["Comparable plan prices"])],
                        expected_answer_elements=["Answer with citations and uncertainty"],
                        comparison_dimensions=["Pricing/value", "Product capabilities", "Customer sentiment"])
    items = []
    for rid, company, price in [("TEST-A", "NexaFlow", 100), ("TEST-B", "OrbitDesk", 120)]:
        finding = f"Growth is {price} USD per seat per month, annual commitment"
        items.append(EvidenceItem(
            record_id=rid, claim_id=None, company_name=company, finding=finding,
            quoted_or_paraphrased_evidence=finding, source_title=f"{company} illustrative price",
            source_type="Official pricing page", source_url="https://synthetic.example/test",
            published_at=CUTOFF - timedelta(days=10), available_at=CUTOFF - timedelta(days=9),
            effective_date=CUTOFF - timedelta(days=5), confidence=0.8,
            limitations=["Unknown onboarding fees and AI entitlements"],
            source_independence_group=company.lower(), verification_status="unverified"))
    evidence = EvidencePack(original_question=question, as_of=CUTOFF, evidence_items=items,
                            coverage_by_task=[TaskCoverage(task_id="T1", coverage_score=None,
                                                           supporting_record_ids=["TEST-A", "TEST-B"])],
                            missing_information=["Onboarding fees and AI entitlements unknown"])
    claims = [ReportClaim(claim_id="C1", text=items[0].finding, supporting_record_ids=["TEST-A"],
                          claim_type="factual", confidence=0.8),
              ReportClaim(claim_id="C2", text=items[1].finding, supporting_record_ids=["TEST-B"],
                          claim_type="factual", confidence=0.8),
              ReportClaim(claim_id="C3", text="The illustrative NexaFlow list price is 20 USD lower per seat per month",
                          supporting_record_ids=["TEST-A", "TEST-B"], claim_type="inference", confidence=0.7)]
    report = CompetitiveReport(
        question=question, as_of=CUTOFF, executive_summary="NexaFlow has the lower supplied comparable list price (C1-C3).",
        key_findings=["C1-C3 compare Growth prices only, not total cost."],
        competitor_comparison=["NexaFlow: 100; OrbitDesk: 120 USD/seat/month, annual commitment (C1-C3)."],
        strategic_implications=["C3 suggests a list-price advantage; wider competitive value is uncertain."],
        limitations=["Unknown onboarding fees and AI entitlements; no complete five-company comparison."],
        claims=claims, sources=["TEST-A", "TEST-B"])
    return plan, evidence, report


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    send = AsyncMock(side_effect=AssertionError("HTTP requests forbidden in Analyst tests"))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    yield
    send.assert_not_called()


@pytest.fixture
def sdk(monkeypatch):
    settings = load_config(environ={"OPENAI_API_KEY": "test-only-secret", "OPENAI_MODEL": "gpt-5-mini",
                                   "DATA_DIR": str(PUBLIC)})
    monkeypatch.setattr(module, "load_config", lambda: settings)
    runner = AsyncMock(return_value=SimpleNamespace(final_output=scenario()[2]))
    monkeypatch.setattr(module.Runner, "run", runner)
    return runner


def run(plan, evidence):
    return asyncio.run(module.analyst(plan, evidence))


def test_valid_report_sdk_compatibility_and_full_input_propagation(sdk):
    plan, evidence, expected = scenario()
    result = run(plan, evidence)
    assert result == expected
    arguments = sdk.call_args
    context = json.loads(arguments.kwargs["input"])
    assert context == {"research_plan": plan.model_dump(mode="json"),
                       "evidence_pack": evidence.model_dump(mode="json")}
    agent = arguments.args[0]
    assert agent.output_type is CompetitiveReport and agent.model.model == "gpt-5-mini"
    assert agent.tools == [] and agent.handoffs == []
    assert agent.model_settings.temperature is None
    assert arguments.kwargs["max_turns"] == 1
    assert arguments.kwargs["run_config"].tracing_disabled
    schema = AgentOutputSchema(CompetitiveReport)
    assert schema.is_strict_json_schema()
    assert schema.validate_json(result.model_dump_json()) == result
    sdk.assert_awaited_once()


def test_q01_like_pricing_data_flow_and_prompt(sdk):
    plan, evidence, expected = scenario()
    result = run(plan, evidence)
    assert result.claims[-1].claim_type == "inference"
    assert result.claims[-1].supporting_record_ids == ["TEST-A", "TEST-B"]
    assert "20 USD" in result.claims[-1].text  # Mocked arithmetic, not model reasoning validation.
    prompt = sdk.call_args.args[0].instructions
    for phrase in ["currency", "billing frequency", "per-seat versus flat-rate", "effective dates",
                   "eligibility", "not automatically correct", "not directly comparable"]:
        assert phrase in prompt


def test_q15_like_ranked_inferences(sdk):
    plan, evidence, report = scenario(Q15)
    report.executive_summary = "A complete top-three ranking is not supported by the supplied coverage."
    report.strategic_implications = ["Conditional ranking criterion: price/value, product capability, and sentiment; "
                                     "NexaFlow may threaten on price, but broader ranking is uncertain (C3)."]
    report.claims[-1].text = "NexaFlow may exert price pressure; ranking is uncertain given narrow evidence"
    sdk.return_value = SimpleNamespace(final_output=report)
    result = run(plan, evidence)
    assert result.claims[-1].claim_type == "inference"
    assert "uncertain" in result.strategic_implications[0]
    prompt = sdk.call_args.args[0].instructions
    assert "rank using" in prompt and "explicit criteria" in prompt
    assert "top-N" in prompt


def test_factual_citations_are_valid(sdk):
    plan, evidence, _ = scenario()
    result = run(plan, evidence)
    allowed = {item.record_id for item in evidence.evidence_items}
    assert all(claim.supporting_record_ids for claim in result.claims if claim.claim_type == "factual")
    assert set(result.sources).issubset(allowed)


def test_missing_factual_citations_rejected_without_regeneration(sdk):
    plan, evidence, report = scenario()
    report.claims[0].supporting_record_ids.clear()
    sdk.return_value = SimpleNamespace(final_output=report)
    with pytest.raises(module.AnalystOutputError, match="factual citations"):
        run(plan, evidence)
    sdk.assert_awaited_once()


@pytest.mark.parametrize("case", ["claim", "sources_only", "inference"])
def test_fabricated_or_absent_sources(sdk, case):
    plan, evidence, report = scenario()
    report.sources.append("UNKNOWN-ID")
    if case == "claim":
        report.claims[0].supporting_record_ids.append("UNKNOWN-ID")
    elif case == "inference":
        report.claims[-1].supporting_record_ids.append("UNKNOWN-ID")
    sdk.return_value = SimpleNamespace(final_output=report)
    with pytest.raises(module.AnalystOutputError, match="UNKNOWN-ID.*not present"):
        run(plan, evidence)


def test_existing_but_unprovided_source_is_rejected(sdk):
    plan, evidence, report = scenario()
    report.sources.append("WEB-00001")  # Valid public corpus ID, absent from this pack.
    sdk.return_value = SimpleNamespace(final_output=report)
    with pytest.raises(module.AnalystOutputError, match="WEB-00001"):
        run(plan, evidence)


def test_historical_cutoff_violation_rejected_before_sdk(sdk):
    plan, evidence, _ = scenario()
    evidence.evidence_items[0].available_at = CUTOFF + timedelta(seconds=1)
    with pytest.raises(module.AnalystInputError, match="post-cutoff evidence"):
        run(plan, evidence)
    sdk.assert_not_called()


def test_future_effective_announcement_is_not_temporal_leak(sdk):
    plan, evidence, report = scenario()
    evidence.evidence_items[0].effective_date = CUTOFF + timedelta(days=5)
    report.limitations = ["NexaFlow announcement is future-effective and cannot establish a current price."]
    sdk.return_value = SimpleNamespace(final_output=report)
    assert run(plan, evidence).limitations == report.limitations
    # Applicability is semantic; validation only checks availability, not source truth.


def test_conflicting_prices_preserved_in_context_and_report(sdk):
    plan, evidence, report = scenario()
    evidence.contradictions = ["Official and third-party Growth prices disagree."]
    evidence.evidence_items[1].source_type = "Third-party comparison"
    report.executive_summary = "The contradictory supplied prices do not establish a definitive winner."
    report.limitations = ["Official and third-party Growth prices disagree; a newer observation is not proof."]
    sdk.return_value = SimpleNamespace(final_output=report)
    result = run(plan, evidence)
    assert len(result.sources) == 2
    context = json.loads(sdk.call_args.kwargs["input"])
    assert context["evidence_pack"]["contradictions"] == evidence.contradictions
    assert "disagree" in result.limitations[0]


def test_missing_information_allows_explicit_abstention(sdk):
    plan, evidence, report = scenario()
    evidence.evidence_items.clear()
    evidence.coverage_by_task[0].supporting_record_ids.clear()
    evidence.missing_information = ["No relevant evidence found"]
    report.claims.clear()
    report.sources.clear()
    report.executive_summary = "Insufficient evidence to identify the lowest price."
    report.key_findings = ["No relevant evidence was supplied."]
    report.competitor_comparison = ["No supported comparison is possible."]
    report.strategic_implications = ["No supported strategic implications can be established."]
    report.limitations = ["No relevant evidence found; do not infer zero prices."]
    sdk.return_value = SimpleNamespace(final_output=report)
    result = run(plan, evidence)
    assert result.claims == [] and result.sources == []
    assert "Unable to answer" in result.executive_summary
    sdk.assert_not_awaited()


def test_factual_citation_requirement_is_in_generated_api_schema():
    # Regression for the actual rejected Analyst response: process/missing-data
    # statements were tagged factual with [] citations. The provider must now
    # enforce this rule too, rather than discovering it only after generation.
    from ci_agents.schemas import FactualClaim,InferenceClaim
    schema=AgentOutputSchema(CompetitiveReport).json_schema()
    factual=schema['$defs']['FactualClaim']
    assert factual['properties']['supporting_record_ids']['minItems']==1
    assert factual['properties']['claim_type']['const']=='factual'
    assert schema['$defs']['InferenceClaim']['properties']['claim_type']['const']=='inference'
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        FactualClaim(claim_id='C1',claim_type='factual',text='No records were supplied.',supporting_record_ids=[],confidence=.5)
    assert InferenceClaim(claim_id='C1',claim_type='inference',text='Evidence is insufficient.',supporting_record_ids=[],confidence=.5)


@pytest.mark.parametrize("field", ["question", "as_of"])
def test_wrong_output_question_or_cutoff(sdk, field):
    plan, evidence, report = scenario()
    setattr(report, field, "Changed question" if field == "question" else CUTOFF - timedelta(days=1))
    sdk.return_value = SimpleNamespace(final_output=report)
    with pytest.raises(module.AnalystOutputError, match="question|cutoff"):
        run(plan, evidence)


@pytest.mark.parametrize("field", ["key_findings", "competitor_comparison", "strategic_implications", "limitations"])
def test_empty_sections_rejected(sdk, field):
    plan, evidence, report = scenario()
    getattr(report, field).clear()
    sdk.return_value = SimpleNamespace(final_output=report)
    with pytest.raises(module.AnalystOutputError, match=field):
        run(plan, evidence)


def test_omitted_section_not_silently_defaulted(sdk):
    plan, evidence, report = scenario()
    report.model_fields_set.remove("key_findings")
    sdk.return_value = SimpleNamespace(final_output=report)
    with pytest.raises(module.AnalystOutputError, match="omitted.*key_findings"):
        run(plan, evidence)


@pytest.mark.parametrize("value", [None, {}, "not JSON", '{"question":"text"}'])
def test_invalid_structured_output(sdk, value):
    plan, evidence, _ = scenario()
    sdk.return_value = SimpleNamespace(final_output=value)
    with pytest.raises(module.AnalystOutputError, match="structured CompetitiveReport"):
        run(plan, evidence)


def test_duplicate_claim_ids_and_invalid_confidence_revalidated(sdk):
    plan, evidence, report = scenario()
    report.claims.append(report.claims[0].model_copy())
    sdk.return_value = SimpleNamespace(final_output=report)
    with pytest.raises(module.AnalystOutputError, match="Invalid CompetitiveReport"):
        run(plan, evidence)
    _, _, report = scenario()
    report.claims[0] = report.claims[0].model_copy(update={"confidence": 2})
    sdk.return_value = SimpleNamespace(final_output=report)
    with pytest.raises(module.AnalystOutputError, match="Invalid CompetitiveReport"):
        run(plan, evidence)


@pytest.mark.parametrize("error,expected", [(RuntimeError("test-only-secret provider payload"), module.AnalystError),
                                           (ModelBehaviorError("bad generated output"), module.AnalystOutputError)])
def test_api_or_sdk_output_failure(sdk, error, expected):
    sdk.side_effect = error
    plan, evidence, _ = scenario()
    with pytest.raises(expected) as captured:
        run(plan, evidence)
    assert "test-only-secret" not in str(captured.value)
    assert captured.value.__suppress_context__
    sdk.assert_awaited_once()


@pytest.mark.parametrize("case", ["question", "cutoff", "type", "duplicate_evidence"])
def test_invalid_handoffs_fail_before_sdk(sdk, case):
    plan, evidence, _ = scenario()
    if case == "question":
        evidence.original_question = "Different"
    elif case == "cutoff":
        evidence.as_of = CUTOFF + timedelta(days=1)
    elif case == "duplicate_evidence":
        evidence.evidence_items.append(evidence.evidence_items[0])
    else:
        evidence = {}
    with pytest.raises(module.AnalystInputError):
        run(plan, evidence)
    sdk.assert_not_called()


def test_equivalent_timezone_is_accepted(sdk):
    plan, evidence, report = scenario()
    report.as_of = CUTOFF.astimezone(timezone(timedelta(hours=-5)))
    sdk.return_value = SimpleNamespace(final_output=report)
    assert run(plan, evidence).as_of == CUTOFF


def test_no_retrieval_dataset_or_private_access(sdk, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected retrieval or dataset access")
    from ci_agents import dataset, retrieval
    monkeypatch.setattr(dataset, "load_dataset", forbidden)
    for name in ["search_records", "get_record", "get_pricing_history"]:
        monkeypatch.setattr(retrieval, name, forbidden)
    original = Path.open
    opened = []
    def guard(path, *args, **kwargs):
        assert path.name == "analyst.md", f"Unexpected file access: {path.name}"
        opened.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    plan, evidence, _ = scenario()
    run(plan, evidence)
    assert opened == ["analyst.md"]
    assert sdk.call_args.args[0].tools == []


def test_configured_model_loading(sdk, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=test-only-secret\nOPENAI_MODEL=configured-model\nDATA_DIR="
                                  + PUBLIC.as_posix() + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "load_config", lambda: load_config(project_root=tmp_path, environ={}))
    plan, evidence, _ = scenario()
    run(plan, evidence)
    assert sdk.call_args.args[0].model.model == "configured-model"
