"""Offline behavior tests; real SDK replays mock only the provider boundary."""
import asyncio
from datetime import datetime, timezone
from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents import Usage, AgentOutputSchema
from openai import AsyncOpenAI, RateLimitError
import httpx
import pytest

from ci_agents.config import load_config, use_settings
from ci_agents.dataset import load_dataset
from ci_agents.evidence_context import evidence_context, numerical_aids
from ci_agents.retrieval import RetrievalStore
from ci_agents.schemas import EvidencePack, ResearchPlan, ResearchTask
from ci_agents.telemetry import PricingConfig, capture_stage, stage_metrics, utc_now
from ci_agents.workflow_runtime import (WorkflowPolicy, RunBudget, WorkflowRuntime, use_workflow,
    BudgetExceeded, BudgetHooks, private_storage, run_structured)
from test_planner_debug import payload, response_for

planner_module = importlib.import_module('ci_agents.agents.planner')
research_module = importlib.import_module('ci_agents.agents.researcher')
CUTOFF = datetime(2026,11,11,23,59,59,tzinfo=timezone.utc)
PUBLIC = Path(__file__).resolve().parents[1] / 'data/public'


def runtime(**limits):
    return WorkflowRuntime(WorkflowPolicy(), RunBudget(**limits), PricingConfig(model='gpt-5-mini',
        input_usd_per_million=.25, output_usd_per_million=2, source='test fixture', source_date='2026-11-15'))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    libraries = [httpx]
    try:
        import httpx2
        libraries.append(httpx2)
    except ImportError:
        pass
    for lib in libraries:
        monkeypatch.setattr(lib.AsyncClient,'send',AsyncMock(side_effect=AssertionError('HTTP forbidden')))
        monkeypatch.setattr(lib.Client,'send',lambda *a,**k: pytest.fail('HTTP forbidden'))


def test_native_planner_repairs_once_and_counts_both_responses(monkeypatch):
    settings = load_config(environ={'OPENAI_API_KEY':'test-secret','OPENAI_MODEL':'gpt-5-mini'})
    bad = payload(); bad['tasks'][0]['companies'] = ['Invented company']
    async def exercise():
        client = AsyncOpenAI(api_key='test-secret',max_retries=0)
        create = AsyncMock(side_effect=[response_for(json.dumps(bad)),response_for()])
        monkeypatch.setattr(client.responses,'create',create)
        monkeypatch.setattr(planner_module,'AsyncOpenAI',lambda **kw:client)
        state = runtime(max_model_requests=2)
        with use_settings(settings),use_workflow(state),capture_stage() as captured:
            result = await planner_module.planner(payload()['original_question'], CUTOFF)
        assert result.tasks[0].companies == ['NexaFlow']
        assert create.await_count == 2 and state.requests == 2
        assert state.total_tokens == 60 and captured.usage['total_tokens'] == 60
        assert state.usage_complete and state.events[0]['kind'] == 'output_repair'
        assert state.events[0]['invalid_output']['tasks'][0]['companies'] == ['Invented company']
        assert create.call_args.kwargs['max_output_tokens'] == 8000
    asyncio.run(exercise())


def test_native_invalid_json_is_bounded_and_preserves_known_usage(monkeypatch):
    settings = load_config(environ={'OPENAI_API_KEY':'test-secret','OPENAI_MODEL':'gpt-5-mini'})
    async def exercise():
        client = AsyncOpenAI(api_key='test-secret',max_retries=0)
        create = AsyncMock(return_value=response_for('not-json'))
        monkeypatch.setattr(client.responses,'create',create)
        monkeypatch.setattr(planner_module,'AsyncOpenAI',lambda **kw:client)
        state = runtime()
        with use_settings(settings),use_workflow(state),capture_stage(),pytest.raises(planner_module.PlannerOutputError):
            await planner_module.planner(payload()['original_question'],CUTOFF)
        assert create.await_count <= 2
        assert state.requests == create.await_count
        # If SDK parsing prevents a hook from reporting usage, it remains unknown.
        assert state.total_tokens == 30 * create.await_count or not state.usage_complete
    asyncio.run(exercise())


def test_quota_failure_is_never_output_repaired(monkeypatch):
    settings = load_config(environ={'OPENAI_API_KEY':'test-secret','OPENAI_MODEL':'gpt-5-mini'})
    response = httpx.Response(429,request=httpx.Request('POST','https://offline.invalid'))
    runner = AsyncMock(side_effect=RateLimitError('quota',response=response,body={'code':'insufficient_quota'}))
    monkeypatch.setattr(planner_module.Runner,'run',runner)
    with use_settings(settings),use_workflow(runtime()),pytest.raises(planner_module.PlannerError):
        asyncio.run(planner_module.planner(payload()['original_question'],CUTOFF))
    assert runner.await_count == 1


def test_repair_cannot_bypass_request_budget(monkeypatch):
    settings=load_config(environ={'OPENAI_API_KEY':'test-secret','OPENAI_MODEL':'gpt-5-mini'})
    bad=payload();bad['tasks'][0]['companies']=['Invented company']
    async def exercise():
        client=AsyncOpenAI(api_key='test-secret',max_retries=0)
        create=AsyncMock(return_value=response_for(json.dumps(bad)))
        monkeypatch.setattr(client.responses,'create',create)
        monkeypatch.setattr(planner_module,'AsyncOpenAI',lambda **kw:client)
        state=runtime(max_model_requests=1)
        with use_settings(settings),use_workflow(state),capture_stage(),pytest.raises(BudgetExceeded,match='request_limit'):
            await planner_module.planner(payload()['original_question'],CUTOFF)
        assert create.await_count==1 and state.requests==1 and state.total_tokens==30
    asyncio.run(exercise())


def test_valid_fail_is_not_repaired():
    agent = SimpleNamespace(name='Reviewer')
    runner = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(final_output={'decision':'FAIL'})))
    with use_workflow(runtime()):
        out = asyncio.run(run_structured(agent,'input',lambda value:value,runner=runner))
    assert out['decision']=='FAIL' and runner.run.await_count==1


@pytest.mark.parametrize('limits,reason',[
    ({'max_total_tokens':30},'token_limit'),({'max_estimated_cost':.000001},'cost_limit'),
    ({'max_model_requests':1},'request_limit')])
def test_shared_budget_blocks_next_stage(limits,reason):
    state = runtime(**limits)
    state.check('Planner',dispatch=True)
    state.record(Usage(requests=1,input_tokens=10,output_tokens=20,total_tokens=30))
    with pytest.raises(BudgetExceeded,match=reason):state.check('Reviewer')


def test_research_reserves_requests_for_analysis_review():
    state = runtime(max_model_requests=4)
    state.check('Planner',dispatch=True); state.check('Researcher',dispatch=True)
    with pytest.raises(BudgetExceeded,match='research_request_reserve'):state.check('Researcher')
    state.check('Analyst',dispatch=True);state.check('Reviewer',dispatch=True)
    assert state.requests==4


def test_unknown_usage_blocks_further_resource_spend():
    state = runtime(max_total_tokens=1000);state.usage_complete=False
    with pytest.raises(BudgetExceeded,match='unknown_usage'):state.check('Analyst')


def test_cost_budget_requires_rates_and_time_budget_expires():
    with pytest.raises(ValueError,match='documented'):
        WorkflowRuntime(WorkflowPolicy(),RunBudget(max_estimated_cost=1),PricingConfig())
    state=runtime(max_seconds=1);state.started-=2
    with pytest.raises(BudgetExceeded,match='time_limit'):state.check('Planner')


def test_cache_and_targeted_bootstrap_preserve_read_budget():
    p = ResearchPlan.model_validate(payload())
    with RetrievalStore(load_dataset(PUBLIC)) as store,use_workflow(runtime()):
        session = research_module.ResearchSession(p,store,research_module.ResearcherLimits(max_tool_calls=2))
        session.bootstrap()
        assert session.calls==2 and len(session.records)==1
        rid=next(iter(session.records))
        session.invoke('get_record',{'record_id':rid})
        assert session.calls==2 and session.log[-1].find('cache_hit')>=0
        outcome=research_module.GapResearchOutcome(stop_reason='sufficient_for_question',requirement_assessments=[
            research_module.RequirementAssessment(task_id='T1',requirement_index=0,status='supported',supporting_record_ids=[rid],explanation='Fixture support')])
        session.validate_outcome(outcome)
        pack=session.assemble(outcome)
        assert pack.evidence_items[0].pricing_observation is not None
        assert any('requirement_assessments' in line for line in pack.retrieval_log)


@pytest.mark.parametrize('case',['missing','unknown_record'])
def test_coverage_checks_do_not_accept_fabricated_completion(case):
    p=ResearchPlan.model_validate(payload())
    with RetrievalStore(load_dataset(PUBLIC)) as store,use_workflow(runtime()):
        session=research_module.ResearchSession(p,store,research_module.ResearcherLimits())
        session.bootstrap()
        rows=[] if case=='missing' else [research_module.RequirementAssessment(task_id='T1',requirement_index=0,
            status='partial' if case=='unresolved' else 'supported',supporting_record_ids=['FAKE'] if case=='unknown_record' else list(session.records),explanation='Check')]
        out=research_module.GapResearchOutcome(stop_reason='sufficient_for_question',requirement_assessments=rows)
        with pytest.raises(research_module.EvidenceValidationError):session.validate_outcome(out)


def test_unresolved_coverage_preserves_useful_evidence_without_claiming_completion():
    p=ResearchPlan.model_validate(payload())
    with RetrievalStore(load_dataset(PUBLIC)) as store,use_workflow(runtime()):
        session=research_module.ResearchSession(p,store,research_module.ResearcherLimits())
        session.bootstrap()
        outcome=research_module.GapResearchOutcome(stop_reason='sufficient_for_question',requirement_assessments=[
            research_module.RequirementAssessment(task_id='T1',requirement_index=0,status='conflicting',
                supporting_record_ids=list(session.records),explanation='Conflict requires explicit treatment')])
        validated=session.validate_outcome(outcome)
        assert validated.stop_reason=='unresolved_requirements'
        pack=session.assemble(validated)
        assert pack.evidence_items and any('conflicting' in text for text in pack.missing_information)
        assert outcome.stop_reason=='sufficient_for_question'  # Original response preserved.


def test_bootstrap_covers_every_company_before_extra_categories():
    p=ResearchPlan.model_validate(payload())
    p.tasks[0].companies=['ClearPilot','NexaFlow','OrbitDesk','PulseGrid','VertexOps']
    p.tasks[0].categories=['Promotions','Pricing','Product launch']
    with RetrievalStore(load_dataset(PUBLIC)) as store,use_workflow(runtime()):
        session=research_module.ResearchSession(p,store,research_module.ResearcherLimits(max_tool_calls=10))
        session.bootstrap()
        assert {r['company_name'] for r in session.records.values()}==set(p.tasks[0].companies)
        assert all(r['event_category']=='Pricing' for r in session.records.values())


def test_invalid_requirement_assessment_does_not_discard_full_sources(monkeypatch):
    settings=load_config(environ={'OPENAI_API_KEY':'test-secret','OPENAI_MODEL':'gpt-5-mini'})
    runner=AsyncMock(return_value=SimpleNamespace(final_output=research_module.GapResearchOutcome(
        stop_reason='budget_exhausted',requirement_assessments=[])))
    monkeypatch.setattr(research_module.Runner,'run',runner)
    state=runtime()
    with RetrievalStore(load_dataset(PUBLIC)) as store,use_settings(settings),use_workflow(state),capture_stage() as captured:
        pack=asyncio.run(research_module.researcher(ResearchPlan.model_validate(payload()),store))
    assert runner.await_count==2 and pack.evidence_items
    assert any('unassessed' in note for note in pack.missing_information)
    assert any('requirement_assessment_unavailable' in line for line in pack.retrieval_log)
    assert [json.loads(line) for line in pack.retrieval_log]==captured.events
    assert any(e['kind']=='output_repair_exhausted' for e in state.events)


def test_compaction_preserves_all_evidence_text_and_metadata():
    with RetrievalStore(load_dataset(PUBLIC)) as store,use_workflow(runtime()):
        session=research_module.ResearchSession(ResearchPlan.model_validate(payload()),store,research_module.ResearcherLimits())
        session.bootstrap();pack=session.assemble(research_module.ResearchOutcome(stop_reason='budget_exhausted'))
        before=pack.model_dump_json();context=evidence_context(pack)
        assert 'retrieval_log' not in context
        for item,wire in zip(pack.evidence_items,context['evidence_items']):
            assert wire['quoted_or_paraphrased_evidence']==item.quoted_or_paraphrased_evidence
            assert wire['supersedes_record_id']==item.supersedes_record_id
        assert pack.model_dump_json()==before
        assert len(json.dumps(context))<len(pack.model_dump_json())


def test_decimal_aids_preserve_units_and_do_not_assume_unknown_fees():
    with RetrievalStore(load_dataset(PUBLIC)) as store,use_workflow(runtime()):
        p=ResearchPlan.model_validate(payload());p.original_question='Compare Growth prices for 50 seats'
        session=research_module.ResearchSession(p,store,research_module.ResearcherLimits())
        session.bootstrap();pack=session.assemble(research_module.ResearchOutcome(stop_reason='budget_exhausted'))
        aids=numerical_aids(p,pack)
        for group in aids['comparison_groups']:
            for row in group['observations']:
                if 'monthly_subscription_baseline' in row:
                    from decimal import Decimal
                    assert Decimal(row['monthly_subscription_baseline'])==Decimal(row['amount'])*50
        assert 'not zero' in ' '.join(aids['limitations'])
        pack.evidence_items[0].effective_date=CUTOFF.replace(year=2027)
        assert any(r['record_id']==pack.evidence_items[0].record_id for r in numerical_aids(p,pack)['excluded_from_arithmetic'])


def test_strict_reviewer_schema_has_enum():
    module=importlib.import_module('ci_agents.agents.reviewer')
    schema=AgentOutputSchema(module.StrictReviewResult).json_schema()
    assert schema['$defs']['StrictReviewIssue']['properties']['issue_type']['enum']==sorted(module.ISSUE_TYPES) or set(schema['$defs']['StrictReviewIssue']['properties']['issue_type']['enum'])==module.ISSUE_TYPES


def test_macos_private_temp_is_not_private_answer_storage(tmp_path):
    assert not private_storage('/private/tmp/ci-public-run')
    assert not private_storage('/private/var/folders/public-runtime')
    assert private_storage(tmp_path/'evaluation_private'/'keys')
    assert private_storage(tmp_path/'private'/'keys')


def test_full_native_v2_after_assessment_repair_preserves_manifest_and_ledger(monkeypatch,tmp_path):
    """All four real wrappers + real SDK parser; only provider responses are mocked."""
    from ci_agents.pipeline_v2 import run_pipeline_v2
    from ci_agents.schemas import CompetitiveReport, ReportClaim, ReviewResult
    from ci_agents.agents.reviewer import report_fingerprint
    from ci_agents.persistence import verify_manifest
    settings=replace(load_config(environ={'OPENAI_API_KEY':'test-secret','OPENAI_MODEL':'gpt-5-mini'}),runs_dir=tmp_path/'runs')
    row=next(r for r in load_dataset(PUBLIC).records if r['record_id']=='WEB-00036')
    plan=ResearchPlan.model_validate(payload())
    report=CompetitiveReport(question=plan.original_question,as_of=CUTOFF,executive_summary=row['summary'],
        key_findings=[row['summary']],competitor_comparison=['One source observation; no market ranking.'],
        strategic_implications=['Unknown fees remain unknown.'],limitations=['No all-in-cost conclusion.'],
        claims=[ReportClaim(claim_id='C1',text=row['summary'],claim_type='factual',supporting_record_ids=[row['record_id']],confidence=.5)],sources=[row['record_id']])
    review=ReviewResult(decision='PASS',completeness_score=1,evidence_support_score=1,temporal_consistency_score=1,
        flagged_issues=[],missing_requirements=[],reviewer_notes=['Offline controlled response'],
        reviewed_question=plan.original_question,reviewed_report_hash=report_fingerprint(report))
    calls=[]
    async def respond(**kwargs):
        props=kwargs['text']['format']['schema']['properties']
        if 'stop_reason' in props:
            value={'stop_reason':'budget_exhausted','requirement_assessments':[]};calls.append('researcher')
        elif 'executive_summary' in props:value=report.model_dump(mode='json');calls.append('analyst')
        elif 'decision' in props:value=review.model_dump(mode='json');calls.append('reviewer')
        else:value=plan.model_dump(mode='json');calls.append('planner')
        return response_for(json.dumps(value))
    def client_factory(**kwargs):
        client=AsyncOpenAI(**kwargs)
        monkeypatch.setattr(client.responses,'create',AsyncMock(side_effect=respond))
        return client
    for name in ['planner','researcher','analyst','reviewer']:
        monkeypatch.setattr(importlib.import_module('ci_agents.agents.'+name),'AsyncOpenAI',client_factory)
    result=asyncio.run(run_pipeline_v2(plan.original_question,CUTOFF,'native-v2',settings=settings))
    assert result.execution_status=='completed'
    assert calls==['planner','researcher','researcher','analyst','reviewer']
    out=Path(result.artifact_directory)
    assert result.total_metrics.input_tokens==50
    assert json.loads((out/'workflow_budget.json').read_text())['requests_dispatched']==5
    assert json.loads((out/'metadata.json').read_text())['agent_code_fingerprints']
    verify_manifest(out)
