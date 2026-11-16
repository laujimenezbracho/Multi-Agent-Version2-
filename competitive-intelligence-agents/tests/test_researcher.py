"""Offline discovery/assembly contracts; no model accuracy is inferred from mocks."""
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

module=importlib.import_module('ci_agents.agents.researcher')
CUTOFF=datetime(2026,11,11,23,59,59,tzinfo=timezone.utc)
PUBLIC=Path(__file__).resolve().parents[1]/'data/public'
Q01='Which competitor has the lowest current Growth plan price?'
Q15='What are the top 3 competitive threats for PulseGrid?'


def plan(question=Q01,cutoff=CUTOFF):
    return ResearchPlan(original_question=question,as_of=cutoff,objective='Gather original sources',
        tasks=[ResearchTask(task_id='T1',research_question=question,companies=['VertexOps','PulseGrid'],
            categories=['Pricing'] if question==Q01 else ['Pricing','Product launch','Financial performance'],
            time_scope='At cutoff',evidence_requirements=['Source-backed comparison'])],
        expected_answer_elements=['Answer original question'],comparison_dimensions=['Pricing'])


def item(row):
    fields={key:row.get(key) for key in ('record_id','claim_id','company_name','source_type','source_url',
        'published_at','available_at','effective_date','end_date','source_independence_group',
        'supersedes_record_id','verification_status','feature_status')}
    return EvidenceItem(**fields,source_title=row['document_title'],finding=row['summary'],
        quoted_or_paraphrased_evidence=row['details'],confidence=.5)


def pack(p,rows=()):
    return EvidencePack(original_question=p.original_question,as_of=p.as_of,evidence_items=[item(r) for r in rows],
        coverage_by_task=[TaskCoverage(task_id='T1',coverage_score=None,supporting_record_ids=[r['record_id'] for r in rows],notes=['Coverage incomplete'])],
        missing_information=[] if rows else ['No record inspected'])


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    send=AsyncMock(side_effect=AssertionError('No HTTP permitted'))
    monkeypatch.setattr(httpx.AsyncClient,'send',send)
    yield
    send.assert_not_called()


@pytest.fixture
def store():
    with RetrievalStore(load_dataset(PUBLIC)) as service:yield service


@pytest.fixture
def sdk(monkeypatch):
    settings=load_config(environ={'OPENAI_API_KEY':'test-secret','OPENAI_MODEL':'gpt-5-mini','DATA_DIR':str(PUBLIC)})
    monkeypatch.setattr(module,'load_config',lambda:settings)
    runner=AsyncMock(return_value=SimpleNamespace(final_output=module.ResearchOutcome(stop_reason='sufficient_for_question')))
    monkeypatch.setattr(module.Runner,'run',runner)
    return runner


@pytest.mark.parametrize('question',[Q01,Q15])
def test_mandatory_research_before_model_stop_and_native_output(sdk,store,question):
    result=asyncio.run(module.researcher(plan(question),store))
    assert result.evidence_items
    events=[json.loads(x) for x in result.retrieval_log]
    assert any(e['tool'] in {'search_records','get_pricing_history'} for e in events)
    assert any(e['tool']=='get_record' and e['result_count'] for e in events)
    agent=sdk.call_args.args[0]
    assert agent.output_type is module.ResearchOutcome
    assert agent.output_type is not EvidencePack
    assert agent.model_settings.parallel_tool_calls is False
    assert len(agent.tools)==3
    context=json.loads(sdk.call_args.kwargs['input'])
    assert context['already_retrieved_full_records']
    assert context['remaining_tool_calls']<60
    schema=AgentOutputSchema(module.ResearchOutcome)
    assert schema.validate_json('{"stop_reason":"sufficient_for_question"}')
    sdk.assert_awaited_once()


def test_original_metadata_and_complete_text_are_assembled_by_code(sdk,store):
    result=asyncio.run(module.researcher(plan(),store))
    for evidence in result.evidence_items:
        source=store.get_record(evidence.record_id)
        assert evidence.source_title==source['document_title']
        assert evidence.finding==source['summary']
        assert source['details'] in evidence.quoted_or_paraphrased_evidence
        assert evidence.source_url==source['source_url']
        assert evidence.available_at==source['available_at']
    assert 'WEB-30009' in {e.record_id for e in result.evidence_items}
    assert 'WEB-30008' in {e.record_id for e in result.evidence_items}


def test_balanced_price_bootstrap_covers_companies_before_optional_history(sdk,store):
    p=plan();p.tasks[0].companies=['NexaFlow','OrbitDesk','VertexOps','PulseGrid','ClearPilot']
    result=asyncio.run(module.researcher(p,store,limits=module.ResearcherLimits(max_tool_calls=16)))
    assert {e.company_name for e in result.evidence_items}==set(p.tasks[0].companies)
    assert len(result.retrieval_log)<=16
    assert all(json.loads(e).get('tool_calls',0)<=16 for e in result.retrieval_log)


def test_empty_historical_corpus_has_logged_searches_and_no_model_call(sdk,store):
    p=plan(cutoff=datetime(2020,1,1,tzinfo=timezone.utc))
    service=Mock(wraps=store)
    result=asyncio.run(module.researcher(p,service))
    assert not result.evidence_items and result.missing_information and result.retrieval_log
    sdk.assert_not_awaited()
    for method in ['search_records','get_pricing_history']:
        for call in getattr(service,method).call_args_list:assert call.kwargs['as_of']==p.as_of
    service.close.assert_not_called()


def test_noop_assembly_cannot_claim_research(store):
    session=module.ResearchSession(plan(),store,module.ResearcherLimits())
    with pytest.raises(module.EvidenceValidationError,match='attempting retrieval'):
        session.assemble(module.ResearchOutcome(stop_reason='no_matching_records'))


@pytest.mark.parametrize('field',['claim_id','source_title','source_url','company_name','source_type',
    'source_independence_group','supersedes_record_id','verification_status','feature_status',
    'published_at','effective_date','end_date'])
def test_existing_metadata_guard_remains_strict(store,field):
    p=plan();session=module.ResearchSession(p,store,module.ResearcherLimits())
    session.invoke('get_record',{'record_id':'WEB-00001'})
    output=pack(p,[store.get_record('WEB-00001')])
    value=datetime(2025,2,1,tzinfo=timezone.utc) if field.endswith(('_at','_date')) else 'wrong'
    setattr(output.evidence_items[0],field,value)
    with pytest.raises(module.EvidenceValidationError,match='metadata mismatch'):session.validate(output)


def test_unread_record_guard_is_not_relaxed(store):
    p=plan();session=module.ResearchSession(p,store,module.ResearcherLimits())
    session.invoke('search_records',{'query':'Growth'})
    with pytest.raises(module.EvidenceValidationError,match='not retrieved in full'):
        session.validate(pack(p,[store.get_record('WEB-00001')]))


def test_ledger_ignores_generated_log_and_forbids_duplicate_ids(store):
    p=plan();session=module.ResearchSession(p,store,module.ResearcherLimits())
    session.invoke('get_record',{'record_id':'WEB-00001'})
    output=pack(p,[store.get_record('WEB-00001')]);output.retrieval_log=['invented']
    assert session.validate(output).retrieval_log==session.log
    output.evidence_items.append(output.evidence_items[0].model_copy())
    with pytest.raises(module.EvidenceValidationError):session.validate(output)


@pytest.mark.parametrize('name',['max_tool_calls','max_full_records','max_evidence_items'])
def test_budget_denials_are_logged_and_no_extra_sources_are_admitted(store,name):
    session=module.ResearchSession(plan(),store,module.ResearcherLimits(**{name:1}))
    session.invoke('get_record',{'record_id':'WEB-00001'})
    denied=json.loads(session.invoke('get_record',{'record_id':'WEB-00002'}))
    assert denied['error']=='budget_exhausted' and denied['budget']==name
    assert set(session.records)=={'WEB-00001'}
    assert json.loads(session.log[-1])['status']=='budget_exhausted'


def test_model_cannot_author_evidence_objects(sdk,store):
    sdk.return_value=SimpleNamespace(final_output=pack(plan(),[store.get_record('WEB-00001')]))
    with pytest.raises(module.EvidenceValidationError,match='ResearchOutcome'):
        asyncio.run(module.researcher(plan(),store))


def test_sdk_failure_never_retries(sdk,store):
    sdk.side_effect=RuntimeError('test-secret')
    with pytest.raises(module.ResearcherError) as exc:asyncio.run(module.researcher(plan(),store))
    assert 'test-secret' not in str(exc.value)
    sdk.assert_awaited_once()


@pytest.mark.parametrize('value',[0,-1,True,201])
def test_invalid_limits(value):
    with pytest.raises(ValidationError):module.ResearcherLimits(max_tool_calls=value)


def test_tool_cutoff_cannot_be_overridden(store):
    session=module.ResearchSession(plan(cutoff=datetime(2020,1,1,tzinfo=timezone.utc)),store,module.ResearcherLimits())
    tool=next(t for t in session.tools() if t.name=='get_record')
    assert 'as_of' not in tool.params_json_schema['properties']
    args=json.dumps({'record_id':'WEB-00001','as_of':'2099-01-01T00:00:00Z'})
    result=asyncio.run(tool.on_invoke_tool(ToolContext(context=None,tool_name='get_record',tool_call_id='test',tool_arguments=args),args))
    assert json.loads(result) is None
