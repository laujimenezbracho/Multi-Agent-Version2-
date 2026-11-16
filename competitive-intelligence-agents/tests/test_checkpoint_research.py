import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ci_agents.checkpoint_research import CheckpointResearcher,NextResearchActions,ResearchAction
from ci_agents.config import load_config
from ci_agents.dataset import load_dataset
from ci_agents.retrieval import RetrievalStore
from ci_agents.schemas import ResearchPlan,ResearchTask

ROOT=Path(__file__).resolve().parents[1]


def action(tool,**kwargs):
    return ResearchAction(tool=tool,query=None,company=None,category=None,record_id=None,
        plan=None,reason='Public evidence requirement',**kwargs) if not kwargs else ResearchAction.model_validate(
            dict(tool=tool,query=None,company=None,category=None,record_id=None,plan=None,reason='Public evidence requirement')|kwargs)


def setup_engine(service,state=None):
    settings=load_config(environ={'OPENAI_API_KEY':'offline-test','OPENAI_MODEL':'gpt-5-mini','DATA_DIR':str(ROOT/'data/public')})
    p=ResearchPlan(original_question='Compare Growth pricing',as_of=settings.evaluation_cutoff,
        objective='Compare public pricing',tasks=[ResearchTask(task_id='T1',research_question='Compare Growth prices',
            companies=['PulseGrid'],categories=['Pricing'],time_scope='At cutoff',evidence_requirements=['Current price'])],
        expected_answer_elements=['Comparison'],comparison_dimensions=['Price'])
    return CheckpointResearcher(p,service,settings,state)


def test_checkpoint_prefix_and_pending_actions_survive_resume(monkeypatch):
    with RetrievalStore(load_dataset(ROOT/'data/public')) as service:
        engine=setup_engine(service)
        async def propose():
            engine.pending=[action('get_pricing_history',company='PulseGrid',plan='Growth') for _ in range(3)]
        monkeypatch.setattr(engine,'propose',propose)
        first=asyncio.run(engine.advance(8))
        assert engine.session.calls==8 and len(engine.pending)==1
        prefix=list(engine.outputs)
        saved=engine.state();resumed=setup_engine(service,saved)
        async def propose_again():resumed.pending=[action('get_pricing_history',company='PulseGrid',plan='Growth') for _ in range(3)]
        monkeypatch.setattr(resumed,'propose',propose_again)
        asyncio.run(resumed.advance(16))
        assert resumed.session.calls==16 and resumed.outputs[:8]==prefix
        assert len([x for x in first.retrieval_log if 'research_outcome' not in x])==8


def test_discovered_ids_required_and_full_text_required():
    with RetrievalStore(load_dataset(ROOT/'data/public')) as service:
        engine=setup_engine(service)
        assert engine.execute(action('get_record',record_id='WEB-00195'))['error']=='undiscovered_record_id'
        assert engine.session.calls==1 and not engine.session.records
        engine.execute(action('get_pricing_history',company='PulseGrid',plan='Growth'))
        assert 'WEB-00195' in engine.known_ids and not engine.session.records
        engine.execute(action('get_record',record_id='WEB-00195'))
        assert engine.session.records['WEB-00195']['document_title']


def test_early_stop_does_not_fill_budget_with_arbitrary_calls(monkeypatch):
    with RetrievalStore(load_dataset(ROOT/'data/public')) as service:
        engine=setup_engine(service)
        async def stop():
            engine.stopped=True;engine.stop_explanation='No purposeful operation remains.'
        monkeypatch.setattr(engine,'propose',stop)
        # An honest zero-operation stop cannot become an unsupported evidence pack.
        with pytest.raises(Exception,match='attempting retrieval'):
            asyncio.run(engine.advance(8))


def test_six_private_keys_have_source_backed_reference_sets(tmp_path):
    path=ROOT/'scripts/prepare_checkpoint_keys.py';spec=importlib.util.spec_from_file_location('checkpoint_keys',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    dest=tmp_path/'evaluation_private'/'keys';manifest=module.prepare(ROOT,dest)
    assert set(manifest['questions'])=={'Q01','Q03','Q05','Q07','Q12','Q15'}
    import json
    for q in manifest['questions']:
        key=json.loads((dest/(q+'.json')).read_text())
        assert len(key['essential_elements'])==4 and key['audited_reference_sources']
    assert '6500' in json.loads((dest/'Q07.json').read_text())['reference_answer']
    assert json.loads((dest/'Q15.json').read_text())['reference_prices']['PulseGrid']['price_usd_per_seat_month']==130
