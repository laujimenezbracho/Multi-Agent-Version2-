"""Offline fairness, provenance and failure-accounting contracts for the screen."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from ci_agents.budget_screen import experiment_limits, public_prediction, stop_details, validate_checkpoint
from ci_agents.config import Settings, agent_model_settings
from ci_agents.dataset import load_dataset
from ci_agents.persistence import fingerprint


def test_only_tool_budget_differs():
    limits=[experiment_limits(b).model_dump() for b in (16,32,60)]
    assert [x.pop('max_tool_calls') for x in limits]==[16,32,60]
    assert limits[0]==limits[1]==limits[2]
    assert limits[0]['max_model_turns']==62
    for value in (0,15,17,100):
        with pytest.raises(ValueError):experiment_limits(value)


def test_optional_model_controls_preserve_defaults_and_freeze_experiment_settings(tmp_path):
    settings=Settings('test-secret','gpt-5-mini',tmp_path,tmp_path,datetime.now(timezone.utc))
    assert agent_model_settings(settings).max_tokens is None
    fixed=replace(settings,reasoning_effort='low',max_output_tokens=12000)
    model=agent_model_settings(fixed,parallel_tool_calls=False)
    assert model.max_tokens==12000 and model.reasoning.effort=='low'
    assert model.parallel_tool_calls is False


def test_budget_exhaustion_and_completed_report_can_coexist(tmp_path):
    (tmp_path/'retrieval_logs.json').write_text(json.dumps([
        {'tool':'get_record','status':'ok','record_ids':['WEB-00001']},
        {'tool':'get_record','status':'budget_exhausted','budget':'max_tool_calls'}]))
    result=stop_details(None,tmp_path)
    assert result['stop_reason']=='budget_exhausted'
    assert result['retrieval_attempts']==2 and result['allowed_tool_operations']==1
    assert result['denied_tool_attempts']==1


def test_invalid_output_is_retained_with_budget_flag(tmp_path):
    (tmp_path/'error.json').write_text(json.dumps({'error_type':'EvidenceValidationError'}))
    (tmp_path/'retrieval_logs.json').write_text(json.dumps([{'status':'budget_exhausted','budget':'max_tool_calls'}]))
    result=stop_details(None,tmp_path)
    assert result['stop_reason']=='invalid_output' and result['budget_exhausted']


def test_checkpoint_reuse_cannot_change_treatment_or_payload():
    binding={'question_id':'Q01','budget':16}; prediction={'report':None}
    record={'binding':binding,'prediction':prediction,'prediction_sha256':fingerprint(prediction)}
    assert validate_checkpoint(record,binding)==prediction
    with pytest.raises(ValueError):validate_checkpoint(record,{**binding,'budget':32})
    record['prediction']['report']='tampered'
    with pytest.raises(ValueError):validate_checkpoint(record,binding)


def test_independent_prediction_excludes_reviewer_and_private_context():
    data=load_dataset(Path(__file__).resolve().parents[1]/'data/public')
    fake=SimpleNamespace(final_report=None,execution_status='execution_failed',review_result='INTERNAL_REVIEWER_SECRET')
    prediction=public_prediction(fake,data)
    assert set(prediction)=={'execution_status','report','original_cited_sources'}
    assert 'INTERNAL_REVIEWER_SECRET' not in json.dumps(prediction)


def test_provisional_keys_are_evaluator_only_and_price_refs_valid(tmp_path):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_budget_rubrics.py'
    spec=importlib.util.spec_from_file_location('rubric_preparer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(ValueError):module.prepare(path.parent.parent/'data/public',tmp_path/'public')
    target=tmp_path/'evaluation_private'/'keys'
    module.prepare(path.parent.parent/'data/public',target)
    q01=json.loads((target/'Q01.json').read_text())
    assert q01['reference_prices']['PulseGrid']['price_usd_per_seat_month']==130
    assert q01['reference_prices']['VertexOps']['price_usd_per_seat_month']==159
    assert 'provisional' in q01['rubric_version']
