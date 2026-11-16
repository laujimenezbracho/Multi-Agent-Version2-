import importlib.util
from pathlib import Path
from types import SimpleNamespace


path=Path(__file__).resolve().parents[1]/'scripts/checkpoint_meter.py'
spec=importlib.util.spec_from_file_location('checkpoint_meter',path)
meter_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(meter_module)


def test_actual_usage_and_cache_discount_are_recorded_without_content(tmp_path):
    meter=meter_module.UsageMeter(tmp_path/'usage.jsonl')
    response=SimpleNamespace(model='gpt-5-mini-2025-08-07',id='response-test',
        usage=SimpleNamespace(input_tokens=1000,output_tokens=100,total_tokens=1100,
            input_tokens_details=SimpleNamespace(cached_tokens=800)))
    with meter_module.label_usage(question_id='Q01',repetition=1,budget=8,stage='researcher'):
        meter.record(response)
    row=meter.rows()[0]
    assert row['stage']=='researcher' and 'content' not in row and 'input' not in row
    assert abs(meter_module.cost(row)-.00027)<1e-10
    assert abs(meter_module.cost(row,True)-.00045)<1e-10


def test_failed_request_does_not_become_zero_tokens_or_known_cost(tmp_path):
    meter=meter_module.UsageMeter(tmp_path/'usage.jsonl')
    meter.record(error=RuntimeError('Sensitive provider body'),requested_model='gpt-5-mini')
    row=meter.rows()[0];summary=meter_module.totals([row])
    assert row['exception_class']=='RuntimeError' and 'Sensitive' not in str(row)
    assert summary['model_request_attempts']==1
    assert summary['total_tokens'] is None and summary['estimated_cost_usd'] is None


def test_native_gateway_usage_is_captured_once_and_retries_are_bounded(tmp_path,monkeypatch):
    from mlflow.genai.judges.adapters import gateway_adapter
    calls=[]
    def fake(**kwargs):
        calls.append(kwargs['num_retries'])
        return {'id':'native-judge-1','model':'gpt-4.1-2025-04-14',
            'usage':{'prompt_tokens':200,'completion_tokens':20,'total_tokens':220,
                'prompt_tokens_details':{'cached_tokens':100}}}
    monkeypatch.setattr(gateway_adapter,'send_chat_request',fake)
    meter=meter_module.UsageMeter(tmp_path/'usage.jsonl');meter.install()
    try:gateway_adapter.send_chat_request(endpoint='offline',headers={},payload={'model':'gpt-4.1'},num_retries=10)
    finally:meter.uninstall()
    assert calls==[0]
    rows=meter.rows();assert len(rows)==1 and rows[0]['input_tokens']==200
    assert rows[0]['cached_input_tokens']==100 and rows[0]['stage']=='external_judge'


def test_sdk_meter_survives_autolog_method_replacement_and_deduplicates(tmp_path,monkeypatch):
    from openai.resources.chat.completions.completions import Completions
    response=SimpleNamespace(model='gpt-5-mini',id='unique-meter-response',
        usage=SimpleNamespace(prompt_tokens=10,completion_tokens=2,total_tokens=12))
    fake=lambda client,**kwargs:response
    monkeypatch.setattr(Completions,'create',fake)
    meter=meter_module.UsageMeter(tmp_path/'usage.jsonl');meter.install()
    try:
        # Simulate MLflow replacing class methods between calibration and prediction.
        Completions.create=fake
        with meter_module.label_usage(stage='planner',question_id='Q01',repetition=1,budget=0):
            Completions.create(None,model='gpt-5-mini')
            meter.record(response)  # A nested observer must not double-count it.
    finally:meter.uninstall()
    rows=meter.rows();assert len(rows)==1
    assert rows[0]['stage']=='planner' and rows[0]['total_tokens']==12
