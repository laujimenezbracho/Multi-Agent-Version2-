"""A confirmed quota failure must stop before calibration or agent execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest


def test_credit_exhaustion_is_reported_safely_without_starting_experiment(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('checkpoint_runner_test',root/'scripts/run_checkpoint_evaluation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    requests=[]
    def fail(**kwargs):
        requests.append(kwargs)
        raise openai.APIStatusError('Sensitive provider text',
            response=httpx.Response(429,request=httpx.Request('POST','https://offline.invalid/v1/chat/completions')),
            body={'code':'credit_balance_exhausted','type':'insufficient_quota'})
    class FakeOpenAI:
        def __init__(self,**kwargs):assert kwargs['max_retries']==0
        def __enter__(self):return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
        def __exit__(self,*args):return False
    monkeypatch.setattr(openai,'OpenAI',FakeOpenAI)
    monkeypatch.setenv('OPENAI_API_KEY','offline-sensitive-test-key')
    with pytest.raises(RuntimeError,match='credit_balance_exhausted'):
        module.billing_preflight(tmp_path)
    record=json.loads((tmp_path/'billing_preflight.json').read_text())
    assert len(requests)==1 and record['agent_experiment_started'] is False
    assert record['http_status']==429
    assert 'Sensitive provider text' not in str(record) and 'offline-sensitive-test-key' not in str(record)
