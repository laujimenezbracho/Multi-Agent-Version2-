"""Measure actual OpenAI response usage, including schema failures and judge calls.

No prompts, response content, API keys or reasoning text enter this ledger. Native
MLflow autologging remains responsible for traces. Meter wrappers call its wrappers.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime,timezone
from functools import wraps
import json
from threading import RLock
from types import SimpleNamespace

_label=ContextVar('checkpoint_usage_label',default={'stage':'external_judge'})
_lock=RLock()
_active_meter=None
RATES={'gpt-5-mini':(.25,.025,2.),'gpt-4.1':(2.,.5,8.)}


@contextmanager
def label_usage(**label):
    token=_label.set(label)
    try:
        # MLflow temporarily reconfigures autolog wrappers between evaluate() calls.
        # Re-establish observation AFTER that reconfiguration, before each stage.
        if _active_meter is not None:_active_meter.ensure_sdk()
        yield
    finally:_label.reset(token)


def cost(row,conservative=False):
    model=row.get('model','')
    rate=next((v for k,v in RATES.items() if model.startswith(k)),None)
    if rate is None or row.get('input_tokens') is None or row.get('output_tokens') is None:return None
    incoming,cached,out=rate;cache=0 if conservative else row.get('cached_input_tokens') or 0
    return ((row['input_tokens']-cache)*incoming+cache*cached+row['output_tokens']*out)/1e6


def totals(rows):
    complete=all(r.get('status')=='response_received' and r.get('input_tokens') is not None for r in rows)
    def total(field):return sum(r.get(field) or 0 for r in rows) if complete else None
    costs=[cost(r) for r in rows];upper=[cost(r,True) for r in rows]
    return {'model_request_attempts':len(rows),'model_requests_with_response':sum(r.get('status')=='response_received' for r in rows),
        'input_tokens':total('input_tokens'),'output_tokens':total('output_tokens'),
        'cached_input_tokens':total('cached_input_tokens'),'total_tokens':total('total_tokens'),
        'estimated_cost_usd':sum(costs) if all(x is not None for x in costs) else None,
        'conservative_uncached_estimate_usd':sum(upper) if all(x is not None for x in upper) else None,
        'known_input_tokens':sum(r.get('input_tokens') or 0 for r in rows),
        'known_output_tokens':sum(r.get('output_tokens') or 0 for r in rows),
        'known_total_tokens':sum(r.get('total_tokens') or 0 for r in rows),
        'known_estimated_cost_usd':sum(x for x in costs if x is not None),
        'requests_with_unknown_usage':sum(r.get('input_tokens') is None for r in rows),
        'usage_complete':complete}


class UsageMeter:
    def __init__(self,path):
        self.path=path;self.originals=[];self.gateway_original=None
        self.seen_ids={r['response_id'] for r in self.rows() if r.get('response_id')}

    def rows(self):
        with _lock:return [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []

    def record(self,result=None,error=None,requested_model=None):
        usage=getattr(result,'usage',None)
        def attr(*names):
            for name in names:
                value=getattr(usage,name,None)
                if value is not None:return value
        detail=getattr(usage,'input_tokens_details',None) or getattr(usage,'prompt_tokens_details',None)
        record={**_label.get(),'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'model':getattr(result,'model',None) or requested_model,
            'response_id':getattr(result,'id',None),'status':'response_received' if result is not None else 'request_failed',
            'input_tokens':attr('input_tokens','prompt_tokens'),'output_tokens':attr('output_tokens','completion_tokens'),
            'total_tokens':attr('total_tokens'),'cached_input_tokens':getattr(detail,'cached_tokens',None),
            'exception_class':type(error).__name__ if error is not None else None,
            'http_status':getattr(error,'status_code',None) if error is not None else None}
        with _lock:
            if record['response_id'] and record['response_id'] in self.seen_ids:return
            if error is not None and getattr(error,'_checkpoint_usage_recorded',False):return
            if record['response_id']:self.seen_ids.add(record['response_id'])
            if error is not None:
                try:setattr(error,'_checkpoint_usage_recorded',True)
                except Exception:pass
            with self.path.open('a') as f:f.write(json.dumps(record)+'\n')

    def ensure_sdk(self):
        from openai.resources.responses.responses import Responses,AsyncResponses
        from openai.resources.chat.completions.completions import Completions,AsyncCompletions
        for cls,is_async in [(Responses,False),(AsyncResponses,True),(Completions,False),(AsyncCompletions,True)]:
            if getattr(cls.create,'_checkpoint_meter_owner',None)==id(self):continue
            original=cls.create;self.originals.append((cls,original));meter=self
            if is_async:
                def factory(original):
                    @wraps(original)
                    async def call(client,*args,**kwargs):
                        try:result=await original(client,*args,**kwargs)
                        except Exception as exc:
                            meter.record(error=exc,requested_model=kwargs.get('model'));raise
                        meter.record(result,requested_model=kwargs.get('model'));return result
                    return call
            else:
                def factory(original):
                    @wraps(original)
                    def call(client,*args,**kwargs):
                        try:result=original(client,*args,**kwargs)
                        except Exception as exc:
                            meter.record(error=exc,requested_model=kwargs.get('model'));raise
                        meter.record(result,requested_model=kwargs.get('model'));return result
                    return call
            wrapped=factory(original);wrapped._checkpoint_meter_owner=id(self)
            cls.create=wrapped

    def install(self):
        global _active_meter
        _active_meter=self
        with _lock:self.ensure_sdk()
        # Native MLflow judges use its Gateway HTTP adapter, bypassing OpenAI SDK.
        # Observe raw response usage here without replacing native judging or tracing.
        from mlflow.genai.judges.adapters import gateway_adapter
        original=gateway_adapter.send_chat_request;self.gateway_original=original
        @wraps(original)
        def gateway_call(*args,**kwargs):
            # Bound native judge HTTP work too. Its default can retry a quota error
            # repeatedly; failed requests must remain visible rather than hidden.
            if len(args)>=4:args=(*args[:3],0,*args[4:])
            else:kwargs['num_retries']=0
            try:result=original(*args,**kwargs)
            except Exception as exc:
                self.record(error=exc,requested_model=(kwargs.get('payload') or {}).get('model'));raise
            usage=result.get('usage') or {}
            details=usage.get('prompt_tokens_details') or {}
            measured=SimpleNamespace(model=result.get('model') or (kwargs.get('payload') or {}).get('model'),
                id=result.get('id'),usage=SimpleNamespace(**(usage|{'prompt_tokens_details':SimpleNamespace(**details)})))
            self.record(measured)
            return result
        gateway_adapter.send_chat_request=gateway_call

    def uninstall(self):
        global _active_meter
        _active_meter=None
        for cls,original in reversed(self.originals):cls.create=original
        self.originals=[]
        if self.gateway_original is not None:
            from mlflow.genai.judges.adapters import gateway_adapter
            gateway_adapter.send_chat_request=self.gateway_original
            self.gateway_original=None
