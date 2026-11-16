"""Native MLflow paired 8/16/32 research checkpoints: six questions, three repeats."""
import argparse
import asyncio
from dataclasses import replace
from datetime import datetime,timezone
from html import escape
import json
import os
from pathlib import Path
import sys
from threading import Lock
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ci_agents.checkpoint_research import CHECKPOINTS,QUESTIONS,CheckpointResearcher,SELECTOR_INSTRUCTIONS
from ci_agents.config import OPENAI_ACCEPT_ENCODING,load_config,use_settings
from ci_agents.dataset import load_dataset
from ci_agents.persistence import dataset_fingerprints,fingerprint,prompt_fingerprints
from ci_agents.schemas import ResearchPlan
from ci_agents.retrieval import RetrievalStore
from ci_agents.agents.planner import planner
from ci_agents.agents.analyst import analyst
from ci_agents.agents.reviewer import reviewer
from ci_agents.telemetry import capture_stage,sanitized_error
from checkpoint_meter import UsageMeter,label_usage,totals
from grading_contract import INSTRUCTIONS,catalogue,normalise

JUDGE_INSTRUCTIONS=INSTRUCTIONS+'''
ADDITIONAL STRICTNESS FOR THIS STUDY:
For anchors select ONLY field_name=summary, details or claim_text with a nonempty text
value. Never use dates, numeric fields, source_title or record IDs as text anchors.
For every claim, supply temporally_correct as true or false (do not omit or use null).
Include alleged_winner_invalidation=null for every defect unless actually alleging a
changed cheapest-price winner. Use the same concrete defect rules for inferences:
an inference label or hedge does not make unsupported competitive advantages valid.
Do not award E02 on a threat answer with unsupported relative pricing or false GA.
An official 159 price is HIGHER than PulseGrid 130. A superseded 149 aggregator price
and denial of an unauthorized 50% offer do not show actual negotiated undercutting.
Accept careful conditional scenarios without pretending those outcomes are observed.
For Q07, cheapest subscription baseline is not established best value or all-in cost.
For Q03, active published Enterprise eligibility windows are not Growth discounts or
independently verified redemption; older promotions may have expired.
Independent source text and explicit key facts take priority over the report's
self-assessed confidence, its explanation of missing data, and its polished style.
Assess prose outside structured claims too, logging material defects as NARRATIVE.
claim_checks objects contain ONLY claim_id, factually_correct, citation_supported,
temporally_correct, explanation and anchors. alleged_winner_invalidation belongs ONLY
inside defects, never inside claim_checks, even when its value is null.
'''

SOURCE_FIELDS=('record_id','company_name','document_title','summary','details','claim_text',
    'source_type','source_url','available_at','published_at','effective_date','end_date',
    'feature_status','verification_status','plan','metric_value','currency',
    'billing_frequency','price_basis','supersedes_record_id')


def compact_source(row):
    """Original fields only; omit duplicate retrieval_text and irrelevant metadata."""
    return {field:row.get(field) for field in SOURCE_FIELDS}


def billing_preflight(root):
    """One minimal request detects exhausted credits before native judge retries."""
    from openai import APIStatusError,OpenAI
    key=os.getenv('OPENAI_API_KEY')
    try:
        with label_usage(stage='billing_probe'):
            with OpenAI(api_key=key,max_retries=0,timeout=30,
                default_headers={'Accept-Encoding':OPENAI_ACCEPT_ENCODING}) as client:
                response=client.chat.completions.create(model='gpt-4.1',
                    messages=[{'role':'user','content':'Reply OK.'}],max_completion_tokens=8)
        save(root/'billing_preflight.json',{'status':'passed','model':response.model,
            'response_id':response.id,'usage':response.usage.model_dump(),
            'purpose':'Detect billing/access failure before starting the controlled study; not an experimental answer.'})
    except APIStatusError as exc:
        value={'status':'blocked','http_status':exc.status_code,'error_code':exc.code,
            'error_type':exc.type,'agent_experiment_started':False}
        save(root/'billing_preflight.json',value)
        raise RuntimeError('Generation preflight blocked: '+str(exc.code)+'. Restore API credits/access before running the study.') from None


def save(path,value):
    from pydantic import TypeAdapter
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(TypeAdapter(object).dump_python(value,mode='json'),indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


def public_output(report,data):
    from pydantic import TypeAdapter
    p=report.model_dump(mode='json') if report else None
    ids=set(p['sources']) if p else set()
    if p:ids.update(i for c in p['claims'] for i in c['supporting_record_ids'])
    result={'execution_status':'report_available' if p else 'no_report','report':p,
        'original_cited_sources':TypeAdapter(object).dump_python([compact_source(r) for r in data.records if r['record_id'] in ids],mode='json')}
    cat=catalogue(p,{'essential_elements':[]})
    result.update(claim_catalogue=cat['claims'],answer_segments=cat['segments'])
    return result


def parse_grade(row,output,rubric,name):
    try:
        value=row.get(name+'/value')
        if not isinstance(value,dict) or set(value)!={'assessment_json'}:raise ValueError('Invalid judge framework envelope')
        assessment=json.loads(value['assessment_json'])
        repairs=[]
        # A misplaced optional null carries no judgement. Remove only this known
        # transport-only artefact and retain an explicit audit trail. Non-null values
        # and every other extra field remain invalid; no substantive decision changes.
        for check in assessment.get('claim_checks',[]):
            if 'alleged_winner_invalidation' in check and check['alleged_winner_invalidation'] is None:
                del check['alleged_winner_invalidation'];repairs.append('Removed misplaced null alleged_winner_invalidation from '+check.get('claim_id','unknown'))
        score=normalise(assessment,output,rubric)
        # Primary endpoint: supported essential coverage, gated against material defects.
        score['decision_safe_coverage']=0 if score['status']=='no_report' else (
            score['completeness']*100 if score['material_error_count']==0 else 0)
        return {'assessment':assessment,'scores':score,'transport_normalization':repairs}
    except (ValueError,TypeError,KeyError) as exc:
        return {'scores':{'status':'invalid_grader_output','decision_safe_coverage':None,
            'approval':'pending_human','error':str(exc)},'raw_feedback':row.get(name+'/value')}


def calibration_fixtures(keys,data):
    byid={r['record_id']:r for r in data.records}
    fixtures={};verdicts={}
    def add(token,q,text,ids,expected,claims=None):
        claim_list=claims if claims is not None else [{'claim_id':'C1','claim_type':'factual','text':text,'supporting_record_ids':ids}]
        p={'question':next(x['user_question'] for x in data.questions if x['question_id']==q),
            'as_of':'2026-11-11T23:59:59Z','executive_summary':text,
            'key_findings':[text],'competitor_comparison':[text],'strategic_implications':[text],
            'limitations':['Unknown negotiated fees and external facts remain unknown; source statements are time-bounded.'],
            'claims':claim_list,'sources':ids}
        from pydantic import TypeAdapter
        out={'execution_status':'report_available','report':p,
            'original_cited_sources':TypeAdapter(object).dump_python([compact_source(byid[i]) for i in ids],mode='json')}
        cat=catalogue(p,keys[q]);out.update(claim_catalogue=cat['claims'],answer_segments=cat['segments'])
        fixtures[token]=(q,out);verdicts[token]=expected
    prices=[v['record_id'] for v in keys['Q01']['reference_prices'].values()]+['WEB-30008']
    add('C01','Q01',keys['Q01']['reference_answer']+' Current official VertexOps 159 supersedes the unverified 149 listing; promotions, unknown fees and add-ons are excluded, not assumed zero.',prices,'valid')
    add('C02','Q03',keys['Q03']['reference_answer'],['WEB-00043','WEB-00096','WEB-00149','WEB-00202','WEB-00255','WEB-30014','WEB-30015'],'valid')
    add('C03','Q12',keys['Q12']['reference_answer'],['WEB-30004','WEB-30005','WEB-30006','WEB-30007','WEB-30011','WEB-30018','WEB-30019'],'valid')
    add('C04','Q01','VertexOps has the lowest current Growth price at USD 159 per seat per month, beating PulseGrid at USD 130 on the same annual-commitment basis.',prices,'material_error')
    add('C05','Q05','OrbitDesk has a verified customer-data breach and confirmed exfiltration. Its outage proves hackers stole customer data.', ['WEB-30001','WEB-30002','WEB-30003'],'material_error')
    add('C06','Q15','The top threats are: 1 VertexOps undercuts PulseGrid because 159 is cheaper than 130; 2 NexaFlow has an authorized active 50% Growth promotion; 3 PulseGrid has audited 18% churn. These facts demonstrate competitors are winning customers.',prices+['WEB-30012','WEB-30013','WEB-30014','WEB-30015'],'material_error')
    add('C07','Q01','No source records were supplied. I cannot identify a price winner, compare current prices or establish total cost. Fees, promotions, taxes and add-on costs remain unknown.',[],'abstention',claims=[])
    return fixtures,verdicts


def export_answers(root):
    cards=[];lines=['# Paired checkpoint answers','',
        'Exact saved reports. Quality grades are provisional; internal reviews are not independent truth.','']
    for q in QUESTIONS:
        for rep in (1,2,3):
            for b in CHECKPOINTS:
                p=root/'cases'/f'{q}-r{rep}-b{b}'/'prediction.json'
                if not p.exists():continue
                d=json.loads(p.read_text());report=d['output']['report'];heading=f'{q} / repeat {rep} / {b} retrieval cap'
                sections=[('Executive answer',[report['executive_summary']]),('Findings',report['key_findings']),
                    ('Comparison',report['competitor_comparison']),('Implications',report['strategic_implications']),
                    ('Limitations',report['limitations'])] if report else [('Execution',['No Analyst report produced.'])]
                body=''.join('<h3>'+title+'</h3><ul>'+''.join('<li>'+escape(str(x))+'</li>' for x in values)+'</ul>' for title,values in sections)
                cards.append('<article id="'+f'{q}-r{rep}-b{b}'+'"><h2>'+heading+'</h2><pre>'+escape(json.dumps(d['metrics'],indent=2))+'</pre>'+body+'</article>')
                lines.extend(['## '+heading,'','```json',json.dumps(d['metrics'],indent=2),'```',''])
                for title,values in sections:lines.extend(['### '+title,'',*['- '+str(x) for x in values],''])
    (root/'answers.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Paired research checkpoints</title><style>body{max-width:1050px;margin:30px auto;padding:20px;font:16px/1.55 system-ui;background:#f4f7f6;color:#17332d}article{background:white;padding:25px;margin:25px 0;border:1px solid #cbdad4;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f1;padding:15px;font-size:12px}li{margin:8px 0}</style><h1>Paired research checkpoint experiment</h1><p>Six questions × three repetitions × 8/16/32 retrieval caps. Provisional source-audited evaluation; reports and failures are preserved.</p>'+''.join(cards)+'</html>')
    (root/'answers.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--key-dir',type=Path,required=True)
    parser.add_argument('--allow-api',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--calibrate-only',action='store_true')
    parser.add_argument('--questions',nargs='+',choices=QUESTIONS,default=list(QUESTIONS))
    parser.add_argument('--repetitions',nargs='+',type=int,choices=[1,2,3],default=[1,2,3])
    args=parser.parse_args()
    if not args.prepare_only and not args.allow_api:parser.error('Live evaluation requires --allow-api')
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    settings=load_config(env_file=ROOT.parent/'.env',require_api_key=not args.prepare_only,
        environ={'OPENAI_MODEL':'gpt-5-mini','DATA_DIR':str(ROOT/'data/public'),'RUNS_DIR':str(root/'trajectories')})
    settings=replace(settings,reasoning_effort='low',max_output_tokens=12000)
    data=load_dataset(settings.data_dir)
    keys={q:json.loads((args.key_dir/(q+'.json')).read_text()) for q in QUESTIONS}
    # Preserve private key files verbatim; reduce only redundant serialized source
    # fields in evaluator context. The facts and source text are unchanged.
    for key in keys.values():key['audited_reference_sources']=[compact_source(r) for r in key['audited_reference_sources']]
    questions={q['question_id']:q['user_question'] for q in data.questions}
    os.environ.setdefault('MLFLOW_ENABLE_TELEMETRY','false')
    os.environ.setdefault('MLFLOW_ENABLE_ASYNC_TRACE_LOGGING','false')
    os.environ.setdefault('MLFLOW_GENAI_EVAL_MAX_WORKERS','3')
    os.environ.setdefault('MLFLOW_SCORER_MAX_WORKERS','2')
    if settings.openai_api_key:os.environ['OPENAI_API_KEY']=settings.openai_api_key
    import mlflow
    from mlflow.genai import datasets
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers import list_scorers
    # set_tracking_uri() also updates process environment. Preserve the caller's
    # configuration BEFORE temporarily discovering the old screen's datasets.
    configured_uri=os.getenv('MLFLOW_TRACKING_URI')
    configured_experiment=os.getenv('MLFLOW_EXPERIMENT_ID')
    # Discover existing native datasets before creating a new six-question design.
    inventory=[]
    prior=ROOT/'runs/reliability_screen_20261113_verified/mlflow.db'
    if prior.exists() and not configured_uri:
        mlflow.set_tracking_uri('sqlite:///'+str(prior))
        inventory=[{'name':d.name,'dataset_id':d.dataset_id,'tracking':'previous_screen'} for d in datasets.search_datasets()]
    uri=configured_uri or 'sqlite:///'+str(root/'mlflow.db')
    mlflow.set_tracking_uri(uri)
    if configured_experiment:
        experiment=mlflow.get_experiment(configured_experiment);mlflow.set_experiment(experiment_id=experiment.experiment_id)
    else:experiment=mlflow.set_experiment('paired-retrieval-threshold-assignment')
    found=datasets.search_datasets(experiment_ids=[experiment.experiment_id])
    inventory += [{'name':d.name,'dataset_id':d.dataset_id,'tracking':'current'} for d in found]
    save(root/'dataset_discovery.json',{'existing':inventory,'selection':'Existing Q01/Q15 datasets do not cover the user-authorized six-question paired design. Reuse their source keys, create a separately named native dataset.'})
    with mlflow.start_span(name='checkpoint_trace_preflight',span_type='CHAIN') as span:
        span.set_inputs({'questions':list(QUESTIONS),'checkpoints':CHECKPOINTS});span.set_outputs({'api_calls':0,'trace_readback_required':True})
    mlflow.flush_trace_async_logging()
    if mlflow.search_traces(experiment_ids=[experiment.experiment_id]).empty:raise RuntimeError('Trace readback failed')
    judge_name='paired_source_audit_'+fingerprint(JUDGE_INSTRUCTIONS)[:8]
    registered=list_scorers(experiment_id=experiment.experiment_id)
    judge=next((s for s in registered if s.name==judge_name),None)
    if judge is None:
        judge=make_judge(name=judge_name,instructions=JUDGE_INSTRUCTIONS,model='openai:/gpt-4.1',
            inference_params={'temperature':0,'max_tokens':12000},feedback_value_type=dict[str,str]).register(experiment_id=experiment.experiment_id)
    # Only native registered judges execute scoring. Protocol normalization is post-processing.
    scorers=list_scorers(experiment_id=experiment.experiment_id)
    from importlib.metadata import version
    plan={'schema_version':'paired-checkpoint-study-v1','questions':QUESTIONS,'checkpoints':CHECKPOINTS,'repetitions':3,
        'agent_model':'gpt-5-mini','judge_model':'gpt-4.1','reasoning_effort':'low','agent_output_limit':12000,'selector_output_limit':4000,
        'cutoff':settings.evaluation_cutoff.isoformat(),'recovery_enabled':False,
        'selector_policy':'Up to four independent model-selected public operations, pending actions preserved across checkpoints. Overall horizon 32 hidden intermediate checkpoints. No analyst/reviewer/evaluator feedback into research.',
        'other_limits':{'search_results':10,'full_records':40,'evidence_items':40},
        'dataset':dataset_fingerprints(settings.data_dir),'prompts':prompt_fingerprints(),
        'source':{str(p.relative_to(ROOT)):fingerprint(p.read_text()) for p in sorted((ROOT/'src/ci_agents').rglob('*.py'))},
        'scripts':{name:fingerprint((ROOT/'scripts'/name).read_text()) for name in ['run_checkpoint_evaluation.py','checkpoint_meter.py','grading_contract.py','prepare_checkpoint_keys.py']},
        'keys':{q:fingerprint((args.key_dir/(q+'.json')).read_text()) for q in QUESTIONS},
        'selector_prompt_sha256':fingerprint(SELECTOR_INSTRUCTIONS),'judge_prompt_sha256':fingerprint(JUDGE_INSTRUCTIONS),
        'judge_inference_params':{'temperature':0,'max_tokens':12000},'judge_gateway_http_retries':0,
        'evaluator_source_fields':SOURCE_FIELDS,
        'versions':{p:version(p) for p in ['openai','openai-agents','mlflow','pydantic','pandas']},
        'decision_rule':{'noninferiority_margin_points':2,'confidence_level':.95,'reference_budget':32,
            'primary':'100 × supported essential coverage, zero when independently identified material defects; no report zero; invalid grader remains missing with sensitivity bounds',
            'uncertainty':'Question-cluster resampling of paired question means; six question clusters, three repetitions. No claim of universal optimality or certified low material-error risk.'},
        'pricing_checked':'2026-11-15','pricing_sources':['https://developers.openai.com/api/docs/models/gpt-5-mini','https://developers.openai.com/api/docs/models/gpt-4.1'],
        'pricing_per_million':{'gpt-5-mini':{'input':.25,'cached':.025,'output':2},'gpt-4.1':{'input':2,'cached':.5,'output':8}},
        'cost_basis':'Measured response usage. Report actual experiment total separately from counterfactual stop-at-checkpoint cost, excluding earlier shadow reports. Incremental research usage counted once.'}
    if (root/'experiment_plan.json').exists() and json.loads((root/'experiment_plan.json').read_text())!=json.loads(json.dumps(plan)):
        raise RuntimeError('Frozen source/settings changed; use a new output directory')
    save(root/'experiment_plan.json',plan)
    design_name='paired-threshold-design-'+fingerprint(plan)[:12]
    design=next((d for d in found if d.name==design_name),None)
    if design is None:
        design=datasets.create_dataset(name=design_name,experiment_id=experiment.experiment_id,
            tags={'design':'complete-54-observation-preregistered-design','agent_access':'public_inputs_only'})
        design.merge_records([{'inputs':{'question_id':q,'repetition':rep,'retrieval_budget':b},
            'expectations':{'rubric':keys[q],'requirement_catalogue':catalogue(None,keys[q])['requirements']}}
            for q in QUESTIONS for rep in (1,2,3) for b in CHECKPOINTS])
    save(root/'design_dataset.json',{'dataset_id':design.dataset_id,'name':design.name,
        'conditions':54,'experiment_id':experiment.experiment_id,'tracking_uri':uri,
        'status':'prepared_not_executed','scorers':[s.name for s in scorers]})
    print('PREFLIGHT trace verified; registered judge',judge_name,'existing datasets',len(inventory),flush=True)
    if args.prepare_only:return
    mlflow.openai.autolog()
    meter=UsageMeter(root/'request_usage.jsonl');meter.install()
    try:
        billing_preflight(root)
        fixtures,expected=calibration_fixtures(keys,data)
        cal_path=root/'calibration_summary.json'
        if not cal_path.exists():
            name='checkpoint-grader-calibration-'+fingerprint(plan)[:10]
            cal=next((d for d in found if d.name==name),None)
            if cal is None:
                cal=datasets.create_dataset(name=name,experiment_id=experiment.experiment_id)
                cal.merge_records([{'inputs':{'case_token':token},'expectations':{'rubric':keys[q],
                    'requirement_catalogue':catalogue(None,keys[q])['requirements']}} for token,(q,out) in fixtures.items()])
            @mlflow.trace(name='checkpoint_grader_calibration',span_type='CHAIN')
            def calibration_predict(case_token):return fixtures[case_token][1]
            result=mlflow.genai.evaluate(data=cal,predict_fn=calibration_predict,scorers=scorers)
            result.result_df.to_json(root/'calibration_raw.json',orient='records',indent=2,force_ascii=False)
            summary=[]
            for row in json.loads((root/'calibration_raw.json').read_text()):
                trace=json.loads(row['trace']);token=json.loads(trace['info']['request_preview'])['case_token']
                q,out=fixtures[token];graded=parse_grade(row,out,keys[q],judge_name);s=graded['scores']
                ok=s['status']=='assessed' and ((expected[token]=='valid' and s['material_error_count']==0 and s['completeness']>=.75)
                    or (expected[token]=='material_error' and s['material_error_count']>=1)
                    or (expected[token]=='abstention' and s['completeness']<=.25))
                summary.append({'case_token':token,'expected':expected[token],'passed':ok,'trace_id':row['trace_id'],**graded})
            save(cal_path,{'passed':all(x['passed'] for x in summary),'cases':summary,'scope':'Seven-case source-audited smoke calibration; not a human-certified estimate of judge accuracy.'})
        if not json.loads(cal_path.read_text())['passed']:
            raise RuntimeError('Grader calibration failed; no agent experiment started. Preserve this version and diagnose.')
        print('CALIBRATION passed seven cases',flush=True)
        if args.calibrate_only:return
        locks={(q,rep):Lock() for q in QUESTIONS for rep in [1,2,3]}

        def traced_tool(engine,action):
            # Do not auto-serialize the engine/settings object; only public tool I/O.
            with mlflow.start_span(name='checkpoint_public_retrieval',span_type='RETRIEVER') as active:
                active.set_inputs({'tool':action.tool,'arguments':action.arguments()})
                response=engine.execute(action);active.set_outputs({'result':response})
                return response

        async def stage_call(q,rep,b,stage,call,directory):
            start=time.perf_counter()
            with label_usage(question_id=q,repetition=rep,budget=b,stage=stage),capture_stage():
                try:
                    value=await call();save(directory/(stage+'.json'),value.model_dump(mode='json'))
                    return value,{'status':'completed','latency_seconds':time.perf_counter()-start}
                except Exception as exc:
                    error=sanitized_error(exc,stage,secrets=(settings.openai_api_key,))
                    save(directory/(stage+'_error.json'),error)
                    return None,{'status':'failed','latency_seconds':time.perf_counter()-start,'error':error}

        async def ensure_trajectory(q,rep,target):
            tdir=root/'trajectories'/f'{q}-r{rep}';tdir.mkdir(parents=True,exist_ok=True)
            with use_settings(settings):
                pp=tdir/'plan.json'
                if pp.exists():p=ResearchPlan.model_validate_json(pp.read_text())
                else:
                    if (tdir/'planner_started.json').exists():raise RuntimeError('Interrupted Planner requires audit; no paid retry')
                    save(tdir/'planner_started.json',{'question_id':q,'repetition':rep})
                    p,pm=await stage_call(q,rep,0,'planner',lambda:planner(questions[q],settings.evaluation_cutoff),tdir)
                    save(tdir/'planner_stage_metrics.json',pm)
                    if p is None:
                        for b in CHECKPOINTS:
                            out=public_output(None,data);save(root/'cases'/f'{q}-r{rep}-b{b}'/'prediction.json',
                                {'binding':fingerprint(plan),'output':out,'output_sha256':fingerprint(out),'metrics':{'question_id':q,'repetition':rep,'budget':b,'execution_status':'planner_failed','retrieval_operations':0,'new_full_records':0,'full_records':0}})
                        return
                    save(pp,p.model_dump(mode='json'))
                state_path=tdir/'research_state.json'
                state=json.loads(state_path.read_text()) if state_path.exists() else None
                with RetrievalStore(data) as service:
                    engine=CheckpointResearcher(p,service,settings,state)
                    for b in CHECKPOINTS:
                        if b>target:break
                        cdir=root/'cases'/f'{q}-r{rep}-b{b}';checkpoint=cdir/'prediction.json'
                        if checkpoint.exists():continue
                        if (cdir/'attempt_started.json').exists():raise RuntimeError('Interrupted checkpoint requires audit; no paid retry')
                        save(cdir/'attempt_started.json',{'binding':fingerprint(plan),'question_id':q,'repetition':rep,'budget':b})
                        print('START',q,'repeat',rep,'checkpoint',b,flush=True)
                        old_calls=engine.session.calls;old_ids=set(engine.session.records);start=time.perf_counter()
                        with label_usage(question_id=q,repetition=rep,budget=b,stage='researcher'):
                            try:
                                pack=await engine.advance(b,trace_tool=traced_tool,save_state=lambda s:save(state_path,s))
                                research_error=engine.error
                            except Exception as exc:
                                engine.error=sanitized_error(exc,'researcher',secrets=(settings.openai_api_key,));engine.stopped=True
                                engine.stop_explanation='Execution failure; no automatic retry.';research_error=engine.error;pack=None
                            save(state_path,engine.state())
                        research_seconds=time.perf_counter()-start
                        if pack:save(cdir/'evidence_pack.json',pack.model_dump(mode='json'))
                        report,am=await stage_call(q,rep,b,'analyst',lambda:analyst(p,pack),cdir) if pack else (None,{'status':'not_run','latency_seconds':0})
                        review,rm=await stage_call(q,rep,b,'reviewer',lambda:reviewer(p,pack,report),cdir) if report else (None,{'status':'not_run','latency_seconds':0})
                        output=public_output(report,data)
                        rows=[x for x in meter.rows() if x.get('question_id')==q and x.get('repetition')==rep]
                        research_rows=[x for x in rows if x['stage']=='researcher' and x.get('budget',0)<=b]
                        incremental_rows=[x for x in research_rows if x['budget']==b]
                        stop_rows=[x for x in rows if x['stage']=='planner' or x['stage']=='researcher' and x['budget']<=b or x['stage'] in {'analyst','reviewer'} and x['budget']==b]
                        metrics={'question_id':q,'repetition':rep,'budget':b,
                            'execution_status':'research_failed' if research_error else 'analyst_failed' if report is None else 'reviewer_invalid' if review is None else 'completed',
                            'retrieval_operations':engine.session.calls,'incremental_retrieval_operations':engine.session.calls-old_calls,
                            'full_records':len(engine.session.records),'new_full_records':len(set(engine.session.records)-old_ids),
                            'full_record_ids':sorted(engine.session.records),'new_full_record_ids':sorted(set(engine.session.records)-old_ids),
                            'research_stopped_early':engine.stopped,'research_stop_explanation':engine.stop_explanation,
                            'research_error':research_error,'internal_reviewer_decision':review.decision if review else None,
                            'incremental_research_usage':totals(incremental_rows),'cumulative_research_usage':totals(research_rows),
                            'cost_if_stopped_here':totals(stop_rows),'incremental_research_latency_seconds':research_seconds,
                            'analyst_stage':am,'reviewer_stage':rm}
                        save(checkpoint,{'binding':fingerprint(plan),'output':output,'output_sha256':fingerprint(output),'metrics':metrics})
                        print('FINISH',q,rep,b,metrics['execution_status'],'retrievals',engine.session.calls,'sources',len(engine.session.records),flush=True)

        @mlflow.trace(name='paired_checkpoint_case',span_type='CHAIN')
        def predict_fn(question_id,repetition,retrieval_budget):
            if question_id not in QUESTIONS or repetition not in (1,2,3) or retrieval_budget not in CHECKPOINTS:raise ValueError('Unregistered condition')
            with locks[question_id,repetition]:asyncio.run(ensure_trajectory(question_id,repetition,retrieval_budget))
            checkpoint=root/'cases'/f'{question_id}-r{repetition}-b{retrieval_budget}'/'prediction.json'
            value=json.loads(checkpoint.read_text())
            if value['binding']!=fingerprint(plan) or value['output_sha256']!=fingerprint(value['output']):raise ValueError('Checkpoint binding mismatch')
            return value['output']

        subset=[{'inputs':{'question_id':q,'repetition':rep,'retrieval_budget':b},
            'expectations':{'rubric':keys[q],'requirement_catalogue':catalogue(None,keys[q])['requirements']}}
            for q in args.questions for rep in args.repetitions for b in CHECKPOINTS
            if not (root/'cases'/f'{q}-r{rep}-b{b}'/'independent_grade.json').exists()]
        if not subset:
            export_answers(root);print('All requested checkpoints already graded; no calls repeated.',flush=True);return
        # All conditions have stable identities; subset runs reuse immutable checkpoints.
        name='paired-threshold-'+fingerprint(plan)[:12]+'-'+fingerprint([x['inputs'] for x in subset])[:6]
        ds=next((d for d in found if d.name==name),None)
        if ds is None:
            ds=datasets.create_dataset(name=name,experiment_id=experiment.experiment_id,
                tags={'design':'nested-checkpoints','agent_access':'public_inputs_only','ground_truth':'provisional_source_audit'})
            ds.merge_records(subset)
        result=mlflow.genai.evaluate(data=ds,predict_fn=predict_fn,scorers=scorers)
        suffix=fingerprint([args.questions,args.repetitions])[:6]
        raw_path=root/f'native_results_{suffix}.json';result.result_df.to_json(raw_path,orient='records',indent=2,force_ascii=False)
        derived=[]
        for row in json.loads(raw_path.read_text()):
            identity=json.loads(json.loads(row['trace'])['info']['request_preview'])
            q,rep,b=identity['question_id'],identity['repetition'],identity['retrieval_budget']
            checkpoint=root/'cases'/f'{q}-r{rep}-b{b}'/'prediction.json'
            if not checkpoint.exists():
                derived.append({'case':identity,'trace_id':row['trace_id'],'scores':{'status':'execution_failure_without_checkpoint','decision_safe_coverage':0}});continue
            case=json.loads(checkpoint.read_text())
            graded=parse_grade(row,case['output'],keys[q],judge_name)
            value={'case':identity,'trace_id':row['trace_id'],**graded}
            save(checkpoint.parent/'independent_grade.json',value);derived.append(value)
        save(root/f'evaluation_manifest_{suffix}.json',{'mlflow_run_id':result.run_id,'dataset_id':ds.dataset_id,
            'experiment_id':experiment.experiment_id,'conditions':len(subset),'judge_name':judge_name,'plan_sha256':fingerprint(plan)})
        save(root/f'derived_scores_{suffix}.json',derived)
        save(root/'actual_experiment_usage.json',totals(meter.rows()))
        export_answers(root)
        mlflow.flush_trace_async_logging()
        print('SAVED',len(derived),'checkpoint assessments;',root/'answers.html',flush=True)
    finally:meter.uninstall()


if __name__=='__main__':main()
