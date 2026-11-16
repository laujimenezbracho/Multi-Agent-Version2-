"""Complete only zero-call deferred cases, then grade all saved outputs with verdicts blinded."""
import argparse
import asyncio
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ci_agents.budget_screen import experiment_limits, public_prediction, stop_details, validate_checkpoint
from ci_agents.config import load_config
from ci_agents.dataset import load_dataset
from ci_agents.persistence import fingerprint
from ci_agents.telemetry import PricingConfig


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    if not args.allow_api:parser.error('Explicit --allow-api is required')
    root=args.output_dir.resolve()
    plan=json.loads((root/'experiment_plan.json').read_text())
    for name,sha in plan['agent_source_sha256'].items():
        if fingerprint((ROOT/'src'/name).read_text())!=sha:
            raise ValueError('Agent source changed since experiment freeze: '+name)
    # Reconstruct the exact original execution settings, not a new architecture.
    settings=load_config(env_file=ROOT.parent/'.env',environ={'OPENAI_MODEL':plan['model'],
        'DATA_DIR':str(ROOT/'data/public'),'RUNS_DIR':str(root/'agent_runs')})
    settings=replace(settings,reasoning_effort=plan['reasoning_effort'],max_output_tokens=plan['max_output_tokens'])
    from ci_agents.persistence import dataset_fingerprints,prompt_fingerprints
    if dataset_fingerprints(settings.data_dir)!=plan['dataset'] or prompt_fingerprints()!=plan['prompts']:
        raise ValueError('Dataset or prompt changed')
    os.environ['OPENAI_API_KEY']=settings.openai_api_key
    os.environ['MLFLOW_GENAI_EVAL_MAX_WORKERS']='1'
    os.environ['MLFLOW_SCORER_MAX_WORKERS']='1'
    os.environ['MLFLOW_ENABLE_ASYNC_TRACE_LOGGING']='false'
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+str(root/'mlflow.db'))
    exp=mlflow.set_experiment('retrieval-budget-screen-v1')
    mlflow.openai.autolog()
    # Recover usage only from ORIGINAL recorded agent response spans, requiring
    # complete token usage on every model span. Do not overwrite sealed artifacts.
    original=json.loads((root/'independent_results.json').read_text())
    recoveries=[]
    for row in original:
        trace=json.loads(row['trace'])
        preview=json.loads(trace['info']['request_preview'])
        path=root/'cases'/f"{preview['question_id']}-b{preview['retrieval_calls']}"
        case=json.loads((path/'prediction.json').read_text())
        if case['summary']['estimated_cost_usd'] is not None:continue
        model_spans=[s for s in trace['data']['spans'] if s['name']=='AsyncResponses']
        usages=[json.loads(s['attributes']['mlflow.chat.tokenUsage']) for s in model_spans
                if 'mlflow.chat.tokenUsage' in s['attributes']]
        if not usages or len(usages)!=len(model_spans):raise ValueError('Cannot recover complete actual usage')
        inputs=sum(u['input_tokens'] for u in usages);outputs=sum(u['output_tokens'] for u in usages)
        recovered={'source':'original MLflow AsyncResponses usage, including response rejected by SDK output validation',
            'trace_id':trace['info']['trace_id'],'trace_payload_sha256':fingerprint(trace),
            'input_tokens':inputs,'output_tokens':outputs,'total_tokens':inputs+outputs,
            'cached_input_tokens':sum(u.get('cache_read_input_tokens',0) for u in usages),
            'model_requests':len(usages),'estimated_cost_usd':inputs*0.25/1e6+outputs*2/1e6,
            'sealed_agent_metrics_modified':False}
        save(path/'usage_recovered_from_trace.json',recovered);recoveries.append(recovered)
    cases=[json.loads(p.read_text()) for p in (root/'cases').glob('*/prediction.json')]
    spent=0
    for c in cases:
        value=c['summary']['estimated_cost_usd']
        if value is None:
            p=root/'cases'/f"{c['summary']['question_id']}-b{c['summary']['retrieval_budget']}"/'usage_recovered_from_trace.json'
            value=json.loads(p.read_text())['estimated_cost_usd']
        spent+=value
    if spent>=plan['max_agent_estimated_usd_between_cases']:raise ValueError('Original monetary safeguard reached')
    spec=importlib.util.spec_from_file_location('screen_runner',ROOT/'scripts/run_budget_screen.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    from mlflow.genai import datasets
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers import list_scorers
    found=datasets.search_datasets(experiment_ids=[exp.experiment_id])
    name='q01-q15-budget-screen-'+fingerprint(plan)[:12]
    dataset=next(d for d in found if d.name==name)
    instructions=runner.JUDGE_INSTRUCTIONS+'''
ADDITIONAL ARITHMETIC AND SCOPE CHECK:
Check numeric ordering explicitly. An outdated competitor price is an error, but does
not automatically make the lowest-price winner wrong: only a corrected LOWER competing
price can change that winner. Count errors once, give consistent denominators, and do
not penalize a correct winner merely because a different price elsewhere is outdated.
For a list-price question do not require a full feature-value matrix as an essential
answer component. Scope follows the original question and rubric, not an overbroad plan.
The execution status has been neutralized and carries no internal Reviewer verdict.
'''
    scorers=list_scorers(experiment_id=exp.experiment_id)
    judge=next((s for s in scorers if s.name=='independent_blind_audit_v2'),None)
    if judge is None:
        judge=make_judge(name='independent_blind_audit_v2',instructions=instructions,
            model='openai:/gpt-4.1',feedback_value_type=dict[str,str])
        judge=judge.register(experiment_id=exp.experiment_id)
    data=load_dataset(settings.data_dir)
    questions={r['question_id']:r['user_question'] for r in data.questions}
    pricing=PricingConfig(model=plan['model'],input_usd_per_million=0.25,output_usd_per_million=2,
        source=plan['pricing_source'],source_date=plan['pricing_checked_date'])
    save(root/'continuation_plan.json',{'original_plan_sha256':fingerprint(plan),'preserved_agent_controls':True,
        'resume_only':'zero-call not_started_budget cases','usage_recoveries':recoveries,
        'known_agent_cost_before_resume':spent,'original_estimated_usd_safeguard':plan['max_agent_estimated_usd_between_cases'],
        'final_grader_model':'gpt-4.1','final_grader_prompt_sha256':fingerprint(instructions),
        'reason':'Captured API usage resolves the earlier unknown-cost stop. Regrade with internal verdict status blinded; provisional first-pass judge feedback is preserved, not treated as truth.'})

    @mlflow.trace(name='budget_screen_completed_blind',span_type='CHAIN')
    def predict(question_id,retrieval_calls):
        case_dir=root/'cases'/f'{question_id}-b{retrieval_calls}'
        checkpoint=case_dir/'prediction.json';case=json.loads(checkpoint.read_text())
        expected={'plan_sha256':fingerprint(plan),'question_id':question_id,'limits':experiment_limits(retrieval_calls).model_dump()}
        validate_checkpoint(case,expected)
        if case['summary']['execution_status']=='not_started_budget':
            if (case_dir/'attempt_started.json').exists():raise ValueError('Interrupted case cannot be silently repeated')
            current=0
            for path in (root/'cases').glob('*/prediction.json'):
                c=json.loads(path.read_text());amount=c['summary']['estimated_cost_usd']
                if amount is None:amount=json.loads((path.parent/'usage_recovered_from_trace.json').read_text())['estimated_cost_usd']
                current+=amount
            if current>=plan['max_agent_estimated_usd_between_cases']:raise ValueError('Cost safeguard reached')
            archive=case_dir/'prediction_not_started.json'
            if not archive.exists():save(archive,case)
            save(case_dir/'attempt_started.json',expected)
            from ci_agents.pipeline import run_pipeline
            from ci_agents.diagnostics import local_diagnostics
            run_id=f'{question_id.lower()}-b{retrieval_calls}-'+fingerprint(plan)[:10]
            print('RESUME UNSTARTED',question_id,retrieval_calls,flush=True)
            with local_diagnostics(True):
                result=asyncio.run(run_pipeline(questions[question_id],settings.evaluation_cutoff,run_id,
                    question_id=question_id,experiment_id=exp.experiment_id,settings=settings,pricing=pricing,
                    researcher_limits=experiment_limits(retrieval_calls)))
            prediction=public_prediction(result,data)
            totals=json.loads((Path(result.artifact_directory)/'metrics.json').read_text())['totals']
            summary={'question_id':question_id,'retrieval_budget':retrieval_calls,'run_id':run_id,
                'execution_status':result.execution_status,'internal_reviewer_decision':result.review_result.decision if result.review_result else None,
                **stop_details(result,result.artifact_directory),'model_requests':totals.get('llm_requests'),
                'input_tokens':totals['input_tokens'],'output_tokens':totals['output_tokens'],'total_tokens':totals.get('total_tokens'),
                'latency_seconds':round(totals['latency_seconds'],3),'estimated_cost_usd':totals['estimated_cost_usd'],
                'artifact_directory':result.artifact_directory}
            case={'binding':expected,'prediction':prediction,'prediction_sha256':fingerprint(prediction),'summary':summary}
            save(checkpoint,case);runner.comparison(root)
            print('FINISH RESUMED',question_id,retrieval_calls,result.execution_status,summary['stop_reason'],flush=True)
        # The grader sees neither PASS/FAIL nor a status encoding that verdict.
        prediction=dict(case['prediction'])
        prediction['execution_status']='report_available' if prediction['report'] is not None else 'no_report'
        return prediction

    result=mlflow.genai.evaluate(data=dataset,predict_fn=predict,scorers=[judge])
    result.result_df.to_json(root/'independent_blind_results.json',orient='records',indent=2,force_ascii=False)
    result.result_df.to_csv(root/'independent_blind_results.csv',index=False)
    save(root/'final_evaluation_manifest.json',{'mlflow_run_id':result.run_id,'original_plan_sha256':fingerprint(plan),
        'final_grader_model':'gpt-4.1','verdict_status_blinded':True,'provisional_key':True,
        'six_unique_agent_cases_expected':True,'agent_retries':0,'source_code_unchanged':True})
    runner.comparison(root)
    mlflow.flush_trace_async_logging()
    print('Complete. All saved answers:',root/'answers.html',flush=True)


if __name__=='__main__':main()
