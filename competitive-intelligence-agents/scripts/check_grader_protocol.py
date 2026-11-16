"""One native MLflow canary against an existing source-audited abstention; no agent run."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ci_agents.config import load_config
from ci_agents.persistence import fingerprint
from grading_contract import INSTRUCTIONS,normalise,catalogue


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--rubric-dir',type=Path,required=True)
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    if not args.allow_api:parser.error('Canary is one paid judge assessment; use --allow-api')
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    settings=load_config(env_file=ROOT.parent/'.env',environ={'OPENAI_MODEL':'gpt-5-mini'})
    os.environ['OPENAI_API_KEY']=settings.openai_api_key
    os.environ['MLFLOW_GENAI_EVAL_MAX_WORKERS']='1'
    os.environ['MLFLOW_ENABLE_ASYNC_TRACE_LOGGING']='false'
    import mlflow
    from mlflow.genai import datasets
    from mlflow.genai.judges import make_judge
    mlflow.set_tracking_uri('sqlite:///'+str(root/'mlflow.db'))
    exp=mlflow.set_experiment('grading-protocol-canary')
    found=datasets.search_datasets(experiment_ids=[exp.experiment_id])
    key=json.loads((args.rubric_dir/'Q15.json').read_text())
    data_name='existing-abstention-canary-'+fingerprint(INSTRUCTIONS)[:12]
    data=next((d for d in found if d.name==data_name),None)
    if data is None:
        data=datasets.create_dataset(name=data_name,experiment_id=exp.experiment_id)
        data.merge_records([{'inputs':{'case':'existing_Q15_abstention'},'expectations':{'rubric':key,
            'requirement_catalogue':catalogue(None,key)['requirements']}}])
    judge=make_judge(name='item_protocol_canary',instructions=INSTRUCTIONS,model='openai:/gpt-4.1',feedback_value_type=dict[str,str])
    judge=judge.register(experiment_id=exp.experiment_id)
    prediction=json.loads((ROOT/'runs/budget_screen_20261113_live/cases/Q15-b32/prediction.json').read_text())['prediction']
    prediction={**prediction,'execution_status':'report_available','claim_catalogue':catalogue(prediction['report'],key)['claims']}
    prediction['answer_segments']=catalogue(prediction['report'],key)['segments']
    frame=data.to_df();frame['outputs']=[prediction]
    result=mlflow.genai.evaluate(data=frame,scorers=[judge])
    result.result_df.to_json(root/'canary_results.json',orient='records',indent=2)
    rows=json.loads((root/'canary_results.json').read_text())
    value=rows[0].get('item_protocol_canary/value')
    if not isinstance(value,dict) or set(value)!={'assessment_json'}:
        raise ValueError('MLflow judge envelope or payload is invalid; inspect canary results')
    score=normalise(json.loads(value['assessment_json']),prediction,key)
    if score['completeness']!=.25:
        raise ValueError('Canary disagrees with source-audited one-of-four essential coverage')
    (root/'canary_passed.json').write_text(json.dumps({'run_id':result.run_id,'derived_scores':score,
        'framework_envelope_valid':True,'agent_executions':0,'used_preexisting_answer':True},indent=2)+'\n')
    print('Canary passed: valid framework envelope, exact essential coverage 1/4, no agent execution.')


if __name__=='__main__':main()
