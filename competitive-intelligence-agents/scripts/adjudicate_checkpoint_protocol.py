"""One predeclared native-MLflow adjudication pass for structurally invalid grades.

Never repeats agent research. Every remaining invalid grade gets exactly one fresh
assessment; valid grades are not rerun, and invalid adjudications remain missing.
"""
import argparse
import json
import os
from pathlib import Path

from run_checkpoint_evaluation import (ROOT,JUDGE_INSTRUCTIONS,billing_preflight,
    catalogue,compact_source,parse_grade,save)
from normalize_checkpoint_grades import normalize
from checkpoint_meter import UsageMeter
from ci_agents.config import load_config
from ci_agents.persistence import fingerprint

ADJUDICATION_INSTRUCTIONS=JUDGE_INSTRUCTIONS+'''
STRICT NULLABILITY AND DEFECT LINKING:
For an inference with supporting_record_ids=[] and no source anchors, set
citation_supported=null. Do not award positive citation support to an uncited claim.
For a pure inference/uncertainty statement, factually_correct may be null; an honest
statement that evidence is missing is not factually false merely because the overall
answer is incomplete. Assess unsupported strategy as a defect where appropriate.
Every false factually_correct, citation_supported or temporally_correct check needs
a defect bearing that SAME claim_id. NARRATIVE defects alone cannot satisfy that link.
Use NARRATIVE only for defects outside structured claims. Do not duplicate one
underlying issue just to add a link. Use temporally_correct=true when no temporal
error applies. All anchors must name existing text fields of original source records.
The four essential requirements and material-defect standards are unchanged. Grade
the supplied answer afresh against its references, without guessing any earlier score.
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--key-dir',type=Path,required=True);p.add_argument('--allow-api',action='store_true')
    a=p.parse_args()
    if not a.allow_api:p.error('Explicit --allow-api is required')
    root=a.run_dir.resolve();adir=root/'protocol_adjudication';adir.mkdir(parents=True,exist_ok=True)
    records={}
    for path in sorted((root/'cases').glob('*/independent_grade.json')):
        candidate=path.parent/'protocol_normalized_grade.json'
        grade=json.loads((candidate if candidate.exists() else path).read_text())
        if grade['scores']['status']!='invalid_grader_output':continue
        if (path.parent/'protocol_adjudicated_grade.json').exists():continue
        records[path.parent.name]={'grade':grade,'checkpoint':json.loads((path.parent/'prediction.json').read_text()),'directory':path.parent}
    if not records:print('No unadjudicated invalid grades; no model calls.');return
    policy={'script_sha256':fingerprint(Path(__file__).read_text()),'prompt_sha256':fingerprint(ADJUDICATION_INSTRUCTIONS),
        'scope':'One fresh native assessment for every structurally invalid grade remaining after mechanical normalization; no selection by score, no agent reruns, no third attempt.',
        'case_ids':list(records),'judge_model':'gpt-4.1','temperature':0}
    if (adir/'attempt_started.json').exists():raise RuntimeError('Interrupted adjudication must be audited; no automatic paid replay')
    save(adir/'policy.json',policy)
    settings=load_config(env_file=ROOT.parent/'.env',environ={'OPENAI_MODEL':'gpt-5-mini','DATA_DIR':str(ROOT/'data/public'),'RUNS_DIR':str(root)})
    os.environ['OPENAI_API_KEY']=settings.openai_api_key
    os.environ.setdefault('MLFLOW_ENABLE_TELEMETRY','false')
    os.environ.setdefault('MLFLOW_GENAI_EVAL_MAX_WORKERS','2')
    os.environ.setdefault('MLFLOW_SCORER_MAX_WORKERS','2')
    import mlflow
    from mlflow.genai import datasets
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers import list_scorers
    from mlflow.entities import AssessmentSource
    mlflow.set_tracking_uri(json.loads((root/'design_dataset.json').read_text())['tracking_uri'])
    experiment=mlflow.set_experiment('paired-checkpoint-protocol-adjudication')
    discovered=datasets.search_datasets(experiment_ids=[experiment.experiment_id])
    save(adir/'dataset_discovery.json',[{'name':d.name,'dataset_id':d.dataset_id} for d in discovered])
    judge_name='protocol_adjudication_'+fingerprint(ADJUDICATION_INSTRUCTIONS)[:8]
    scorers=list_scorers(experiment_id=experiment.experiment_id)
    judge=next((x for x in scorers if x.name==judge_name),None)
    if judge is None:
        judge=make_judge(name=judge_name,instructions=ADJUDICATION_INSTRUCTIONS,
            model='openai:/gpt-4.1',inference_params={'temperature':0,'max_tokens':12000},
            feedback_value_type=dict[str,str]).register(experiment_id=experiment.experiment_id)
    scorers=list_scorers(experiment_id=experiment.experiment_id)
    keys={}
    for entry in records.values():
        q=entry['grade']['case']['question_id']
        key=json.loads((a.key_dir/(q+'.json')).read_text())
        key['audited_reference_sources']=[compact_source(r) for r in key['audited_reference_sources']];keys[q]=key
    name='invalid-protocol-cases-'+fingerprint(policy)[:12]
    ds=next((d for d in discovered if d.name==name),None)
    if ds is None:
        ds=datasets.create_dataset(name=name,experiment_id=experiment.experiment_id)
        ds.merge_records([{'inputs':{'case_id':cid},'expectations':{'rubric':keys[entry['grade']['case']['question_id']],
            'requirement_catalogue':catalogue(None,keys[entry['grade']['case']['question_id']])['requirements']}}
            for cid,entry in records.items()])
    mlflow.openai.autolog();meter=UsageMeter(root/'adjudication_usage.jsonl');meter.install()
    try:
        billing_preflight(adir);save(adir/'attempt_started.json',policy)
        @mlflow.trace(name='frozen_answer_protocol_adjudication',span_type='CHAIN')
        def prediction(case_id):return records[case_id]['checkpoint']['output']
        result=mlflow.genai.evaluate(data=ds,predict_fn=prediction,scorers=scorers)
        result.result_df.to_json(adir/'native_results.json',orient='records',indent=2,force_ascii=False)
        summary=[]
        for row in json.loads((adir/'native_results.json').read_text()):
            cid=json.loads(json.loads(row['trace'])['info']['request_preview'])['case_id'];entry=records[cid]
            q=entry['grade']['case']['question_id'];output=entry['checkpoint']['output']
            graded=parse_grade(row,output,keys[q],judge_name)
            value={'case':entry['grade']['case'],'trace_id':row['trace_id'],**graded}
            value=normalize(value,output,keys[q])
            value.update(adjudication_policy=policy,previous_trace_id=entry['grade']['trace_id'],
                source='one_pass_native_protocol_adjudication',approval='pending_human')
            save(entry['directory']/'protocol_adjudicated_grade.json',value)
            mlflow.log_feedback(trace_id=entry['grade']['trace_id'],name='protocol_adjudication_quality',
                value=value['scores'].get('decision_safe_coverage') if value['scores'].get('decision_safe_coverage') is not None else 'still_invalid',
                rationale='One fixed adjudication pass for structurally invalid grading, with original research and feedback preserved.',
                source=AssessmentSource(source_type='LLM_JUDGE',source_id=judge_name))
            summary.append({'case_id':cid,'status':value['scores']['status'],'quality':value['scores'].get('decision_safe_coverage')})
        save(adir/'summary.json',{'mlflow_run_id':result.run_id,'dataset_id':ds.dataset_id,'cases':summary})
        print(json.dumps(summary,indent=2))
    finally:meter.uninstall()


if __name__=='__main__':main()
