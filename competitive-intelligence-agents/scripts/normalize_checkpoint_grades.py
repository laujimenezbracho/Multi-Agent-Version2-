"""Audited, non-semantic protocol normalization of saved native judge feedback.

Never invents correctness decisions, source anchors, defects, or fractions. Raw
native feedback and the runner's strict grades remain unchanged. Nullable citation
checks on uncited inferences are not a licence to omit material inference defects.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from grading_contract import normalise


def normalize(grade,output,rubric):
    original=deepcopy(grade)
    if grade['scores']['status']!='invalid_grader_output':
        return original|{'protocol_normalization':[],'raw_grade_status':grade['scores']['status']}
    raw=grade.get('raw_feedback')
    if not isinstance(raw,dict) or set(raw)!={'assessment_json'}:return original
    assessment=json.loads(raw['assessment_json']);changes=[]
    claims={c['claim_id']:c for c in output['report']['claims']} if output['report'] else {}
    for check in assessment.get('claim_checks',[]):
        if check.get('alleged_winner_invalidation','absent') is None:
            del check['alleged_winner_invalidation'];changes.append({'claim_id':check['claim_id'],'field':'misplaced optional null removed'})
        claim=claims.get(check.get('claim_id'))
        if (claim and claim['claim_type']=='inference' and not claim['supporting_record_ids']
            and not check.get('anchors') and check.get('citation_supported') is True):
            check['citation_supported']=None
            changes.append({'claim_id':check['claim_id'],'field':'citation_supported',
                'old':True,'new':None,'reason':'An uncited inference has no citation whose support can be scored; no positive credit is awarded. Other judgments and all defects remain unchanged.'})
    for defect in assessment.get('defects',[]):
        if 'alleged_winner_invalidation' not in defect:
            defect['alleged_winner_invalidation']=None
            changes.append({'claim_id':defect['claim_id'],'field':'absent optional rank-allegation metadata defaulted to null'})
    if not changes:return original
    try:
        scores=normalise(assessment,output,rubric)
        scores['decision_safe_coverage']=0 if scores['status']=='no_report' else scores['completeness']*100 if scores['material_error_count']==0 else 0
    except (ValueError,TypeError,KeyError) as exc:
        return original|{'protocol_normalization':changes,'normalization_still_invalid':str(exc)}
    return {'case':grade['case'],'trace_id':grade['trace_id'],'assessment':assessment,'scores':scores,
        'raw_grade_status':'invalid_grader_output','protocol_normalization':changes,
        'source':'deterministic_contract_normalization_not_new_quality_judging','approval':'pending_human'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--key-dir',type=Path,required=True);p.add_argument('--log-feedback',action='store_true')
    a=p.parse_args();root=a.run_dir.resolve();summary=[]
    if a.log_feedback:
        import mlflow
        from mlflow.entities import AssessmentSource
        mlflow.set_tracking_uri('sqlite:///'+str(root/'mlflow.db'))
    for path in sorted((root/'cases').glob('*/independent_grade.json')):
        grade=json.loads(path.read_text());checkpoint=json.loads((path.parent/'prediction.json').read_text())
        q=grade['case']['question_id'];key=json.loads((a.key_dir/(q+'.json')).read_text())
        normalized=normalize(grade,checkpoint['output'],key)
        (path.parent/'protocol_normalized_grade.json').write_text(json.dumps(normalized,indent=2,ensure_ascii=False)+'\n')
        entry={'case':grade['case'],'raw_status':grade['scores']['status'],'final_status':normalized['scores']['status'],
            'changes':normalized.get('protocol_normalization',[])};summary.append(entry)
        if a.log_feedback and entry['changes']:
            mlflow.log_feedback(trace_id=grade['trace_id'],name='citation_protocol_normalization',
                value=normalized['scores']['status']!='invalid_grader_output',rationale=json.dumps(entry['changes']),
                source=AssessmentSource(source_type='CODE',source_id='checkpoint-nullability-normalizer-v1'))
    record={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Only schema nullability/optional metadata normalization; all substantive judgments and defects preserved. No raw feedback overwritten.',
        'cases':summary}
    (root/'protocol_normalization_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'cases':len(summary),'raw_invalid':sum(x['raw_status']=='invalid_grader_output' for x in summary),
        'still_invalid':sum(x['final_status']=='invalid_grader_output' for x in summary)},indent=2))


if __name__=='__main__':main()
