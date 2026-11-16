"""Decision-rule calibration with labelled synthetic fixtures, never study results."""
import json
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]


def test_known_plateau_selects_sixteen_and_preserves_all_paired_extensions(tmp_path):
    run=tmp_path/'fixture_only';run.mkdir()
    for q in ['Q01','Q03','Q05','Q07','Q12','Q15']:
        for rep in [1,2,3]:
            for b in [8,16,32]:
                d=run/'cases'/f'{q}-r{rep}-b{b}';d.mkdir(parents=True)
                score=50 if b==8 else 100
                requirements=[{'requirement_id':f'E{i:02d}','met':i<=(2 if b==8 else 4)} for i in range(1,5)]
                usage={'model_request_attempts':2,'total_tokens':100*b,'estimated_cost_usd':b*.001,'known_estimated_cost_usd':b*.001}
                checkpoint={'output':{'report':{'sources':['TEST-1'],'claims':[]}},'metrics':{
                    'execution_status':'completed','internal_reviewer_decision':'PASS',
                    'retrieval_operations':b,'full_record_ids':['TEST-1'],
                    'cost_if_stopped_here':usage,'incremental_research_usage':usage,
                    'incremental_research_latency_seconds':1}}
                grade={'scores':{'status':'assessed','decision_safe_coverage':score,
                    'completeness':score/100,'material_error_count':0},
                    'assessment':{'requirement_checks':requirements,'defects':[]}}
                (d/'prediction.json').write_text(json.dumps(checkpoint))
                (d/'independent_grade.json').write_text(json.dumps(grade))
    (run/'request_usage.jsonl').write_text('')
    result=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_checkpoint_study.py'),'--run-dir',str(run)],
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    summary=json.loads((run/'threshold_results.json').read_text())
    assert summary['recommendation']['provisional_default']==16
    assert summary['recommendation']['statistically_supported_within_this_study'] is True
    assert [x['quality_mean'] for x in summary['budget_summary']]==[50,100,100]
    changes=json.loads((run/'incremental_benefits.json').read_text())
    assert len(changes)==36
    assert all(x['quality_change']==50 for x in changes if x['from_budget']==8)
    assert all(x['quality_change']==0 for x in changes if x['from_budget']==16)
    contrast=summary['paired_contrasts_vs_32']
    assert contrast[0]['cluster_interval_with_missing_sensitivity']==[-50,-50]
    assert contrast[1]['cluster_interval_with_missing_sensitivity']==[0,0]
