import importlib.util
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def load(name,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fixture():
    text='Unknown fees remain unknown; no source-backed ranking can be established.'
    output={'report':{'executive_summary':text,'key_findings':[text],'competitor_comparison':[text],
        'strategic_implications':[text],'limitations':[text],'sources':[],
        'claims':[{'claim_id':'I1','claim_type':'inference','text':text,'supporting_record_ids':[]}]},
        'original_cited_sources':[]}
    assessment={'status':'assessed','requirement_checks':[{'requirement_id':f'E{i:02d}',
        'met':i==4,'explanation':'Only uncertainty disclosed.','answer_segment_id':'EXEC'} for i in range(1,5)],
        'claim_checks':[{'claim_id':'I1','factually_correct':None,'citation_supported':True,
            'temporally_correct':True,'explanation':'Missing data statement.','anchors':[]}],
        'defects':[],'narrative_scanned':True}
    grade={'case':{'question_id':'Q01','repetition':1,'retrieval_budget':8},'trace_id':'fixture',
        'scores':{'status':'invalid_grader_output'},'raw_feedback':{'assessment_json':json.dumps(assessment)}}
    rubric={'essential_elements':['Answer','Compare','Verify','Disclose unknowns'],
        'essential_requires_evidence':[True,True,True,False],'audited_reference_sources':[]}
    return grade,output,rubric,assessment


def test_uncited_inference_gets_no_positive_citation_credit_and_raw_stays_unchanged(monkeypatch):
    module=load('normalize_checkpoint_grades',monkeypatch);g,o,k,a=fixture();raw=g['raw_feedback']['assessment_json']
    result=module.normalize(g,o,k)
    assert result['scores']['status']=='assessed'
    assert result['scores']['completeness']==.25 and result['scores']['evidence_support'] is None
    assert result['assessment']['claim_checks'][0]['citation_supported'] is None
    assert g['raw_feedback']['assessment_json']==raw


def test_normalization_preserves_material_errors(monkeypatch):
    module=load('normalize_checkpoint_grades',monkeypatch);g,o,k,a=fixture()
    a['defects']=[{'claim_id':'I1','severity':'material','explanation':'Fixture material defect remains.',
        'reference_record_ids':[],'alleged_winner_invalidation':None}]
    g['raw_feedback']['assessment_json']=json.dumps(a)
    result=module.normalize(g,o,k)
    assert result['scores']['material_error_count']==1 and result['scores']['decision_safe_coverage']==0


def test_uncited_factual_claim_is_not_repaired(monkeypatch):
    module=load('normalize_checkpoint_grades',monkeypatch);g,o,k,a=fixture()
    o['report']['claims'][0]['claim_type']='factual'
    assert module.normalize(g,o,k)['scores']['status']=='invalid_grader_output'


def test_failed_checkpoint_keeps_actual_planner_cost(monkeypatch):
    module=load('analyze_checkpoint_study',monkeypatch)
    rows=[{'question_id':'Q01','repetition':1,'budget':0,'stage':'planner','status':'response_received',
        'model':'gpt-5-mini','input_tokens':1000,'output_tokens':100,'total_tokens':1100,'cached_input_tokens':800}]
    metrics={'execution_status':'planner_failed'}
    usage=module.counterfactual_usage(metrics,rows,'Q01',1,16)
    assert usage['total_tokens']==1100 and usage['model_request_attempts']==1
    assert abs(usage['estimated_cost_usd']-.00027)<1e-10
    assert 'cost_if_stopped_here' not in metrics
