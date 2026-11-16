"""Evaluator protocol invariants; fixtures do not prove semantic judge quality."""
import copy
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/grading_contract.py'
spec=importlib.util.spec_from_file_location('grading_contract',path)
grading=importlib.util.module_from_spec(spec);spec.loader.exec_module(grading)


def example(empty=False):
    text='No evidence was supplied; no ranking is possible.' if empty else 'The current Growth list price is 130 USD per seat per month.'
    report={'executive_summary':text,'key_findings':[text],'competitor_comparison':[text],
        'strategic_implications':[text],'limitations':['Unknown contract fees.'],
        'claims':[] if empty else [{'claim_id':'C1','claim_type':'factual','text':text,'supporting_record_ids':['TEST-1']}],
        'sources':[] if empty else ['TEST-1']}
    source={'record_id':'TEST-1','summary':'Observed list price is 130 USD per seat per month.'}
    prediction={'report':report,'original_cited_sources':[] if empty else [source]}
    rubric={'essential_elements':['Direct answer','Evidence-linked comparison','Supported implications','Disclose gaps'],
        'essential_requires_evidence':[True,True,True,False],
        'reference_prices':{'PulseGrid':{'price_usd_per_seat_month':130},'VertexOps':{'price_usd_per_seat_month':159}},
        'audited_reference_sources':[source]}
    assessment={'status':'assessed','requirement_checks':[{'requirement_id':f'E{i:02d}',
        'met':not empty or i==4,'explanation':'Fixture judgement.',
        'answer_excerpt':text if not empty or i==4 else ''} for i in range(1,5)],
        'claim_checks':[] if empty else [{'claim_id':'C1','factually_correct':True,'citation_supported':True,
            'temporally_correct':True,'explanation':'Fixture judgement.',
            'anchors':[{'record_id':'TEST-1','excerpt':source['summary']}]}],
        'defects':[],'narrative_scanned':True}
    return prediction,rubric,assessment


def test_abstention_has_one_of_four_essential_elements_not_three():
    p,r,a=example(empty=True);score=grading.normalise(a,p,r)
    assert score['completeness']==.25
    assert score['factual_accuracy'] is None and score['temporal_accuracy'] is None
    assert score['approval']=='pending_human'


def test_uncited_abstention_cannot_satisfy_competitive_requirement():
    p,r,a=example(empty=True);a['requirement_checks'][0].update(met=True,answer_excerpt=p['report']['executive_summary'])
    with pytest.raises(ValueError,match='uncited abstention'):grading.normalise(a,p,r)


def test_invented_direct_scores_are_rejected():
    p,r,a=example();a['completeness']=.75
    with pytest.raises(ValueError):grading.normalise(a,p,r)


def test_quote_and_catalogue_must_be_exact():
    p,r,a=example();a['requirement_checks'].pop()
    with pytest.raises(ValueError,match='essential requirements'):grading.normalise(a,p,r)
    p,r,a=example();a['claim_checks'][0]['anchors'][0]['excerpt']='Invented source text that was not supplied.'
    with pytest.raises(ValueError,match='exact original'):grading.normalise(a,p,r)


def test_correcting_a_more_expensive_competitor_does_not_invalidate_winner():
    p,r,a=example();a['defects']=[{'claim_id':'C1','severity':'material','explanation':'Alleged wrong winner.',
        'reference_record_ids':['TEST-1'],'alleged_winner_invalidation':{'claimed_winner':'PulseGrid','competing_company':'VertexOps'}}]
    with pytest.raises(ValueError,match='does not undercut'):grading.normalise(a,p,r)


def test_failed_checks_cannot_disappear_from_defects():
    p,r,a=example();a['claim_checks'][0]['factually_correct']=False
    with pytest.raises(ValueError,match='consistently linked defect'):grading.normalise(a,p,r)


def test_no_report_is_not_perfect_abstention():
    p,r,a=example();p['report']=None
    a={'status':'no_report','requirement_checks':[],'claim_checks':[],'defects':[],'narrative_scanned':False}
    score=grading.normalise(a,p,r)
    assert score['task_completion']==0 and score['completeness'] is None


def test_derived_scores_are_counts_not_model_fractions():
    p,r,a=example();a['requirement_checks'][1].update(met=False,answer_excerpt='')
    score=grading.normalise(a,p,r)
    assert score['completeness']==.75 and score['factual_accuracy']==1
    assert score['denominators']['essential_elements']==4


def test_stable_pointers_resolve_original_text_without_copying_quotes():
    p,r,a=example()
    for check in a['requirement_checks']:
        check.pop('answer_excerpt')
        check['answer_segment_id']='EXEC'
    a['claim_checks'][0]['anchors']=[{'record_id':'TEST-1','field_name':'summary'}]
    assert grading.normalise(a,p,r)['completeness']==1


def test_unknown_answer_pointer_is_rejected():
    p,r,a=example()
    a['requirement_checks'][0]['answer_segment_id']='invented'
    with pytest.raises(ValueError,match='exact answer excerpt'):
        grading.normalise(a,p,r)
