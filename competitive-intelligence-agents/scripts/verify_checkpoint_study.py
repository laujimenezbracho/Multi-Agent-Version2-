"""Offline integrity checks for the 54-observation paired checkpoint experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def verify(root,project,key_dir):
    plan=json.loads((root/'experiment_plan.json').read_text())
    for relative,expected in plan['source'].items():assert fingerprint((project/relative).read_text())==expected,relative
    for name,expected in plan['scripts'].items():assert fingerprint((project/'scripts'/name).read_text())==expected,name
    for q,expected in plan['keys'].items():assert fingerprint((key_dir/(q+'.json')).read_text())==expected,q
    sources={r['record_id']:r for r in map(json.loads,(project/'data/public/public_web_records.jsonl').read_text().splitlines())}
    results=[]
    for q in plan['questions']:
        for rep in (1,2,3):
            previous=[];previous_ids=set();previous_calls=0
            for b in plan['checkpoints']:
                directory=root/'cases'/f'{q}-r{rep}-b{b}'
                value=json.loads((directory/'prediction.json').read_text());m=value['metrics']
                assert value['binding']==fingerprint(plan)
                assert value['output_sha256']==fingerprint(value['output'])
                assert m['retrieval_operations']<=b and m['retrieval_operations']>=previous_calls
                assert (directory/'independent_grade.json').exists()
                pp=directory/'evidence_pack.json'
                if pp.exists():
                    pack=json.loads(pp.read_text())
                    events=[json.loads(x) for x in pack['retrieval_log'] if json.loads(x).get('kind')!='research_outcome']
                    assert events[:len(previous)]==previous,'Non-nested retrieval history'
                    full={rid for e in events if e.get('tool')=='get_record' and e.get('status')=='ok' for rid in e.get('record_ids',[])}
                    ids={e['record_id'] for e in pack['evidence_items']}
                    assert previous_ids<=ids<=full
                    assert len(events)==m['retrieval_operations']
                    for item in pack['evidence_items']:
                        source=sources[item['record_id']]
                        assert item['source_title']==source['document_title']
                        assert item['source_url']==source['source_url']
                        assert item['finding']==source['summary']
                        assert source['details'] in item['quoted_or_paraphrased_evidence']
                        assert source['available_at']<=plan['cutoff'].replace('+00:00','Z')
                    previous=events;previous_ids=ids
                    report=value['output']['report']
                    if report:
                        assert set(report['sources'])<=ids
                        for claim in report['claims']:
                            assert set(claim['supporting_record_ids'])<=ids
                            if claim['claim_type']=='factual':assert claim['supporting_record_ids']
                previous_calls=m['retrieval_operations']
                results.append({'question_id':q,'repetition':rep,'budget':b,'execution_status':m['execution_status'],
                    'report_available':value['output']['report'] is not None,'integrity_checks_passed':True})
    assert len(results)==54
    usage=[json.loads(x) for x in (root/'request_usage.jsonl').read_text().splitlines()]
    response_ids=[x['response_id'] for x in usage if x['response_id']]
    assert len(response_ids)==len(set(response_ids)),'Duplicate counted model responses'
    meter_by_id={x['response_id']:x for x in usage if x['response_id']}
    traced_responses=0
    database=root/'mlflow.db'
    if database.exists():
        connection=sqlite3.connect(database)
        for raw, in connection.execute("select content from spans where name='AsyncResponses'"):
            attributes=json.loads(raw).get('attributes',{})
            try:response=json.loads(attributes.get('mlflow.spanOutputs','{}'))
            except (TypeError,ValueError):continue
            response_id=response.get('id')
            if not response_id:continue
            assert response_id in meter_by_id,'Traced model response missing from usage ledger'
            measured=meter_by_id[response_id];reported=response.get('usage') or {}
            for field in ['input_tokens','output_tokens','total_tokens']:
                if reported.get(field) is not None:assert measured[field]==reported[field],field
            traced_responses+=1
        connection.close()
    summary={'conditions':54,'trajectories':18,'checkpoints':[8,16,32],'source_and_key_hashes_unchanged':True,
        'nested_retrieval_prefixes_verified':True,'full_record_source_provenance_verified':True,
        'model_response_ids_unique':True,'reports_produced':sum(x['report_available'] for x in results),
        'native_sdk_response_spans_cross_checked':traced_responses,
        'execution_failures':sum(x['execution_status']!='completed' for x in results),'cases':results}
    (root/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--key-dir',type=Path,required=True);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args();verify(a.run_dir,a.project,a.key_dir)
