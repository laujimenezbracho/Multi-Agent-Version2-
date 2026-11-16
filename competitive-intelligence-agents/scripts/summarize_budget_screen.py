"""Offline presentation of exact answers, measured efficiency and disputed model grading."""
import argparse
from html import escape
import json
from pathlib import Path


def summarize(root):
    root=Path(root)
    grading={}
    for row in json.loads((root/'independent_blind_results.json').read_text()):
        trace=json.loads(row['trace']); inputs=json.loads(trace['info']['request_preview'])
        grading[(inputs['question_id'],inputs['retrieval_calls'])]={
            'value':row.get('independent_blind_audit_v2/value'),
            'rationale':row.get('independent_blind_audit_v2/rationale'), 'trace_id':row['trace_id']}
    rows=[];sections=[];notes=[]
    for question in ('Q01','Q15'):
        for budget in (16,32,60):
            path=root/'cases'/f'{question}-b{budget}'
            case=json.loads((path/'prediction.json').read_text()); row=dict(case['summary'])
            row['usage_source']='SDK stage totals'
            recovery=path/'usage_recovered_from_trace.json'
            if row['estimated_cost_usd'] is None and recovery.exists():
                recovered=json.loads(recovery.read_text())
                for key in ['input_tokens','output_tokens','total_tokens','model_requests','estimated_cost_usd','cached_input_tokens']:
                    row[key]=recovered[key]
                row['usage_source']='Actual original MLflow response token usage (sealed SDK metrics preserved)'
            report=case['prediction']['report']
            row['answer_type']='No report' if report is None else 'Abstention' if not report['sources'] else 'Substantive answer'
            row['independent_model_feedback']=grading.get((question,budget))
            row['human_review_note']=''
            if question=='Q01' and budget==60:
                row['human_review_note']='Source-audit check: PulseGrid USD130 is correctly identified as lowest. VertexOps USD174 is outdated; WEB-30009 gives USD159. This correction does NOT change the lowest-price winner. The model grader’s rationale incorrectly implies that it does; numeric ratings require human adjudication.'
            if question=='Q15' and report is not None:
                row['human_review_note']='No three threats are ranked and no sources support a comparison. Gap disclosure is the only one of the four predeclared essential elements addressed. The 60-call automatic completeness score of 0.75 over-credits abstention relative to that rubric; do not treat it as validated quality.' if budget==60 else 'No three threats are ranked and no sources support a comparison. Honest abstention avoids invented facts but does not complete the requested task.'
            rows.append(row)
            title=f'{question} / {budget}-call ceiling'
            notes.extend([f'## {title}', '', f"Outcome: {row['answer_type']}; stop: {row['stop_reason']}; actual tool attempts: {row['retrieval_attempts']}.",''])
            body='<h2>'+escape(title)+'</h2><p><b>'+escape(row['answer_type'])+'</b> · Internal Reviewer: '+escape(str(row['internal_reviewer_decision']))+'</p>'
            if report is None:
                error=row.get('error') or {}
                where=error.get('stage','unknown')
                origin=error.get('diagnostic',{}).get('origin',{})
                text='The Researcher cited a record it had not retrieved in full; validation stopped the workflow before the Analyst.' if origin.get('module')=='ci_agents.agents.researcher' and origin.get('line')==148 else 'The Analyst’s returned output was rejected by the SDK; no validated final report exists.'
                body+='<p>'+escape(text)+'</p>';notes.extend([text,''])
            else:
                for heading,values in [('Executive answer',[report['executive_summary']]),('Key findings',report['key_findings']),
                    ('Competitor comparison',report['competitor_comparison']),('Strategic implications',report['strategic_implications']),('Limitations',report['limitations'])]:
                    body+='<h3>'+heading+'</h3><ul>'+''.join('<li>'+escape(x)+'</li>' for x in values)+'</ul>'
                    notes.extend([f'### {heading}','',*['- '+x for x in values],''])
                body+='<details><summary>Original structured claims and citations</summary><pre>'+escape(json.dumps(report['claims'],ensure_ascii=False,indent=2))+'</pre></details>'
                review_path=Path(row['artifact_directory'])/'review.json'
                if review_path.exists():
                    body+='<details><summary>Internal Reviewer feedback — not ground truth</summary><pre>'+escape(review_path.read_text())+'</pre></details>'
            if row['human_review_note']:
                body+='<p class="warning">'+escape(row['human_review_note'])+'</p>'
                notes.extend(['**Human adjudication note:** '+row['human_review_note'],''])
            body+='<details><summary>Independent model grading — provisional and disputed</summary><pre>'+escape(json.dumps(row['independent_model_feedback'],ensure_ascii=False,indent=2))+'</pre></details>'
            sections.append('<article id="'+question+'-'+str(budget)+'">'+body+'</article>')
    if len(rows)!=6 or any(r['execution_status']=='not_started_budget' for r in rows):
        raise ValueError('All six actual conditions must be executed before publishing this final summary')
    total={'unique_agent_executions':6,'retrieval_attempts':sum(r['retrieval_attempts'] for r in rows),
        'model_requests':sum(r['model_requests'] for r in rows),'input_tokens':sum(r['input_tokens'] for r in rows),
        'output_tokens':sum(r['output_tokens'] for r in rows),'total_tokens':sum(r['total_tokens'] for r in rows),
        'estimated_agent_usd_uncached_input':sum(r['estimated_cost_usd'] for r in rows),
        'independent_grading_cost_included':False,'independent_grading_cost':'Unavailable in the per-agent stage telemetry; additional provider usage occurred.',
        'no_report_count':sum(r['answer_type']=='No report' for r in rows),
        'abstention_count':sum(r['answer_type']=='Abstention' for r in rows),
        'substantive_answer_count':sum(r['answer_type']=='Substantive answer' for r in rows),
        'budget_exhaustions':sum(r.get('budget_exhausted',False) for r in rows),
        'conclusion':'No tested ceiling became binding; this screen cannot validate an optimal retrieval limit. Model/source handling and abstention failures need attention first.',
        'automatic_grade_status':'Provisional and disputed; known arithmetic/scope inconsistencies are preserved and annotated for human review.'}
    (root/'pilot_summary.json').write_text(json.dumps({'totals':total,'cases':rows},ensure_ascii=False,indent=2)+'\n')
    (root/'answers.md').write_text('# Exact saved answers — six-case screen\n\n'+total['conclusion']+'\n\n'+'\n'.join(notes))
    table='<table><tr><th>Question</th><th>Ceiling</th><th>Actual tools</th><th>Model requests</th><th>Tokens</th><th>Seconds</th><th>Agent USD estimate*</th><th>Answer</th></tr>'
    for row in rows:
        table+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [row['question_id'],row['retrieval_budget'],row['retrieval_attempts'],row['model_requests'],row['total_tokens'],row['latency_seconds'],f"{row['estimated_cost_usd']:.4f}",row['answer_type']])+'</tr>'
    table+='</table>'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Six-run budget screen: exact answers</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1120px;margin:36px auto;padding:0 20px;background:#f5f8f7;color:#17343b}article{padding:24px;margin:28px 0;background:white;border:1px solid #d0ddd7;border-radius:12px}h1,h2{line-height:1.25}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f1;padding:14px;font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border:1px solid #c9d7d1;text-align:left}.scroll{overflow-x:auto}.warning{background:#fff0c9;padding:16px;border-radius:8px}a{color:#007064}li{margin-bottom:9px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}</style><h1>Six-run retrieval-budget screen</h1>'''
    page+='<p class="warning">'+escape(total['conclusion'])+' Three runs produced no final report; two abstained; one gave a partially correct substantive answer. No failed condition was retried.</p>'
    page+='<p>All agents: GPT-5 mini, low reasoning, 12,000 output-token cap; same dataset, prompts and cutoff. Recovery disabled. Each ceiling was tested once.</p>'
    page+='<p><b>Q01:</b> Which competitor has the lowest current Growth plan price?<br><b>Q15:</b> What are the top 3 competitive threats for PulseGrid?</p>'
    page+='<p>'+ ' · '.join(f'<a href="#{q}-{b}">{q}: {b}</a>' for q in ('Q01','Q15') for b in (16,32,60))+'</p><div class="scroll">'+table+'</div>'
    page+='<p>*Estimated six-run agent cost: USD '+f"{total['estimated_agent_usd_uncached_input']:.4f}"+'. All input conservatively priced as uncached; grader API usage is additional and excluded. One failed case’s actual usage was recovered from its original response trace without changing sealed metrics.</p>'
    page+='<p class="warning">Independent model scores are provisional, not ground truth. Spot-checking found inconsistent arithmetic/scope judgments. Raw feedback is preserved below with explicit adjudication notes; use the source-audited rubric and original answers to judge them yourself.</p>'+''.join(sections)+'</html>'
    (root/'answers.html').write_text(page,encoding='utf-8')
    print(json.dumps(total,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    summarize(parser.parse_args().output_dir)
