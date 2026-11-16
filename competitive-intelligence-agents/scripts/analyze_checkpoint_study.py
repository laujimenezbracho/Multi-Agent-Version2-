"""Offline paired statistical analysis of native MLflow checkpoint observations.

No model calls or custom evaluation execution: this consumes saved native judgments,
validated item assessments and measured resource ledgers.
"""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
from statistics import mean

QUESTIONS=('Q01','Q03','Q05','Q07','Q12','Q15')
BUDGETS=(8,16,32)


def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def percentile(values,p):
    values=sorted(values);position=(len(values)-1)*p;i=int(position);fraction=position-i
    return values[i]*(1-fraction)+values[min(i+1,len(values)-1)]*fraction


def cluster_interval(values,alpha=.05):
    """Exact enumeration of the ordinary empirical cluster-bootstrap distribution.

    Exact enumeration avoids Monte Carlo noise; coverage remains an approximate
    bootstrap inference with only six heterogeneous question clusters.
    """
    distribution=[sum(sample)/len(values) for sample in product(values,repeat=len(values))]
    return [percentile(distribution,alpha/2),percentile(distribution,1-alpha/2)]


def interval_delta(cases,smaller,larger,alpha=.05):
    low=[];high=[];points=[]
    for q in QUESTIONS:
        question_low=[];question_high=[];question_points=[]
        for rep in (1,2,3):
            a,b=cases[q,rep,smaller],cases[q,rep,larger]
            question_low.append(a['quality_low']-b['quality_high'])
            question_high.append(a['quality_high']-b['quality_low'])
            if a['quality'] is not None and b['quality'] is not None:question_points.append(a['quality']-b['quality'])
        low.append(mean(question_low));high.append(mean(question_high))
        points.append(mean(question_points) if len(question_points)==3 else None)
    return {'smaller':smaller,'reference':larger,'mean_difference':mean(points) if all(x is not None for x in points) else None,
        'mean_difference_bounds':[mean(low),mean(high)],
        'cluster_interval_with_missing_sensitivity':[cluster_interval(low,alpha)[0],cluster_interval(high,alpha)[1]],
        'confidence_level':1-alpha,'question_clusters':6,'repetitions_per_cluster':3,
        'paired_question_differences':dict(zip(QUESTIONS,points))}


def counterfactual_usage(metrics,usage_rows,q,rep,budget):
    if metrics.get('cost_if_stopped_here') is not None:return metrics['cost_if_stopped_here']
    # Planner-failure checkpoints contain no downstream stages. Its actual returned
    # API usage is still known in the append-only meter and must not be discarded.
    selected=[x for x in usage_rows if x.get('question_id')==q and x.get('repetition')==rep
        and (x['stage']=='planner' or x['stage']=='researcher' and x.get('budget',0)<=budget
             or x['stage'] in {'analyst','reviewer'} and x.get('budget')==budget)]
    if not selected:return {'usage_complete':False,'model_request_attempts':None,'total_tokens':None,
        'estimated_cost_usd':None,'known_estimated_cost_usd':0,'usage_basis':'No measured request ledger available'}
    from checkpoint_meter import totals
    return totals(selected)|{'usage_basis':'Recovered from original append-only request ledger for failed checkpoint'}


def incremental_usage(metrics,usage_rows,q,rep,budget):
    if metrics.get('incremental_research_usage') is not None:return metrics['incremental_research_usage']
    from checkpoint_meter import totals
    selected=[x for x in usage_rows if x.get('question_id')==q and x.get('repetition')==rep
        and x.get('budget')==budget and x.get('stage')=='researcher']
    if selected:return totals(selected)
    if metrics['execution_status']=='planner_failed':
        return totals([])|{'usage_basis':'Research deterministically not started after Planner failure'}
    return {'total_tokens':None,'estimated_cost_usd':None,'usage_complete':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--readme',type=Path)
    p.add_argument('--grade-layer',choices=['raw','normalized','adjudicated'],default='adjudicated')
    a=p.parse_args();root=a.run_dir.resolve()
    usage_rows=[json.loads(x) for x in (root/'request_usage.jsonl').read_text().splitlines()]
    cases={};records=[]
    for q in QUESTIONS:
        for rep in (1,2,3):
            for budget in BUDGETS:
                directory=root/'cases'/f'{q}-r{rep}-b{budget}'
                checkpoint=json.loads((directory/'prediction.json').read_text())
                grade_path=directory/'independent_grade.json'
                if a.grade_layer in {'normalized','adjudicated'} and (directory/'protocol_normalized_grade.json').exists():
                    grade_path=directory/'protocol_normalized_grade.json'
                if a.grade_layer=='adjudicated' and (directory/'protocol_adjudicated_grade.json').exists():
                    grade_path=directory/'protocol_adjudicated_grade.json'
                grade=json.loads(grade_path.read_text())
                s=grade['scores'];m=checkpoint['metrics'];report=checkpoint['output']['report']
                quality=s.get('decision_safe_coverage')
                if report is None:quality=0
                requirements={x['requirement_id']:x['met'] for x in grade.get('assessment',{}).get('requirement_checks',[])}
                record={'question_id':q,'repetition':rep,'budget':budget,'quality':quality,
                    'quality_low':0 if quality is None else quality,'quality_high':100 if quality is None else quality,
                    'grade_status':s['status'],'completeness':s.get('completeness'),
                    'factual_accuracy':s.get('factual_accuracy'),'evidence_support':s.get('evidence_support'),
                    'temporal_accuracy':s.get('temporal_accuracy'),
                    'material_error_count':s.get('material_error_count'),'requirements':requirements,
                    'defects':grade.get('assessment',{}).get('defects',[]),
                    'execution_status':m['execution_status'],'report_available':report is not None,
                    'internal_reviewer_decision':m.get('internal_reviewer_decision'),
                    'retrieval_operations':m['retrieval_operations'],'full_record_ids':m.get('full_record_ids',[]),
                    'cited_ids':sorted(set(report['sources'])|{rid for c in report['claims'] for rid in c['supporting_record_ids']}) if report else [],
                    'cost_if_stopped_here':counterfactual_usage(m,usage_rows,q,rep,budget),
                    'incremental_research_usage':incremental_usage(m,usage_rows,q,rep,budget),
                    'incremental_research_latency_seconds':m.get('incremental_research_latency_seconds',0 if m['execution_status']=='planner_failed' else None),
                    'source_checkpoint':str(directory/'prediction.json'),'source_grade':str(grade_path)}
                planner_metrics=root/'trajectories'/f'{q}-r{rep}'/'planner_stage_metrics.json'
                if planner_metrics.exists():
                    planner_seconds=json.loads(planner_metrics.read_text())['latency_seconds']
                    research_seconds=0
                    for prior_budget in BUDGETS:
                        if prior_budget>budget:break
                        prior_metrics=json.loads((root/'cases'/f'{q}-r{rep}-b{prior_budget}'/'prediction.json').read_text())['metrics']
                        research_seconds+=prior_metrics.get('incremental_research_latency_seconds') or 0
                    record['stop_at_budget_stage_seconds']=planner_seconds+research_seconds+sum(
                        m.get(stage,{}).get('latency_seconds',0) for stage in ['analyst_stage','reviewer_stage'])
                else:record['stop_at_budget_stage_seconds']=None
                cases[q,rep,budget]=record;records.append(record)
    assert len(cases)==54
    aggregated=[]
    for budget in BUDGETS:
        rows=[v for v in records if v['budget']==budget]
        costs=[v['cost_if_stopped_here'].get('estimated_cost_usd') for v in rows]
        upper_costs=[v['cost_if_stopped_here'].get('conservative_uncached_estimate_usd') for v in rows]
        token_counts=[v['cost_if_stopped_here'].get('total_tokens') for v in rows]
        model_counts=[v['cost_if_stopped_here'].get('model_request_attempts') for v in rows]
        durations=[v['stop_at_budget_stage_seconds'] for v in rows]
        low=[mean(cases[q,r,budget]['quality_low'] for r in (1,2,3)) for q in QUESTIONS]
        high=[mean(cases[q,r,budget]['quality_high'] for r in (1,2,3)) for q in QUESTIONS]
        aggregated.append({'budget':budget,'observations':18,
            'quality_mean':mean(v['quality'] for v in rows) if all(v['quality'] is not None for v in rows) else None,
            'quality_mean_bounds':[mean(low),mean(high)],
            'cluster_95_interval_with_missing_sensitivity':[cluster_interval(low)[0],cluster_interval(high)[1]],
            'ungradable_answers':sum(v['quality'] is None for v in rows),
            'material_error_answers':sum((v['material_error_count'] or 0)>0 for v in rows),
            'material_errors_total':sum(v['material_error_count'] or 0 for v in rows),
            'no_report':sum(not v['report_available'] for v in rows),
            'execution_failures':sum(v['execution_status']!='completed' for v in rows),
            'internal_reviewer_pass':sum(v['internal_reviewer_decision']=='PASS' for v in rows),
            'retrieval_operations_mean':mean(v['retrieval_operations'] for v in rows),
            'full_records_mean':mean(len(v['full_record_ids']) for v in rows),
            'stop_at_budget_mean_tokens':mean(token_counts) if all(x is not None for x in token_counts) else None,
            'stop_at_budget_mean_model_requests':mean(model_counts) if all(x is not None for x in model_counts) else None,
            'stop_at_budget_mean_stage_seconds':mean(durations) if all(x is not None for x in durations) else None,
            'stop_at_budget_token_range':[min(token_counts),max(token_counts)] if all(x is not None for x in token_counts) else None,
            'stop_at_budget_mean_estimated_usd':mean(costs) if all(x is not None for x in costs) else None,
            'stop_at_budget_mean_uncached_upper_usd':mean(upper_costs) if all(x is not None for x in upper_costs) else None,
            'known_cost_subtotal_usd':sum(v['cost_if_stopped_here'].get('known_estimated_cost_usd',0) for v in rows)})
        for field in ['completeness','factual_accuracy','evidence_support','temporal_accuracy']:
            applicable=[v[field] for v in rows if v[field] is not None]
            aggregated[-1][field+'_conditional_mean']=mean(applicable) if applicable else None
            aggregated[-1][field+'_applicable_observations']=len(applicable)
    bybudget={v['budget']:v for v in aggregated}
    increments=[]
    for q in QUESTIONS:
        for rep in (1,2,3):
            for before,after in [(8,16),(16,32)]:
                old,new=cases[q,rep,before],cases[q,rep,after]
                increments.append({'question_id':q,'repetition':rep,'from_budget':before,'to_budget':after,
                    'quality_change':new['quality']-old['quality'] if new['quality'] is not None and old['quality'] is not None else None,
                    'quality_change_bounds':[new['quality_low']-old['quality_high'],new['quality_high']-old['quality_low']],
                    'requirements_gained':[k for k,v in new['requirements'].items() if v and old['requirements'].get(k) is False],
                    'requirements_lost':[k for k,v in old['requirements'].items() if v and new['requirements'].get(k) is False],
                    'new_full_record_ids':sorted(set(new['full_record_ids'])-set(old['full_record_ids'])),
                    'new_cited_record_ids':sorted(set(new['cited_ids'])-set(old['cited_ids'])),
                    'extra_retrieval_operations':new['retrieval_operations']-old['retrieval_operations'],
                    'new_full_evidence_obtained':bool(set(new['full_record_ids'])-set(old['full_record_ids'])),
                    'research_vs_regeneration_note':'No extra retrieval: any answer change reflects synthesis/review variability or an early-stop decision, not additional retrieved evidence.' if new['retrieval_operations']==old['retrieval_operations'] else 'Additional operations occurred; inspect whether they yielded new full evidence rather than only discovery/noise.',
                    'incremental_research_usage':new['incremental_research_usage'],
                    'incremental_research_latency_seconds':new['incremental_research_latency_seconds'],
                    'material_defects_before':old['material_error_count'],'material_defects_after':new['material_error_count'],
                    'previous_defects':old['defects'],'current_defects':new['defects'],
                    'interpretation':'Defect sets are retained verbatim. Regenerated claim IDs are not assumed to align across checkpoints.'})
    contrasts=[interval_delta(cases,b,32,alpha=.025) for b in (8,16)]
    # Bonferroni: two 97.5% intervals for the two candidate-vs-reference comparisons.
    eligible=[]
    for c in contrasts:
        b=c['smaller'];stats=bybudget[b]
        c['within_predeclared_2_point_margin']=c['cluster_interval_with_missing_sensitivity'][0]>=-2
        c['observed_error_and_execution_gate']=stats['material_error_answers']==0 and stats['execution_failures']==0 and stats['ungradable_answers']==0
        if c['within_predeclared_2_point_margin'] and c['observed_error_and_execution_gate']:eligible.append(b)
    best=max(aggregated,key=lambda v:v['quality_mean_bounds'][0])
    if eligible:
        recommendation={'provisional_default':min(eligible),'basis':'Smallest candidate meeting the predeclared paired noninferiority and observed error/execution gates.',
            'statistically_supported_within_this_study':True}
    else:
        near=[v for v in aggregated if v['quality_mean_bounds'][0]>=best['quality_mean_bounds'][0]-2 and v['material_error_answers']<=best['material_error_answers']]
        chosen=min(near,key=lambda v:v['budget'])
        recommendation={'provisional_default':chosen['budget'],
            'basis':'Best observed quality/cost compromise: smallest budget within two observed mean points of the best conservative mean, with no more observed material-error answers. Smaller-cap noninferiority was not established under the predeclared uncertainty/error gates.',
            'statistically_supported_within_this_study':False}
    recommendation.update(universal_optimality_proved=False,low_error_risk_certified=False,
        limitations=['Only six distinct selected synthetic questions and three repetitions each.',
            'No budgets above 32 tested; no held-out validation performed.',
            'Experimental resumable selector differs from production early-stop policy.',
            'Independent judges and source-audited keys remain fallible and not human-owner approved.',
            'CI is a small-cluster empirical bootstrap, not a distribution-free guarantee.'])
    from checkpoint_meter import totals
    usage_rows=[json.loads(x) for x in (root/'request_usage.jsonl').read_text().splitlines()]
    if (root/'adjudication_usage.jsonl').exists():
        usage_rows += [json.loads(x) for x in (root/'adjudication_usage.jsonl').read_text().splitlines()]
    usage={'whole_study_including_calibration_and_judges':totals(usage_rows),
        'agent_work_including_shadow_reports':totals([x for x in usage_rows if x.get('stage') in {'planner','researcher','analyst','reviewer'}]),
        'external_judging_and_calibration':totals([x for x in usage_rows if x.get('stage')=='external_judge']),
        'billing_preflight':totals([x for x in usage_rows if x.get('stage')=='billing_probe'])}
    save(root/'case_analysis.json',records);save(root/'incremental_benefits.json',increments)
    marginal=[]
    for before,after in [(8,16),(16,32)]:
        pairs=[x for x in increments if x['from_budget']==before]
        gains=[x['quality_change'] for x in pairs]
        gain_bounds=[mean(x['quality_change_bounds'][0] for x in pairs),mean(x['quality_change_bounds'][1] for x in pairs)]
        extra_costs=[x['incremental_research_usage'].get('estimated_cost_usd') for x in pairs]
        stop_cost_changes=[]
        for q in QUESTIONS:
            for rep in (1,2,3):
                old=cases[q,rep,before]['cost_if_stopped_here'].get('estimated_cost_usd')
                new=cases[q,rep,after]['cost_if_stopped_here'].get('estimated_cost_usd')
                stop_cost_changes.append(new-old if old is not None and new is not None else None)
        gain=mean(gains) if all(x is not None for x in gains) else None
        added=mean(extra_costs) if all(x is not None for x in extra_costs) else None
        end_to_end=mean(stop_cost_changes) if all(x is not None for x in stop_cost_changes) else None
        marginal.append({'from_budget':before,'to_budget':after,'mean_quality_gain':gain,
            'mean_quality_gain_bounds':gain_bounds,
            'mean_additional_research_cost_usd':added,'mean_change_in_stop_at_budget_workflow_cost_usd':end_to_end,
            'research_usd_per_quality_point':added/gain if added is not None and gain is not None and gain>0 else None,
            'workflow_usd_per_quality_point_bounds':[end_to_end/gain_bounds[1],end_to_end/gain_bounds[0]] if end_to_end is not None and gain_bounds[0]>0 else None,
            'interpretation':'Observed ratio, not a guaranteed return or business valuation. Nonpositive quality gain has no positive quality-return ratio.'})
    result={'conditions':54,'research_trajectories':18,'grade_layer':a.grade_layer,'budget_summary':aggregated,
        'paired_contrasts_vs_32':contrasts,'marginal_value':marginal,'recommendation':recommendation,'actual_usage':usage,
        'score_scope':'Decision-safe essential coverage; claim fractions describe structured catalogue, with prose defects checked separately.'}
    save(root/'threshold_results.json',result)
    with (root/'case_analysis.csv').open('w',newline='') as f:
        columns=['question_id','repetition','budget','quality','quality_low','quality_high','grade_status','material_error_count',
            'completeness','factual_accuracy','evidence_support','temporal_accuracy','execution_status','retrieval_operations',
            'full_records','cited_records','internal_reviewer_decision','model_requests','input_tokens','output_tokens','total_tokens',
            'cached_input_tokens','estimated_workflow_usd','uncached_workflow_upper_usd','measured_stage_seconds']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for row in records:
            measured=row['cost_if_stopped_here']
            writer.writerow(row|{'full_records':len(row['full_record_ids']),'cited_records':len(row['cited_ids']),
                'model_requests':measured.get('model_request_attempts'),'input_tokens':measured.get('input_tokens'),
                'output_tokens':measured.get('output_tokens'),'total_tokens':measured.get('total_tokens'),
                'cached_input_tokens':measured.get('cached_input_tokens'),'estimated_workflow_usd':measured.get('estimated_cost_usd'),
                'uncached_workflow_upper_usd':measured.get('conservative_uncached_estimate_usd'),
                'measured_stage_seconds':row['stop_at_budget_stage_seconds']})
    with (root/'incremental_benefits.csv').open('w',newline='') as f:
        columns=['question_id','repetition','from_budget','to_budget','quality_change','quality_change_low','quality_change_high',
            'extra_retrieval_operations','new_full_records','new_full_record_ids','new_cited_record_ids','requirements_gained','requirements_lost',
            'material_defects_before','material_defects_after','extra_research_model_requests','extra_research_input_tokens',
            'extra_research_output_tokens','extra_research_total_tokens','extra_research_usd','extra_research_seconds',
            'previous_material_defects','current_material_defects','research_vs_regeneration_note']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for row in increments:
            measured=row['incremental_research_usage']
            writer.writerow(row|{'quality_change_low':row['quality_change_bounds'][0],
                'quality_change_high':row['quality_change_bounds'][1],'new_full_records':len(row['new_full_record_ids']),
                'extra_research_model_requests':measured.get('model_request_attempts'),
                'extra_research_input_tokens':measured.get('input_tokens'),'extra_research_output_tokens':measured.get('output_tokens'),
                'extra_research_total_tokens':measured.get('total_tokens'),'extra_research_usd':measured.get('estimated_cost_usd'),
                'extra_research_seconds':row['incremental_research_latency_seconds'],
                'previous_material_defects':' | '.join(x['explanation'] for x in row['previous_defects'] if x['severity']=='material'),
                'current_material_defects':' | '.join(x['explanation'] for x in row['current_defects'] if x['severity']=='material')})
    lines=['## Results — completed study','',
        '**Recommendation: use '+str(recommendation['provisional_default'])+' as the provisional retrieval ceiling for this tested continuation policy.** '+recommendation['basis'],'',
        'This is a retrieval-operation ceiling, not a token ceiling or total API-request limit. It is not proof of a universally optimal budget.','',
        '| Ceiling | Answer snapshots | Quality score / 100 | Material-error answers | Execution failures | Ungradable | Mean actual retrievals | Mean model requests if stopped here | Mean workflow tokens if stopped here | Mean workflow USD: observed cache / uncached upper |',
        '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in aggregated:
        quality=f"{s['quality_mean']:.2f}" if s['quality_mean'] is not None else f"{s['quality_mean_bounds'][0]:.2f}–{s['quality_mean_bounds'][1]:.2f} (missing-grade bounds)"
        tokens=f"{s['stop_at_budget_mean_tokens']:.0f}" if s['stop_at_budget_mean_tokens'] is not None else 'unknown'
        requests=f"{s['stop_at_budget_mean_model_requests']:.2f}" if s['stop_at_budget_mean_model_requests'] is not None else 'unknown'
        dollars=f"{s['stop_at_budget_mean_estimated_usd']:.5f}" if s['stop_at_budget_mean_estimated_usd'] is not None else 'unknown'
        dollars+=' / '+(f"{s['stop_at_budget_mean_uncached_upper_usd']:.5f}" if s['stop_at_budget_mean_uncached_upper_usd'] is not None else 'unknown')
        lines.append(f"| {s['budget']} | 18 | {quality} | {s['material_error_answers']} | {s['execution_failures']} | {s['ungradable_answers']} | {s['retrieval_operations_mean']:.2f} | {requests} | {tokens} | {dollars} |")
    lines.extend(['','### Incremental benefits','',
        '| Extension | Mean quality change, all 18 paired attempts | Newly met essential items, total | Lost essential items, total | New full records, total | Measured additional research tokens |',
        '| --- | --- | ---: | ---: | ---: | ---: |'])
    for before,after in [(8,16),(16,32)]:
        rows=[x for x in increments if x['from_budget']==before]
        known=[x['quality_change'] for x in rows if x['quality_change'] is not None]
        delta=f'{mean(known):+.2f}' if len(known)==18 else f"{mean(x['quality_change_bounds'][0] for x in rows):+.2f} to {mean(x['quality_change_bounds'][1] for x in rows):+.2f} (missing-grade bounds)"
        token_values=[x['incremental_research_usage'].get('total_tokens') for x in rows]
        tokens=str(sum(token_values)) if all(x is not None for x in token_values) else 'unknown; see measured subtotals'
        lines.append(f"| {before}→{after} | {delta} | {sum(len(x['requirements_gained']) for x in rows)} | {sum(len(x['requirements_lost']) for x in rows)} | {sum(len(x['new_full_record_ids']) for x in rows)} | {tokens} |")
    lines.extend(['','### Separate quality dimensions','',
        'These are conditional means among applicable valid assessments, with denominators shown. Missing reports and invalid grading are not assigned perfect accuracy. The primary quality score above retains all 18 attempts per ceiling and penalizes material defects. Claim fractions cover structured claims; prose defects are checked separately.','',
        '| Ceiling | Completeness | Factual correctness | Citation support | Temporal correctness |',
        '| ---: | --- | --- | --- | --- |'])
    for s in aggregated:
        dimensions=[]
        for field in ['completeness','factual_accuracy','evidence_support','temporal_accuracy']:
            value=s[field+'_conditional_mean'];n=s[field+'_applicable_observations']
            dimensions.append(f'{value:.3f} (n={n})' if value is not None else f'undefined (n={n})')
        lines.append('| '+str(s['budget'])+' | '+' | '.join(dimensions)+' |')
    lines.extend(['','Each individual extension, its new source IDs, requirement gains/losses, defect sets, and incremental usage are in `incremental_benefits.json` and `.csv`. Gains do not imply that every extra source improved the answer.','',
        'Requirement gain/loss counts use only available valid item judgments; unknown judgments are not invented. Source and resource totals include all attempted trajectories, including zero additional research after Planner failure.','',
        'Extensions with no added retrieval are explicitly marked: answer changes there cannot be attributed to extra evidence. Repeated synthesis itself is stochastic.','',
        'Observed marginal quality/cost ratios (a business must decide its willingness to pay per quality point):','',
        '```json',json.dumps(marginal,indent=2),'```','',
        '### Paired uncertainty and decision rule','',
        'Comparisons use paired question means, preserving three repetitions within each question. The intervals below are 97.5% empirical question-cluster bootstrap intervals; Bonferroni adjustment covers the two candidate-vs-32 comparisons at an approximate family-wise 95% level. With six question clusters, this is limited evidence, not guaranteed coverage. Missing-grade sensitivity bounds are included.',''])
    for c in contrasts:
        lo,hi=c['cluster_interval_with_missing_sensitivity']
        lines.append(f"- {c['smaller']} minus 32: mean difference {c['mean_difference']}; interval [{lo:.2f}, {hi:.2f}]. Two-point noninferiority margin met: {c['within_predeclared_2_point_margin']}. Observed material-error/execution gate met: {c['observed_error_and_execution_gate']}.")
    lines.extend(['','### Per-question quality','', '| Question | 8 | 16 | 32 |','| --- | ---: | ---: | ---: |'])
    for q in QUESTIONS:
        vals=[]
        for b in BUDGETS:
            rows=[cases[q,r,b] for r in (1,2,3)]
            vals.append(f"{mean(x['quality_low'] for x in rows):.1f}–{mean(x['quality_high'] for x in rows):.1f}")
        lines.append('| '+q+' | '+' | '.join(vals)+' |')
    lines.extend(['','### Total experiment resources','', '```json',json.dumps(usage,indent=2),'```','',
        'Actual study cost includes shadow checkpoint reports and independent evaluation. It differs from the counterfactual cost of answering one question at a selected ceiling. Cache warming and execution order can affect the observed-cache estimate; the all-input-uncached estimate provides a conservative comparison. Unknown failed-request usage is not assumed to be zero. Earlier engineering/calibration attempts outside this study directory are excluded from these totals.','',
        '### Benefits and limitations of the ceilings','',
        '- **8:** least permitted research; useful only when essential evidence is already found. Inspect omitted competitors, unresolved conflicts and unsupported early conclusions.',
        '- **16:** intermediate allowance; evaluate whether the 8→16 gains cover the question and whether 16→32 adds meaningful evidence or merely more tokens.',
        '- **32:** largest tested allowance and most opportunity to resolve gaps. It can add noise, weaker inferences, more cost or Reviewer failures; it is not automatically correct or optimal.',
        '', '### Assignment interpretation','',
        'Report the threshold as a provisional empirical choice for these six synthetic benchmark questions and this frozen continuation policy. Do not claim a proven token limit, universal optimum, or certified material-error probability. Keep retrieval calls, model requests and tokens separate. Follow-up validation should use the held-out questions and the actual production stopping policy.',''])
    report='\n'.join(lines)+'\n';(root/'RESULTS.md').write_text(report)
    if a.readme:
        original=a.readme.read_text();marker='## Results — completed study'
        original=original.split(marker)[0]
        heading='## Question and predeclared design'
        if heading in original:
            original=('# Retrieval threshold evaluation — assignment study\n\n'
                'Study date: 15 November 2026, America/Toronto.\n\n'
                '**Status: all 54 checkpoint observations completed, including recorded failures.** '
                'See the measured results, provisional recommendation and limitations below.\n\n'
                +heading+original.split(heading,1)[1])
        a.readme.write_text(original.rstrip()+'\n\n'+report)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,5))
        for q in QUESTIONS:
            ys=[mean(cases[q,r,b]['quality_low'] for r in (1,2,3)) for b in BUDGETS]
            ax.plot(BUDGETS,ys,marker='o',alpha=.65,label=q)
        average=[s['quality_mean_bounds'][0] for s in aggregated]
        upper=[s['quality_mean_bounds'][1] for s in aggregated]
        ax.fill_between(BUDGETS,average,upper,color='grey',alpha=.22,label='Range from ungradable cases')
        ax.plot(BUDGETS,average,color='black',linewidth=3,marker='o',label='Overall mean lower bound')
        ax.set(xlabel='Retrieval-operation ceiling',ylabel='Material-error-gated essential coverage / 100',ylim=(-3,103),xticks=BUDGETS,
            title='Paired research checkpoints: 6 questions × 3 repetitions')
        ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=7)
        fig.tight_layout();fig.savefig(root/'quality_curve.png',dpi=200);fig.savefig(root/'quality_curve.svg');plt.close(fig)
    except ImportError:print('Plot dependency unavailable; numeric results are complete.')
    print(json.dumps({'recommendation':recommendation,'budget_summary':aggregated,'paired_contrasts':contrasts},indent=2))


if __name__=='__main__':main()
