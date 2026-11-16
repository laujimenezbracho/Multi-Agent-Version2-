"""Offline execution comparison; no quality judging or model calls."""
import argparse
import json
from pathlib import Path


def cases(root):
    result={}
    for path in sorted((root/'cases').glob('*/prediction.json')):
        value=json.loads(path.read_text())
        summary=value['summary']
        report=value['prediction'].get('report')
        result[(summary['question_id'],summary['retrieval_budget'])]={
            'execution_status':summary['execution_status'],
            'stop_reason':summary['stop_reason'],
            'report_available':report is not None,
            'structured_claims':len(report['claims']) if report else None,
            'cited_records':len(report['sources']) if report else None,
            'retrieval_attempts':summary['retrieval_attempts'],
            'model_requests':summary['model_requests'],
            'estimated_agent_cost_usd':summary['estimated_cost_usd'],
            'internal_reviewer_decision':summary.get('internal_reviewer_decision')}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    args=parser.parse_args()
    before,after=cases(args.before),cases(args.after)
    if len(before)!=6 or set(before)!=set(after):
        raise ValueError('Expected the same six completed condition checkpoints')
    rows=[{'question_id':q,'retrieval_budget':b,'before':before[q,b],'after':after[q,b]}
          for q,b in sorted(before)]
    (args.after/'before_after.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Before/after execution comparison','',
        'Each version has one run per condition. These are execution observations, not quality scores. '
        'The workflow changed between versions; differences cannot be attributed solely to retrieval budget. '
        'Raw answers and provisional independent judgments require human review.','',
        '| Case | Previous outcome | Corrected outcome | Previous → corrected retrievals | Previous → corrected cited records |',
        '| --- | --- | --- | ---: | ---: |']
    for row in rows:
        old,new=row['before'],row['after']
        lines.append(f"| {row['question_id']} / {row['retrieval_budget']} | {old['execution_status']} | "
            f"{new['execution_status']} | {old['retrieval_attempts']} → {new['retrieval_attempts']} | "
            f"{old['cited_records']} → {new['cited_records']} |")
    (args.after/'before_after.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
