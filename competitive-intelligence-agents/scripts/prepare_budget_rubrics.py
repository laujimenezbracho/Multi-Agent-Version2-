"""Write provisional evaluator-only Q01/Q15 rubrics, never agent runtime input."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(data_dir, output_dir):
    output_dir = Path(output_dir)
    if 'evaluation_private' not in output_dir.parts:
        raise ValueError('Rubrics must be stored under evaluation_private')
    rows = [json.loads(line) for line in (Path(data_dir)/'public_web_records.jsonl').read_text().splitlines() if line.strip()]
    sources = {r['record_id']: r for r in rows}
    prices = {'PulseGrid': (130, 'WEB-00195'), 'OrbitDesk': (148, 'WEB-00089'),
        'ClearPilot': (152, 'WEB-00248'), 'VertexOps': (159, 'WEB-30009'), 'NexaFlow': (162, 'WEB-00036')}
    for company, (price, rid) in prices.items():
        row = sources[rid]
        if row['company_name'] != company or str(price) not in (row['details']+' '+row['claim_text']):
            raise ValueError('Source snapshot differs from audited provisional pricing rubric')
    common = {'provenance': 'Provisional source-audited rubric derived from supplied synthetic public records; not the benchmark owner’s official private answer key.',
        'cutoff': '2026-11-11T23:59:59Z', 'rubric_version': 'budget-screen-provisional-v2',
        'essential_requires_evidence':[True,True,True,False],
        'measures': {
            'factual_accuracy': 'Correct factual assertions / all checkable factual assertions, including prose outside structured claims; distinguish a source statement from established fact.',
            'completeness': 'Met essential answer elements / predeclared essential elements; explain partial credit and missing elements.',
            'evidence_support': 'Factual assertions actually supported by cited ORIGINAL public records / all factual assertions. Existing record IDs alone are not support.',
            'temporal_accuracy': 'Correct date-dependent conclusions / all date-dependent conclusions. Distinguish availability, publication, effective dates and expiry.',
            'material_errors': 'Count decision-changing errors, including wrong cheapest price, invalid price basis, unsubstantiated breach/churn, or roadmap/preview described as GA.',
            'efficiency': 'Measured separately from execution artifacts: actual tokens, requests, retrieval attempts/denials, wall time and documented estimated USD. Never inferred from answer length.'},
        'grading_rules': ['No private rubric or expected answer is passed to any of the four agents.',
            'No report means not_gradable, not a perfect abstention; the run still counts as a failure in the six-run screen.',
            'Use supplied original records and explicit reference facts. Novel claims beyond reference facts must still have real citation support.',
            'Do not require every controlled trap record unless relevant to an assertion or essential question element.',
            'Internal Reviewer decisions are excluded from the independent judge input. Judge feedback remains fallible and is available for human review.',
            'Do not claim single-run budget differences establish causality or an optimal limit.']}
    q01 = {**common, 'question_id': 'Q01',
        'essential_elements': ['Directly identify the lowest comparable current Growth annual-commitment list price at the cutoff.',
            'Compare all five companies on USD per seat per month with annual commitment; do not substitute Starter or a discount screenshot.',
            'Use the latest applicable source history, including VertexOps official clarification rather than stale third-party listing.',
            'Keep list price separate from promotions, taxes, onboarding, seat minima and undisclosed AI/add-on costs; unknown costs remain unknown.'],
        'reference_answer': 'PulseGrid has the lowest comparable Growth list price: USD 130 per seat per month, billed annually. Other comparable prices: OrbitDesk 148, ClearPilot 152, VertexOps 159, NexaFlow 162. This does not establish the lowest all-in cost for every contract.',
        'reference_prices': {company: {'price_usd_per_seat_month': price, 'record_id': rid} for company, (price, rid) in prices.items()},
        'critical_traps': ['VertexOps WEB-30008 USD149 is superseded by WEB-30009 USD159; do not use the older USD174 as current.',
            'NexaFlow 50% screenshot WEB-30014 is not a verified active offer; WEB-30015 denies a public authorized offer.',
            'Do not count expired Growth promotions or impute unknown onboarding or AI charges as zero.']}
    q15 = {**common, 'question_id': 'Q15',
        'essential_elements': ['Give three prioritized competitive threats to PulseGrid, identifying named rivals or clearly defined competitive pressures.',
            'Explain explicit ranking criteria and connect each threat to actual cited comparative evidence, rather than providing an unsupported ranking.',
            'Explain why each threat matters to PulseGrid and give evidence-bounded strategic implications.',
            'Disclose material evidence gaps and uncertainty; distinguish previews, pilots and GA and avoid turning sentiment into audited metrics.'],
        'reference_answer': 'There is no single canonical top-three ranking. Accept alternative rankings when three coherent threats, their relative priority, competitor comparisons and implications are supported by available evidence. Do not force a particular vendor order.',
        'conditional_reference_facts': ['PulseGrid AI Summaries is GA for Growth/Enterprise from 2026-10-05 (WEB-30007), superseding beta-only descriptions (WEB-30006).',
            'NexaFlow Agent Builder remains private preview (WEB-30004/30005). VertexOps autonomous controls are a 2027 roadmap, not GA (WEB-30018).',
            'ClearPilot predictive scoring is included in Enterprise but an optional Growth add-on (WEB-30011); explanation cards are Enterprise GA (WEB-30019).',
            'PulseGrid forum 18% churn is unverified (WEB-30012); no verified annual logo-churn is published (WEB-30013).',
            'PulseGrid Nov7 file-sync incident resolved in 47 minutes, no data loss reported (WEB-30020).',
            'OrbitDesk alleged exfiltration is unsubstantiated; disruption is supported but breach is not confirmed (WEB-30001/30002/30003).',
            'Use current comparable pricing if invoked, not outdated list prices or unsupported discounts.'],
        'reference_prices_if_used': q01['reference_prices']}
    ref_ids = set(rid for _, rid in prices.values()) | {f'WEB-{i}' for i in range(30001,30021)}
    for q in (q01,q15):
        q['audited_reference_sources'] = [sources[rid] for rid in sorted(ref_ids)]
    output_dir.mkdir(parents=True, exist_ok=False)
    for q in (q01,q15):
        (output_dir/(q['question_id']+'.json')).write_text(json.dumps(q, ensure_ascii=False, indent=2)+'\n')
    manifest = {'kind': 'provisional_not_official', 'source_file_sha256': hashlib.sha256((Path(data_dir)/'public_web_records.jsonl').read_bytes()).hexdigest(),
        'rubric_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output_dir.glob('Q*.json'))}}
    (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/public'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_dir,args.output_dir), indent=2))
