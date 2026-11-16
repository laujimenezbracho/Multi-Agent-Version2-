"""Evaluator-only source-audited keys for the agreed six-question assignment."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

QUESTIONS=('Q01','Q03','Q05','Q07','Q12','Q15')
CUTOFF='2026-11-11T23:59:59Z'


def prepare(root, destination):
    root,destination=Path(root),Path(destination)
    if 'evaluation_private' not in destination.parts:
        raise ValueError('Keys belong outside the agent runtime under evaluation_private')
    rows=[json.loads(x) for x in (root/'data/public/public_web_records.jsonl').read_text().splitlines()]
    source={r['record_id']:r for r in rows}
    previous=root/'evaluation_private/budget_screen_v2'
    q01=json.loads((previous/'Q01.json').read_text())
    q15=json.loads((previous/'Q15.json').read_text())
    keys={'Q01':deepcopy(q01),'Q15':deepcopy(q15)}
    keys['Q03']={
        'essential_elements':[
            'Identify the five published November Enterprise offers: NexaFlow 18%, OrbitDesk 10%, VertexOps 20%, PulseGrid 15%, ClearPilot 25%; qualify these as source-stated offers, not independently verified redemption.',
            'Apply availability/effective/expiry at the cutoff: November 2 through November 30, 2026; do not count expired Starter/Growth or December 2025 offers as active.',
            'Explain applicable Enterprise tier, eligible new accounts, first twelve months, five-seat minimum, annual billing, no stacking, implementation/AI-overage exclusions and subsequent prevailing renewal price; codes are useful but not a separate mandatory element.',
            'Reject the uncorroborated NexaFlow 50% Growth screenshot using the official November 6 denial; distinguish published eligibility from guaranteed availability and note source uncertainty.'],
        'reference_answer':'Five source-stated Enterprise offers are in their published eligibility window on November 11: NexaFlow 18% NEX2611, OrbitDesk 10% ORB2611, VertexOps 20% VER2611, PulseGrid 15% PUL2611, ClearPilot 25% CLE2611. Each runs November 2–30 for eligible new accounts, first twelve months, minimum five seats, annual billing, no stacking, excludes implementation/AI overages. Source metadata is unverified, so do not guarantee checkout redemption. Older Starter/Growth offers expired; the alleged 50% Growth offer is unauthorized.',
        'critical_traps':['Enterprise discounts are not Growth discounts.','A future end_date alone does not authenticate a promotion.','The available-at date must be at or before the cutoff.']}
    keys['Q05']={
        'essential_elements':[
            'Answer that no verified customer-data breach is established by the supplied evidence at the cutoff; do not assert that a breach is impossible or never happened.',
            'Resolve the September OrbitDesk allegation with the official no-evidence-of-exfiltration statement and independent report of disruption rather than confirmed breach (WEB-30001/2/3).',
            'Cover the other vendors: their August anonymous configuration/security allegations have company responses reporting no evidence of unauthorized access; security-feature announcements and SOC2 claims are not proof of a breach.',
            'Explain source reliability and time-bounded investigations; distinguish outage, configuration problem, allegation and confirmed breach, with explicit uncertainty.'],
        'reference_answer':'No verified customer-data breach is established in the available corpus. All five vendors have August anonymous allegations and subsequent vendor responses saying no evidence of unauthorized access. OrbitDesk also has a September unsupported exfiltration allegation, a vendor investigation update, and independent coverage confirming disruption without verifying a breach. Vendor denials are time-bounded statements, not proof of absence.',
        'critical_traps':['Do not equate service disruption with data exfiltration.','Do not treat anonymous allegations or the verification_status field alone as proof.']}
    keys['Q07']={
        'essential_elements':[
            'Calculate comparable annual-commitment Growth subscription baselines for fifty seats across five vendors: PulseGrid 6500/month or 78000/year; OrbitDesk 7400/88800; ClearPilot 7600/91200; VertexOps 7950/95400; NexaFlow 8100/97200 USD. Keep monthly-equivalent versus annual payment units clear.',
            'Use current applicable pricing including VertexOps official 159, not stale 174 or unverified 149; do not apply expired Growth, Enterprise-only or unauthorized 50% discounts.',
            'Assess value-relevant feature/AI entitlement limits and unknown extra costs: PulseGrid AI Summaries GA with a Growth quota; ClearPilot predictive scoring is an optional Growth add-on. Unknown add-on/onboarding/overage or negotiated costs are not zero.',
            'Give a qualified conclusion: PulseGrid is cheapest on the comparable published subscription baseline, while definitive best value or total buyer-paid cost depends on requirements and missing fees/entitlements. State assumptions and actionable information needed.'],
        'reference_answer':'For 50 seats at published annual-commitment Growth rates, PulseGrid 6500 monthly-equivalent/78000 annually is cheapest; OrbitDesk 7400/88800, ClearPilot 7600/91200, VertexOps 7950/95400, NexaFlow 8100/97200 USD. This does not establish best value or final invoice: feature requirements, quotas, AI add-ons, onboarding, overages and negotiated terms are incomplete. Enterprise promotions do not apply to Growth; prior Growth promotions are expired. Do not invent a free AI entitlement.',
        'reference_prices':deepcopy(q01['reference_prices']),
        'critical_traps':['Fifty times monthly-equivalent price is not the annual invoice.','Cheapest baseline is not a definitive best-value ranking.']}
    keys['Q12']={
        'essential_elements':[
            'Explain announced/roadmap versus private preview/beta versus GA; availability must be assessed at the cutoff, for the relevant feature, tier and eligibility, rather than a universal vendor status.',
            'NexaFlow Agent Builder remains invite-only private preview (WEB-30004/5); VertexOps autonomous controls are a 2027 roadmap objective (WEB-30018), neither GA.',
            'PulseGrid AI Summaries moved from opt-in beta (WEB-30006) to GA for Growth/Enterprise October 5 (WEB-30007), with quotas/tier limits; do not retain the outdated beta-only conclusion.',
            'ClearPilot predictive scoring is GA, Enterprise included and optional Growth add-on (WEB-30011); explanation cards are Enterprise GA (WEB-30019). Generic AI previews, product announcements or missing OrbitDesk release evidence do not prove broader GA or free entitlement.'],
        'reference_answer':'Availability is feature- and tier-specific. NexaFlow Agent Builder is still private preview; VertexOps autonomous controls remain a 2027 roadmap. PulseGrid AI Summaries became GA October 5 for Growth/Enterprise with usage limits, superseding beta-only descriptions. ClearPilot predictive scoring is GA but Enterprise included/Growth paid add-on, and explanation cards are Enterprise GA. Generic preview announcements alone do not establish GA.',
        'critical_traps':['Confirmed announcement does not mean generally available product.','A beta record can be outdated even without an explicit supersedes pointer.']}
    keys['Q15']['essential_elements'][1]='Explain ranking criteria and support each threat with actual comparative evidence. Relative pricing claims require PulseGrid baseline comparison; no unsupported advantage from VertexOps 159 versus PulseGrid 130, superseded aggregator 149 or denial of a public discount. Do not manufacture three strong threats when only tentative ones are supported.'
    keys['Q15']['critical_traps']=[
        'An official 159 price does not undercut PulseGrid 130; price-pressure as a leading threat requires additional evidence, not merely any competitor price.',
        'Denial of an unauthorized 50% discount and a stale aggregator listing do not demonstrate negotiated undercutting.',
        'Outage evidence does not demonstrate actual competitor wins or measured churn; a clearly conditional scenario is allowed, with lower confidence.',
        'Three paraphrases of the same pricing concern are not three distinct supported threats.']
    # The strict rank-arithmetic validator uses the same field across question types.
    keys['Q15']['reference_prices']=deepcopy(q01['reference_prices'])
    price_ids={x['record_id'] for x in q01['reference_prices'].values()}
    trap_ids={f'WEB-{i}' for i in range(30001,30021)}
    for q,k in keys.items():
        refs=price_ids|trap_ids
        if q in {'Q03','Q07'}:refs|={r['record_id'] for r in rows if r['event_category']=='Promotion'}
        if q=='Q05':refs|={r['record_id'] for r in rows if r['event_category']=='Security'}
        if q=='Q12':refs|={r['record_id'] for r in rows if r['event_category'] in {'AI investment','Product launch'}}
        if q=='Q15':
            for company in sorted({r['company_name'] for r in rows}):
                for category in ['Partnership','Customer win','Market expansion','Financial performance','Hiring']:
                    candidates=[r for r in rows if r['company_name']==company and r['event_category']==category and r['available_at']<=CUTOFF]
                    if candidates:refs.add(max(candidates,key=lambda r:r['available_at'])['record_id'])
        k.update(question_id=q,cutoff=CUTOFF,rubric_version='paired-checkpoint-source-audit-v1',
            provenance='Evaluator-only provisional key audited against the supplied synthetic corpus. Existing Q01/Q15 rubrics reused and corrected for strategic-inference over-credit. Not owner-certified ground truth.',
            essential_requires_evidence=[True,True,True,False],
            audited_reference_sources=[{field:source[rid].get(field) for field in ['record_id','company_name','document_title','summary','details','claim_text','source_type','available_at','effective_date','end_date','feature_status','verification_status','plan','promotion_code','promotion_discount_pct','billing_frequency','price_basis','supersedes_record_id']} for rid in sorted(refs)])
        # Q12's final element contains substantive availability facts, not merely gaps.
        if q=='Q12':k['essential_requires_evidence']=[True]*4
    # Assertions prevent silent drift of the reference calculations and active offers.
    assert source['WEB-30009']['metric_value']=='159'
    offers=[r for r in rows if r['event_category']=='Promotion' and r.get('end_date') and r['effective_date']<='2026-11-11'<=r['end_date'] and r['available_at']<=CUTOFF and r['record_id']!='WEB-30014']
    assert len(offers)==5 and all(r['plan']=='Enterprise' for r in offers)
    destination.mkdir(parents=True,exist_ok=False)
    for q in QUESTIONS:(destination/(q+'.json')).write_text(json.dumps(keys[q],indent=2,ensure_ascii=False)+'\n')
    manifest={'questions':QUESTIONS,'cutoff':CUTOFF,'source_sha256':hashlib.sha256((root/'data/public/public_web_records.jsonl').read_bytes()).hexdigest(),
        'reused_keys':[str(previous/'Q01.json'),str(previous/'Q15.json')],
        'rubric_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.glob('Q*.json'))}}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.output_dir),indent=2))
