"""Six public-input V1 executions. No private rubric is accepted by this module."""
from collections import Counter
import json
from pathlib import Path

from .agents.researcher import ResearcherLimits
from .persistence import fingerprint


BUDGETS = (16, 32, 60)
QUESTIONS = ('Q01', 'Q15')


def experiment_limits(budget):
    if budget not in BUDGETS:
        raise ValueError('Screen accepts only the predeclared budgets 16, 32 and 60')
    # Freeze the SDK-turn cap instead of changing it with the retrieval treatment.
    return ResearcherLimits(max_tool_calls=budget, max_search_results=10,
        max_full_records=40, max_evidence_items=40, max_model_turns=62)


def stop_details(result, directory):
    root = Path(directory)
    events = json.loads((root/'retrieval_logs.json').read_text()) if (root/'retrieval_logs.json').exists() else []
    outcomes=[event for event in events if event.get('kind')=='research_outcome']
    events=[event for event in events if event.get('kind')!='research_outcome']
    denied = [e for e in events if e.get('status') == 'budget_exhausted']
    error = json.loads((root/'error.json').read_text()) if (root/'error.json').exists() else None
    schema_error = error and (error.get('error_type') in {'EvidenceValidationError', 'AnalystOutputError',
        'ReviewerOutputError', 'PlannerOutputError', 'ValidationError', 'ModelBehaviorError'} or
        error.get('diagnostic', {}).get('category') == 'schema_validation')
    exhausted = bool(denied)
    reason = 'invalid_output' if schema_error else 'budget_exhausted' if exhausted else 'execution_failure' if error else 'completed_research'
    # Budget exhaustion and a usable report can coexist. Preserve both, never
    # silently drop a failed run from the comparison denominator.
    return {'stop_reason': reason, 'budget_exhausted': exhausted,
        'research_completion':outcomes[-1] if outcomes else None,
        'exhausted_budgets': sorted({e.get('budget', 'unknown') for e in denied}),
        'denied_tool_attempts': len(denied), 'retrieval_attempts': len(events),
        'allowed_tool_operations': sum(e.get('status') != 'budget_exhausted' for e in events),
        'tool_counts': dict(Counter(e.get('tool', 'unknown') for e in events)),
        'full_record_ids': sorted({rid for e in events if e.get('tool') == 'get_record' and e.get('status') == 'ok' for rid in e.get('record_ids', [])}),
        'error': error}


def public_prediction(result, dataset):
    report = result.final_report.model_dump(mode='json') if result.final_report else None
    ids = set(report['sources']) if report else set()
    if report:
        ids.update(rid for claim in report['claims'] for rid in claim['supporting_record_ids'])
    # These are ORIGINAL public records, not the Researcher's generated paraphrases.
    from pydantic import TypeAdapter
    sources = TypeAdapter(object).dump_python([r for r in dataset.records if r['record_id'] in ids], mode='json')
    # Do not leak Reviewer PASS/FAIL indirectly through completed_review_failed.
    return {'execution_status': 'report_available' if report is not None else 'no_report', 'report': report, 'original_cited_sources': sources}


def validate_checkpoint(record, binding):
    if record['binding'] != binding:
        raise ValueError('Checkpoint settings changed; use a new experiment directory')
    if record['prediction_sha256'] != fingerprint(record['prediction']):
        raise ValueError('Prediction checkpoint hash mismatch')
    return record['prediction']
