"""Deterministic V2 routing/context construction. No model calls or evaluator inputs."""
from dataclasses import dataclass
import json

from .schemas import EvidencePack, ResearchPlan, ResearchTask

POLICY_VERSION = "targeted_recovery.v1"
UNACTIONABLE_CORRECTIONS = {"n/a", "none", "no actionable correction", "manual review required", "human review required"}
REVISION_TYPES = {"unsupported_factual_claim", "citation_mismatch", "factual_contradiction",
                  "temporal_inconsistency", "numerical_or_unit_error", "unresolved_evidence_conflict",
                  "overstated_certainty", "weak_strategic_inference"}


@dataclass(frozen=True)
class RecoveryAction:
    route: str
    issue_ids: tuple[str, ...]
    research_issue_ids: tuple[str, ...]
    ignored_issue_ids: tuple[str, ...]
    reason: str

    def as_dict(self):
        return {"policy_version": POLICY_VERSION, "max_cycles": 1, "route": self.route,
                "trigger_issue_ids": list(self.issue_ids), "research_issue_ids": list(self.research_issue_ids),
                "ignored_issue_ids": list(self.ignored_issue_ids), "reason": self.reason}


def select_action(review):
    if review.decision == "PASS":
        return RecoveryAction("skip", (), (), (), "initial_review_passed")
    actionable, research, ignored = [], [], []
    for issue in review.flagged_issues:
        correction = issue.suggested_correction.strip()
        if issue.severity not in {"major", "critical"} or not correction or correction.casefold() in UNACTIONABLE_CORRECTIONS:
            ignored.append(issue.issue_id)
            continue
        if issue.issue_type == "missing_required_information":
            actionable.append(issue.issue_id)
            # missing_record_ids are known retrieved records omitted from the report.
            # Such omissions can be addressed without additional retrieval.
            if not issue.missing_record_ids:
                research.append(issue.issue_id)
        elif issue.issue_type in REVISION_TYPES:
            actionable.append(issue.issue_id)
        else:
            ignored.append(issue.issue_id)
    route = "research_then_revise" if research else "revise_only" if actionable else "skip"
    return RecoveryAction(route, tuple(actionable), tuple(research), tuple(ignored),
                          "structured_material_issues" if actionable else "no_actionable_material_issues")


def targeted_research_plan(plan, evidence, report, review, action):
    claims = {claim.claim_id: claim for claim in report.claims}
    sources = {item.record_id: item for item in evidence.evidence_items}
    all_companies = sorted({company for task in plan.tasks for company in task.companies})
    categories = sorted({category for task in plan.tasks for category in task.categories})
    tasks = []
    for issue in review.flagged_issues:
        if issue.issue_id not in action.research_issue_ids:
            continue
        ids = set(issue.related_record_ids)
        for cid in issue.affected_claim_ids:
            if cid in claims:
                ids.update(claims[cid].supporting_record_ids)
        companies = sorted({sources[rid].company_name for rid in ids if rid in sources}) or all_companies
        tasks.append(ResearchTask(task_id="REC-" + issue.issue_id,
            research_question=issue.explanation + " Requested correction: " + issue.suggested_correction,
            companies=companies, categories=categories,
            time_scope="Available on or before " + plan.as_of.isoformat() + "; retain effective/expiry distinctions",
            evidence_requirements=[issue.suggested_correction,
                "Use only permitted public records. If unavailable, explicitly record the remaining gap; do not fetch URLs or obtain outside quotes."]))
    return ResearchPlan(original_question=plan.original_question, as_of=plan.as_of,
        objective="One targeted evidence correction cycle for Reviewer issues: " + ", ".join(action.research_issue_ids),
        tasks=tasks, expected_answer_elements=["Evidence or explicit unresolved gaps for the targeted issues"],
        comparison_dimensions=list(plan.comparison_dimensions))


def revision_plan(plan, initial_report, review, action):
    revised = plan.model_copy(deep=True)
    feedback = {"initial_report": initial_report.model_dump(mode="json"),
                "reviewer_issues": [issue.model_dump(mode="json") for issue in review.flagged_issues if issue.issue_id in action.issue_ids],
                "reviewer_missing_requirements": list(review.missing_requirements)}
    revised.objective += ("\nV2 TARGETED REVISION CONTEXT (feedback, not additional source evidence): "
        "Revise the supplied initial report only as needed to address the selected issues. "
        "Preserve IDs for unchanged claims where possible. Use only the supplied EvidencePack for factual support. "
        "Do not invent evidence or treat feedback as ground truth. Preserve unresolved gaps and uncertainty. "
        + json.dumps(feedback, ensure_ascii=False))
    return revised


def merge_evidence(initial, additional):
    if initial.original_question != additional.original_question or initial.as_of != additional.as_of:
        raise ValueError("Recovery evidence changed the original question or cutoff")
    merged = initial.model_copy(deep=True)
    by_id = {item.record_id: item for item in merged.evidence_items}
    metadata = ("claim_id", "company_name", "source_title", "source_type", "source_url", "published_at", "available_at",
                "effective_date", "end_date", "source_independence_group", "supersedes_record_id", "verification_status", "feature_status")
    for item in additional.evidence_items:
        if item.record_id not in by_id:
            copied = item.model_copy(deep=True)
            merged.evidence_items.append(copied)
            by_id[item.record_id] = copied
        else:
            existing = by_id[item.record_id]
            if any(getattr(existing, field) != getattr(item, field) for field in metadata):
                raise ValueError("Duplicate recovery record has inconsistent source metadata")
            for field in ("finding", "quoted_or_paraphrased_evidence"):
                if getattr(existing, field) != getattr(item, field):
                    setattr(existing, field, getattr(existing, field) + "\nAdditional retrieved finding: " + getattr(item, field))
            existing.limitations = list(dict.fromkeys(existing.limitations + item.limitations))
            existing.confidence = min(existing.confidence, item.confidence)
    merged.missing_information = (["Initial gap (may be resolved by supplemental evidence): " + gap for gap in initial.missing_information]
                                  + ["Recovery gap: " + gap for gap in additional.missing_information])
    merged.contradictions = list(dict.fromkeys(initial.contradictions + additional.contradictions))
    merged.retrieval_log = initial.retrieval_log + additional.retrieval_log
    for coverage in merged.coverage_by_task:
        coverage.coverage_score = None
        coverage.notes.append("Original task coverage retained; supplemental evidence has not been independently rescored.")
    return EvidencePack.model_validate(merged.model_dump())


def claim_changes(initial, revised):
    old = {claim.claim_id: claim.model_dump(mode="json") for claim in initial.claims}
    new = {claim.claim_id: claim.model_dump(mode="json") for claim in revised.claims}
    return {"added": [new[cid] for cid in new if cid not in old],
            "removed": [old[cid] for cid in old if cid not in new],
            "modified": [{"claim_id": cid, "before": old[cid], "after": new[cid]} for cid in new if cid in old and new[cid] != old[cid]],
            "unchanged_claim_ids": [cid for cid in new if cid in old and new[cid] == old[cid]],
            "matching_rule": "Exact stable claim ID; no inferred semantic alignment"}
