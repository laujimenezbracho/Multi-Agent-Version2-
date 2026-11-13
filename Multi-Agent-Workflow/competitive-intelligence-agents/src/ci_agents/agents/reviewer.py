"""Terminal Version 1 review: no new evidence, corrections, or retries."""
from hashlib import sha256
from importlib.resources import files
import json

from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig, Runner
from agents.exceptions import ModelBehaviorError
from openai import AsyncOpenAI
from pydantic import ValidationError

from ..telemetry import observe_sdk_result
from ..config import OPENAI_ACCEPT_ENCODING, load_config
from ..schemas import CompetitiveReport, EvidencePack, ResearchPlan, ReviewResult

SCORE_THRESHOLDS = {"completeness_score": 0.85, "evidence_support_score": 0.90,
                    "temporal_consistency_score": 0.90}
ISSUE_TYPES = frozenset({
    "unsupported_factual_claim", "citation_mismatch", "factual_contradiction",
    "temporal_inconsistency", "numerical_or_unit_error", "missing_required_information",
    "unresolved_evidence_conflict", "overstated_certainty", "weak_strategic_inference",
})


class ReviewerError(RuntimeError):
    """Sanitized SDK failure, distinct from a valid FAIL review."""


class ReviewerInputError(ValueError):
    """Invalid or inconsistent input handoffs."""


class ReviewerOutputError(ReviewerError):
    """Invalid review identity, references, structure, or decision policy."""


def report_fingerprint(report: CompetitiveReport) -> str:
    """SHA-256 of canonical JSON; includes every field of the report snapshot."""
    payload = json.dumps(report.model_dump(mode="json"), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _inputs(plan, evidence_pack, report):
    if not isinstance(plan, ResearchPlan) or not isinstance(evidence_pack, EvidencePack) or not isinstance(report, CompetitiveReport):
        raise ReviewerInputError("Expected ResearchPlan, EvidencePack, and CompetitiveReport")
    try:
        plan = ResearchPlan.model_validate(plan.model_dump())
        evidence_pack = EvidencePack.model_validate(evidence_pack.model_dump())
        report = CompetitiveReport.model_validate(report.model_dump())
    except ValidationError:
        raise ReviewerInputError("Invalid structured handoff, duplicate IDs, or post-cutoff evidence") from None
    if evidence_pack.original_question != plan.original_question or report.question != plan.original_question:
        raise ReviewerInputError("Plan, evidence, and report must refer to the same original question")
    if evidence_pack.as_of != plan.as_of or report.as_of != plan.as_of:
        raise ReviewerInputError("Plan, evidence, and report cutoffs must match")
    # Do not reject unknown report citations here: these are potential review defects.
    # Schema-invalid inputs cannot be reviewed, but schema-valid incomplete reports can.
    return plan, evidence_pack, report


def _validate_review(output, plan, evidence_pack, report, fingerprint):
    if not isinstance(output, ReviewResult):
        raise ReviewerOutputError("Reviewer did not return a structured ReviewResult")
    try:
        review = ReviewResult.model_validate(output.model_dump())
    except ValidationError:
        raise ReviewerOutputError("Invalid ReviewResult: check scores, severity, decision, and duplicate issue IDs") from None
    if review.reviewed_question != plan.original_question or review.reviewed_report_hash != fingerprint:
        raise ReviewerOutputError("Review question or report fingerprint does not match the input")
    claims = {claim.claim_id for claim in report.claims}
    records = {item.record_id for item in evidence_pack.evidence_items}
    signatures = set()
    for issue in review.flagged_issues:
        if issue.issue_type not in ISSUE_TYPES:
            raise ReviewerOutputError(f"Invalid issue_type: {issue.issue_type}")
        if not set(issue.affected_claim_ids).issubset(claims):
            raise ReviewerOutputError(f"Issue {issue.issue_id} references an unknown report claim ID")
        for field in ("related_record_ids", "missing_record_ids"):
            refs = getattr(issue, field)
            if len(set(refs)) != len(refs) or not set(refs).issubset(records):
                raise ReviewerOutputError(f"Issue {issue.issue_id} has duplicate or unknown evidence IDs in {field}")
        if len(set(issue.affected_claim_ids)) != len(issue.affected_claim_ids):
            raise ReviewerOutputError(f"Issue {issue.issue_id} repeats affected claim IDs")
        signature = (issue.issue_type, tuple(sorted(issue.affected_claim_ids)),
                     tuple(sorted(issue.related_record_ids)), tuple(sorted(issue.missing_record_ids)),
                     " ".join(issue.explanation.casefold().split()))
        if signature in signatures:
            raise ReviewerOutputError("Duplicate review defects with distinct issue IDs")
        signatures.add(signature)
    material = any(issue.severity in {"critical", "major"} for issue in review.flagged_issues)
    low_scores = any(getattr(review, name) < threshold for name, threshold in SCORE_THRESHOLDS.items())
    expected = "FAIL" if material or review.missing_requirements or low_scores else "PASS"
    if review.decision != expected:
        raise ReviewerOutputError(f"Review decision violates policy: expected {expected}")
    if low_scores and not (material or review.missing_requirements or review.reviewer_notes):
        raise ReviewerOutputError("Below-threshold scores require an explicit explanation")
    return review


async def reviewer(plan: ResearchPlan, evidence_pack: EvidencePack,
                   report: CompetitiveReport) -> ReviewResult:
    """Review exactly once. Valid FAIL returns normally without modifying inputs.

    The caller will later persist outputs and stop; this function does not orchestrate
    any other stage or act as an independent ground-truth evaluator.
    """
    plan, evidence_pack, report = _inputs(plan, evidence_pack, report)
    fingerprint = report_fingerprint(report)
    settings = load_config()
    instructions = files("ci_agents").joinpath("prompts/reviewer.md").read_text(encoding="utf-8")
    context = json.dumps({"research_plan": plan.model_dump(mode="json"),
                          "evidence_pack": evidence_pack.model_dump(mode="json"),
                          "report": report.model_dump(mode="json"),
                          "report_fingerprint": fingerprint,
                          "review_policy": {"minimum_scores": SCORE_THRESHOLDS,
                                            "issue_types": sorted(ISSUE_TYPES)}})
    async with AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0,
                       default_headers={"Accept-Encoding": OPENAI_ACCEPT_ENCODING}) as client:
        agent = Agent(name="Reviewer", instructions=instructions,
                      model=OpenAIResponsesModel(model=settings.openai_model, openai_client=client),
                      model_settings=ModelSettings(), output_type=ReviewResult, tools=[], handoffs=[])
        try:
            result = await Runner.run(agent, input=context, max_turns=1,
                                      run_config=RunConfig(tracing_disabled=True))
            observe_sdk_result(result)
        except (ModelBehaviorError, ValidationError):
            raise ReviewerOutputError("SDK could not validate the structured ReviewResult") from None
        except Exception:
            raise ReviewerError("Reviewer SDK execution failed; no retry was attempted") from None
    return _validate_review(result.final_output, plan, evidence_pack, report, fingerprint)
