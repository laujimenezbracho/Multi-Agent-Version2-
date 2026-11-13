"""Versioned evaluator-only contracts. Agents must not import or consume results."""
from typing import Literal
from pydantic import Field
from .schemas import Contract, NonNegative, Score, Text, TokenCount, UTCTimestamp


class ClaimErrorAnnotation(Contract):
    claim_id: Text
    error_category: Text
    explanation: Text
    ground_truth_reference: Text | None = None
    matched_reviewer_issue_ids: list[Text] = Field(default_factory=list)


class IndependentEvaluation(Contract):
    schema_version: Literal["independent_evaluation.v1"] = "independent_evaluation.v1"
    run_id: Text
    evaluated_at: UTCTimestamp | None = None
    evaluator_version: Text | None = None
    ground_truth_version: Text | None = None
    factual_accuracy: Score | None = None
    answer_completeness: Score | None = None
    citation_support_rate: Score | None = None
    unsupported_claim_rate: Score | None = None
    temporal_accuracy: Score | None = None
    contradiction_error_count: TokenCount | None = None
    numerical_error_count: TokenCount | None = None
    missing_essential_answer_elements: list[Text] | None = None
    independent_error_annotations: list[ClaimErrorAnnotation] | None = None
    reviewer_error_detection_precision: Score | None = None
    reviewer_error_detection_recall: Score | None = None
    reviewer_false_positives: TokenCount | None = None
    error_correction_rate: Score | None = None
    new_error_introduction_rate: Score | None = None
    quality_improvement_per_additional_cost: float | None = Field(default=None, allow_inf_nan=False)


class BaselineReference(Contract):
    schema_version: Literal["baseline_reference.v1"] = "baseline_reference.v1"
    baseline_run_id: Text
    manifest_sha256: Text
    artifact_fingerprints: dict[str, str]
    question: Text
    question_id: Text | None
    as_of: UTCTimestamp
    comparison_key: Text


class PairedComparison(Contract):
    """Future evaluator input; no recovery execution or metric calculation."""
    schema_version: Literal["paired_comparison.v1"] = "paired_comparison.v1"
    baseline: BaselineReference
    recovered_run_id: Text | None = None
    recovered_report_fingerprint: Text | None = None
    recovered_review_fingerprint: Text | None = None
    incremental_input_tokens: TokenCount | None = None
    incremental_output_tokens: TokenCount | None = None
    incremental_latency_seconds: NonNegative | None = None
    incremental_estimated_cost_usd: NonNegative | None = None
