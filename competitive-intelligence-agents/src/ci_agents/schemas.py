"""Pydantic v2 handoff contracts; no agent behavior or network operations.

Agent-facing objects avoid arbitrary dictionaries. Nullable evidence fields are
required in generated output: unknown values must be emitted explicitly as null.
Cross-artifact citation checks against retrieved evidence belong to the controller.
"""
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator,field_validator

Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
TokenCount = Annotated[int, Field(ge=0, strict=True)]


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


UTCTimestamp = Annotated[AwareDatetime, AfterValidator(_utc)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _unique(values: list[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {label} are not allowed")


class ResearchTask(Contract):
    task_id: Text
    research_question: Text
    companies: list[Text] = Field(default_factory=list)
    categories: list[Text] = Field(default_factory=list)
    time_scope: Text = Field(description="Explicit natural-language time period and relevant date semantics")
    evidence_requirements: list[Text] = Field(default_factory=list)
    priority: Literal['essential', 'optional'] = 'essential'


class ResearchPlan(Contract):
    original_question: Text
    as_of: UTCTimestamp
    objective: Text
    tasks: list[ResearchTask] = Field(min_length=1)
    expected_answer_elements: list[Text] = Field(default_factory=list)
    comparison_dimensions: list[Text] = Field(default_factory=list)
    optional_context: list[Text] = Field(default_factory=list,
        description='Useful enrichment that must not become a requirement for answering the original question.')

    @model_validator(mode="after")
    def unique_tasks(self):
        _unique([task.task_id for task in self.tasks], "task IDs")
        return self


class PricingObservation(Contract):
    """Verbatim structured source assertions; never a canonical current price."""
    plan: Text | None = None
    amount: Annotated[str, Field(pattern=r"^\d+(?:\.\d+)?$")] | None = None
    currency: Text | None = None
    billing_frequency: Text | None = None
    price_basis: Text | None = None
    seat_minimum: TokenCount | None = None


class EvidenceItem(Contract):
    record_id: Text
    claim_id: Text | None
    company_name: Text
    finding: Text
    quoted_or_paraphrased_evidence: Text
    source_title: Text
    source_type: Text
    source_url: Text  # Source provenance only; never fetched by these contracts.
    published_at: UTCTimestamp
    available_at: UTCTimestamp
    effective_date: UTCTimestamp | None
    confidence: Score
    limitations: list[Text] = Field(default_factory=list)
    source_independence_group: Text | None = None
    supersedes_record_id: Text | None = None
    end_date: UTCTimestamp | None = None
    verification_status: Text | None = None
    feature_status: Text | None = None
    pricing_observation: PricingObservation | None = None


class TaskCoverage(Contract):
    task_id: Text
    coverage_score: Score | None
    supporting_record_ids: list[Text] = Field(default_factory=list)
    notes: list[Text] = Field(default_factory=list)


class EvidencePack(Contract):
    original_question: Text
    as_of: UTCTimestamp
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    missing_information: list[Text] = Field(default_factory=list)
    contradictions: list[Text] = Field(default_factory=list)
    retrieval_log: list[Text] = Field(default_factory=list)
    coverage_by_task: list[TaskCoverage] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence(self):
        _unique([item.record_id for item in self.evidence_items], "evidence record IDs")
        _unique([item.task_id for item in self.coverage_by_task], "coverage task IDs")
        if any(item.available_at > self.as_of for item in self.evidence_items):
            raise ValueError("Evidence available_at must be on or before EvidencePack.as_of")
        return self


class ReportClaim(Contract):
    claim_id: Text
    text: Text
    supporting_record_ids: list[Text] = Field(default_factory=list)
    claim_type: Literal["factual", "inference"]
    confidence: Score

    @model_validator(mode="after")
    def factual_support(self):
        _unique(self.supporting_record_ids, "supporting record IDs")
        if self.claim_type == "factual" and not self.supporting_record_ids:
            raise ValueError("Factual claims require supporting_record_ids")
        return self


class FactualClaim(ReportClaim):
    claim_type:Literal['factual']
    supporting_record_ids:list[Text]=Field(min_length=1)


class InferenceClaim(ReportClaim):
    claim_type:Literal['inference']


class CompetitiveReport(Contract):
    question: Text
    as_of: UTCTimestamp
    executive_summary: Text
    key_findings: list[Text] = Field(default_factory=list)
    competitor_comparison: list[Text] = Field(default_factory=list)
    strategic_implications: list[Text] = Field(default_factory=list)
    limitations: list[Text] = Field(default_factory=list)
    claims: list[FactualClaim|InferenceClaim] = Field(default_factory=list)
    sources: list[Text] = Field(default_factory=list, description="Source record IDs, not URLs")

    @field_validator('claims',mode='before')
    @classmethod
    def validate_legacy_claim_construction(cls,values):
        # Preserve public Python construction through ReportClaim while the API
        # schema now encodes minItems=1 specifically for factual claims.
        if isinstance(values,list):
            return [value.model_dump() if isinstance(value,ReportClaim) else value for value in values]
        return values

    @model_validator(mode="after")
    def validate_claims(self):
        _unique([claim.claim_id for claim in self.claims], "report claim IDs")
        _unique(self.sources, "source record IDs")
        cited = {rid for claim in self.claims for rid in claim.supporting_record_ids}
        if not cited.issubset(set(self.sources)):
            raise ValueError("All supporting_record_ids must appear in report sources")
        return self


class ReviewIssue(Contract):
    issue_id: Text
    severity: Literal["critical", "major", "minor"]
    issue_type: Text
    affected_claim_ids: list[Text] = Field(default_factory=list)
    explanation: Text
    suggested_correction: Text
    missing_record_ids: list[Text] = Field(default_factory=list)
    related_record_ids: list[Text] = Field(default_factory=list)


class ReviewResult(Contract):
    decision: Literal["PASS", "FAIL"]
    completeness_score: Score
    evidence_support_score: Score
    temporal_consistency_score: Score
    flagged_issues: list[ReviewIssue] = Field(default_factory=list)
    missing_requirements: list[Text] = Field(default_factory=list)
    reviewer_notes: list[Text] = Field(default_factory=list)
    reviewed_question: Text | None = None
    reviewed_report_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None

    @model_validator(mode="after")
    def unique_issues(self):
        _unique([issue.issue_id for issue in self.flagged_issues], "review issue IDs")
        return self


class StageMetrics(Contract):
    """Deterministic telemetry. Missing measurements stay unknown, including cost."""
    stage_name: Text
    model: Text | None = None
    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    latency_seconds: NonNegative | None = None
    estimated_cost_usd: NonNegative | None = None
    status: Literal["pending", "running", "success", "failed", "skipped"]
    error_message: Text | None = None
    error_type: Text | None = None
    started_at: UTCTimestamp | None = None
    finished_at: UTCTimestamp | None = None
    total_tokens: TokenCount | None = None
    llm_requests: TokenCount | None = None
    cached_input_tokens: TokenCount | None = None
    estimated_input_cost_usd: NonNegative | None = None
    estimated_output_cost_usd: NonNegative | None = None
    usage_source: Text | None = None
    retrieval_tool_calls: TokenCount | None = None
    full_record_retrievals: TokenCount | None = None
    search_calls: TokenCount | None = None
    pricing_history_calls: TokenCount | None = None


class TotalMetrics(Contract):
    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    latency_seconds: NonNegative | None = None
    estimated_cost_usd: NonNegative | None = None


class RunResult(Contract):
    run_id: Text
    experiment_id: Text
    question_id: Text | None
    execution_status: Literal["completed", "completed_review_failed", "execution_failed"] | None = None
    artifact_directory: Text | None = None
    configuration: Literal["single", "v1", "v2"]
    model: Text
    as_of: UTCTimestamp
    stage_metrics: list[StageMetrics] = Field(default_factory=list)
    final_report: CompetitiveReport | None = None
    review_result: ReviewResult | None = None
    total_metrics: TotalMetrics = Field(default_factory=TotalMetrics)

    @model_validator(mode="after")
    def consistent_cutoff(self):
        if self.final_report is not None and self.final_report.as_of != self.as_of:
            raise ValueError("Run and final report as_of timestamps must match")
        return self
