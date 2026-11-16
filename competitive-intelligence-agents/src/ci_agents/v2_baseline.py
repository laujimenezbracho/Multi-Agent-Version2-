"""Read-only V1 compatibility checks. No independent evaluation results are loaded."""

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path

from .agents.analyst import _validate_report
from .agents.reviewer import _validate_review, report_fingerprint
from .dataset import _date, load_dataset
from .persistence import baseline_reference, dataset_fingerprints, prompt_fingerprints
from .schemas import (
    CompetitiveReport,
    EvidencePack,
    ResearchPlan,
    ReviewResult,
    RunResult,
)
from .workflow_runtime import private_storage, code_fingerprints
from .source_observations import validate_evidence_items

ALLOWED_FILES = {
    "started.json",
    "metadata.json",
    "plan.json",
    "evidence_pack.json",
    "report.json",
    "review.json",
    "metrics.json",
    "retrieval_logs.json",
    "report.md",
    "claim_traceability.json",
    "evaluation_pending.json",
    "run_result.json",
    "planner_metrics.json",
    "researcher_metrics.json",
    "analyst_metrics.json",
    "reviewer_metrics.json",
    "error.json",
    "diagnostics.json",
    "workflow_policy.json",
    "workflow_budget.json",
    "agent_source.json",
}
REQUIRED_FILES = {
    "metadata.json",
    "plan.json",
    "evidence_pack.json",
    "report.json",
    "review.json",
    "metrics.json",
    "retrieval_logs.json",
    "run_result.json",
    "planner_metrics.json",
    "researcher_metrics.json",
    "analyst_metrics.json",
    "reviewer_metrics.json",
}


@dataclass
class V1Baseline:
    directory: Path
    reference: object
    metadata: dict
    metrics: dict
    result: RunResult
    plan: ResearchPlan
    evidence: EvidencePack
    report: CompetitiveReport
    review: ReviewResult


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_manifest_names(directory, *, completed=True):
    supplied = Path(directory).expanduser()
    path = supplied.resolve()
    if supplied.is_symlink() or private_storage(path):
        raise ValueError(
            "Baseline must be a public runtime artifact directory, not private evaluation storage"
        )
    if (path / "manifest.json").is_symlink():
        raise ValueError("Baseline manifest must not be a symbolic link")
    manifest = read_json(path / "manifest.json")
    names = set(manifest.get("artifact_fingerprints", {}))
    if not names.issubset(ALLOWED_FILES) or (
        completed and not REQUIRED_FILES.issubset(names)
    ):
        raise ValueError("Baseline manifest includes unsupported/missing V1 artifacts")
    return names


def load_baseline(directory, settings, as_of=None):
    supplied = Path(directory).expanduser()
    safe_manifest_names(supplied)
    path = supplied.resolve()
    reference = baseline_reference(path.parent, path.name)
    metadata = read_json(path / "metadata.json")
    if metadata.get("pipeline_version") != "v1_fixed":
        raise ValueError(
            "Only completed V1 baselines may be continued; V2 chaining is prohibited"
        )
    if (
        metadata.get("run_id") != path.name
        or metadata.get("model") != settings.openai_model
    ):
        raise ValueError("Baseline run ID or model configuration mismatch")
    if metadata.get("dataset") != dataset_fingerprints(settings.data_dir):
        raise ValueError("Baseline public dataset fingerprint mismatch")
    if metadata.get("prompt_fingerprints") != prompt_fingerprints():
        raise ValueError("Baseline prompt fingerprints mismatch")
    if (
        metadata.get("agent_code_fingerprints") is not None
        and metadata["agent_code_fingerprints"] != code_fingerprints()
    ):
        raise ValueError("Baseline agent code fingerprints mismatch")
    if "agent_source.json" in reference.artifact_fingerprints:
        snapshot = read_json(path / "agent_source.json")
        from hashlib import sha256

        actual = {
            name: sha256(content.encode("utf-8")).hexdigest()
            for name, content in snapshot["files"].items()
        }
        if (
            snapshot.get("schema_version") != "agent_source.v1"
            or actual != snapshot.get("sha256")
            or actual != metadata.get("agent_code_fingerprints")
        ):
            raise ValueError(
                "Baseline source snapshot differs from recorded code fingerprints"
            )
    if metadata.get("openai_agents_sdk_version") != version("openai-agents"):
        raise ValueError("Baseline Agents SDK version differs from current environment")
    generation = metadata.get("generation_settings")
    if generation is not None and generation != {
        "reasoning_effort": settings.reasoning_effort,
        "max_output_tokens": settings.max_output_tokens,
    }:
        raise ValueError("Baseline generation settings differ from current environment")
    cutoff = _date(settings.evaluation_cutoff if as_of is None else as_of)
    if reference.as_of != cutoff:
        raise ValueError("Baseline evaluation cutoff mismatch")
    models = (
        ("plan.json", ResearchPlan),
        ("evidence_pack.json", EvidencePack),
        ("report.json", CompetitiveReport),
        ("review.json", ReviewResult),
        ("run_result.json", RunResult),
    )
    plan, evidence, report, review, result = [
        schema.model_validate_json((path / name).read_text(encoding="utf-8"))
        for name, schema in models
    ]
    if (
        result.configuration != "v1"
        or result.run_id != path.name
        or result.execution_status != metadata["execution_status"]
    ):
        raise ValueError("Baseline RunResult identity/status mismatch")
    if (
        result.as_of != cutoff
        or plan.as_of != cutoff
        or evidence.as_of != cutoff
        or report.as_of != cutoff
    ):
        raise ValueError("Baseline artifact cutoffs disagree")
    if (
        metadata["question"] != plan.original_question
        or evidence.original_question != plan.original_question
        or report.question != plan.original_question
    ):
        raise ValueError("Baseline artifact questions disagree")
    if result.model != settings.openai_model or result.question_id != metadata.get(
        "question_id"
    ):
        raise ValueError("Baseline model/question metadata mismatch")
    if result.final_report != report or result.review_result != review:
        raise ValueError("Baseline report/review differs from stored RunResult")
    expected_status = (
        "completed" if review.decision == "PASS" else "completed_review_failed"
    )
    if result.execution_status != expected_status:
        raise ValueError("Baseline completion status disagrees with Reviewer decision")
    _validate_report(report, plan, evidence)
    _validate_review(review, plan, evidence, report, report_fingerprint(report))
    metrics = read_json(path / "metrics.json")
    if metrics.get("stages") != [
        stage.model_dump(mode="json") for stage in result.stage_metrics
    ]:
        raise ValueError("Baseline stage telemetry differs from stored RunResult")
    if [stage.stage_name for stage in result.stage_metrics] != [
        "planner",
        "researcher",
        "analyst",
        "reviewer",
    ]:
        raise ValueError("Baseline does not contain exactly the four initial V1 stages")
    if any(
        stage.status != "success" or stage.model != settings.openai_model
        for stage in result.stage_metrics
    ):
        raise ValueError(
            "Baseline stages did not all complete under the configured model"
        )
    if read_json(path / "retrieval_logs.json") != [
        json.loads(entry) for entry in evidence.retrieval_log
    ]:
        raise ValueError("Baseline retrieval logs differ from initial EvidencePack")
    public = load_dataset(settings.data_dir)
    if result.question_id is not None and not any(
        q["question_id"] == result.question_id
        and q["user_question"] == plan.original_question
        for q in public.questions
    ):
        raise ValueError("Baseline question ID does not match public question text")
    records = {record["record_id"]: record for record in public.records}
    retrieved_ids = {
        rid
        for event in read_json(path / "retrieval_logs.json")
        if event.get("tool") == "get_record" and event.get("status") == "ok"
        for rid in event.get("record_ids", [])
    }
    validate_evidence_items(evidence.evidence_items, records, cutoff, retrieved_ids)
    if "evaluation_pending.json" in reference.artifact_fingerprints:
        pending = read_json(path / "evaluation_pending.json")
        if any(
            value is not None
            for key, value in pending.items()
            if key not in {"schema_version", "run_id"}
        ):
            raise ValueError(
                "Baseline contains populated independent evaluation results"
            )
    return V1Baseline(
        path, reference, metadata, metrics, result, plan, evidence, report, review
    )
