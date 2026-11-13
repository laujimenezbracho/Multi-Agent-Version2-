"""Controlled Q01/Q15 pilot around the existing Version 1 pipeline; never auto-executes."""
import ast
import csv
from dataclasses import dataclass, replace
from datetime import datetime
from importlib.resources import files
import io
import json
import math
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from agents import AgentOutputSchema

from .config import Settings
from .dataset import _date, load_dataset
from .evaluation_contracts import IndependentEvaluation
from .persistence import (RunStore, baseline_reference, dataset_fingerprints,
                          prompt_fingerprints, verify_manifest)
from .pipeline import run_pipeline
from .retrieval import RetrievalStore
from .schemas import CompetitiveReport, EvidencePack, ResearchPlan, ReviewResult
from .telemetry import PricingConfig, sanitized_error

PILOT_IDS = ("Q01", "Q15")
PUBLIC_FILES = ("README.md", "PUBLIC_competitive_intelligence.xlsx", "public_web_records.csv",
                "public_web_records.jsonl", "pricing_events.csv", "public_questions.csv")
COLUMNS = ("question_id", "run_id", "execution_status", "reviewer_decision",
           "reviewer_completeness_score", "reviewer_evidence_support_score", "reviewer_temporal_consistency_score",
           "total_input_tokens", "total_output_tokens", "total_tokens", "total_latency_seconds",
           "estimated_cost_usd", "retrieval_calls", "cited_evidence_records", "report_claims",
           "reviewer_issues", "artifact_directory", "evaluation_ready", "diagnostic")


class PilotPreflightError(ValueError):
    """Offline readiness failure, without provider or credential details."""


@dataclass(frozen=True)
class PilotPreflight:
    settings: Settings
    questions: tuple[tuple[str, str], ...]
    as_of: datetime
    output_dir: Path
    pilot_id: str
    run_ids: tuple[str, ...]
    pricing: PricingConfig
    dataset_hash: str
    prompts: dict[str, str]

    def summary(self):
        return {"preflight": "passed", "model": self.settings.openai_model,
                "question_ids": [qid for qid, _ in self.questions], "as_of": self.as_of.isoformat(),
                "output_directory": str(self.output_dir / self.pilot_id), "run_ids": list(self.run_ids),
                "pricing_rates_configured": rates_available(self.pricing),
                "cost_warning": None if rates_available(self.pricing) else "Estimated cost unavailable: no pricing rates configured",
                "dataset_sha256": self.dataset_hash, "prompt_fingerprints": self.prompts,
                "checks": ["public_files", "dataset_validation", "pilot_questions", "fts5", "prompts",
                           "sdk_output_schemas", "model_configuration", "writable_output", "no_overwrite", "public_tool_surface"],
                "privacy_scope": "Public-only application tools; no OS filesystem sandbox",
                "budget_scope": "Between questions only; in-flight calls can exceed budgets"}


def rates_available(pricing):
    return pricing.input_usd_per_million is not None and pricing.output_usd_per_million is not None


def _check_output(output, data_dir):
    if output == data_dir or data_dir in output.parents or output in data_dir.parents:
        raise PilotPreflightError("Output directory must be separate from the public dataset")
    if "evaluation_private" in {part.casefold() for part in output.parts}:
        raise PilotPreflightError("Pilot output cannot be inside private evaluation storage")
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise PilotPreflightError("Output path is blocked by a file")
    if (output / "manifest.json").exists() or (output / "metadata.json").exists():
        raise PilotPreflightError("Output directory is an existing baseline run; choose a separate pilot root")
    try:
        # Reversible probe only; no run directories, reports, or summaries in dry-run.
        with tempfile.TemporaryDirectory(prefix=".ci-preflight-", dir=ancestor) as probe:
            source = Path(probe) / "probe"
            source.write_bytes(b"write-and-link-check")
            os.link(source, Path(probe) / "linked")
    except OSError:
        raise PilotPreflightError("Output filesystem must be writable and support atomic hard-link publication") from None


def _public_agent_surface():
    package = files("ci_agents")
    for name in ("planner", "researcher", "analyst", "reviewer"):
        tree = ast.parse(package.joinpath(f"agents/{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and any(
                part.startswith("private") or part == "evaluation_contracts" for part in node.module.split(".")
            ):
                raise PilotPreflightError("Agent imports an evaluation/private module")
            if isinstance(node, ast.Import) and any("private" in entry.name or "evaluation_contracts" in entry.name for entry in node.names):
                raise PilotPreflightError("Agent imports an evaluation/private module")
    # This audit is deliberately not represented as OS-level access control.


def preflight(settings, question_ids=PILOT_IDS, as_of=None, output_dir=None, *, live=False, pricing=None):
    if not isinstance(settings, Settings) or not settings.openai_model.strip():
        raise PilotPreflightError("A configured model is required")
    if live and not settings.openai_api_key.strip():
        raise PilotPreflightError("OPENAI_API_KEY is required for live execution")
    selected = tuple(question_ids)
    if not selected or len(set(selected)) != len(selected) or any(qid not in PILOT_IDS for qid in selected):
        raise PilotPreflightError("Select distinct pilot question IDs from Q01 and Q15; held-out questions are excluded")
    data_dir = settings.data_dir.resolve()
    if "evaluation_private" in {part.casefold() for part in data_dir.parts}:
        raise PilotPreflightError("Public DATA_DIR cannot point to private evaluation storage")
    for name in PUBLIC_FILES:
        if (data_dir / name).is_symlink() or (data_dir / name).resolve().parent != data_dir:
            raise PilotPreflightError("Public input files must stay inside DATA_DIR without symbolic links")
        if not (data_dir / name).is_file():
            raise PilotPreflightError(f"Missing public dataset file: {name}")
    try:
        dataset = load_dataset(data_dir)
    except Exception:
        raise PilotPreflightError("Public dataset validation failed; check required fields, IDs, counts and dates") from None
    questions = {row["question_id"]: row["user_question"] for row in dataset.questions}
    if not all(qid in questions for qid in PILOT_IDS):
        raise PilotPreflightError("Both Q01 and Q15 must be present in public questions")
    try:
        cutoff = _date(settings.evaluation_cutoff if as_of is None else as_of)
    except (TypeError, ValueError):
        raise PilotPreflightError("as_of must include a timezone") from None
    try:
        with RetrievalStore(dataset):
            pass
    except Exception:
        raise PilotPreflightError("SQLite FTS5 retrieval initialization failed") from None
    try:
        prompts = prompt_fingerprints()
        for name in prompts:
            if not files("ci_agents").joinpath(f"prompts/{name}.md").read_text(encoding="utf-8").strip():
                raise ValueError("Empty prompt")
    except Exception:
        raise PilotPreflightError("All four non-empty agent prompts must be available") from None
    try:
        for schema in (ResearchPlan, EvidencePack, CompetitiveReport, ReviewResult):
            AgentOutputSchema(schema)
    except Exception:
        raise PilotPreflightError("Agent schemas failed SDK strict structured-output conversion") from None
    _public_agent_surface()
    output = Path(output_dir).expanduser().resolve() if output_dir is not None else (settings.runs_dir / "pilots").resolve()
    _check_output(output, data_dir)
    pilot_id = "pilot-" + uuid4().hex
    run_ids = tuple(qid.lower() + "-" + uuid4().hex for qid in selected)
    if (output / pilot_id).exists() or any((output / pilot_id / "runs" / rid).exists() for rid in run_ids):
        raise PilotPreflightError("Proposed pilot or run directory already exists")
    rates = pricing or PricingConfig(model=settings.openai_model)
    if rates.model != settings.openai_model:
        raise PilotPreflightError("Pricing model must match configured model")
    return PilotPreflight(settings, tuple((qid, questions[qid]) for qid in selected), cutoff, output,
                          pilot_id, run_ids, rates, dataset_fingerprints(data_dir)["sha256"], prompts)


def evaluation_readiness(directory):
    """Validate initial artifacts without opening or calculating any ground truth."""
    directory = Path(directory)
    manifest = verify_manifest(directory)
    placeholder = IndependentEvaluation.model_validate_json((directory / "evaluation_pending.json").read_text(encoding="utf-8"))
    if any(value is not None for key, value in placeholder.model_dump().items() if key not in {"schema_version", "run_id"}):
        raise ValueError("Independent evaluation placeholder is not entirely null")
    loaded = {}
    for filename, schema in (("report.json", CompetitiveReport), ("evidence_pack.json", EvidencePack), ("review.json", ReviewResult)):
        if filename not in manifest["artifact_fingerprints"]:
            return False
        loaded[filename] = schema.model_validate_json((directory / filename).read_text(encoding="utf-8"))
    report, evidence, review = loaded["report.json"], loaded["evidence_pack.json"], loaded["review.json"]
    from .agents.reviewer import report_fingerprint
    claim_ids = {claim.claim_id for claim in report.claims}
    evidence_ids = {item.record_id for item in evidence.evidence_items}
    if not set(report.sources).issubset(evidence_ids) or any(not set(claim.supporting_record_ids).issubset(evidence_ids) for claim in report.claims):
        raise ValueError("Report references evidence absent from initial EvidencePack")
    for issue in review.flagged_issues:
        if not set(issue.affected_claim_ids).issubset(claim_ids) or not set(issue.related_record_ids + issue.missing_record_ids).issubset(evidence_ids):
            raise ValueError("Review issue references are not preserved in initial artifacts")
    if review.reviewed_question != report.question or review.reviewed_report_hash != report_fingerprint(report):
        raise ValueError("Review identity does not match the preserved report")
    if report.question != evidence.original_question or report.as_of != evidence.as_of:
        raise ValueError("Initial report/evidence identity mismatch")
    reference = baseline_reference(directory.parent, directory.name)
    return bool(reference.artifact_fingerprints["report.json"] and reference.artifact_fingerprints["review.json"])


def _sum(stages, field):
    values = [getattr(stage, field) for stage in stages]
    return sum(values) if values and all(value is not None for value in values) else None


def _row(qid, rid, result=None, diagnostic=None, directory=None):
    row = dict.fromkeys(COLUMNS)
    row.update(question_id=qid, run_id=rid, execution_status="execution_failed", diagnostic=diagnostic,
               artifact_directory=str(directory) if directory is not None else None)
    if result is None:
        return row
    review, report = result.review_result, result.final_report
    row.update(execution_status=result.execution_status, total_input_tokens=result.total_metrics.input_tokens,
               total_output_tokens=result.total_metrics.output_tokens, total_tokens=_sum(result.stage_metrics, "total_tokens"),
               total_latency_seconds=result.total_metrics.latency_seconds, estimated_cost_usd=result.total_metrics.estimated_cost_usd,
               retrieval_calls=_sum(result.stage_metrics, "retrieval_tool_calls"),
               cited_evidence_records=len(set(report.sources)) if report is not None else None,
               report_claims=len(report.claims) if report is not None else None,
               reviewer_issues=len(review.flagged_issues) if review is not None else None,
               artifact_directory=result.artifact_directory)
    if review is not None:
        row.update(reviewer_decision=review.decision, reviewer_completeness_score=review.completeness_score,
                   reviewer_evidence_support_score=review.evidence_support_score,
                   reviewer_temporal_consistency_score=review.temporal_consistency_score)
    return row


def summary_csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader()
    # csv writes None as empty cells: unknown, not zero.
    writer.writerows(rows)
    return stream.getvalue()


def summary_markdown(rows):
    def safe(value):
        if value is None:
            return "null"
        import html
        return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ").replace("`", "&#96;").replace("[", "&#91;").replace("]", "&#93;")
    lines = ["# Controlled pilot summary", "", "Reviewer scores are internal quality assessments, not independent factual accuracy.", "",
             "| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    lines += ["| " + " | ".join(safe(row.get(column)) for column in COLUMNS) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def validate_budgets(pricing, max_total_tokens=None, max_estimated_cost=None):
    if max_total_tokens is not None and (isinstance(max_total_tokens, bool) or not isinstance(max_total_tokens, int) or max_total_tokens < 1):
        raise PilotPreflightError("max_total_tokens must be a positive integer")
    if max_estimated_cost is not None and (isinstance(max_estimated_cost, bool) or not math.isfinite(max_estimated_cost) or max_estimated_cost <= 0):
        raise PilotPreflightError("max_estimated_cost must be positive and finite")
    if max_estimated_cost is not None and not rates_available(pricing):
        raise PilotPreflightError("A cost budget requires documented input and output pricing rates")


async def run_pilot(prepared, *, allow_api=False, max_total_tokens=None, max_estimated_cost=None):
    """Sequential delegation only. Optional budgets stop NEXT questions, not in-flight calls."""
    if not allow_api:
        raise PilotPreflightError("Live pilot requires explicit allow_api=True")
    if not prepared.settings.openai_api_key.strip():
        raise PilotPreflightError("OPENAI_API_KEY is required for live execution")
    validate_budgets(prepared.pricing, max_total_tokens, max_estimated_cost)
    # Recheck immediately before any API-capable execution.
    if dataset_fingerprints(prepared.settings.data_dir)["sha256"] != prepared.dataset_hash or prompt_fingerprints() != prepared.prompts:
        raise PilotPreflightError("Dataset or prompts changed since preflight")
    _check_output(prepared.output_dir, prepared.settings.data_dir.resolve())
    store = RunStore(prepared.output_dir, prepared.pilot_id, secrets=(prepared.settings.openai_api_key,))
    store.write_json("preflight.json", prepared.summary())
    settings = replace(prepared.settings, runs_dir=store.directory / "runs")
    rows, errors = [], []
    spent_tokens, spent_cost, budget_stop = 0, 0.0, None
    for (qid, question), rid in zip(prepared.questions, prepared.run_ids, strict=True):
        directory = settings.runs_dir / rid
        if budget_stop:
            row = _row(qid, rid, diagnostic=budget_stop)
            row["execution_status"] = "not_started_budget"
        else:
            try:
                result = await run_pipeline(question, prepared.as_of, rid, question_id=qid,
                    experiment_id=prepared.pilot_id, settings=settings, pricing=prepared.pricing)
                row = _row(qid, rid, result)
                if result.execution_status == "execution_failed":
                    row["diagnostic"] = "Execution failed; inspect preserved run artifacts. No retry attempted."
                try:
                    row["evaluation_ready"] = evaluation_readiness(result.artifact_directory)
                except Exception:
                    row["evaluation_ready"] = False
                    row["diagnostic"] = "Artifact readiness check failed; inspect run artifacts. No retry attempted."
            except Exception as exc:
                error = sanitized_error(exc, "pilot_question")
                errors.append({"question_id": qid, "run_id": rid, **error})
                row = _row(qid, rid, diagnostic=error["message"], directory=directory if directory.exists() else None)
            tokens, cost = row["total_tokens"], row["estimated_cost_usd"]
            if tokens is None and max_total_tokens is not None:
                budget_stop = "Token usage unavailable; remaining questions stopped conservatively"
            if cost is None and max_estimated_cost is not None:
                budget_stop = "Estimated cost unavailable; remaining questions stopped conservatively"
            spent_tokens += tokens or 0
            spent_cost += cost or 0
            if max_total_tokens is not None and spent_tokens >= max_total_tokens:
                budget_stop = "Between-question total-token budget reached"
            if max_estimated_cost is not None and spent_cost >= max_estimated_cost:
                budget_stop = "Between-question estimated-cost budget reached"
        rows.append(row)
        # Append-only progress snapshot survives a later question failure/crash.
        store.write_json(f"progress_{len(rows)}.json", row)
    store.write_text("pilot_summary.csv", summary_csv(rows))
    store.write_text("pilot_summary.md", summary_markdown(rows))
    store.write_json("pilot_errors.json", errors)
    store.write_json("pilot_safeguards.json", {"max_total_tokens": max_total_tokens, "max_estimated_cost": max_estimated_cost,
        "enforcement": "between_questions_only_not_a_hard_cap", "stop_reason": budget_stop})
    store.finalize()
    return {"pilot_directory": str(store.directory), "rows": rows}
