"""Native MLflow initial/final evaluation and a source-audited Reviewer challenge.

Private rubrics belong only to native evaluation expectations, never predict_fn.
Use a new output directory after code/policy changes. --prepare-only is offline.
"""

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ci_agents.config import load_config, use_settings
from ci_agents.dataset import load_dataset
from ci_agents.persistence import (
    fingerprint,
    dataset_fingerprints,
    prompt_fingerprints,
    render_report,
    verify_manifest,
)
from ci_agents.pipeline_v2 import run_pipeline_v2
from ci_agents.agents.researcher import (
    ResearcherLimits,
    ResearchSession,
    ResearchOutcome,
)
from ci_agents.agents.reviewer import reviewer
from ci_agents.retrieval import RetrievalStore
from ci_agents.schemas import (
    ResearchPlan,
    ResearchTask,
    CompetitiveReport,
    ReportClaim,
    InferenceClaim,
    RunResult,
)
from ci_agents.telemetry import PricingConfig, capture_stage, sanitized_error
from ci_agents.workflow_runtime import (
    WorkflowRuntime,
    WorkflowPolicy,
    RunBudget,
    use_workflow,
)
from run_checkpoint_evaluation import (
    public_output,
    parse_grade,
    JUDGE_INSTRUCTIONS,
    compact_source,
)
from grading_contract import catalogue
from checkpoint_meter import UsageMeter, label_usage, totals

EVALUATION_INSTRUCTIONS = (
    """If outputs contains recovery_comparison=true, compare the
initial and final answers against the private rubric and original sources. Return
assessment_json encoding {"status":"recovery_comparison", "corrected_defects":[{
"explanation":string,"reference_record_ids":[string],"initial_claim_ids":[string],
"final_claim_ids":[string]}], "introduced_defects":[same shape],
"remaining_defects":[same shape], "explanation":string}.
Classify MATERIAL defects semantically, not by changed claim IDs or word choice.
Corrected means an initial material defect is absent or responsibly qualified in the
final answer; introduced means a new material defect appears; remaining means the
same defect persists. Use NARRATIVE for uncatalogued prose. Ground every defect in
exact reference/source IDs, and never treat Reviewer PASS as truth. If an answer is
missing, report it as a material completion defect with empty claim IDs and source
IDs where none apply; do not fabricate a successful correction.
If outputs contains reviewer_challenge=true, this is an
internal Reviewer detection test, not a CompetitiveReport to grade. Use the supplied
source observation and expected decision. Return the framework result envelope with
assessment_json encoding {"status":"reviewer_challenge", "decision_correct":boolean,
"seeded_material_error_detected":boolean|null,"explanation":string}.
Check the Reviewer's actual issues: a FAIL for an unrelated defect is not detection
of the seeded price error. An invalid review cannot receive detection credit.
For all other outputs apply the source-assessment contract below unchanged.
"""
    + JUDGE_INSTRUCTIONS
)


class RecoveryDefect(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    explanation: str = Field(min_length=1)
    reference_record_ids: list[str]
    initial_claim_ids: list[str]
    final_claim_ids: list[str]


class RecoveryAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["recovery_comparison"]
    corrected_defects: list[RecoveryDefect]
    introduced_defects: list[RecoveryDefect]
    remaining_defects: list[RecoveryDefect]
    explanation: str = Field(min_length=1)


def validate_recovery_grade(payload, case, rubric):
    value = RecoveryAssessment.model_validate(payload)
    records = {r["record_id"] for r in rubric["audited_reference_sources"]}
    ids = {}
    for variant in ["initial", "final"]:
        report = case[variant]["report"]
        ids[variant] = {"NARRATIVE"} | (
            {c["claim_id"] for c in report["claims"]} if report else set()
        )
        records.update(r["record_id"] for r in case[variant]["original_cited_sources"])
    for defect in (
        value.corrected_defects + value.introduced_defects + value.remaining_defects
    ):
        if not set(defect.reference_record_ids).issubset(records):
            raise ValueError("Unknown recovery reference")
        if not set(defect.initial_claim_ids).issubset(ids["initial"]) or not set(
            defect.final_claim_ids
        ).issubset(ids["final"]):
            raise ValueError("Unknown recovery claim IDs")
        if (
            all(case[variant]["report"] is not None for variant in ["initial", "final"])
            and not defect.reference_record_ids
        ):
            raise ValueError(
                "A material defect between available reports needs a source anchor"
            )
    if case["initial"]["report"] is not None and any(
        not defect.initial_claim_ids for defect in value.corrected_defects
    ):
        raise ValueError(
            "A corrected defect must identify its initial claim or NARRATIVE"
        )
    if case["final"]["report"] is not None and any(
        not defect.final_claim_ids for defect in value.introduced_defects
    ):
        raise ValueError(
            "An introduced defect must identify its final claim or NARRATIVE"
        )
    return value.model_dump()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, ensure_ascii=False) + "\n")


def ensure_new_evaluation(root, *, prepare_only=False):
    if not prepare_only and (root / "evaluation_manifest.json").exists():
        raise ValueError(
            "Evaluation already completed. Use a new output directory; prior grades will not be overwritten."
        )


def validate_cached_challenge(prediction):
    value = dict(prediction)
    recorded = value.pop("prediction_sha256", None)
    if recorded is None or recorded != fingerprint(value):
        raise ValueError("Cached Reviewer prediction changed or has no integrity hash")


def validate_cached_case(case, directory, data):
    """Resume only predictions that still match their sealed agent artifacts."""
    if not (directory / "manifest.json").exists():
        if case.get("execution_status") == "execution_failed" and all(
            case[variant]["report"] is None for variant in ["initial", "final"]
        ):
            return  # A pre-artifact failure cannot receive positive answer credit.
        raise ValueError("Cached reports require a sealed agent run")
    verify_manifest(directory)
    run_path = directory / "run_result.json"
    run = (
        RunResult.model_validate_json(run_path.read_text())
        if run_path.exists()
        else None
    )
    for variant, filename in [
        ("initial", "initial_report.json"),
        ("final", "final_report.json"),
    ]:
        path = directory / filename
        report = (
            CompetitiveReport.model_validate_json(path.read_text())
            if path.exists()
            else run.final_report
            if run
            else None
        )
        if case[variant] != public_output(report, data):
            raise ValueError(
                "Cached prediction differs from its sealed " + variant + " report"
            )
    metrics_path = directory / "metrics.json"
    expected_resources = (
        json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    )
    if case.get("resources") != expected_resources:
        raise ValueError("Cached resource usage differs from sealed metrics")
    metadata = json.loads((directory / "metadata.json").read_text())
    if case["execution_status"] != metadata["execution_status"]:
        raise ValueError("Cached execution status differs from sealed metadata")


def reviewer_challenge(data, cutoff, wrong=False):
    """Controlled mutations of one public source observation, never a private key."""
    variant = (
        "wrong_price"
        if wrong is True
        else "valid_source_price"
        if wrong is False
        else wrong
    )
    row = next(r for r in data.records if r["record_id"] == "WEB-00002")
    q = "What Growth price is stated in source WEB-00002? Report that historical source observation, not a current market winner; unknown fees are excluded."
    plan = ResearchPlan(
        original_question=q,
        as_of=cutoff,
        objective="Report the stated observation only",
        tasks=[
            ResearchTask(
                task_id="T1",
                research_question=q,
                companies=[row["company_name"]],
                categories=["Pricing"],
                time_scope="Historical source observation available by cutoff",
                evidence_requirements=[
                    "Correct stated price, units and annual commitment"
                ],
            )
        ],
        expected_answer_elements=[
            "Stated Growth price, units and annual commitment, with fees unknown"
        ],
        comparison_dimensions=["Historical source price"],
    )
    with RetrievalStore(data) as store:
        session = ResearchSession(plan, store, ResearcherLimits())
        session.invoke("get_record", {"record_id": row["record_id"]})
        pack = session.assemble(ResearchOutcome(stop_reason="sufficient_for_question"))
    amount = "1" if variant == "wrong_price" else str(row["metric_value"])
    fact = f"Source WEB-00002 states NexaFlow Growth at USD {amount} per user per month with annual commitment; tax and onboarding are excluded. This is a historical source observation."
    if variant == "wrong_billing_unit":
        fact = fact.replace("per user per month", "per entire company per month")
    if variant == "wrong_effective_date":
        fact += " The source states this price becomes effective on 2027-01-06."
    report = CompetitiveReport(
        question=q,
        as_of=cutoff,
        executive_summary=fact,
        key_findings=[fact],
        competitor_comparison=["No current cross-company comparison requested."],
        strategic_implications=[
            "No current market ranking follows from this historical observation."
        ],
        limitations=[
            "Unknown negotiated terms, taxes and onboarding are not included or assumed zero."
        ],
        claims=[
            ReportClaim(
                claim_id="C1",
                text=fact,
                claim_type="factual",
                supporting_record_ids=["WEB-00002"],
                confidence=0.5,
            )
        ],
        sources=["WEB-00002"],
    )
    if variant == "unsupported_strategy":
        text = "This proves NexaFlow is currently cheaper than PulseGrid and will win its customers."
        report.strategic_implications = [text]
        report.claims.append(
            InferenceClaim(
                claim_id="C2",
                text=text,
                claim_type="inference",
                supporting_record_ids=["WEB-00002"],
                confidence=0.5,
            )
        )
    return plan, pack, report


def paired_summary(rows, cases):
    grouped = {}
    for row in rows:
        key = (row["question_id"], row["repetition"])
        grouped.setdefault(key, {})[row["variant"]] = row
    result = []
    for (q, rep), pair in grouped.items():
        a, b = pair.get("initial"), pair.get("final")
        first = a["scores"] if a else {}
        last = b["scores"] if b else {}
        valid = all(s.get("status") == "assessed" for s in [first, last])
        case = cases.get(f"{q}-r{rep}", {})
        result.append(
            {
                "question_id": q,
                "repetition": rep,
                "comparison_status": "assessed" if valid else "incomplete_or_invalid",
                "initial_quality": first.get("decision_safe_coverage"),
                "final_quality": last.get("decision_safe_coverage"),
                "quality_delta": last["decision_safe_coverage"]
                - first["decision_safe_coverage"]
                if valid
                else None,
                "initial_material_errors": first.get("material_error_count"),
                "final_material_errors": last.get("material_error_count"),
                "initial_requirement_checks": a.get("assessment", {}).get(
                    "requirement_checks"
                )
                if a
                else None,
                "final_requirement_checks": b.get("assessment", {}).get(
                    "requirement_checks"
                )
                if b
                else None,
                "initial_defects": a.get("assessment", {}).get("defects")
                if a
                else None,
                "final_defects": b.get("assessment", {}).get("defects") if b else None,
                "defect_alignment": "No automatic semantic matching across rewritten claim IDs; inspect before/after defects for corrected/new errors.",
                "resources": case.get("resources"),
                "routing": case.get("routing"),
                "approval": "pending_human",
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--questions", nargs="+", default=["Q01", "Q15"])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--retrieval-budget", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and not args.allow_api:
        parser.error("Execution requires --allow-api")
    if args.repetitions < 1:
        parser.error("At least one repetition is required")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ensure_new_evaluation(root, prepare_only=args.prepare_only)
    settings = load_config(
        env_file=ROOT.parent / ".env",
        require_api_key=not args.prepare_only,
        environ={
            "OPENAI_MODEL": "gpt-5-mini",
            "DATA_DIR": str(ROOT / "data/public"),
            "RUNS_DIR": str(root / "agent_runs"),
        },
    )
    settings = replace(settings, reasoning_effort="low", max_output_tokens=8000)
    data = load_dataset(settings.data_dir)
    questions = {r["question_id"]: r["user_question"] for r in data.questions}
    if not set(args.questions).issubset(questions):
        parser.error("Unknown public question ID")
    keys = {
        q: json.loads((args.key_dir / (q + ".json")).read_text())
        for q in args.questions
    }
    for key in keys.values():
        key["audited_reference_sources"] = [
            compact_source(r) for r in key["audited_reference_sources"]
        ]
    policy = WorkflowPolicy()
    budget = RunBudget(
        max_total_tokens=750000,
        max_estimated_cost=2,
        max_seconds=600,
        max_model_requests=32,
        reserve_final_tokens=40000,
    )
    rates = PricingConfig(
        model=settings.openai_model,
        input_usd_per_million=0.25,
        output_usd_per_million=2,
        source="https://developers.openai.com/api/docs/models/gpt-5-mini",
        source_date="2026-11-15",
    )
    plan = {
        "questions": args.questions,
        "repetitions": args.repetitions,
        "retrieval_budget": args.retrieval_budget,
        "policy": policy.model_dump(),
        "budget": budget.model_dump(),
        "model": settings.openai_model,
        "judge": "gpt-4.1",
        "reasoning_effort": settings.reasoning_effort,
        "max_output_tokens": settings.max_output_tokens,
        "cutoff": settings.evaluation_cutoff.isoformat(),
        "dataset": dataset_fingerprints(settings.data_dir),
        "source": {
            str(p.relative_to(ROOT)): fingerprint(p.read_text())
            for p in sorted((ROOT / "src/ci_agents").rglob("*.py"))
        },
        "evaluation_scripts": {
            name: fingerprint((ROOT / "scripts" / name).read_text())
            for name in [
                "evaluate_v2_reliability.py",
                "run_checkpoint_evaluation.py",
                "checkpoint_meter.py",
                "grading_contract.py",
            ]
        },
        "script": fingerprint(Path(__file__).read_text()),
        "prompts": prompt_fingerprints(),
        "rubric_hashes": {q: fingerprint(key) for q, key in keys.items()},
        "quality_tolerance_points": 2,
        "decision_rule": "Screening only. No material-error increase; assess independent quality and incremental cost. Requires repetitions, human adjudication and held-out validation before a policy conclusion.",
    }
    frozen = root / "experiment_plan.json"
    if frozen.exists() and json.loads(frozen.read_text()) != json.loads(
        json.dumps(plan)
    ):
        raise ValueError("Frozen experiment changed; use a new output directory")
    if not frozen.exists():
        save(frozen, plan)
    source_path = root / "evaluation_source.json"
    if not source_path.exists():
        save(
            source_path,
            {
                "files": {
                    name: (ROOT / "scripts" / name).read_text()
                    for name in plan["evaluation_scripts"]
                },
                "hash_definition": "Use experiment_plan.evaluation_scripts; no private answer keys or environment files included",
            },
        )
    os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")
    os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")
    os.environ.setdefault("MLFLOW_SCORER_MAX_WORKERS", "1")
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    import mlflow
    from mlflow.genai import datasets
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers import list_scorers

    configured = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(configured or "sqlite:///" + str(root / "mlflow.db"))
    expid = os.environ.get("MLFLOW_EXPERIMENT_ID")
    experiment = (
        mlflow.set_experiment(experiment_id=expid)
        if expid
        else mlflow.set_experiment("v2-reliability-paired-pilot")
    )
    existing = datasets.search_datasets(experiment_ids=[experiment.experiment_id])
    save(
        root / "dataset_discovery.json",
        {
            "existing": [
                {"name": d.name, "dataset_id": d.dataset_id} for d in existing
            ],
            "selection": "Reuse matching frozen design; earlier threshold data does not contain paired V2 recovery outputs or this Reviewer challenge.",
        },
    )
    with mlflow.start_span(name="v2_evaluation_preflight", span_type="CHAIN") as span:
        span.set_inputs({"design": plan})
        span.set_outputs({"api_calls": 0, "status": "trace_check"})
    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(experiment_ids=[experiment.experiment_id])
    if traces.empty:
        raise RuntimeError("MLflow trace readback failed")
    judge_name = "v2_source_audit_" + fingerprint(EVALUATION_INSTRUCTIONS)[:8]
    registered = list_scorers(experiment_id=experiment.experiment_id)
    if not any(s.name == judge_name for s in registered):
        make_judge(
            name=judge_name,
            instructions=EVALUATION_INSTRUCTIONS,
            model="openai:/gpt-4.1",
            inference_params={"temperature": 0, "max_tokens": 12000},
            feedback_value_type=dict[str, str],
        ).register(experiment_id=experiment.experiment_id)
    scorers = list_scorers(experiment_id=experiment.experiment_id)
    name = "v2-paired-" + fingerprint(plan)[:12]
    ds = next((d for d in existing if d.name == name), None)
    if ds is None:
        ds = datasets.create_dataset(name=name, experiment_id=experiment.experiment_id)
        records = [
            {
                "inputs": {"question_id": q, "repetition": rep, "variant": variant},
                "expectations": {
                    "rubric": keys[q],
                    "requirement_catalogue": catalogue(None, keys[q])["requirements"],
                },
            }
            for q in args.questions
            for rep in range(1, args.repetitions + 1)
            for variant in ["initial", "final", "comparison"]
        ]
        challenge_variants = {
            "valid_source_price": ("PASS", "No deliberately seeded error"),
            "wrong_price": ("FAIL", "USD 1 substituted for the source price USD 149"),
            "wrong_billing_unit": (
                "FAIL",
                "Per-company price substituted for per-user price",
            ),
            "wrong_effective_date": (
                "FAIL",
                "Invented 2027 effective date contradicts the 2025 source",
            ),
            "unsupported_strategy": (
                "FAIL",
                "One historical vendor price does not establish current undercutting or customer wins",
            ),
        }
        records += [
            {
                "inputs": {
                    "question_id": "REVIEWER",
                    "repetition": 1,
                    "variant": variant,
                },
                "expectations": {
                    "expected_decision": decision,
                    "seeded_error": description,
                    "reference_source": compact_source(
                        next(r for r in data.records if r["record_id"] == "WEB-00002")
                    ),
                },
            }
            for variant, (decision, description) in challenge_variants.items()
        ]
        ds.merge_records(TypeAdapter(object).dump_python(records, mode="json"))
    save(
        root / "readiness.json",
        {
            "dataset_id": ds.dataset_id,
            "experiment_id": experiment.experiment_id,
            "trace_count": len(traces),
            "scorers": [s.name for s in scorers],
            "status": "prepared",
            "paid_execution": False,
        },
    )
    print(
        "PREPARED native dataset and registered scorers; traces found:",
        len(traces),
        flush=True,
    )
    if args.prepare_only:
        return
    mlflow.openai.autolog()
    meter = UsageMeter(root / "request_usage.jsonl")
    meter.install()
    cases = {}
    try:

        @mlflow.trace(name="v2_paired_prediction", span_type="CHAIN")
        def predict_fn(question_id, repetition, variant):
            if question_id == "REVIEWER":
                path = root / "reviewer_challenge" / f"{variant}.json"
                if path.exists():
                    prediction = json.loads(path.read_text())
                    validate_cached_challenge(prediction)
                    return prediction
                p, e, r = reviewer_challenge(data, settings.evaluation_cutoff, variant)
                state = WorkflowRuntime(
                    policy, RunBudget(max_model_requests=2, max_seconds=90), rates
                )
                row = {
                    "reviewer_challenge": True,
                    "case": variant,
                    "report": r.model_dump(mode="json"),
                    "source": compact_source(
                        next(x for x in data.records if x["record_id"] == "WEB-00002")
                    ),
                }
                try:
                    with (
                        use_settings(settings),
                        use_workflow(state),
                        capture_stage(),
                        label_usage(stage="reviewer_challenge", case=variant),
                    ):
                        review = asyncio.run(reviewer(p, e, r))
                    row.update(review=review.model_dump(mode="json"), valid_output=True)
                except Exception as exc:
                    row.update(
                        valid_output=False,
                        error=sanitized_error(exc, "reviewer"),
                        review=None,
                    )
                row["usage"] = state.snapshot()
                row = TypeAdapter(object).dump_python(row, mode="json")
                row["prediction_sha256"] = fingerprint(row)
                save(path, row)
                return row
            token = f"{question_id}-r{repetition}"
            path = root / "cases" / token / "case.json"
            if path.exists():
                case = json.loads(path.read_text())
                validate_cached_case(case, settings.runs_dir / token, data)
            else:
                run = None
                failure = None
                try:
                    with label_usage(
                        stage="v2_workflow",
                        question_id=question_id,
                        repetition=repetition,
                    ):
                        run = asyncio.run(
                            run_pipeline_v2(
                                questions[question_id],
                                settings.evaluation_cutoff,
                                token,
                                question_id=question_id,
                                settings=settings,
                                pricing=rates,
                                policy=policy,
                                budget=budget,
                                initial_research_limits=ResearcherLimits(
                                    max_tool_calls=args.retrieval_budget
                                ),
                            )
                        )
                except Exception as exc:
                    failure = sanitized_error(
                        exc, "workflow_evaluation", secrets=(settings.openai_api_key,)
                    )
                directory = (
                    Path(run.artifact_directory) if run else settings.runs_dir / token
                )
                initial_path = directory / "initial_report.json"
                final_path = directory / "final_report.json"
                initial = (
                    CompetitiveReport.model_validate_json(initial_path.read_text())
                    if initial_path.exists()
                    else run.final_report
                    if run
                    else None
                )
                final = (
                    CompetitiveReport.model_validate_json(final_path.read_text())
                    if final_path.exists()
                    else run.final_report
                    if run
                    else None
                )
                case = {
                    "initial": public_output(initial, data),
                    "final": public_output(final, data),
                    "execution_status": run.execution_status
                    if run
                    else "execution_failed",
                    "resources": json.loads((directory / "metrics.json").read_text())
                    if (directory / "metrics.json").exists()
                    else None,
                    "routing": json.loads(
                        (directory / "recovery_routing.json").read_text()
                    )
                    if (directory / "recovery_routing.json").exists()
                    else {"route": "initial_execution_failed"},
                    "reviewer": run.review_result.model_dump(mode="json")
                    if run and run.review_result
                    else None,
                    "execution_error": failure,
                }
                save(path, case)
                for label, report in [("initial", initial), ("final", final)]:
                    (path.parent / (label + ".md")).write_text(
                        render_report(report, None)
                        if report
                        else "No report produced.\n"
                    )
            cases[token] = case
            if variant == "comparison":
                return {
                    "recovery_comparison": True,
                    "initial": case["initial"],
                    "final": case["final"],
                }
            return case[variant]

        result = mlflow.genai.evaluate(data=ds, predict_fn=predict_fn, scorers=scorers)
        result.result_df.to_json(
            root / "native_results.json", orient="records", indent=2
        )
        graded = []
        challenge = []
        recovery = []
        for row in json.loads((root / "native_results.json").read_text()):
            identity = json.loads(json.loads(row["trace"])["info"]["request_preview"])
            q, rep, variant = (
                identity["question_id"],
                identity["repetition"],
                identity["variant"],
            )
            if q == "REVIEWER":
                out = json.loads(
                    (root / "reviewer_challenge" / f"{variant}.json").read_text()
                )
                expected = "PASS" if variant == "valid_source_price" else "FAIL"
                review = out.get("review") or {}
                value = row.get(judge_name + "/value")
                try:
                    assessment = json.loads(value["assessment_json"])
                    if (
                        set(assessment)
                        != {
                            "status",
                            "decision_correct",
                            "seeded_material_error_detected",
                            "explanation",
                        }
                        or assessment["status"] != "reviewer_challenge"
                        or type(assessment["decision_correct"]) is not bool
                    ):
                        raise ValueError("Invalid challenge judge output")
                    judge_status = "assessed"
                except (ValueError, KeyError, TypeError):
                    assessment = None
                    judge_status = "invalid_grader_output"
                challenge.append(
                    {
                        "case": variant,
                        "expected_decision": expected,
                        "valid_output": out["valid_output"],
                        "decision_matches_source_audit": review.get("decision")
                        == expected
                        if out["valid_output"]
                        else None,
                        "seeded_issue_flagged": any(
                            i["severity"] in {"major", "critical"}
                            and ("C2" if variant == "unsupported_strategy" else "C1")
                            in i["affected_claim_ids"]
                            for i in review.get("flagged_issues", [])
                        )
                        if expected == "FAIL"
                        else None,
                        "independent_assessment": assessment,
                        "grading_status": judge_status,
                        "trace_id": row["trace_id"],
                        "approval": "pending_human",
                    }
                )
                continue
            case = json.loads(
                (root / "cases" / f"{q}-r{rep}" / "case.json").read_text()
            )
            cases[f"{q}-r{rep}"] = case
            if variant == "comparison":
                value = row.get(judge_name + "/value")
                try:
                    assessment = validate_recovery_grade(
                        json.loads(value["assessment_json"]), case, keys[q]
                    )
                    recovery.append(
                        {
                            **identity,
                            "status": "assessed",
                            "assessment": assessment,
                            "trace_id": row["trace_id"],
                            "approval": "pending_human",
                        }
                    )
                except (ValueError, KeyError, TypeError):
                    recovery.append(
                        {
                            **identity,
                            "status": "invalid_grader_output",
                            "assessment": None,
                            "trace_id": row["trace_id"],
                            "approval": "pending_human",
                        }
                    )
                continue
            graded.append(
                {
                    **identity,
                    "trace_id": row["trace_id"],
                    **parse_grade(row, case[variant], keys[q], judge_name),
                }
            )
        save(root / "independent_grades.json", graded)
        save(root / "paired_summary.json", paired_summary(graded, cases))
        save(root / "recovery_defect_changes.json", recovery)
        valid = [r for r in challenge if r["valid_output"]]
        confusion = {
            "true_positive_fail": sum(
                r["expected_decision"] == "FAIL" and r["decision_matches_source_audit"]
                for r in valid
            ),
            "false_negative_pass": sum(
                r["expected_decision"] == "FAIL"
                and not r["decision_matches_source_audit"]
                for r in valid
            ),
            "false_positive_fail": sum(
                r["expected_decision"] == "PASS"
                and not r["decision_matches_source_audit"]
                for r in valid
            ),
            "true_negative_pass": sum(
                r["expected_decision"] == "PASS" and r["decision_matches_source_audit"]
                for r in valid
            ),
            "invalid_review_outputs": len(challenge) - len(valid),
        }
        save(
            root / "reviewer_challenge_summary.json",
            {
                "cases": challenge,
                "decision_confusion_counts": confusion,
                "limitations": "Five source-audited fixtures are a smoke test, not calibrated production precision/recall. FAIL may concern an unrelated issue; inspect independent seeded-error assessments. Human adjudication remains pending.",
            },
        )
        save(
            root / "evaluation_manifest.json",
            {
                "mlflow_run_id": result.run_id,
                "dataset_id": ds.dataset_id,
                "trace_count": len(
                    mlflow.search_traces(experiment_ids=[experiment.experiment_id])
                ),
                "approval": "pending_human",
                "status": "pilot_completed",
            },
        )
        save(root / "actual_usage.json", totals(meter.rows()))
        print("SAVED paired results and Reviewer challenge:", root, flush=True)
    finally:
        save(root / "actual_usage.json", totals(meter.rows()))
        meter.uninstall()
        mlflow.flush_trace_async_logging()


if __name__ == "__main__":
    main()
