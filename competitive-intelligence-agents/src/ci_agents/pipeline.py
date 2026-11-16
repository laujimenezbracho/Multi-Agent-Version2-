"""Version 1 fixed-order controller. No retries, corrections, or independent grading."""

from datetime import datetime
import asyncio
from importlib.metadata import version
import json
import platform
from time import perf_counter
from uuid import uuid4

from .agents.planner import planner
from .agents.researcher import researcher
from .agents.analyst import analyst
from .agents.reviewer import reviewer
from .config import Settings, load_config, use_settings
from .dataset import _date, load_dataset
from .evaluation_contracts import IndependentEvaluation
from .persistence import (
    RunStore,
    dataset_fingerprints,
    fingerprint,
    prompt_fingerprints,
    render_report,
    validate_run_id,
)
from .retrieval import RetrievalStore
from .schemas import (
    CompetitiveReport,
    EvidencePack,
    ResearchPlan,
    ReviewResult,
    RunResult,
)
from .telemetry import (
    PricingConfig,
    aggregate_metrics,
    capture_stage,
    retrieval_summary,
    sanitized_error,
    stage_metrics,
    utc_now,
)
from .workflow_runtime import active_runtime, code_fingerprints, source_snapshot
from .source_observations import validate_evidence_items

OUTPUTS = (
    ("planner", "plan.json", ResearchPlan),
    ("researcher", "evidence_pack.json", EvidencePack),
    ("analyst", "report.json", CompetitiveReport),
    ("reviewer", "review.json", ReviewResult),
)


def validate_request(question, as_of):
    if not isinstance(question, str) or not question.strip() or len(question) > 10000:
        raise ValueError("question must be non-empty text of at most 10000 characters")
    if not isinstance(as_of, datetime):
        raise ValueError("as_of must be a timezone-aware datetime")
    return _date(as_of)


async def run_pipeline(
    question: str,
    as_of: datetime,
    run_id: str | None = None,
    *,
    question_id: str | None = None,
    experiment_id: str | None = None,
    settings: Settings | None = None,
    pricing: PricingConfig | None = None,
    researcher_limits=None,
) -> RunResult:
    """Execute each agent once in order. Explicit invocation can make paid calls.

    CLI requires --allow-api. Execution failures are returned as failed RunResults;
    invalid requests, existing run IDs, and persistence failures raise before/after
    recording everything the filesystem permits. No background execution is used.
    """
    cutoff = validate_request(question, as_of)
    settings = settings or load_config()
    run_id = validate_run_id(run_id if run_id is not None else uuid4().hex)
    if question_id is not None and (
        not isinstance(question_id, str) or not question_id.strip()
    ):
        raise ValueError("question_id must be non-empty text or None")
    if experiment_id is not None and (
        not isinstance(experiment_id, str) or not experiment_id.strip()
    ):
        raise ValueError("experiment_id must be non-empty text or None")
    pricing = pricing or PricingConfig(model=settings.openai_model)
    if researcher_limits is not None:
        from .agents.researcher import ResearcherLimits

        researcher_limits = ResearcherLimits.model_validate(
            researcher_limits.model_dump()
        )
    if pricing.model != settings.openai_model:
        raise ValueError("Pricing model must match configured model")
    store = RunStore(settings.runs_dir, run_id, secrets=(settings.openai_api_key,))
    started = utc_now()
    clock = perf_counter()
    metadata = {
        "schema_version": "run_metadata.v1",
        "run_id": run_id,
        "experiment_id": experiment_id or run_id,
        "question_id": question_id,
        "question": question,
        "as_of": cutoff.isoformat(),
        "started_at": started.isoformat(),
        "pipeline_version": "v1_fixed",
        "model": settings.openai_model,
        "openai_agents_sdk_version": version("openai-agents"),
        "python_version": platform.python_version(),
        "dataset": None,
        "prompt_fingerprints": None,
        "comparison_key": fingerprint(
            {"question": question, "as_of": cutoff.isoformat()}
        ),
        "pricing": pricing.model_dump(mode="json"),
        "baseline_run_id": run_id,
        "hash_definition": "Artifact SHA-256 hashes use exact persisted bytes; comparison key uses canonical JSON",
        "independent_evaluation_status": "not_evaluated",
        "generation_settings": {
            "reasoning_effort": settings.reasoning_effort,
            "max_output_tokens": settings.max_output_tokens,
        },
    }
    if researcher_limits is not None:
        metadata["researcher_limits"] = researcher_limits.model_dump()
    store.write_json("started.json", metadata)
    runtime = active_runtime()
    if runtime:
        metadata["workflow_policy"] = runtime.policy.model_dump()
        snapshot = source_snapshot()
        metadata["agent_code_fingerprints"] = snapshot["sha256"]
        store.write_json("agent_source.json", snapshot)
        store.write_json("workflow_policy.json", runtime.policy)
    outputs, metrics, retrieval_events = {}, [], []
    service, error, status = None, None, "execution_failed"
    current = "initialization"
    try:
        metadata["dataset"] = dataset_fingerprints(settings.data_dir)
        metadata["prompt_fingerprints"] = prompt_fingerprints()
        dataset = load_dataset(settings.data_dir)
        public_records = {row["record_id"]: row for row in dataset.records}
        if dataset_fingerprints(settings.data_dir) != metadata["dataset"]:
            raise ValueError("Public dataset changed while loading")
        if question_id is not None:
            matches = [q for q in dataset.questions if q["question_id"] == question_id]
            if len(matches) != 1 or matches[0]["user_question"] != question:
                raise ValueError(
                    "question_id and original question must match the public benchmark"
                )
        service = RetrievalStore(dataset)
        with use_settings(settings):
            for name, filename, schema in OUTPUTS:
                current = name
                stage_start, stage_clock = utc_now(), perf_counter()
                failure = None
                with capture_stage() as captured:
                    try:
                        if runtime:
                            runtime.check(name.title())
                        if (
                            dataset_fingerprints(settings.data_dir)
                            != metadata["dataset"]
                            or prompt_fingerprints() != metadata["prompt_fingerprints"]
                        ):
                            raise ValueError(
                                "Public dataset or agent prompts changed during the run"
                            )
                        if (
                            runtime
                            and code_fingerprints()
                            != metadata["agent_code_fingerprints"]
                        ):
                            raise ValueError("Agent source changed during the run")
                        if name == "planner":
                            output = await planner(question, cutoff)
                        elif name == "researcher":
                            if researcher_limits is None:
                                output = await researcher(
                                    outputs["planner"].model_copy(deep=True), service
                                )
                            else:
                                output = await researcher(
                                    outputs["planner"].model_copy(deep=True),
                                    service,
                                    limits=researcher_limits,
                                )
                        elif name == "analyst":
                            output = await analyst(
                                outputs["planner"].model_copy(deep=True),
                                outputs["researcher"].model_copy(deep=True),
                            )
                        else:
                            output = await reviewer(
                                outputs["planner"].model_copy(deep=True),
                                outputs["researcher"].model_copy(deep=True),
                                outputs["analyst"].model_copy(deep=True),
                            )
                        if not isinstance(output, schema):
                            raise ValueError(
                                f"{name} returned the wrong structured output type"
                            )
                        output = schema.model_validate(output.model_dump())
                        if name == "researcher":
                            events = captured.events or [
                                json.loads(entry) for entry in output.retrieval_log
                            ]
                            full_ids = {
                                rid
                                for event in events
                                if event.get("tool") == "get_record"
                                and event.get("status") == "ok"
                                for rid in event.get("record_ids", [])
                            }
                            validate_evidence_items(
                                output.evidence_items, public_records, cutoff, full_ids
                            )
                        if name in {"planner", "researcher"}:
                            if (
                                output.original_question != question
                                or output.as_of != cutoff
                            ):
                                raise ValueError(
                                    "Stage output changed the question or cutoff"
                                )
                        elif name == "analyst" and (
                            output.question != question or output.as_of != cutoff
                        ):
                            raise ValueError("Report changed the question or cutoff")
                        outputs[name] = output
                        store.write_json(filename, output)
                    except asyncio.CancelledError as exc:
                        failure = sanitized_error(
                            exc, name, secrets=(settings.openai_api_key,)
                        )
                        raise
                    except Exception as exc:
                        failure = sanitized_error(
                            exc, name, secrets=(settings.openai_api_key,)
                        )
                    finally:
                        finish = utc_now()
                        elapsed = perf_counter() - stage_clock
                        if name == "researcher":
                            # Live observer preserves events even on failure; mocked
                            # agents may supply the same authoritative log in EvidencePack.
                            if not captured.events and name in outputs:
                                for entry in outputs[name].retrieval_log:
                                    captured.events.append(json.loads(entry))
                            retrieval_events = list(captured.events)
                        measurement = stage_metrics(
                            name,
                            settings.openai_model,
                            stage_start,
                            finish,
                            elapsed,
                            captured,
                            "failed" if failure else "success",
                            pricing,
                            failure,
                        )
                        metrics.append(measurement)
                        store.write_json(name + "_metrics.json", measurement)
                        if name == "researcher":
                            store.write_json("retrieval_logs.json", retrieval_events)
                if failure:
                    error = failure
                    break
        if error is None:
            status = (
                "completed"
                if outputs["reviewer"].decision == "PASS"
                else "completed_review_failed"
            )
    except Exception as exc:
        error = sanitized_error(exc, current, secrets=(settings.openai_api_key,))
        status = "execution_failed"
    finally:
        if service is not None:
            service.close()
    if error is not None:
        status = "execution_failed"
        store.write_json(
            "error.json",
            {**error, "stage": current, "timestamp": utc_now().isoformat()},
        )
        store.write_json(
            "diagnostics.json",
            {"schema_version": "run_diagnostics.v1", "failures": [error["diagnostic"]]},
        )
    report = outputs.get("analyst")
    review = outputs.get("reviewer")
    if report is not None:
        store.write_text("report.md", render_report(report, review))
        # Exact claims are also in report.json; context fields unavailable in the
        # existing ReportClaim contract stay unknown, not inferred from citations.
        store.write_json(
            "claim_traceability.json",
            [
                {
                    **claim.model_dump(mode="json"),
                    "relevant_companies": None,
                    "relevant_time_scope": None,
                }
                for claim in report.claims
            ],
        )
    finished = utc_now()
    totals = aggregate_metrics(metrics, perf_counter() - clock)
    selected = (
        [item.record_id for item in outputs["researcher"].evidence_items]
        if "researcher" in outputs
        else []
    )

    def complete_sum(field):
        values = [getattr(stage, field) for stage in metrics]
        return sum(values) if values and all(v is not None for v in values) else None

    store.write_json(
        "metrics.json",
        {
            "schema_version": "run_metrics.v1",
            "stages": [m.model_dump(mode="json") for m in metrics],
            "totals": {
                **totals.model_dump(mode="json"),
                "total_tokens": complete_sum("total_tokens"),
                "llm_requests": complete_sum("llm_requests"),
            },
            "pricing": pricing.model_dump(mode="json"),
            "retrieval": retrieval_summary(retrieval_events, selected),
            "failure_diagnostics": [error["diagnostic"]] if error else [],
            "usage_note": "Null means unavailable; failed SDK calls may have incurred unreported usage. No character-based estimates.",
        },
    )
    metadata.update(
        finished_at=finished.isoformat(),
        execution_status=status,
        reviewer_decision=review.decision if review else None,
        error=error,
        initial_artifact_fingerprints=dict(store.fingerprints),
        initial_report_fingerprint=store.fingerprints.get("report.json"),
        initial_review_fingerprint=store.fingerprints.get("review.json"),
    )
    store.write_json("metadata.json", metadata)
    # This is an empty contract placeholder, never internal reviewer scores or truth.
    store.write_json("evaluation_pending.json", IndependentEvaluation(run_id=run_id))
    result = RunResult(
        run_id=run_id,
        experiment_id=metadata["experiment_id"],
        question_id=question_id,
        configuration="v1",
        model=settings.openai_model,
        as_of=cutoff,
        stage_metrics=metrics,
        final_report=report,
        review_result=review,
        total_metrics=totals,
        execution_status=status,
        artifact_directory=str(store.directory),
    )
    store.write_json("run_result.json", result)
    if runtime:
        store.write_json("workflow_budget.json", runtime.snapshot())
    store.finalize()
    return result
