"""Additive, deterministic one-cycle V2 controller. V1 and all agents remain unchanged."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from . import pipeline as v1
from .agents.researcher import researcher, ResearcherLimits
from .agents.analyst import analyst, _validate_report
from .agents.reviewer import reviewer, _validate_review, report_fingerprint
from .config import load_config, use_settings
from .dataset import load_dataset
from .evaluation_contracts import IndependentEvaluation
from .persistence import RunStore, dataset_fingerprints, prompt_fingerprints, render_report, validate_run_id, verify_manifest
from .recovery import POLICY_VERSION, select_action, targeted_research_plan, revision_plan, merge_evidence, claim_changes
from .retrieval import RetrievalStore
from .schemas import EvidencePack, CompetitiveReport, ReviewResult, RunResult, TotalMetrics
from .telemetry import PricingConfig, capture_stage, stage_metrics, sanitized_error, utc_now
from .v2_baseline import load_baseline, read_json, safe_manifest_names


@dataclass
class V2StageFailure(Exception):
    error: dict


def usage_totals(stages, latency):
    """Empty means no calls were executed (known zero); missing measurements stay null."""
    fields = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens", "llm_requests",
              "retrieval_tool_calls", "full_record_retrievals", "search_calls", "pricing_history_calls", "estimated_cost_usd")
    result = {}
    for name in fields:
        values = [getattr(stage, name) for stage in stages]
        result[name] = sum(values) if all(value is not None for value in values) else None
    result["latency_seconds"] = latency
    return result


def combine_totals(initial, incremental):
    return {name: None if initial.get(name) is None or incremental.get(name) is None else initial[name] + incremental[name]
            for name in incremental}


def _copy_initial(store, directory):
    names = safe_manifest_names(directory, completed=False)
    verify_manifest(directory)
    mapping = {}
    for name in sorted(names | {"manifest.json"}):
        copied = "initial_" + name
        store.write_text(copied, (directory / name).read_text(encoding="utf-8"))
        original_hash = sha256((directory / name).read_bytes()).hexdigest()
        if store.fingerprints[copied] != original_hash:
            raise ValueError("Initial snapshot copy differs from baseline bytes")
        mapping[copied] = original_hash
    return mapping


async def run_pipeline_v2(question=None, as_of=None, run_id=None, *, baseline_dir=None,
                          question_id=None, settings=None, pricing=None, recovery_limits=None):
    """Paid-capable explicit entry point; CLI separately requires --allow-api.

    From scratch delegates initial execution to v1.run_pipeline. Paired mode never
    calls Planner or reruns the initial four stages. Only one correction cycle exists.
    """
    settings = settings or load_config()
    run_id = validate_run_id(run_id if run_id is not None else "v2-" + uuid4().hex)
    cutoff = as_of if as_of is not None else settings.evaluation_cutoff
    mode = "paired_continuation" if baseline_dir is not None else "from_scratch"
    baseline = None
    if baseline_dir is not None:
        if question is not None or question_id is not None:
            raise ValueError("Paired continuation takes its question and ID only from the baseline")
        baseline = load_baseline(baseline_dir, settings, cutoff)
        output_root = settings.runs_dir.expanduser().resolve()
        if output_root == baseline.directory or baseline.directory in output_root.parents:
            raise ValueError("V2 outputs must not be created inside the read-only V1 baseline")
    else:
        cutoff = v1.validate_request(question, cutoff)
    rates = pricing or (PricingConfig.model_validate(baseline.metadata["pricing"]) if baseline else PricingConfig(model=settings.openai_model))
    if rates.model != settings.openai_model:
        raise ValueError("Pricing model differs from configured model")
    limits = recovery_limits or ResearcherLimits(max_tool_calls=20, max_search_results=10, max_full_records=15, max_evidence_items=15)
    limits = ResearcherLimits.model_validate(limits.model_dump())
    output_root = settings.runs_dir.expanduser().resolve()
    if settings.data_dir.resolve() == output_root or settings.data_dir.resolve() in output_root.parents:
        raise ValueError("V2 outputs must be separate from public benchmark data")
    if {"private", "evaluation_private"} & {part.casefold() for part in output_root.parts}:
        raise ValueError("V2 outputs must not use private evaluation storage")
    store = RunStore(settings.runs_dir, run_id, secrets=(settings.openai_api_key,))
    observed_clock = perf_counter()
    metadata = {"schema_version": "v2_metadata.v1", "run_id": run_id, "pipeline_version": "v2_bounded",
                "mode": mode, "started_at": utc_now().isoformat(), "model": settings.openai_model,
                "recovery_policy_version": POLICY_VERSION, "max_recovery_cycles": 1,
                "recovery_limits": limits.model_dump(), "independent_evaluation_status": "not_evaluated"}
    store.write_json("started.json", metadata)
    if baseline is None:
        initial_id = "v1-initial-" + uuid4().hex
        try:
            initial = await v1.run_pipeline(question, cutoff, initial_id, question_id=question_id,
                                            experiment_id=run_id, settings=settings, pricing=rates)
        except Exception as exc:
            error = sanitized_error(exc, "initialization", secrets=(settings.openai_api_key,))
            store.write_json("error.json", error)
            store.write_json("metadata.json", {**metadata, "execution_status": "execution_failed", "initial_run_id": initial_id,
                                               "recovery_state": "skipped", "reason": "initial_pipeline_exception", "error": error})
            store.finalize()
            raise V2StageFailure(error) from None
        initial_dir = Path(initial.artifact_directory)
        if initial.execution_status == "execution_failed":
            _copy_initial(store, initial_dir)
            inherited = usage_totals(initial.stage_metrics, initial.total_metrics.latency_seconds)
            store.write_json("metrics.json", {"initial_v1": inherited, "incremental_total": usage_totals([], 0),
                                               "final_totals": inherited, "initial_execution_failed": True})
            result = initial.model_copy(deep=True, update={"run_id": run_id, "configuration": "v2", "artifact_directory": str(store.directory)})
            store.write_json("run_result.json", result)
            store.write_json("metadata.json", {**metadata, "execution_status": "execution_failed", "baseline_run_id": initial_id,
                "recovery_state": "skipped", "reason": "initial_execution_failed", "finished_at": utc_now().isoformat()})
            store.finalize()
            return result
        baseline = load_baseline(initial_dir, settings, cutoff)
    initial_hashes = _copy_initial(store, baseline.directory)
    store.write_json("baseline_reference.json", baseline.reference)
    plan, evidence = baseline.plan.model_copy(deep=True), baseline.evidence.model_copy(deep=True)
    initial_report, initial_review = baseline.report.model_copy(deep=True), baseline.review.model_copy(deep=True)
    action = select_action(initial_review)
    store.write_json("recovery_routing.json", action.as_dict())
    store.write_json("recovery_feedback.json", {"trigger_issue_ids": list(action.issue_ids),
        "issues": [issue.model_dump(mode="json") for issue in initial_review.flagged_issues if issue.issue_id in action.issue_ids]})
    recovery_metrics, recovery_events = [], []
    error, final_report, final_review = None, initial_report, initial_review
    state, attempted = "skipped", False
    reason = action.reason
    recovery_elapsed = 0.0
    merged = evidence

    async def execute(name, filename, schema, operation):
        nonlocal attempted, recovery_events
        if dataset_fingerprints(settings.data_dir) != baseline.metadata["dataset"] or prompt_fingerprints() != baseline.metadata["prompt_fingerprints"]:
            raise ValueError("Dataset or prompts changed during recovery")
        verify_manifest(baseline.directory)
        started, clock, failure, output = utc_now(), perf_counter(), None, None
        attempted = True
        with capture_stage() as captured:
            try:
                output = await operation()
                if not isinstance(output, schema):
                    raise ValueError("Recovery returned wrong output type")
                output = schema.model_validate(output.model_dump())
                if name == "researcher":
                    if output.original_question != plan.original_question or output.as_of != plan.as_of:
                        raise ValueError("Recovery Researcher changed question or cutoff")
                elif name == "analyst":
                    _validate_report(output, plan, merged)
                else:
                    _validate_review(output, plan, merged, final_report, report_fingerprint(final_report))
                store.write_json(filename, output)
            except Exception as exc:
                failure = sanitized_error(exc, name, secrets=(settings.openai_api_key,))
            finally:
                if name == "researcher":
                    if not captured.events and output is not None:
                        captured.events.extend(json.loads(entry) for entry in output.retrieval_log)
                    recovery_events = list(captured.events)
                    store.write_json("recovery_retrieval_logs.json", recovery_events)
                measurement = stage_metrics(name, settings.openai_model, started, utc_now(), perf_counter()-clock,
                    captured, "failed" if failure else "success", rates, failure)
                measurement.stage_name = "recovery_" + name
                recovery_metrics.append(measurement)
                store.write_json("recovery_" + name + "_metrics.json", measurement)
        if failure:
            raise V2StageFailure(failure)
        return output

    if action.route != "skip":
        cycle_clock = perf_counter()
        state = "failed"
        try:
            with use_settings(settings):
                if action.route == "research_then_revise":
                    target = targeted_research_plan(plan, evidence, initial_report, initial_review, action)
                    store.write_json("recovery_plan.json", target)
                    with RetrievalStore(load_dataset(settings.data_dir)) as service:
                        additional = await execute("researcher", "recovery_evidence_pack.json", EvidencePack,
                            lambda: researcher(target.model_copy(deep=True), service, limits=limits))
                    merged = merge_evidence(evidence, additional)
                store.write_json("merged_evidence_pack.json", merged)
                context = revision_plan(plan, initial_report, initial_review, action)
                store.write_json("revision_plan.json", context)
                final_report = await execute("analyst", "revised_report.json", CompetitiveReport,
                    lambda: analyst(context.model_copy(deep=True), merged.model_copy(deep=True)))
                final_review = None  # Initial review must not be misattributed to revised text.
                final_review = await execute("reviewer", "second_review.json", ReviewResult,
                    lambda: reviewer(plan.model_copy(deep=True), merged.model_copy(deep=True), final_report.model_copy(deep=True)))
                state, reason = "completed", "one_cycle_completed_stop_regardless_of_decision"
        except V2StageFailure as exc:
            error, reason = exc.error, "recovery_stage_exception_no_retry"
        except Exception as exc:
            error, reason = sanitized_error(exc, "initialization", secrets=(settings.openai_api_key,)), "recovery_controller_exception_no_retry"
        finally:
            recovery_elapsed = perf_counter() - cycle_clock
    if error:
        store.write_json("error.json", error)
    delta = claim_changes(initial_report, final_report)
    store.write_json("claim_changes.json", delta)
    original_ids = {item.record_id for item in evidence.evidence_items}
    initial_events = [json.loads(entry) for entry in evidence.retrieval_log]
    initial_seen = {rid for event in initial_events for rid in event.get("record_ids", [])}
    initial_full = {rid for event in initial_events if event.get("tool")=="get_record" and event.get("status")=="ok" for rid in event.get("record_ids", [])}
    recovery_seen = {rid for event in recovery_events for rid in event.get("record_ids", [])}
    recovery_full = {rid for event in recovery_events if event.get("tool")=="get_record" and event.get("status")=="ok" for rid in event.get("record_ids", [])}
    store.write_json("evidence_changes.json", {"newly_discovered_record_ids": sorted(recovery_seen-initial_seen),
        "newly_fetched_full_record_ids": sorted(recovery_full-initial_full),
        "new_evidence_ids": sorted({item.record_id for item in merged.evidence_items}-original_ids),
        "initial_evidence_ids": sorted(original_ids), "final_evidence_ids": sorted({item.record_id for item in merged.evidence_items})})
    store.write_json("final_report.json", final_report)
    if final_review is not None:
        store.write_json("final_review.json", final_review)
    store.write_text("report.md", render_report(final_report, final_review))
    inherited_latency = baseline.result.total_metrics.latency_seconds
    inherited = usage_totals(baseline.result.stage_metrics, inherited_latency)
    work_stages = [stage for stage in recovery_metrics if stage.stage_name != "recovery_reviewer"]
    review_stages = [stage for stage in recovery_metrics if stage.stage_name == "recovery_reviewer"]
    incremental = usage_totals(recovery_metrics, recovery_elapsed)
    total = combine_totals(inherited, incremental)
    store.write_json("metrics.json", {"schema_version": "v2_metrics.v1", "initial_v1": inherited,
        "initial_v1_source": "inherited_archive_not_new_calls" if mode=="paired_continuation" else "executed_via_unchanged_v1",
        "recovery_work": usage_totals(work_stages, sum(stage.latency_seconds for stage in work_stages)),
        "re_review": usage_totals(review_stages, sum(stage.latency_seconds for stage in review_stages)),
        "incremental_total": incremental, "final_totals": total,
        "executed_this_invocation": incremental if mode=="paired_continuation" else total,
        "recovery_stages": [stage.model_dump(mode="json") for stage in recovery_metrics],
        "initial_pricing": baseline.metadata.get("pricing"), "incremental_pricing": rates.model_dump(mode="json"),
        "note": "Re-review is a subset of incremental_total, not an additional cost. Unknown usage/cost stays null."})
    status = "execution_failed" if error else "completed" if final_review and final_review.decision=="PASS" else "completed_review_failed"
    metadata.update(baseline_run_id=baseline.result.run_id, baseline_directory=str(baseline.directory),
        experiment_id=baseline.result.experiment_id,
        openai_agents_sdk_version=baseline.metadata["openai_agents_sdk_version"], question=plan.original_question,
        question_id=baseline.result.question_id, as_of=plan.as_of.isoformat(), dataset=baseline.metadata["dataset"],
        prompt_fingerprints=baseline.metadata["prompt_fingerprints"], comparison_key=baseline.reference.comparison_key,
        baseline_manifest_sha256=baseline.reference.manifest_sha256, initial_artifact_fingerprints=initial_hashes,
        initial_reviewer_decision=initial_review.decision, final_reviewer_decision=final_review.decision if final_review else None,
        recovery_attempted=attempted, recovery_state=state, recovery_reason=reason, trigger_issue_ids=list(action.issue_ids),
        execution_status=status, finished_at=utc_now().isoformat(), observed_controller_wall_clock=perf_counter()-observed_clock,
        inherited_latency_seconds=inherited_latency, incremental_latency_seconds=recovery_elapsed,
        code_compatibility_note="Historical runs lack agent-source hashes; dataset, prompts, model, cutoff and recorded SDK are verified")
    store.write_json("metadata.json", metadata)
    result = RunResult(run_id=run_id, experiment_id=baseline.result.experiment_id, question_id=baseline.result.question_id,
        configuration="v2", model=settings.openai_model, as_of=plan.as_of, execution_status=status,
        artifact_directory=str(store.directory), stage_metrics=baseline.result.stage_metrics+recovery_metrics,
        final_report=final_report, review_result=final_review,
        total_metrics=TotalMetrics(input_tokens=total["input_tokens"], output_tokens=total["output_tokens"],
                                   estimated_cost_usd=total["estimated_cost_usd"], latency_seconds=total["latency_seconds"]))
    store.write_json("evaluation_pending.json", IndependentEvaluation(run_id=run_id))
    store.write_json("run_result.json", result)
    store.finalize()
    return result
