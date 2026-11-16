"""Explicit foreground CLI: dry-run is offline; live runs require --allow-api."""
import argparse
import asyncio
import json

from .config import load_config
from .dataset import _date, load_dataset
from .persistence import dataset_fingerprints, prompt_fingerprints, validate_run_id
from .pipeline import run_pipeline, validate_request


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ci-agents")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the fixed Version 1 pipeline")
    choice = run.add_mutually_exclusive_group(required=True)
    choice.add_argument("--question-id")
    choice.add_argument("--question")
    run.add_argument("--as-of")
    run.add_argument("--run-id")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--allow-api", action="store_true")
    pilot = sub.add_parser("pilot", help="Preflight or explicitly execute Q01/Q15 sequentially")
    pilot.add_argument("--dry-run", action="store_true")
    pilot.add_argument("--allow-api", action="store_true")
    pilot.add_argument("--question-ids", nargs="+", default=["Q01", "Q15"])
    pilot.add_argument("--as-of")
    pilot.add_argument("--output-dir")
    pilot.add_argument("--max-total-tokens", type=int)
    pilot.add_argument("--max-estimated-cost", type=float)
    v2 = sub.add_parser("v2", help="One-cycle V2, fresh or paired with a read-only V1 baseline")
    source = v2.add_mutually_exclusive_group(required=True)
    source.add_argument("--question-id")
    source.add_argument("--question")
    source.add_argument("--baseline-dir")
    v2.add_argument("--as-of")
    v2.add_argument("--run-id")
    v2.add_argument("--dry-run", action="store_true")
    v2.add_argument("--allow-api", action="store_true")
    v2.add_argument("--recovery-max-tool-calls", type=int, default=20)
    v2.add_argument("--recovery-max-full-records", type=int, default=15)
    v2.add_argument("--recovery-max-evidence-items", type=int, default=15)
    v2.add_argument("--legacy-policy", action="store_true", help="Use original V2 behavior without reliability improvements")
    v2.add_argument("--initial-max-tool-calls", type=int, default=60)
    v2.add_argument("--max-total-tokens", type=int)
    v2.add_argument("--max-estimated-cost", type=float)
    v2.add_argument("--max-seconds", type=float)
    v2.add_argument("--max-model-requests", type=int)
    v2.add_argument("--reserve-final-tokens", type=int, default=0)
    v2.add_argument("--max-output-tokens", type=int, default=8000)
    v2.add_argument("--input-usd-per-million", type=float)
    v2.add_argument("--output-usd-per-million", type=float)
    v2.add_argument("--pricing-source")
    v2.add_argument("--pricing-date")
    pdf = sub.add_parser("report", help="Generate a two-page PDF offline from a completed run")
    pdf.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        from .reporting import generate_pdf
        try:
            print(str(generate_pdf(args.run_dir)))
            return 0
        except Exception as exc:
            parser.exit(2, f"PDF generation failed ({type(exc).__name__}); check completed run artifacts and output permissions.\n")
    if not args.dry_run and not args.allow_api:
        parser.error("Live execution is blocked; use --dry-run or explicitly supply --allow-api")
    if args.command == "pilot":
        return _pilot_main(parser, args)
    if args.command == "v2":
        return _v2_main(parser, args)
    try:
        settings = load_config()
        dataset = load_dataset(settings.data_dir)
        if args.question_id:
            matches = [q for q in dataset.questions if q["question_id"] == args.question_id]
            if len(matches) != 1:
                raise ValueError("Unknown public question ID")
            question = matches[0]["user_question"]
        else:
            question = args.question
        cutoff = _date(args.as_of) if args.as_of else settings.evaluation_cutoff
        validate_request(question, cutoff)
        if args.run_id is not None:
            validate_run_id(args.run_id)
            if (settings.runs_dir / args.run_id).exists():
                raise ValueError("Run ID already exists")
        if args.dry_run:
            # No artifacts, retrieval index, SDK runner, or API clients are created.
            print(json.dumps({"dry_run": True, "question_id": args.question_id,
                "as_of": cutoff.isoformat(), "model": settings.openai_model,
                "record_count": len(dataset.records), "question_count": len(dataset.questions),
                "dataset_sha256": dataset_fingerprints(settings.data_dir)["sha256"],
                "prompt_fingerprints": prompt_fingerprints()}))
            return 0
        result = asyncio.run(run_pipeline(question, cutoff, args.run_id,
                                          question_id=args.question_id, settings=settings))
        print(json.dumps({"run_id": result.run_id, "status": result.execution_status,
                          "reviewer_decision": result.review_result.decision if result.review_result else None}))
        return 1 if result.execution_status == "execution_failed" else 0
    except Exception as exc:
        # Never print raw configuration, SDK exception messages, or authentication data.
        parser.exit(2, f"Run could not proceed ({type(exc).__name__}); check configuration, public inputs, cutoff and run ID.\n")


def _pilot_main(parser, args):
    from .config import ConfigurationError
    from .pilot import PilotPreflightError, preflight, run_pilot, validate_budgets
    from .telemetry import redact_text
    try:
        try:
            settings = load_config(require_api_key=not args.dry_run)
        except ConfigurationError:
            raise PilotPreflightError(
                "Configuration invalid: check OPENAI_MODEL, DATA_DIR, RUNS_DIR and EVALUATION_CUTOFF; "
                "OPENAI_API_KEY is required only for live execution") from None
        ids = [qid for group in args.question_ids for qid in group.split(",")]
        prepared = preflight(settings, ids, args.as_of, args.output_dir, live=not args.dry_run)
        validate_budgets(prepared.pricing, args.max_total_tokens, args.max_estimated_cost)
        summary = {"dry_run": args.dry_run, **prepared.summary()}
        print(redact_text(json.dumps(summary), (settings.openai_api_key,)), flush=True)
        if args.dry_run:
            return 0
        result = asyncio.run(run_pilot(prepared, allow_api=args.allow_api,
            max_total_tokens=args.max_total_tokens, max_estimated_cost=args.max_estimated_cost))
        print(redact_text(json.dumps(result), (settings.openai_api_key,)))
        return 1 if any(row["execution_status"] in {"execution_failed", "not_started_budget"} or row["evaluation_ready"] is False
                        for row in result["rows"]) else 0
    except PilotPreflightError as exc:
        parser.exit(2, f"Pilot preflight failed: {exc}\n")
    except Exception as exc:
        parser.exit(2, f"Pilot could not proceed ({type(exc).__name__}); inspect output permissions and public inputs.\n")


def _v2_main(parser, args):
    from .agents.researcher import ResearcherLimits
    from .pipeline_v2 import run_pipeline_v2
    from .v2_baseline import load_baseline
    from .recovery import select_action
    from .telemetry import redact_text
    from .telemetry import PricingConfig
    from .workflow_runtime import WorkflowPolicy, RunBudget, WorkflowRuntime
    try:
        settings = load_config(require_api_key=not args.dry_run)
        cutoff = _date(args.as_of) if args.as_of else settings.evaluation_cutoff
        limits = ResearcherLimits(max_tool_calls=args.recovery_max_tool_calls,
            max_full_records=args.recovery_max_full_records, max_evidence_items=args.recovery_max_evidence_items,
            max_search_results=10)
        initial_limits = ResearcherLimits(max_tool_calls=args.initial_max_tool_calls)
        policy = WorkflowPolicy(max_output_tokens=args.max_output_tokens)
        budget = RunBudget(max_total_tokens=args.max_total_tokens, max_estimated_cost=args.max_estimated_cost,
            max_seconds=args.max_seconds, max_model_requests=args.max_model_requests,
            reserve_final_tokens=args.reserve_final_tokens)
        if args.legacy_policy and any((args.max_total_tokens, args.max_estimated_cost, args.max_seconds, args.max_model_requests)):
            raise ValueError('Run-wide budgets require improved V2')
        rates = None
        if args.input_usd_per_million is not None or args.output_usd_per_million is not None:
            rates = PricingConfig(model=settings.openai_model, source=args.pricing_source, source_date=args.pricing_date,
                input_usd_per_million=args.input_usd_per_million, output_usd_per_million=args.output_usd_per_million)
        if args.run_id is not None:
            validate_run_id(args.run_id)
            if (settings.runs_dir / args.run_id).exists():
                raise ValueError("Output run ID already exists")
        question, qid, preview = None, None, None
        if args.baseline_dir:
            baseline = load_baseline(args.baseline_dir, settings, cutoff)
            if rates is None: rates = PricingConfig.model_validate(baseline.metadata['pricing'])
            preview = {"baseline_run_id": baseline.result.run_id, **select_action(baseline.review).as_dict()}
        else:
            data = load_dataset(settings.data_dir)
            if args.question_id:
                matches = [row for row in data.questions if row["question_id"] == args.question_id]
                if len(matches) != 1:
                    raise ValueError("Unknown public question ID")
                question, qid = matches[0]["user_question"], args.question_id
            else:
                question = args.question
            validate_request(question, cutoff)
            prompt_fingerprints()
        if not args.legacy_policy:
            WorkflowRuntime(policy, budget, rates or PricingConfig(model=settings.openai_model))
        print(redact_text(json.dumps({"architecture": "v2_bounded", "dry_run": args.dry_run,
            "model": settings.openai_model, "as_of": cutoff.isoformat(), "baseline": preview,
            "recovery_limits": limits.model_dump(), "max_recovery_cycles": 1,
            "workflow_policy": None if args.legacy_policy else policy.model_dump(),
            "run_budget": None if args.legacy_policy else budget.model_dump(),
            "initial_research_limits": initial_limits.model_dump()}), (settings.openai_api_key,)), flush=True)
        if args.dry_run:
            return 0
        result = asyncio.run(run_pipeline_v2(question, cutoff, args.run_id, baseline_dir=args.baseline_dir,
                            question_id=qid, settings=settings, recovery_limits=limits, pricing=rates,
                            initial_research_limits=initial_limits, enable_improvements=not args.legacy_policy,
                            policy=policy, budget=None if args.legacy_policy else budget))
        print(json.dumps({"run_id": result.run_id, "status": result.execution_status,
                          "final_reviewer_decision": result.review_result.decision if result.review_result else None}))
        return 1 if result.execution_status == "execution_failed" else 0
    except Exception as exc:
        parser.exit(2, f"V2 could not proceed ({type(exc).__name__}); check baseline compatibility, public inputs, cutoff and new run ID. No retry attempted.\n")


if __name__ == "__main__":
    raise SystemExit(main())
