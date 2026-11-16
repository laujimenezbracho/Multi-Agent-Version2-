# V2 research-readiness code review

Review date: 16 November 2026. Scope: the Planner → Researcher → Analyst →
Reviewer workflow, one-cycle recovery, evidence handoffs, resource accounting,
baseline compatibility, CLI packaging and the native MLflow evaluation runner.
The separate beverage-news CrewAI implementation is outside this review.

## Findings and corrections

| Severity | Finding | Correction and verification |
| --- | --- | --- |
| High | Price-plan normalization could discard an explicit country, exclusion, historical period or single-company restriction and broaden the research question. | Normalization now preserves plans with additional qualifiers or a single named company. Regression cases cover Germany, excluded competitors, a named subset and a historical date. The original simple Q01 normalization remains tested. |
| High | A future-effective replacement could suppress the currently applicable price in numerical aids. | A successor only suppresses an earlier observation when the successor is available and effective at the cutoff and not expired. The future announcement remains visible in source evidence, but is excluded from current arithmetic. |
| High | Structured price fields were not independently revalidated at every controller handoff and baseline load. Correct IDs alone could therefore accompany an altered amount or unit. | A shared source validator checks full-retrieval provenance, cutoff, source metadata and structured prices before initial/recovery analysis and during baseline loading. Tests deliberately alter a price and confirm downstream agents are not called. A rehashed baseline is still rejected when its structured value contradicts the public source. |
| High | Default SDK zero usage, partial usage or inconsistent token totals could appear as a valid zero-cost or incorrectly priced request. Some custom-runner fallback paths could retain a previous known subtotal as though usage were complete. | Runtime and stage telemetry share a completeness check. Missing/invalid usage remains unknown, known subtotals remain separately identifiable, and further token/cost-budgeted work is blocked when usage cannot establish compliance. Deterministic paths that make no model call still record known zero usage. |
| Medium | Cancellation could pass through a `finally` block with the stage status still set to success. | Both initial and recovery controllers record a failed stage with `CancelledError` and re-raise cancellation. Partial artifacts remain partial; a completed run is not fabricated. |
| Medium | Paired continuation did not record/check common reasoning-effort and output-limit settings. | New runs record generation settings. Continuation rejects mismatches when those fields are available. Historical runs lacking them cannot retrospectively prove the settings used. |
| Medium | A completed evaluation could be rerun in place, replacing its visible grades; cached predictions could be edited and silently reused. | Completed evaluations require a new output directory. Resumed workflow predictions must match sealed reports, metrics and execution status. Reviewer challenge caches carry an integrity hash. Invalid independent grades remain missing; the review does not change judgments into passes. |
| Medium | Source hashes identified a version but could not reconstruct its code after later edits. Imported evaluator helpers were also missing from the frozen script inventory. | New improved runs save manifest-covered `agent_source.json` with source text and per-file hashes. Controllers detect source changes between stages. Evaluations archive their script/helper sources and include helper hashes in the frozen plan. No environment files or private answer keys are included in the agent source snapshot. |
| Medium | Documented `ci-agents` commands were missing their package entry point. | Added the entry point and verified an isolated editable installation, `ci-agents v2 --help`, and an offline V2 dry run. |
| Low | Stale imports, inconsistent formatting and an initial-run label incorrectly implied unchanged agent behavior when a V2 policy was active. | Removed 15 unused imports, formatted reviewed modules and clarified the execution label. Static checks cover undefined names and unused imports. |

The pilot test fixtures now include source metadata and the full-record retrieval
event actually produced by their mock service. This updates the fixtures to the
stronger handoff contract; production validation was not relaxed to accept them.

## Validation

The pre-review suite passed 437 tests, with two skipped. The final pinned-runtime
suite passed **457 tests, with two skipped**: twenty additional regression cases.
The reviewed files pass Ruff checks for undefined names and unused imports, and
the installed V2 CLI passes an offline dry run. Review-specific regressions cover
failure paths and real SDK parsing with provider responses mocked. No paid model
calls were made during this code review.

The two existing skips concern a historical Brotli incompatibility that is no longer
present in the installed backend. Remaining warnings are from the PDF dependency's
deprecated import/SWIG types, rather than failed agent handoffs.

`requirements-research.lock` pins the research runtime dependencies and distribution
hashes. It was generated and tested with Python 3.11 on macOS. Recompile and verify
the lock for a different Python/platform target rather than assuming identical
resolution. The installed editable package supplies the CLI entry point.

Example in a configured project folder:

```bash
uv run --no-project --with-requirements requirements-research.lock --with-editable . ci-agents v2 --question-id Q01 --initial-max-tool-calls 32 --dry-run
```

`--dry-run` does not call a model. A live invocation requires a funded API key and
`--allow-api`. Use a new run ID/output directory for each research condition.

## Research limits that remain

- This is a correctness and reproducibility review, not proof that model-generated
  answers are always accurate. Existing 15 November quality/cost figures describe
  their frozen code versions; they were not recomputed for this revision.
- Q15's earlier independent grades remain invalid because their citation anchors
  failed validation. Its quality and the general benefit of recovery are not proven.
- Reviewer PASS is internal routing, not independent ground truth. Private-key
  evaluation and human adjudication remain separate from the runtime.
- Provider aliases may change. Configure an appropriate fixed model snapshot for
  controlled research and retain version/configuration artifacts; repeated trials
  are still needed to measure model variability.
- Token/cost budgets gate subsequent calls on reported usage. An in-flight request
  may overshoot, and cancellation may leave unknown billed usage. They are not an
  exact provider billing cap.
- Integrity hashes detect accidental/tampered artifacts against recorded hashes;
  they are not signatures or OS-enforced write-once storage against the file owner.
- Source archives improve reproducibility for new runs. Older hash-only runs cannot
  retroactively acquire their original source, settings or complete dependency state.

The remaining recommendation is controlled research with fresh versioned runs,
repeated/held-out questions and independently adjudicated results. No code review
can certify that an LLM workflow is free of every error.
