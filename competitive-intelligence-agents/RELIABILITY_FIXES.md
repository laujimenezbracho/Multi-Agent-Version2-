# Research and evaluation reliability changes

The original six-case screen exposed execution and measurement defects before it
established a useful retrieval limit. These changes address those defects; they do
not establish that 16, 32 or 60 is the optimal budget.

## Research before abstention

Every Researcher session starts with deterministic discovery across the essential
task companies, followed by full-record retrieval. Pricing questions use the
requested tier's effective pricing history; other questions start with short company
queries. Explicit superseded records are fetched when budget remains. Every attempt
consumes the same retrieval budget as a model-selected operation. This bootstrap uses
the public question and public corpus, never evaluator references or question IDs.

If discovery yields no full records, the workflow produces an explicit empty pack
and an honest Analyst abstention. It records zero model requests for that code path.
Finding some records does not prove sufficient competitive coverage.

## Citations come from full records

The Researcher model selects additional retrievals and a stop reason. Application
code builds the evidence pack from the session's successful full-record ledger. It
copies source IDs, content and metadata directly; search snippets and pricing-history
rows alone cannot become cited evidence. Runtime validation still checks the cutoff
and source provenance. Evidence confidence is a neutral structural placeholder, not
an empirically calibrated probability of correctness.

## Analyst output contract

The provider JSON schema now separates factual and inference claims. A factual
claim requires at least one supporting record ID in the generated schema as well as
runtime validation. Missing-data statements belong in limitations or qualified
inference, rather than uncited factual claims. An empty evidence pack produces a
deterministic abstention with no factual claims. There are no silent model retries.

During verification, a substantive report still failed because one missing-data
statement was labeled factual without citations. That failed run is preserved in
`runs/reliability_screen_20261113/`; the final frozen rerun includes the schema fix.

## Question scope and Reviewer

Plans distinguish essential tasks from optional context. A published-list-price
question requires comparable current prices, billing basis, effective dates and
conflict resolution. Unrequested SLA, feature-value and sales-quote investigations
are optional. Both Planner validation and Reviewer context carry this distinction.
Essential pricing tasks may also consult promotion or product metadata to resolve
dates and conflicts; their source-category tags do not make them unrelated research.
Reviewer instructions also require checking whether a corrected competitor price
actually changes the winner. The internal verdict remains an observation, not the
benchmark's ground truth.

The live rerun still exposes Reviewer limitations. In Q01/32 it rejects a qualified,
source-correct list-price comparison for missing fees/minimums/promotions requested by
the plan. In Q15/16 it returns missing-record IDs outside the full-record evidence
items; the existing runtime guard rejects that review and preserves the Analyst
report. Neither case is a validated Reviewer success. Stronger instructions are an
implemented change, not proof that false rejections or invalid references are solved.

## Independent grading and source audit

Native MLflow datasets, registered judges, tracing and `mlflow.genai.evaluate` own
execution and judging. The evaluator receives private provisional source rubrics;
the agent does not. The judge returns per-requirement and per-claim decisions,
original source-field pointers, and explicit defects. Code validates the fixed item
catalogue and derives fractions from those decisions. It rejects invented scores,
unsupported anchors, missing checks and impossible price-rank allegations.

Stable answer-segment and source-field pointers avoid unreliable copying of long
quotes. The judge uses MLflow's required result/rationale response envelope. A live
canary confirmed that an abstention meeting only gap disclosure receives 1/4
completeness, rather than the earlier incorrect 3/4. Empty claim denominators are
undefined, not perfect accuracy. Invalid grader output remains explicitly invalid.

The initial six outputs have an evaluator-side source audit under
`evaluation_private/reliability_source_audit/`. It corrects the obvious arithmetic
and abstention-scoring errors while preserving the original judgments. This is a
coding-assistant source audit, **not owner/human approval**. All independent scores
remain pending human adjudication; schema validity does not prove semantic judging
reliability.

Derived claim fractions cover the structured claim catalogue. Narrative defects are
reported separately; these fractions are not an exhaustive clause-level accuracy
audit of the prose. The corrected screen's `source_audit_notes.json` records follow-up
spot-checks, also logged as native MLflow feedback with coding-assistant attribution.

## Verification artifacts

The original screen is preserved at `runs/budget_screen_20261113_live/`. The corrected
six-case screen is at `runs/reliability_screen_20261113_verified/`:

- `experiment_plan.json`: frozen public-data, source, prompt, rubric and judge hashes;
  model/version settings and treatment limits.
- `answers.html` and `answers.md`: exact reports and execution/cost summaries.
- `comparison.json`: actual retrieval attempts, model requests, tokens, latency,
  internal verdict and stop reason.
- `independent_results.json`: raw native evaluation outputs and traces.
- `derived_scores.json`: validated item judgments or explicit grading failures.
- `before_after.md` and `.json`: original/corrected execution observations, not quality scores.
- `cases/` and `agent_runs/`: per-condition checkpoints and original stage artifacts.

The earlier `runs/reliability_screen_20261113_final/` attempt is also preserved. Its
scope guard rejected legitimate pricing tasks with mixed source categories. The
verified screen includes that guard correction, with all six conditions run afresh.

All six cases use V1 with recovery disabled, Q01/Q15, limits 16/32/60, GPT-5 mini,
low reasoning, 12,000 output tokens and otherwise fixed controls. This corrected
workflow is a new version: before/after differences cannot be attributed solely to
budget. One run per condition is screening evidence. Model alias variation and run
order/caching remain limitations. Text-token costs conservatively price input as
uncached and exclude independent-judge calls; they are estimates, not invoices.

Local run artifacts and private rubrics are ignored by Git. Reproduction commands
and dependency setup are in [BUDGET_SCREEN_README.md](BUDGET_SCREEN_README.md).

## Corrected screen observations

The final offline suite passes **391 tests**, with **2 skipped**. All six frozen cases
executed once and produced Analyst reports with citations drawn from full retrievals.
Five completed the Reviewer stage; Q15/16 failed Reviewer reference validation while
preserving its report. Internal verdicts were one PASS, four FAIL and one invalid
review. The original screen produced three reports, two of which were empty-evidence
abstentions. Report availability is an execution improvement, not a quality score.

All three Q01 answers now state the five correct comparable prices, including the
VertexOps correction to USD159 and PulseGrid as the USD130 winner. Q15 now contains
source-backed findings, but its price-pressure and hidden-discount inferences remain
weak: the packs lack a PulseGrid pricing baseline, and superseded/unauthorized prices
do not establish actual competitive undercutting. The external judge's perfect item
scores on those reports do not settle their strategic quality.

Two external grades fail protocol validation (a missing temporal judgement and an
invalid source anchor). They remain invalid and were not silently regraded. All
quality judgments need owner adjudication. The six-case agent totals are **24 model
requests**, **179,734 tokens**, and **USD0.11500175 estimated agent text-token cost**,
excluding evaluator calls and earlier verification versions. This is not a billed
amount or an implementation-wide cost total.

Actual retrieval attempts range from **9 to 11**. No case exhausted its retrieval
allowance, so these observations still cannot validate 16, 32 or 60 as an optimal
limit. `verification_summary.json` records frozen-hash checks, original source-text
and metadata checks, full-retrieval citation membership, and the test results.
