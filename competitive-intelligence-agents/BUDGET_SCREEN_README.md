# Six-run retrieval-budget screen

This experiment executes **V1 only**, with no recovery: Q01 and Q15 at retrieval-tool
allowances of 16, 32 and 60. There is one execution per condition. This is a screening
study, not statistical proof of an optimal limit. No single-agent baseline is implied.

## Success criteria and private expectations

The benchmark owner's official private keys were not supplied. With user approval,
`scripts/prepare_budget_rubrics.py` creates explicitly **provisional source-audited**
Q01/Q15 reference rubrics under ignored `evaluation_private/`. Q01 has a documented
current-price comparison and conflict traps. Q15 accepts alternative supported rankings
rather than pretending that one strategic ranking is canonical. Each rubric specifies
factual accuracy, completeness, actual citation support, temporal accuracy, material
errors and efficiency definitions. Reference records and rubric hashes are frozen.

The four runtime agents receive only public benchmark inputs and their normal structured
handoffs. No agent accepts a rubric parameter or private evaluation file path. Runtime
tools can only search/get public records and pricing history; there is no general file
reader. This is an application boundary, not an OS sandbox.

Independent grading uses native **MLflow datasets**, a registered `make_judge()` scorer,
and `mlflow.genai.evaluate()`. Private rubrics are evaluation `expectations`; prediction
receives only question ID and retrieval budget. The final external judge sees the answer and
the original cited public records, not the Researcher's generated paraphrases, internal
Reviewer verdict, or budget label. Feedback is a fallible model assessment against a
provisional key, not independently verified benchmark truth. Its rationale and claim-level
error notes remain available for human review. Internal Reviewer decisions are recorded
separately so false approvals and questionable rejections can be inspected.

Efficiency is measured from runtime telemetry, not graded by an LLM: model requests,
input/output/cached tokens where available, tool attempts and denials, wall-clock latency
and documented estimated cost. Failed and missing-output runs stay in the denominator;
their quality is not graded as successful abstention. This screen does not establish
Reviewer precision/recall; that requires adjudicated claim-level defect matching.

## Frozen controls

| Control | All six cases |
| --- | --- |
| Model | `gpt-5-mini` (same configured alias; actual responses remain in local traces) |
| Reasoning effort | `low` |
| Maximum output tokens per request | 12,000 |
| Cutoff | 2026-11-11 23:59:59 UTC |
| Researcher model-turn ceiling | 62 |
| Full-record attempts | 40 |
| Selected evidence items | 40 |
| Results per search | 10 |
| Recovery / HTTP retries | Disabled / zero |
| Treatment | Maximum retrieval operations: 16 vs 32 vs 60 |

Originally, the Researcher turn ceiling was `max_tool_calls + 2`. The optional
`max_model_turns` override keeps that separate control constant during this screen;
when absent, existing production behavior is preserved. Optional frozen model settings
also leave default V1/V2 behavior unchanged when not supplied.

The complete public dataset, prompt text, source code, model/SDK versions and provisional
rubrics are fingerprinted before execution. Each arm gets a fresh V1 run with a Planner
execution. Planner/model nondeterminism is not eliminated, and input caching/order can
affect cost and latency; one observation per condition cannot isolate every such effect.
The recorded run plan is authoritative. Exactly six unique prediction checkpoints prevent
MLflow's initial trace-validation call or an evaluation resume from repeating paid cases.
An interrupted case is not silently rerun. Agent artifacts retain their original manifests.

## Cost accounting

The GPT-5 mini official model page was checked on 13 November 2026:
https://developers.openai.com/api/docs/models/gpt-5-mini

Standard text rates used: USD 0.25 per million input tokens, 0.025 cached input, and 2.00
output. Existing V1 cost estimates conservatively price all input as uncached; they are
not billed totals. Cached tokens remain separately observable in stage metrics. Evaluator
requests are additional work and are not counted as agent workflow requests or cost.
Unknown usage/cost remains null. A USD 10 agent estimate safeguard checks between cases;
it is **not** a hard in-flight provider spending cap. Turn and output limits bound work,
but a monetary hard cap would require additional per-request reservation/accounting.

## Commands

The project imports `httpx` directly for diagnostics even when the OpenAI SDK uses
`httpx2`; `httpx` is now an explicit dependency. The experiment requirements also include
MLflow and the existing optional PDF-test dependency.

```bash
# Evaluator-only keys: no model calls.
python scripts/prepare_budget_rubrics.py \
  --output-dir evaluation_private/budget_screen_v2

# No model calls: registers/discovers local datasets and scorers, verifies trace readback.
uv run --no-project --with-requirements requirements-experiment.txt \
  python scripts/run_budget_screen.py \
  --output-dir runs/budget_screen_new \
  --rubric-dir evaluation_private/budget_screen_v2 --prepare-only

# Explicit paid execution. Uses the cloned repository root .env without copying secrets.
uv run --no-project --with-requirements requirements-experiment.txt \
  python scripts/run_budget_screen.py \
  --output-dir runs/budget_screen_new \
  --rubric-dir evaluation_private/budget_screen_v2 --allow-api
```

Set `OPENAI_API_KEY` in the cloned repository root `.env`.
No key is printed or passed as an MLflow prediction argument. If MLflow tracking URI and
experiment ID are configured, they are respected; otherwise local SQLite tracking is
created under the selected output directory. No external tracing exporter is enabled.

`answers.html` and `answers.md` show the exact saved answer sections for all completed
conditions; `comparison.json` contains efficiency and stop details. Each run preserves
plan, evidence, report, review, metrics and retrieval logs. `independent_results.json` and
`.csv` contain native MLflow grading feedback. `experiment_plan.json` binds the controls.
Private rubrics, results, database files and traces remain Git-ignored.

Offline verification:

```bash
uv run --no-project --with-requirements requirements-experiment.txt \
  python -m pytest -q -p no:cacheprovider --basetemp runs/offline_test_tmp
```

Using the workspace for temporary files avoids an existing V2 check treating macOS's
`/private/var` system temporary directory as private evaluation storage. No V2 policy
was weakened for this V1-only experiment.

## Observed screen: 13 November 2026

`runs/budget_screen_20261113_live/answers.html` contains every original answer section and
explicit failures. `pilot_summary.json` adds measured efficiency and adjudication notes.
All six conditions executed exactly once; recovery was disabled and failed cases were
not retried. Two initially deferred zero-call cases were later completed after original
captured response usage resolved an unknown-cost safeguard. Their earlier not-started
checkpoints are archived. A failed Analyst SDK response had no returned SDK Usage object;
its actual token/request counts were recovered from all eight original MLflow response
spans, without replacing the sealed agent metrics. The continuation binds to the original
source, model, prompt and data fingerprints.

| Question | Retrieval ceiling | Actual retrieval attempts | Outcome |
| --- | ---: | ---: | --- |
| Q01 | 16 | 15 | Researcher cited a record not retrieved in full; no Analyst report |
| Q01 | 32 | 11 | Same provenance failure; no Analyst report |
| Q01 | 60 | 12 | Report delivered; correct lowest-price winner, outdated VertexOps price |
| Q15 | 16 | 5 | Analyst output rejected by SDK; no final report |
| Q15 | 32 | 0 | Empty evidence pack and abstention |
| Q15 | 60 | 3 | Empty evidence pack and abstention |

No retrieval budget was exhausted. The screen therefore **does not validate 16, 32 or 60**
as an optimal ceiling; the observed failures arose before those ceilings became binding.
There were 65 agent model requests and 437,278 tokens. Estimated agent cost at the recorded
uncached-input rates was USD 0.1844995; evaluator calls were additional and excluded.
Evaluator monetary usage is not available in the existing per-agent stage telemetry and
must not be presented as part of that cost estimate.

The initial evaluation's execution-status text could indirectly reveal the internal
Reviewer verdict. Its results were preserved as preliminary. Final grading uses native
MLflow `independent_blind_audit_v2`, GPT-4.1, and neutral report-availability status, with
no repeated agent generation. Files are `independent_blind_results.json` / `.csv` and
`final_evaluation_manifest.json`.

Spot-checking also found inconsistencies in automatic grading, including a rationale
suggesting that changing VertexOps from USD174 to USD159 could invalidate the USD130
lowest-price winner, and over-crediting a Q15 abstention. Raw model scores are preserved
and explicitly **disputed**, not silently corrected or treated as truth. The human-facing
report states that PulseGrid remains lowest, that VertexOps is outdated, and that neither
Q15 abstention provides three threats. Human adjudication is required before using the
numeric scores for a budget decision.

Subsequent Researcher, Analyst, scope and grading corrections are documented in
[RELIABILITY_FIXES.md](RELIABILITY_FIXES.md). The corrected frozen six-case output
directory is `runs/reliability_screen_20261113_verified/`. Its item-level
`derived_scores.json` validates source pointers and derives fractions from fixed
catalogues; invalid judgments stay invalid, and all scores need human adjudication.
`scripts/check_grader_protocol.py` runs a single paid native-MLflow canary against a
saved abstention without regenerating agent answers. `scripts/compare_budget_screens.py`
compares execution observations offline. The old summarization script is specific to
the original screen and should not be used to label new runs.

`scripts/finish_budget_screen.py` documents the exact continuation and blinded evaluation;
`scripts/summarize_budget_screen.py` is an offline presentation step. It performs no quality
model calls and never rewrites sealed agent artifacts. The offline suite passed 389 tests,
with two environment-specific skips, before live execution.
