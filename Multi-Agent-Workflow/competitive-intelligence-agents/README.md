# competitive-intelligence-agents

Python 3.11+ implementation of the Multi-Agent Competitive Intelligence project through
Task 10 plus additive V2 recovery. Version 1 executes Planner -> Researcher -> Analyst -> Reviewer -> STOP exactly
once in order. Reviewer FAIL is a completed run with review failure, not an exception
or a reason to retry. V2 is a separate bounded controller described below. The single-agent baseline,
background execution, and independent evaluator remain unimplemented.

## Windows PowerShell setup

```powershell
cd .\competitive-intelligence-agents
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
notepad .env
python -m pytest -q -p no:cacheprovider
```

Python must be 3.11 or newer. If activation is restricted, run
`.\.venv\Scripts\python.exe` directly for pip, tests, and CLI commands. SQLite FTS5
uses the standard-library sqlite3 module; no external database dependency is needed.

Set OPENAI_API_KEY and OPENAI_MODEL in .env. Neither has a hardcoded credential/model
default. DATA_DIR defaults to data/public, RUNS_DIR to runs, and EVALUATION_CUTOFF to
2026-11-11T23:59:59Z. Environment variables override .env; relative paths resolve against
the project root. Missing/blank configuration and invalid paths/timestamps raise clear
configuration errors. Loading configuration never makes an API call or creates files.
The controller freezes validated settings in a context-local use_settings scope so
all four agents use the model/data configuration recorded for the run.

## CLI

Implemented commands, from the project directory after editable installation:

```powershell
python -m ci_agents.cli run --question-id Q01 --dry-run
python -m ci_agents.cli run --question-id Q15 --as-of 2026-11-11T23:59:59Z --dry-run
python -m ci_agents.cli run --question "Compare the supplied Growth pricing evidence" --dry-run
python -m ci_agents.cli --help
```

Choose exactly one of --question-id and --question. --as-of overrides the configured
cutoff and must include a timezone. --run-id accepts 1-80 ASCII letters, digits,
underscores, or hyphens, starting with a letter/digit; Windows reserved names and
existing run directories are rejected. IDs otherwise default to random UUID hex.

Dry-run validates configuration (including required key/model presence), all public
inputs, question selection, cutoff, dataset hashes, and prompt hashes. It performs no
agent execution, API calls, retrieval indexing, or artifact writes. It does not prove
account/model access. Dry-run takes precedence even when --allow-api is also present.

Live CLI execution is blocked unless --allow-api is supplied explicitly. This example
would incur API usage; it was **not executed** as part of Task 9:

```powershell
python -m ci_agents.cli run --question-id Q01 --run-id pilot-q01 --allow-api
```

CLI exit codes: 0 for dry-run or completed execution (including Reviewer FAIL), 1 for
execution_failed, 2 for argument/configuration/preflight/persistence errors. Secrets and
raw provider exception messages are not printed. Library run_pipeline is an explicit
paid-capable entry point; only the CLI implements the --allow-api gate.

## Project layout and public corpus

```text
src/ci_agents/
    agents/            # planner.py, researcher.py, analyst.py, reviewer.py
    prompts/           # packaged reusable Markdown prompts
    config.py
    dataset.py
    retrieval.py
    schemas.py
    telemetry.py
    persistence.py
    pipeline.py
    cli.py
    evaluation_contracts.py
data/public/
evaluation_private/
tests/
runs/
```

The six supplied Data files are unchanged copies; originals remain in the parent
workspace. The corpus contains 310 source records, 77 price observations, and 15 questions
across NexaFlow, OrbitDesk, VertexOps, PulseGrid, and ClearPilot. The workbook duplicates
public records/pricing/questions. CSVs use semicolon delimiters. All companies, events,
and source URLs are fictional; URLs are inert provenance and are never fetched.
Private keys were not supplied and are never opened by the application. Keep future
private annotations/results under evaluation_private and outside agent runtimes/indexes.
.env, runs, evaluation_private, caches, and database files are Git-ignored.

## Dataset and retrieval

```python
from ci_agents.dataset import load_dataset
from ci_agents.retrieval import RetrievalStore

public = load_dataset()
with RetrievalStore(public) as store:
    matches = store.search_records("Growth pricing", company="VertexOps", category="Pricing")
    source = store.get_record("WEB-00001")
    history = store.get_pricing_history(company="VertexOps", plan="Growth")
```

Convenience search_records/get_record/get_pricing_history functions are also exported
from ci_agents.retrieval. They construct and close a validated in-memory index each call.
They read DATA_DIR and EVALUATION_CUTOFF from the process environment, not .env; agent
and pipeline calls use load_config and explicit cutoffs instead.

Loading opens only public_web_records.jsonl, pricing_events.csv, and public_questions.csv.
It validates counts, IDs, types, dates, and pricing references. Rows remain dictionaries;
numbers use Decimal, seat counts use int, and booleans are explicit. Empty/unknown markers
become None, never zero/false. ISO timestamps require timezones and normalize to UTC.
Date-only fields use midnight UTC; slash dates are explicitly DD/MM/YYYY.

Search uses parameterized SQLite FTS5, literal-word AND queries, weighted BM25 (title
highest), and stable record-ID tie breaks. Company/category filters are case-insensitive
exact matches; category maps to event_category. Empty queries return []; bad inputs
raise RetrievalError. Search returns 20 by default, at most 50, with 600-character summary
limits; get_record returns complete source details or None for missing/unavailable IDs.
Pricing history uses pandas, preserves conflicting observations and Decimal/None amounts,
and sorts oldest effective date first. It raises if more than 100 observations match,
requiring narrower filters instead of silent truncation.

Availability filtering is inclusive: available_at <= as_of. Publication, availability,
effectiveness, and expiry remain distinct. Available future-effective announcements are
valid evidence but do not establish current prices or GA features. Promotions require
eligibility and valid dates. Newer/official/confirmed sources are not automatically true;
source independence and supersession remain visible. No canonical current price is
silently inferred by retrieval.

## Structured handoffs and agents

Pydantic v2 contracts cover ResearchTask, ResearchPlan, EvidenceItem, EvidencePack,
ReportClaim, CompetitiveReport, ReviewIssue, ReviewResult, StageMetrics, and RunResult.
TaskCoverage and TotalMetrics provide typed nested structures. All scores are finite
[0,1]; naive timestamps, empty IDs, duplicates, and unsupported factual claims are rejected.
Use model_dump_json/model_validate_json for round trips. Closed objects and typed lists
support SDK strict output conversion, verified offline with the installed SDK. SDK
conversion makes generated fields required, including nullable values; local Python
construction retains compatible defaults. In-place list mutation can bypass Pydantic
validation, so handoffs are snapshotted and revalidated.

ResearchPlan.time_scope is descriptive text. Report sections are text lists; sources is
a list of record IDs. EvidenceItem preserves source_independence_group,
supersedes_record_id, end_date, verification_status, and feature_status as nullable fields.
ReportClaim has no competitor/time fields: claim_traceability.json preserves the exact
claim contract and leaves those additional context fields null instead of inferring them.

- planner(question, as_of): metadata-only context, no tools. Rejects changed question/
  cutoff, unsupported companies/categories, empty task requirements, duplicate IDs,
  and repeated normalized task questions with matching company/category sets.
- researcher(plan, retrieval_service=None, *, limits=None): three cutoff-bound SDK tools
  reuse existing retrieval. Every cited source requires a full get_record response in
  the current run; search/history alone cannot authorize citations. Exact metadata,
  availability, coverage references, and unique IDs are validated. Its log is replaced
  by the authoritative runtime ledger. ResearcherLimits defaults are 60 tool calls,
  10 search results/call, 40 evidence items, and 40 full-record attempts. Limits are
  provisional; live benchmark adequacy has not been measured. Budget denials are logged
  and returned as tool errors, not retry instructions. Injected services are trusted
  synchronous public-only sources; the caller owns their lifecycle.
- analyst(plan, evidence_pack): evidence-only context, no tools. Validates unchanged
  question/cutoff, all structured source IDs, factual citations, and required prose
  sections. Unsupported sections must state limitations rather than being empty.
  Explicit abstention may have empty claims/sources. ID membership is not semantic support.
- reviewer(plan, evidence_pack, report): terminal internal quality control, no tools.
  Returns PASS/FAIL and issues; it never rewrites the report or calls another agent.

Agents use native SDK output types, the same configured model (currently gpt-5-mini),
no unsupported temperature overrides, disabled tracing, and zero HTTP retries. Planner,
Analyst, and Reviewer have one SDK turn; Researcher has bounded tool-use turns, not a
recovery loop. No live model calls were used in development/tests.

Reviewer policy is fixed across questions: PASS iff completeness >= 0.85, evidence support
>= 0.90, temporal consistency >= 0.90, no critical/major issues, and no essential missing
requirements. FAIL otherwise. Minor presentation issues alone never cause FAIL.
Below-threshold scores require explanatory issues, missing requirements, or notes.
These thresholds are provisional internal routing policy, not validated objective metrics.

Reviewer taxonomy: unsupported_factual_claim, citation_mismatch, factual_contradiction,
temporal_inconsistency, numerical_or_unit_error, missing_required_information,
unresolved_evidence_conflict, overstated_certainty, weak_strategic_inference.
related_record_ids links known evidence; missing_record_ids means retrieved evidence
omitted from the report, not fabricated unseen records. Unknown evidence needs are text.
reviewed_question and reviewed_report_hash bind the review to its exact report snapshot.
The latter is SHA-256 of canonical report JSON, distinct from the report.json file hash
(which includes its final newline). Duplicate/unknown references, invalid policy decisions,
and exact normalized duplicate defects are rejected. Semantic duplicates and missed errors
still require independent assessment. Old construction of these nullable/list fields
remains compatible; Reviewer output requires matching identity/provenance.

## Version 1 execution and artifacts

```python
from ci_agents.pipeline import run_pipeline
# Explicit invocation can incur cost; only do this with authorization.
# result = await run_pipeline(question, cutoff, question_id="Q01")
```

The controller initializes public data and retrieval, executes each predecessor before
its successor, and returns RunResult. All stage inputs are deep snapshots. It freezes
settings, hashes dataset and prompts, and rejects changes between stages. Public question
IDs, when supplied, must match their original question text. Direct questions have null
question_id; no benchmark identity is invented. experiment_id defaults to run_id unless
provided explicitly through the Python interface.

Execution outcomes are completed (PASS), completed_review_failed (FAIL), and
execution_failed (exception/invalid handoff). Valid Reviewer FAIL returns normally and
stops. Exceptions stop immediately, preserve prior outputs, and produce sanitized error
records; no downstream successful output is fabricated. Only attempted stages have
metrics. Invalid requests/existing IDs fail before agents execute. Filesystem failures
can prevent full finalization; existing stage files are retained and are never replaced.

Each runs/<run_id>/ contains, when the relevant stage completed:

- started.json: initial run identity, timestamp, model and runtime versions.
- plan.json, evidence_pack.json, report.json, review.json: exact structured stage snapshots.
- planner_metrics.json, researcher_metrics.json, analyst_metrics.json, reviewer_metrics.json.
- retrieval_logs.json: authoritative tool events, also saved if Researcher execution fails.
- report.md: escaped readable report, claim citations, review decision/scores/issues; if
  Reviewer execution failed, review status is explicitly unavailable.
- claim_traceability.json: stable claim IDs, exact text/type/confidence/citations, with
  unavailable competitor/time context null.
- metrics.json: attempted-stage telemetry, totals, pricing assumptions, retrieval audit.
- metadata.json: final status, UTC timestamps, model/SDK/Python versions, dataset/prompt
  hashes, question/cutoff identity, error type/message, initial artifact fingerprints.
- error.json only on execution failure.
- evaluation_pending.json: a versioned all-null evaluation placeholder, never internal
  review scores or populated independent results.
- run_result.json: serialized RunResult.
- manifest.json: SHA-256 of exact bytes for every previously written artifact.

Artifacts use sorted canonical UTF-8 JSON plus a final newline. Stage artifacts and
telemetry are written immediately, before the next agent. A run directory must be new.
Files publish through a flushed/fsynced same-directory temporary file and atomic
no-overwrite hard link; NTFS/POSIX hard-link support is required. No overwrite fallback
is allowed. Finalization seals the application writer. verify_manifest detects external
modification; this is application-level append-only storage and tamper detection, not
OS-enforced WORM protection against a user with filesystem write access. Processes
killed abruptly may leave partial runs without final metadata/manifest. Stage files
already published survive. Cancellation/crash durability is not a recovery mechanism.

Secret values from configuration are redacted before persistence; raw exception messages,
tracebacks, headers, and provider bodies are not serialized. Markdown escapes HTML and
Markdown syntax, keeping URLs inert and preserving displayed factual text. JSON artifacts
are the authoritative structured record.

## Telemetry and cost

StageMetrics now adds UTC start/end, monotonic wall-clock duration, actual input/output/
total tokens, LLM requests, cached input tokens when available, usage provenance, estimated
input/output/total USD cost, retrieval call counts, and sanitized error type/message.
RunResult adds execution_status and artifact_directory and allows question_id=None.
The original four-field TotalMetrics contract remains unchanged for compatibility;
metrics.json additionally stores total tokens and requests. Totals are null if any
attempted stage has unknown usage/cost; individual known stage measurements remain visible.

Passive context-local observe_sdk_result calls in all four wrappers capture the returned
SDK context_wrapper.usage object. ResearchSession.event also publishes its existing event
to the local capture. These are the only agent implementation changes; no signatures,
prompts, tools, output contracts, settings, or stage behavior were changed for telemetry.
No tracing exporter is enabled. Unknown usage is null, never estimated from characters.
Default SDK zeros without any requests are treated as unknown. SDK calls that fail before
returning may have incurred unreported cost; no usage is fabricated for them.

Retrieval audit records tool names, query/filter arguments, fixed cutoff, returned IDs,
selected evidence IDs, retrieved-but-not-selected IDs, and budget denials. Counts include
tool attempts (including misses/denials); full_record_ids separately identifies successful
full retrievals. No duplicate full source corpus is stored. Non-Researcher stages have
zero retrieval by construction; missing Researcher instrumentation remains unknown.

PricingConfig is the versioned model_pricing.v1 configuration, persisted in metadata and
metrics. Default gpt-5-mini rates/source/date are null because no reliable rates were
supplied in the codebase. No real monetary cost is calculated. Explicit test/user rates
require source/date and must match the selected model. Estimates are tokens * supplied
USD-per-million / 1,000,000, with separate input/output costs. The current simple policy
prices all input tokens at one rate, including cached input, and records that assumption.
These estimates are not billed amounts. No price discovery API calls are made.

## Evaluation readiness and immutable paired baselines

IndependentEvaluation in evaluation_contracts.py is independent_evaluation.v1. All results
remain null until a separately authorized evaluator supplies ground truth: factual
accuracy, answer completeness, citation support, unsupported-claim rate, temporal accuracy,
contradiction/numerical counts, missing essential elements, per-claim error annotations,
reviewer detection precision/recall, false positives, error correction/new-error rates,
and quality improvement per additional cost. Unknown annotations/lists are null, not
empty lists implying an evaluated zero-error run. Reviewer PASS is never copied into
these fields. Populated independent results belong outside agent runs/contexts/indexes,
under a separately authorized evaluation process. No evaluation loader or scorer exists.

baseline_reference(runs_dir, run_id) verifies a completed initial run's manifest and
returns its exact run ID, manifest hash, artifact fingerprints (including initial report,
review, plan, evidence, logs and stage telemetry), question, cutoff, and comparison key.
verify_baseline_reference checks it again before future use. comparison_key is SHA-256
of canonical question + normalized cutoff JSON; experiment_id controls repeated-trial
grouping. Dataset/prompt/model fingerprints remain available for fairness checks.

PairedComparison (paired_comparison.v1) can reference the immutable baseline, recovered
run/report/review fingerprints, and incremental input/output tokens, latency, and cost.
All recovered and incremental values start null. This is only a future comparison data
contract: it does not run recovery, calculate ground-truth metrics, or overwrite baselines.

## Verification and limitations

Tests mock all paid execution and block HTTP. They cover agents, SDK strict output/tool
adapters, fixed order, PASS/FAIL stopping, partial failures, timing/usage, explicit fixture
pricing rates, retrieval budgets, no-overwrite artifacts, hashes, baseline references,
claim/issue traceability, null evaluation placeholders, CLI gating, and private-data
isolation. One integration test uses all four real wrappers with a mocked SDK Runner.
No mocked scenario proves semantic accuracy, error-detection recall/precision, live account
access, or adequate benchmark retrieval coverage. Those require future authorized pilots
and independent evaluation. Original public data and all four prompts are unchanged.
Task 9 remains the frozen Version 1 controller; Task 10 adds only the controlled runner below.

## Task 10: controlled pilot readiness

The `pilot` command is a thin sequential runner around run_pipeline, not a second
pipeline. It defaults to Q01 and Q15, loading their exact wording from public_questions.csv.
Only these designated pilot IDs may be selected; held-out final-evaluation questions
remain excluded. All four prompts and the six public files remain unchanged.

```powershell
python -m ci_agents.cli pilot --dry-run
python -m ci_agents.cli pilot --dry-run --question-ids Q15
python -m ci_agents.cli pilot --dry-run --question-ids Q01 Q15 --as-of 2026-11-11T23:59:59Z --output-dir runs/pilots
```

Comma-separated IDs are also accepted. `--as-of` otherwise uses the configured cutoff,
not today's date. `--output-dir` is a pilot root: each actual execution gets a fresh
`pilot-<uuid>` child and a `runs/<question-id>-<uuid>` directory per question. Existing
baseline directories cannot be used as the output root and are never overwritten.

Pilot dry-run uses `load_config(require_api_key=False)`; model, data paths, output paths,
and cutoff still require valid configuration, but an API key may be missing or blank.
Existing agents and the single-run command still require a key as before. Live pilot
execution requires both explicit --allow-api and a non-empty configured credential.
Preflight cannot validate credential/account/model access without a live call and
therefore makes no claim to have done so. Dry-run takes precedence over --allow-api.

Preflight checks all six public files, dataset validation, both designated questions,
FTS5 initialization, all four non-empty prompts, four SDK strict output schemas, model
configuration, writable/link-capable output storage, and fresh run paths. A temporary
write/hard-link probe is cleaned up immediately; no run directories, reports, summaries,
model outputs, baselines, or evaluation annotations are created in dry-run. Public input
symlinks outside DATA_DIR and private evaluation paths are rejected. Agent imports are
audited for direct private/evaluation dependencies. Tools and context construction remain
public-only; this is application isolation, not an OS filesystem sandbox. The process
may have filesystem permissions beyond the tools exposed to the model. No private
ground-truth files are opened for preflight or execution.

Before API-capable execution, the CLI prints model, pricing availability/cost warning,
cutoff, selected questions, fingerprints, proposed output path, and budget scope. This
command is prepared for future explicitly approved live testing and was not executed:

```powershell
python -m ci_agents.cli pilot --allow-api --question-ids Q01 Q15 --output-dir runs/pilots
```

A question failure preserves its partial run and is reported; the next selected question
may run once, without retrying the failed question. Reviewer FAIL remains a normal
completed_review_failed result. No agent feedback, correction, recovery, or Version 2
execution is added. Each question has its own immutable Task 9 artifacts and the shared
pilot ID as experiment_id. Progress JSON snapshots survive later failures.

After an actual (or mocked test) execution, the new pilot directory contains
preflight.json, progress_1.json/progress_2.json as applicable, pilot_summary.csv,
pilot_summary.md, pilot_errors.json, pilot_safeguards.json, and an append-only manifest.
Each summary row includes question/run IDs, execution status, Reviewer decision and
three internal scores, total input/output/combined tokens, latency, estimated USD cost,
retrieval calls, cited evidence record count, claim count, issue count, artifact directory,
evaluation readiness, and diagnostic text. CSV uses empty cells for nulls; Markdown
uses `null`. Scores are explicitly labelled Reviewer assessments, never independent
factual accuracy. No fake successful outputs are generated by dry-run.

Evaluation readiness verifies manifests, preserved claim/evidence/issue references,
report/review identity, a fully null evaluation placeholder, and a verifiable immutable
baseline reference. Incomplete failed runs are not marked ready. It does not load truth,
calculate correctness metrics, or mutate initial artifacts.

Optional `--max-total-tokens` and `--max-estimated-cost` stop later questions after a
budget is reached. They are between-question safeguards, NOT exact SDK hard caps; a
single question can exceed either budget. Unknown usage/cost stops remaining questions
conservatively when the corresponding budget is enabled. Such rows have status
not_started_budget, null metrics and no report/artifact directory. A cost limit requires
documented input/output pricing rates; default rates remain null. PricingConfig can
be supplied through the Python preflight interface; the CLI has no pricing-rate override.
Supplying --max-estimated-cost with default unknown rates therefore fails before any
API execution. No billed-cost or token cap guarantee is made. Failed SDK calls can
have unreported usage, as described in Task 9.

Tests mock agents/pipeline calls and block HTTP, using temporary outputs. They exercise
preflight errors, exact CSV question selection, no-key dry-run, live gating, sequential
execution, unique IDs, partial failures, no retries, null metrics, summaries, baseline
readiness/preservation, and unchanged public data. Task 10 performs only tests and the
required Q01/Q15 dry-run. No live/paid calls, independent evaluator, Version 2, single-agent
baseline, or Task 11 work is authorized or executed. Live execution awaits explicit approval.

## Safe failure diagnostics

Planner now captures allowlisted metadata before replacing an SDK exception with its
existing generic PlannerError. The user-facing message and no-retry behavior remain
unchanged. New failed runs save `diagnostics.json` and include the same record under
error.json/metadata.error.diagnostic and metrics.failure_diagnostics. This does not
modify existing finalized runs or restore previously discarded information.

The sdk_failure.v1 record contains only original_exception_class, http_status_code,
api_error_code, request_id, stage, and category (plus schema_version). Unknown values
are null. Categories distinguish authentication, quota, rate_limit, invalid_request,
model_access, network, schema_validation, server_error, and unknown using exception
types and scalar SDK metadata. Categories are best-effort clues, not conclusive root-
cause determinations: for example, a 403 may reflect permissions beyond model access.
Invalid server-side JSON-schema codes and local SDK/Pydantic validation failures are
classified as schema_validation. The metadata collector never reads/serializes provider
messages, exception strings, request/response bodies, headers, or traceback details.
Opaque codes/IDs are length/character checked and suppressed if credential-like or
containing the configured key. Logs use no exc_info or stack_info.

The pipeline also recognizes suppressed in-memory exception contexts (`raise ... from
None`) from other stages, without changing their existing error behavior. Once the
original process has exited, its SDK exception cannot be recovered from an old generic
error.json. Planner local post-output validation failures receive schema diagnostics.
Usage may remain unknown on failures; zero retrieval calls do not prove zero API calls
or zero billed usage.

Offline inspection of the first Q01 failure found PlannerError, no plan artifact, zero
retrieval, and unknown SDK usage. The saved configuration selected gpt-5-mini. Local
inspection confirms explicit model/key loading, Responses-model use, disabled temperature
overrides/retries, native ResearchPlan output, and successful SDK strict-schema conversion.
It cannot verify credential validity, billing/quota, live model permissions, networking,
or server-side acceptance. No live request was made to diagnose this failure. New mocked
tests inject real SDK authentication/quota/rate-limit/invalid-request/network/model-access
exceptions and schema-validation errors, and verify persisted diagnostics and redaction.

## Opt-in local Planner origin diagnostics

Set `CI_DIAGNOSTIC_MODE=1` in the process environment to include safe local origin
metadata in existing failure diagnostics. This does not execute or authorize a live run:

```powershell
$env:CI_DIAGNOSTIC_MODE = "1"
# No API command is executed here. Live testing still requires separate approval.
```

Python tests/tools may instead use `with local_diagnostics():` from ci_agents.diagnostics.
The mode defaults off; it is read from process environment, not .env. It does not enable
SDK/HTTP debug logging. It adds only phase and the innermost available Python origin
(module, function, line number). No file paths, source lines, frame locals, prompts,
credentials, request payloads, raw exception messages, or full traceback chains are
serialized. Origin metadata cannot reveal native/C frames or reconstruct lost tracebacks.

Planner phases are sdk_execution (including native structured parsing inside Runner),
telemetry_processing (observe_sdk_result after Runner returns), and output_validation
(local ResearchPlan postprocessing). Public generic errors, existing exception types,
no-retry controls, prompts, model selection, and ResearchPlan schema remain unchanged.
Unknown local TypeErrors remain category unknown rather than being blamed on the API.
Attached safe origin/phase information survives the pipeline's existing error persistence.

For the third Q01 failure, the saved TypeError has no origin or phase information.
Its approximately 33-second duration and absent plan/usage do not establish where it
occurred. Offline replays passed through both a mocked Responses resource and an
in-memory HTTP response using the real OpenAI client, model adapter, Runner, Pydantic
parsing, UTC conversion, Planner validation, and telemetry. These tests do not reproduce
the original provider response or establish the live root cause. Injected TypeErrors
verify diagnostic phase/origin handling only. No speculative SDK/parsing/schema fix,
package upgrade, or model change was applied. A failure after a remote generation may
have incurred API charges even without plan.json; the saved unknown usage cannot
establish billing. No live or paid calls were used for debugging.

## HTTP decoder investigation and compression configuration

Offline investigation reproduced the Planner's reported TypeError at
`httpx2._decoders.BrotliDecoder.decode`, line 191 in installed httpx2 2.16.0:
`self._decompress(data, output_buffer_limit=MAX_DECODE_CHUNK_SIZE // 2)`. The configured
limit is 524,288 bytes. On this interpreter, _decompress is Brotli 1.0.9's
Decompressor.process, which accepts the compressed bytes but rejects the additional
keyword with TypeError: process() takes no keyword arguments. Valid synthetic Brotli
bytes decode successfully with its one-argument API, so malformed input is not required
to reproduce the failure. This is a confirmed local optional-backend API mismatch and
matches the recorded live module/function/line; the original response headers/body were
not retained and have not been inspected.

Interpreter: C:\Users\maria\anaconda3\python.exe, Python 3.13.5. Before and after this
fix: openai 3.49.0, openai-agents 0.24.0, httpx 0.28.1, httpcore 1.0.9, httpx2 2.16.0,
httpcore2 2.16.0, Brotli 1.0.9, zstandard 0.23.0; zlib compiled/runtime 1.2.13.
No packages were installed, uninstalled, or upgraded. `python -m pip check` reports no
broken declared requirements. Installed distribution `httpx2` owns
C:\Users\maria\anaconda3\Lib\site-packages\httpx2\_decoders.py. Both OpenAI packages
explicitly require/import httpx2; openai's installed requirement is >=2.12.0,<3, and
Agents requires >=2.12.0. httpx2 pins httpcore2 2.16.0. The separate httpx/httpcore
packages are not the default transport used by this installed OpenAI SDK.

httpx2's optional Brotli extra requires Brotli >=1.2.0 on CPython, but a separately
installed older Brotli is auto-detected even when the extra was not requested. Thus
pip check can pass while this optional runtime API combination fails. The decoder
module is not unexpectedly imported from project code, and the wrappers did not supply
a custom HTTP client. The SDK constructs its default httpx2 client. No OPENAI_BASE_URL
or HTTP_PROXY/HTTPS_PROXY/ALL_PROXY environment overrides were present during inspection;
credential values and raw HTTP data were not printed.

The smallest application-level fix leaves the dependency stack untouched: all four
AsyncOpenAI constructors now use
`default_headers={"Accept-Encoding": OPENAI_ACCEPT_ENCODING}`, where the shared constant
in config.py is `gzip, deflate`. This overrides httpx2's automatically advertised
`gzip, deflate, br`, avoiding negotiation of the incompatible optional decoder. Model
configuration, prompts, ResearchPlan/all other schemas, retry settings, and agent
interfaces are unchanged. This is HTTP negotiation configuration, not a decoder patch.

Regression tests in test_http_decoding.py reproduce the exact installed-backend failure,
exercise synthetic Brotli negotiation through the real OpenAI client, verify that the
configured client selects gzip, verify gzip/deflate response deserialization, and check
all four wrappers' headers. In-memory transports and blocked external transports ensure
no live or paid API calls. Backend-specific reproduction tests skip if a future environment
removes the old Brotli incompatibility.

The fix depends on normal server compliance with Accept-Encoding; an endpoint returning
Brotli despite its exclusion can still hit the old decoder. Other applications in this
shared interpreter that negotiate Brotli remain affected. No live endpoint behavior was
verified, no global dependency repair was attempted, and no paid retry is authorized by
this investigation. Unknown billed usage for the earlier live failure remains unknown.


## Additive Version 2: one targeted recovery cycle

V1's pipeline.py, all four agent implementations, all four prompts, schemas, model
configuration, retrieval, persistence and telemetry modules remain byte-for-byte
unchanged for this implementation. Only the CLI gains a separate `v2` branch. The
existing gzip/deflate HTTP compression workaround is preserved in the shared agents.

New pipeline_v2.py delegates its complete initial pass to the existing V1 run_pipeline.
It does not duplicate the initial four-agent loop. Initial V1 artifacts are saved in
a separate immutable V1 run directory, then copied byte-for-byte under initial_* names
in the new V2 directory. An initial execution failure stops before recovery. An initial
PASS stops immediately. A FAIL may trigger at most one Researcher/Analyst/Reviewer cycle;
the second Reviewer is terminal even on FAIL. Exceptions stop without retry or repair.

Recovery routing is deterministic and versioned as targeted_recovery.v1:

- Major/critical missing_required_information with no known omitted record IDs: one
  targeted Researcher call, then Analyst revision and the existing Reviewer.
- Known retrieved-but-omitted missing_record_ids: Analyst revision without retrieval.
- Major/critical unsupported claims, citation mismatches, contradictions, temporal/unit
  errors, overstated certainty or weak inference: Analyst revision and Reviewer, using
  existing evidence unless another selected issue requests new evidence.
- Mixed material issues share the same single cycle; no separate loops per issue.
- Initial PASS, no material structured issues, or unactionable issues: skip and record
  the reason. Scores or vague missing-requirement text alone do not create research tasks.
  Minor-only issues do not trigger an otherwise unexplained failed review's recovery.
  Exact correction values N/A, none, no actionable correction, manual review required,
  or human review required are also skipped. The schema lacks an explicit actionability
  flag, so arbitrary free-text human-only instructions cannot be perfectly classified.

The policy is not specialized to any question, vendor, answer key, or previous audit.
It does not decide whether a Reviewer requirement is benchmark-valid. A bad/overbroad
Reviewer request may remain unresolvable; FAIL is retained instead of inventing facts.
The existing Researcher sees only targeted tasks derived from selected issue text and
public company/category context. Defaults limit recovery to 20 tool calls, 10 search
results/call, 15 full-record attempts and 15 selected evidence items. Override the call,
full-record and evidence limits with the corresponding --recovery-max-* flags.
This bounds tool work, not billed tokens or monetary cost.

The same Analyst interface takes only ResearchPlan and EvidencePack. Its V2-only plan
objective therefore carries clearly labelled recovery feedback and the original report
as revision context, explicitly not as additional factual evidence. No system prompt or
schema is changed. The original plan is retained and used for the final Reviewer.
Unchanged claim IDs are requested, but model compliance is not guaranteed. All added,
removed and modified claims are recorded by exact ID; no inferred semantic alignment
is fabricated. New evidence is merged by record ID with metadata consistency checks,
preserving old snapshots and supplementary findings. Original coverage scores are not
silently upgraded. The separate targeted pack retains recovery-task coverage.

### CLI examples (commands only; no live execution performed)

From the project directory after the existing PowerShell setup:

```powershell
# Existing V1 command remains unchanged.
python -m ci_agents.cli run --question-id Q01 --run-id v1-q01-next --dry-run

# V2 readiness from scratch, with no agent calls or artifacts.
python -m ci_agents.cli v2 --question-id Q01 --run-id v2-q01-fresh --dry-run

# Existing Q01 baseline: read-only compatibility and routing preview.
python -m ci_agents.cli v2 --baseline-dir ".\runs\pilots\pilot-f93e7472269f463480dc6fa2a73ddc33\runs\q01-c992a857b15446478a66dc2771c0b7b5" --run-id v2-q01-paired --dry-run
```

The following examples are paid-capable and require explicit authorization before use;
they were not run during implementation:

```powershell
python -m ci_agents.cli run --question-id Q01 --run-id v1-q01-next --allow-api
python -m ci_agents.cli v2 --question-id Q01 --run-id v2-q01-fresh --allow-api
python -m ci_agents.cli v2 --baseline-dir ".\runs\pilots\pilot-f93e7472269f463480dc6fa2a73ddc33\runs\q01-c992a857b15446478a66dc2771c0b7b5" --run-id v2-q01-paired --allow-api
```

V2 also accepts --question for a direct question, --as-of for an aware cutoff, and
--recovery-max-tool-calls / --recovery-max-full-records / --recovery-max-evidence-items.
A new run ID is always required if the chosen ID already exists. Without --dry-run or
--allow-api the CLI blocks execution. The library entry point run_pipeline_v2, like
run_pipeline, is explicitly paid-capable when invoked by a caller. No background work
or automatic live runs are registered. Dry-run does not invoke agents.

### Paired continuation and compatibility

v2_baseline.py accepts only a completed v1_fixed baseline. It verifies the manifest,
model, configured/explicit cutoff, public dataset hash, all four prompt fingerprints,
recorded Agents SDK version, question identity, stage order/status/telemetry, initial
report/review identity, public evidence metadata, and full-retrieval provenance. It
checks the unchanged Reviewer policy and report contracts. V2 baselines are rejected,
so a second recovery cannot be chained through the same interface. Historical baselines
lack agent source hashes and provider-resolved model snapshot IDs; those identities
cannot be retrospectively established. The model alias and recorded SDK are checked.

Independent evaluation/audit files are not parsed or fed to agents. Unexpected manifest
entries are rejected; unrelated unmanifested files are ignored. Private evaluation paths
and output directories inside the original baseline or public dataset are rejected.
No private benchmark keys are opened. Every original baseline file remains untouched;
future comparison should use baseline_reference.json and its manifest/artifact hashes.

### V2 artifacts and measurements

A new V2 directory contains initial_* snapshots, baseline_reference.json,
recovery_routing.json, recovery_feedback.json, and, as applicable:

- recovery_plan.json and recovery_evidence_pack.json for targeted new evidence;
- recovery_retrieval_logs.json and recovery_*_metrics.json, including partial failures;
- merged_evidence_pack.json, revision_plan.json, revised_report.json, second_review.json;
- claim_changes.json and evidence_changes.json, including newly discovered/full-fetched IDs;
- final_report.json, final_review.json when available, report.md, run_result.json;
- metrics.json, metadata.json, null evaluation_pending.json, manifest.json, and error.json on failure.

Metrics distinguish initial_v1 (inherited archive in paired mode), recovery_work
(Researcher/Analyst), re_review, incremental_total (includes re-review), and final_totals
(initial plus incremental). executed_this_invocation excludes inherited API costs in
paired mode. Do not add re_review a second time. No executed recovery stages means known
zero incremental usage/cost; unknown attempted-stage usage stays null and propagates.
Cached input tokens, LLM requests, retrieval calls and per-stage timing are preserved.
No stage-level measurements are inferred from character counts. Existing V1 PricingConfig
semantics are reused without changing its estimator: unknown rates leave costs null;
configured rates retain the existing all-input-at-one-rate assumption, not a new cached
pricing guarantee. Original pricing/cost records are never recomputed or overwritten.

metadata records initial/final Reviewer decisions, trigger IDs, recovery attempted/skipped/
completed/failed and reason, baseline run ID and hashes, experiment/comparison identity,
limits/policy version, and observed controller wall time. Paired final total latency is
inherited initial latency plus measured incremental recovery; observed controller time
is separately reported, not falsely presented as a newly executed baseline. A completed
recovery with final FAIL remains completed_review_failed, not independently successful.
If re-review execution fails after revision, final ReviewResult is null rather than
misattributing the original review to changed report text. Initial outputs stay available.

No independent evaluation scores, PDF generation, single-agent baseline, or additional
recovery cycles are implemented in this task. The prior five-dimension evaluation/PDF
design remains a separate future implementation. Tests mock all stage/API execution.
The incomplete literal pytest option `-p no` cannot load a plugin; use the project's
established `python -m pytest -q -p no:cacheprovider` to run the complete suite.


## Two-page PDF reporting (offline rendering)

Completed V1, pilot, and V2 runs automatically publish `reports/report-0001.pdf`
inside the run directory, after the artifact manifest is sealed. Regeneration
creates the next numbered PDF without replacing anything. The renderer uses
ReportLab only, never an LLM or private evaluation materials. Both architectures
share the same five performance dimensions. Internal Reviewer scores are labeled;
independent metrics remain Pending. Missing usage/cost is Unavailable; V1 recovery
is N/A. Cached tokens are a subset of input tokens. V2 totals include inherited
baseline work, with incremental recovery and observed controller time separate.

```powershell
python -m pip install -e .
ci-agents report --run-dir ".\runs\pilots\pilot-f93e7472269f463480dc6fa2a73ddc33\runs\q01-c992a857b15446478a66dc2771c0b7b5"
ci-agents run --question-id Q01 --run-id q01-v1-pdf --allow-api
ci-agents v2 --question-id Q01 --run-id q01-v2-pdf --allow-api
ci-agents v2 --baseline-dir ".\runs\pilots\pilot-f93e7472269f463480dc6fa2a73ddc33\runs\q01-c992a857b15446478a66dc2771c0b7b5" --run-id q01-v2-paired-pdf --allow-api
```

Only the `report` command above is fully offline. Execution commands explicitly
authorize API calls when you run them. Output is under configured RUNS_DIR.
PDF publication is best effort: a sanitized warning reports presentation failure
without changing experiment status; retry with `ci-agents report`.
To keep exactly two pages, sections over 3,000 characters use labeled excerpts
and the layout scales content to fit; full original text remains in JSON.
PDFs are presentation copies, outside the immutable experiment manifest.

PDF text/page regression tests use PyMuPDF: `python -m pip install -e ".[test]"`.
