# Retrieval threshold evaluation — assignment study

Study date: 15 November 2026, America/Toronto.

**Status: all 54 checkpoint observations completed, including recorded failures.** See the measured results, provisional recommendation and limitations below.

## Executive findings

**Use 32 as the provisional research ceiling among the three tested options. This study does not justify 16 as an equivalent lower-cost substitute.** Treat the ceiling as permission to continue purposeful research, rather than a requirement to spend every call.

| Retrieval ceiling | Mean quality score / 100 | Mean workflow tokens | Mean model requests | Mean model cost per attempted answer |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 8.3–19.4 | 24,235 | 4.17 | USD 0.0176 |
| 16 | 29.2–34.7 | 54,451 | 6.17 | USD 0.0300 |
| 32 | 44.4 | 130,260 | 9.61 | USD 0.0532 |

The score measures essential-answer coverage and is set to zero for identified material defects; **it is not factual accuracy**. Ranges reflect three remaining ungradable assessments, not confidence intervals. Each ceiling has 18 planned observations and includes failures. Costs are estimated agent text-token costs, excluding independent grading and real website/search-provider charges; retrieval here is a local synthetic corpus.

Going from 16 to 32 bought **9.7–15.3 observed mean quality points** for approximately **USD 0.0232 more per attempted answer**, about **33 additional measured stage seconds** on average. Tokens increased about **2.4×**. This is a better observed tradeoff under the stated two-point quality tolerance, despite diminishing evidence yield and higher cost per quality point.

**The limits of that conclusion matter:** 32 was the largest tested option, so this is not proof that benefits stop after call 32. Across-question uncertainty is wide; neither smaller cap met the noninferiority rule. The workflow also remains unreliable: three of eighteen Planner attempts failed, affecting nine checkpoint observations; eleven Reviewer responses were invalid. A higher cap does not fix these defects.

Read [all answers](runs/request_threshold_20261115_v4/answers.html), [the 36 incremental comparisons](runs/request_threshold_20261115_v4/incremental_benefits.csv), [source checks](runs/request_threshold_20261115_v4/SOURCE_AUDIT.md), and [the full machine-readable results](runs/request_threshold_20261115_v4/threshold_results.json).

![Quality versus retrieval ceiling](runs/request_threshold_20261115_v4/quality_curve.png)

Figure: question lines use lower bounds for ungradable cases. The grey band reflects the effect of those missing grades on the overall mean; it is not a sampling confidence interval. Question-cluster intervals are reported below.

## Question and predeclared design

Does continuing purposeful research beyond 8 or 16 retrieval operations improve
competitive-intelligence answers enough to justify the extra work, up to a cap of 32?
The treatment is retrieval operations, not a token or total API-request ceiling.

Questions: Q01 current Growth price; Q03 active promotions; Q05 verified breaches;
Q07 value for fifty Growth seats; Q12 announced versus available AI; Q15 threats to
PulseGrid. Each is repeated three times: 18 research trajectories, with 8/16/32
checkpoints, producing 54 answer observations. Remaining benchmark questions are held
out and are not used for threshold selection in this study.

Within each trajectory, the same original plan, discoveries, complete records and
pending research actions continue across checkpoints. Later research never receives
the checkpoint Analyst report, Reviewer feedback, private key or external grade.
The Researcher sees an overall horizon of 32 and not the intermediate checkpoints.
Independent operations are proposed in batches of up to four and executed by the
public-only controller. Pending operations survive checkpoint boundaries. This is
an experimental continuation policy, not the production Researcher's stopping policy.
No V2 recovery is used. Three repetitions estimate some model variability; they do
not turn six distinct questions into eighteen independent question samples.

## Private answer keys and scoring

Existing Q01/Q15 private provisional rubrics were reused; all six keys were audited
against original synthetic records and expanded under ignored
`evaluation_private/checkpoint_threshold_20261115_v2/`. Q15 explicitly rejects
unsupported price-advantage and hidden-discount inferences. There is no forced
canonical ranking of three threats. Q07 separates subscription arithmetic from
unknown all-in costs. Keys are provisional source audits, not owner-certified keys.

Native MLflow datasets, registered independent judges and `mlflow.genai.evaluate`
execute evaluation. A seven-case grader calibration includes correct answers,
deliberately wrong prices/breaches/strategy, and an abstention. Failures remain visible.
No model receives the intended calibration verdict. Structured output is checked for
complete item catalogues, valid source pointers, explicit defects and arithmetic.
Mechanical protocol normalization can remove a misplaced optional null, default
absent optional rank-allegation metadata to null, and mark citation support on an
uncited inference with no anchors as not applicable. This awards no positive citation
credit. All correctness decisions and material defects remain unchanged. Raw feedback
and normalization logs are preserved.

The pilot exposed remaining structural grading failures not covered by the seven
calibration examples. Every grade still structurally invalid after normalization
receives **one** fresh native-MLflow adjudication against the same private rubric,
under a frozen protocol prompt. Valid grades are not rerun, no agent research is
repeated, and no best-of-several score is selected. An adjudication that remains
invalid stays missing. Each layer is saved separately, with its model usage and
source trace; all quality judgments remain pending human-owner approval.

Primary endpoint: supported essential coverage on a 0–100 scale, gated to zero for
independently identified material defects. No Analyst report receives zero task
completion. Invalid external grading is missing, with 0–100 sensitivity bounds; it
does not receive an invented quality score. Report quality and operational success
are separate: a Reviewer validation failure can preserve a gradeable Analyst report.
Internal PASS/FAIL is recorded but is never the independent success criterion.

Predeclared practical equivalence margin: **2 quality points** versus the 32-call
reference. This is a policy tolerance, not a scientifically universal constant.
Paired question means average the three repeats. Uncertainty is computed by resampling
whole question clusters, preserving checkpoint pairing. Six clusters give limited
precision and generalizability. Missing grades require sensitivity analysis; material
errors and execution failures are additional decision gates. The 32-call answer is
checked against the private key and is not assumed correct.

## Resources and incremental benefit

Actual response usage is captured for agent SDK calls and the native MLflow judge's
Gateway requests, including responses later rejected by schema validation. HTTP
failures without returned usage remain unknown. Tokens, cache usage, resolved model
IDs, request attempts, estimated cost and latency are recorded without credentials.
Rates checked against official OpenAI documentation on 15 November 2026:

| Model | Input / cached input / output, USD per million tokens |
| --- | --- |
| GPT-5 mini | 0.25 / 0.025 / 2.00 |
| GPT-4.1 independent judge | 2.00 / 0.50 / 8.00 |

Sources: [GPT-5 mini](https://developers.openai.com/api/docs/models/gpt-5-mini),
[GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1).
Estimates are not provider invoices; missing cache data is conservatively priced as
uncached input. Actual experiment spending includes every checkpoint's shadow
analysis/review and all grader calls. Counterfactual stop-at-checkpoint cost includes
the original Planner, research through that point, and that checkpoint's final
analysis/review; it excludes earlier shadow reports because they do not feed research.

For each 8→16 and 16→32 extension, record new full sources, newly met/lost essential
requirements, changes in independent material defects, research tokens/requests/cost,
and checkpoint answer-quality change. Keep raw defects rather than pretending that
different regenerated claim IDs identify the same error across answers.

## Reproduction and artifacts

Use the cached Python runtime in `requirements-experiment.txt` and native local MLflow
SQLite tracking. No external tracing service is needed. The runner supports a small
subset first, then the complete design using the same frozen checkpoints; completed
agent work is not repeated. Interrupted paid stages require an audit rather than an
automatic retry. All source/prompt/key/configuration hashes are frozen before calls.

```bash
uv run --no-project --with-requirements requirements-experiment.txt \
  python scripts/prepare_checkpoint_keys.py \
  --output-dir evaluation_private/checkpoint_threshold_20261115_v2

uv run --no-project --with-requirements requirements-experiment.txt \
  python scripts/run_checkpoint_evaluation.py \
  --output-dir runs/request_threshold_20261115_v4 \
  --key-dir evaluation_private/checkpoint_threshold_20261115_v2 \
  --questions Q01 Q15 --repetitions 1 --allow-api

# After the small subset passes operational checks, complete all 54 observations.
uv run --no-project --with-requirements requirements-experiment.txt \
  python scripts/run_checkpoint_evaluation.py \
  --output-dir runs/request_threshold_20261115_v4 \
  --key-dir evaluation_private/checkpoint_threshold_20261115_v2 --allow-api
```

Run artifacts, MLflow's local `mlruns/` artifact store and private keys are ignored by Git. The final recommendation and its
limitations will be recorded in this README; exact answers, per-case deltas, grades,
usage ledgers and frozen manifests remain in the study's run directory.

## Preparation checks and safeguards

- The complete 54-condition design is stored as a native MLflow dataset before any
  agent work. `--prepare-only` verifies tracing and freezes the design with no model calls.
- The Researcher's pending operations and evidence survive checkpoint resume; full
  retrieval is required for citation eligibility. Tests cover this boundary.
- Model usage is recorded from returned API usage, including cached tokens and
  responses whose output later fails schema validation. Unknown failed-request usage
  remains unknown rather than becoming zero.
- Native judge HTTP retries are set to zero. A minimal generation preflight detects
  missing credits before a long judge retry loop; authentication secrets are not logged.
- Redundant `retrieval_text` and irrelevant metadata are removed from evaluator-only
  source serialization; original facts, source text, dates, provenance and IDs remain.
- Analysis is tested on an explicitly synthetic plateau fixture: known improvement
  from 8 to 16 and none from 16 to 32 must select 16, preserving all 36 paired deltas.
  These fixture outputs are tests, **not assignment results**.
- The final integrity verifier checks source/key hashes, all 54 identities, nested
  retrieval histories, source-text provenance, citation membership and unique counted
  model response IDs. It runs only after genuine study observations exist.

The final analysis commands are:

```bash
uv run --no-project --with-requirements requirements-experiment.txt \
  python scripts/verify_checkpoint_study.py \
  --run-dir runs/request_threshold_20261115_v4 \
  --key-dir evaluation_private/checkpoint_threshold_20261115_v2

uv run --no-project --with-requirements requirements-experiment.txt \
  python scripts/analyze_checkpoint_study.py \
  --run-dir runs/request_threshold_20261115_v4 \
  --readme BUDGET_THRESHOLD_EVALUATION_README.md
```

Method references: [native MLflow evaluation datasets](https://mlflow.org/docs/latest/genai/datasets/),
[paired evaluation comparisons](https://mlflow.org/docs/latest/genai/eval-monitor/comparing-evaluation-runs/).

## Results — completed study

**Recommendation: use 32 as the provisional retrieval ceiling for this tested continuation policy.** Best observed quality/cost compromise: smallest budget within two observed mean points of the best conservative mean, with no more observed material-error answers. Smaller-cap noninferiority was not established under the predeclared uncertainty/error gates.

This is a retrieval-operation ceiling, not a token ceiling or total API-request limit. It is not proof of a universally optimal budget.

| Ceiling | Answer snapshots | Quality score / 100 | Material-error answers | Execution failures | Ungradable | Mean actual retrievals | Mean model requests if stopped here | Mean workflow tokens if stopped here | Mean workflow USD: observed cache / uncached upper |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 18 | 8.33–19.44 (missing-grade bounds) | 10 | 9 | 2 | 6.67 | 4.17 | 24235 | 0.01758 / 0.01762 |
| 16 | 18 | 29.17–34.72 (missing-grade bounds) | 8 | 7 | 1 | 13.33 | 6.17 | 54451 | 0.02999 / 0.03018 |
| 32 | 18 | 44.44 | 7 | 4 | 0 | 26.00 | 9.61 | 130260 | 0.05315 / 0.05392 |

### Incremental benefits

| Extension | Mean quality change, all 18 paired attempts | Newly met essential items, total | Lost essential items, total | New full records, total | Measured additional research tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| 8→16 | +9.72 to +26.39 (missing-grade bounds) | 11 | 1 | 43 | 450307 |
| 16→32 | +9.72 to +15.28 (missing-grade bounds) | 9 | 3 | 55 | 1271965 |

### Separate quality dimensions

These are conditional means among applicable valid assessments, with denominators shown. Missing reports and invalid grading are not assigned perfect accuracy. The primary quality score above retains all 18 attempts per ceiling and penalizes material defects. Claim fractions cover structured claims; prose defects are checked separately.

| Ceiling | Completeness | Factual correctness | Citation support | Temporal correctness |
| ---: | --- | --- | --- | --- |
| 8 | 0.481 (n=13) | 0.939 (n=11) | 0.947 (n=11) | 0.939 (n=11) |
| 16 | 0.732 (n=14) | 0.922 (n=13) | 0.963 (n=13) | 0.821 (n=12) |
| 32 | 0.800 (n=15) | 0.959 (n=15) | 0.981 (n=15) | 0.870 (n=14) |

Each individual extension, its new source IDs, requirement gains/losses, defect sets, and incremental usage are in `incremental_benefits.json` and `.csv`. Gains do not imply that every extra source improved the answer.

Requirement gain/loss counts use only available valid item judgments; unknown judgments are not invented. Source and resource totals include all attempted trajectories, including zero additional research after Planner failure.

Extensions with no added retrieval are explicitly marked: answer changes there cannot be attributed to extra evidence. Repeated synthesis itself is stochastic.

Observed marginal quality/cost ratios (a business must decide its willingness to pay per quality point):

```json
[
  {
    "from_budget": 8,
    "to_budget": 16,
    "mean_quality_gain": null,
    "mean_quality_gain_bounds": [
      9.722222222222221,
      26.38888888888889
    ],
    "mean_additional_research_cost_usd": 0.008814472222222222,
    "mean_change_in_stop_at_budget_workflow_cost_usd": 0.012409736111111111,
    "research_usd_per_quality_point": null,
    "workflow_usd_per_quality_point_bounds": [
      0.0004702636842105263,
      0.00127643
    ],
    "interpretation": "Observed ratio, not a guaranteed return or business valuation. Nonpositive quality gain has no positive quality-return ratio."
  },
  {
    "from_budget": 16,
    "to_budget": 32,
    "mean_quality_gain": null,
    "mean_quality_gain_bounds": [
      9.722222222222221,
      15.277777777777779
    ],
    "mean_additional_research_cost_usd": 0.022008800000000002,
    "mean_change_in_stop_at_budget_workflow_cost_usd": 0.02316533611111111,
    "research_usd_per_quality_point": null,
    "workflow_usd_per_quality_point_bounds": [
      0.0015162765454545455,
      0.002382720285714286
    ],
    "interpretation": "Observed ratio, not a guaranteed return or business valuation. Nonpositive quality gain has no positive quality-return ratio."
  }
]
```

### Paired uncertainty and decision rule

Comparisons use paired question means, preserving three repetitions within each question. The intervals below are 97.5% empirical question-cluster bootstrap intervals; Bonferroni adjustment covers the two candidate-vs-32 comparisons at an approximate family-wise 95% level. With six question clusters, this is limited evidence, not guaranteed coverage. Missing-grade sensitivity bounds are included.

- 8 minus 32: mean difference None; interval [-76.39, 16.67]. Two-point noninferiority margin met: False. Observed material-error/execution gate met: False.
- 16 minus 32: mean difference None; interval [-33.33, 15.28]. Two-point noninferiority margin met: False. Observed material-error/execution gate met: False.

### Per-question quality

| Question | 8 | 16 | 32 |
| --- | ---: | ---: | ---: |
| Q01 | 0.0–0.0 | 33.3–33.3 | 66.7–66.7 |
| Q03 | 0.0–0.0 | 66.7–66.7 | 100.0–100.0 |
| Q05 | 0.0–33.3 | 33.3–33.3 | 66.7–66.7 |
| Q07 | 8.3–8.3 | 8.3–41.7 | 0.0–0.0 |
| Q12 | 0.0–0.0 | 0.0–0.0 | 0.0–0.0 |
| Q15 | 41.7–75.0 | 33.3–33.3 | 33.3–33.3 |

### Total experiment resources

| Work | Request attempts | Responses with usage | Known tokens | Estimated cost of known responses, USD | Unknown-usage attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Agent work, including shadow reports | 229 | 229 | 2,889,512 | 1.335866 | 0 |
| Independent grading and calibration | 76 | 69 | 1,315,858 | 3.048446 | 7 |
| Billing checks | 3 | 3 | 36 | 0.000108 | 0 |
| Whole final study | 308 | 301 | 4,205,406 | 4.384420 | 7 |

Actual study cost includes shadow checkpoint reports and independent evaluation. It differs from the counterfactual cost of answering one question at a selected ceiling. Cache warming and execution order can affect the observed-cache estimate; the all-input-uncached estimate provides a conservative comparison. Unknown failed-request usage is not assumed to be zero. Earlier engineering/calibration attempts outside this study directory are excluded from these totals.

### Benefits and limitations of the ceilings

- **8 — cheapest, but too restrictive in this experiment.** It averaged 1.83 full records per planned observation and had ten confirmed material-defect answers. Some research sessions spent their first eight operations discovering sources without reading any full records. It did not provide acceptable coverage for this question mix.
- **16 — meaningful improvement at moderate cost.** The 8→16 extensions added 43 full sources and eleven known essential requirement gains, with one known loss. Mean workflow cost was about USD0.030. It still missed important current prices or competitor coverage in some cases, and its most optimistic mean score remained below the 32-call score by 9.7 points. It was not established as equivalent within the two-point tolerance.
- **32 — strongest observed overall, with real diminishing returns.** The 16→32 extensions added 55 full sources and nine known requirement gains, with three known losses. Marginal full-source yield fell from 43/120 = 35.8% in the first extension to 55/228 = 24.1% in the second. Pricing, promotions and security had higher observed question means, while value, AI availability and threat synthesis did not consistently improve. Thirty-two used about 2.4× the tokens and 1.8× the estimated workflow cost of sixteen. Seven material-defect answers and four execution failures remained at this ceiling.

All fifteen trajectories with valid plans reached the 8- and 16-call observation points. Fourteen used 32 operations; one pricing trajectory stopped at 20. Three planned trajectories never researched because their Planner failed. These failures stay in every applicable denominator.

The first-pass grader produced fourteen structurally invalid assessments. Mechanical normalization resolved six, and the one fixed adjudication pass resolved another five. Three remain unknown. The recommendation also favored 32 in the preserved normalization-only sensitivity analysis, before the fresh adjudications. All original and revised grading layers are available for audit.

**No universal token limit or point of zero marginal benefit was established.** This is evidence for choosing 32 from the tested options under this experimental continuation policy. The production Researcher uses a different early-stop/bootstrap policy; validating its default requires a separate matched test. Larger budgets above 32, held-out questions and real article retrieval were not evaluated here.

### Assignment interpretation

Report the threshold as a provisional empirical choice for these six synthetic benchmark questions and this frozen continuation policy. Do not claim a proven token limit, universal optimum, or certified material-error probability. Keep retrieval calls, model requests and tokens separate. Follow-up validation should use the held-out questions and the actual production stopping policy.


### Verification and artifacts

The final offline suite passes **405 tests**, with **2 skipped**. Integrity checks confirm 54 unique observations, frozen source/key hashes, nested retrieval prefixes and original full-source citation provenance. Token counts from **219 native SDK response spans** were cross-checked against the append-only usage ledger. Every traced response was present and agreed on token usage; no model response ID was counted twice.

- `experiment_plan.json`, `design_dataset.json`: frozen controls and the native 54-condition dataset.
- `verification_summary.json`: source/provenance and usage integrity checks.
- `case_analysis.csv` / `.json`: each outcome and its resources.
- `incremental_benefits.csv` / `.json`: all 36 extensions, essential-item changes, new sources and defect sets.
- `retrieval_diagnostics.json`: empty/duplicate lookups and source discovery behavior.
- `request_usage.jsonl`, `adjudication_usage.jsonl`: actual returned model usage, with no credentials.
- `protocol_normalization_manifest.json`, `analysis_protocol.json`, `protocol_adjudication/`: grading revisions and the one-pass policy.
- `normalized_layer_threshold_results.json`: sensitivity results before fresh adjudication.
- `SOURCE_AUDIT.md`: source spot-checks and the three grades needing human review.
- `answers.html`, `quality_curve.png`, `quality_curve.svg`: readable outputs and an exportable figure.

Private reference answers are in `evaluation_private/checkpoint_threshold_20261115_v2/ANSWER_KEY_README.md`. If reproducing on a fresh checkout without the earlier private base rubrics, first run `scripts/prepare_budget_rubrics.py --output-dir evaluation_private/budget_screen_v2`; do not overwrite an existing private key directory. Use a new output directory for an independent replication. Existing final checkpoints are resumed/reused rather than repeating paid agent work.
