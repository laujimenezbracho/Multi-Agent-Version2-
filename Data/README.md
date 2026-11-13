# Competitive Intelligence Synthetic Benchmark v2

All companies, events, URLs, and source statements are fictional. The URLs are inert placeholders.

## PUBLIC INPUT (give these to agents)
- public_web_records.csv / .jsonl: simulated retrieved web pages and source metadata; do not treat `verification_status` as ground truth.
- pricing_events.csv: source-level price observations, including deliberately stale third-party data.
- public_questions.csv: benchmark prompts and question IDs (without answer keys).

## PRIVATE EVALUATION (never give to planner/researcher/analyst/reviewer)
- private_benchmark_answer_key.csv: task rubrics and expected elements.
- private_fact_ledger.csv: verified canonical claims and record provenance.
- private_current_price_truth.csv: canonical current official pricing.
- private_computed_price_checks.csv: price calculations.
- private_conflict_tests.csv: seeded contradiction assertions.

## TIME RULES
- Evaluation cutoff: 2026-11-11 23:59:59 UTC. `available_at` represents earliest agent access.
- For point-in-time questions filter available_at <= as-of timestamp AND effective_date <= requested date.
- Do not use future corrections to answer earlier historical questions.
- Promotion active status requires start/end dates, eligibility, and trustworthy corroboration.

## EXPERIMENTAL FAIRNESS
- All configurations receive the same public snapshot, retrieval tools, model settings and report requirements.
- Freeze answer keys before final evaluation. Q01/Q15 are designated pilot and should not be used for confirmatory statistics.
- Use independent evaluation against private keys. Reviewer scores are internal routing signals only.
- Interpretive tasks do not have a single true ranking; grade coverage, factual grounding and uncertainty.
- The original 290 records have simulated source credibility; many are unverified. The 20 controlled records create deliberate conflict/availability tests.
- Unknown onboarding fees and AI entitlements must remain unknown, not imputed as zero.

## SUGGESTED RETRIEVAL API
search_records(query, company=None, category=None, before=None, limit=20)
get_record(record_id)
get_pricing_history(company, plan, as_of)
All responses should return record_id, title, summary, dates, and source_url.
