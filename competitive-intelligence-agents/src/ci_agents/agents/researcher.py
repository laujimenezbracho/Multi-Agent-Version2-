"""Bounded Researcher with cutoff-bound local SDK tools and provenance checks."""

from copy import deepcopy
from importlib.resources import files
import json
import logging
import re
from typing import Literal, Protocol

from agents import Agent, OpenAIResponsesModel, Runner, function_tool
from agents.exceptions import MaxTurnsExceeded
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ..telemetry import observe_retrieval_event, observe_no_model_call
from ..config import OPENAI_ACCEPT_ENCODING, load_config, agent_model_settings
from ..dataset import _date, load_dataset
from ..retrieval import RetrievalError, RetrievalStore
from ..schemas import (
    Contract,
    EvidenceItem,
    EvidencePack,
    ResearchPlan,
    TaskCoverage,
    Text,
)
from ..scope import scope_contract
from ..source_observations import pricing_observation, validate_pricing_observation
from ..workflow_runtime import active_runtime, run_structured, BudgetExceeded

logger = logging.getLogger(__name__)


class ResearcherError(RuntimeError):
    """Sanitized SDK execution failure; no retry is attempted."""


class EvidenceValidationError(ResearcherError):
    """Evidence could not be traced to full records retrieved during this run."""


class RequirementAssessmentError(EvidenceValidationError):
    """Invalid self-assessment, distinct from a source-provenance violation."""


class ResearchOutcome(Contract):
    stop_reason: Literal[
        "sufficient_for_question", "budget_exhausted", "no_matching_records"
    ]


class RequirementAssessment(Contract):
    task_id: Text
    requirement_index: int = Field(ge=0)
    status: Literal["supported", "partial", "missing", "conflicting", "unassessed"]
    supporting_record_ids: list[Text]
    explanation: Text


class GapResearchOutcome(ResearchOutcome):
    stop_reason: Literal[
        "sufficient_for_question",
        "budget_exhausted",
        "no_matching_records",
        "unresolved_requirements",
    ]
    requirement_assessments: list[RequirementAssessment]


class ResearcherLimits(BaseModel):
    """Explicit per-run configuration, independent of model-generated content."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_tool_calls: int = Field(default=60, ge=1, le=200, strict=True)
    max_search_results: int = Field(default=10, ge=1, le=50, strict=True)
    max_evidence_items: int = Field(default=40, ge=1, le=100, strict=True)
    max_full_records: int = Field(default=40, ge=1, le=100, strict=True)
    max_model_turns: int | None = Field(default=None, ge=2, le=202, strict=True)


class RetrievalService(Protocol):
    """Trusted synchronous public retrieval service (normally RetrievalStore)."""

    def search_records(
        self, query, company=None, category=None, as_of=None, limit=20
    ): ...
    def get_record(self, record_id, as_of=None): ...
    def get_pricing_history(self, company=None, plan=None, as_of=None): ...


def _json(value):
    return TypeAdapter(object).dump_json(value).decode("utf-8")


class ResearchSession:
    """Per-run provenance ledger. Never shared between Researcher invocations."""

    def __init__(self, plan, service, limits):
        self.plan = plan
        self.service = service
        self.limits = limits
        self.calls = 0
        self.full_calls = 0
        self.records = {}
        self.log = []
        self.cache = {}

    def event(self, **fields):
        self.log.append(json.dumps(fields, sort_keys=True))
        observe_retrieval_event(fields)

    def invoke(self, name, arguments):
        runtime = active_runtime()
        if runtime:
            runtime.check("Researcher", retrieval=True)
            key = (name, json.dumps(arguments, sort_keys=True, default=str))
            if key in self.cache:
                cached = self.cache[key]
                self.event(
                    kind="retrieval_cache",
                    requested_tool=name,
                    arguments=arguments,
                    status="cache_hit",
                    note="Per-session cache, same corpus object and cutoff; no new retrieval operation",
                )
                return cached
        if self.calls >= self.limits.max_tool_calls:
            return self.exhausted(name, "max_tool_calls")
        self.calls += 1
        if name == "get_record":
            if (
                arguments.get("record_id") not in self.records
                and len(self.records) >= self.limits.max_evidence_items
            ):
                return self.exhausted(name, "max_evidence_items")
            if self.full_calls >= self.limits.max_full_records:
                return self.exhausted(name, "max_full_records")
            self.full_calls += 1  # Attempts, including missing IDs, consume budget.
        args = dict(arguments)
        if name == "search_records":
            limit = args.get("limit", self.limits.max_search_results)
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                self.event(tool=name, status="invalid_limit")
                return _json({"error": "limit must be a positive integer"})
            args["limit"] = min(limit, self.limits.max_search_results)
        args["as_of"] = self.plan.as_of
        try:
            result = getattr(self.service, name)(**args)
        except RetrievalError:
            self.event(tool=name, arguments=arguments, status="invalid_input")
            return _json(
                {
                    "error": "Invalid retrieval input; use supported filters and query text"
                }
            )
        if name == "get_record":
            if result is not None:
                if (
                    result.get("record_id") != args["record_id"]
                    or _date(result["available_at"]) > self.plan.as_of
                ):
                    raise EvidenceValidationError(
                        "Retrieval service returned an invalid or unavailable source"
                    )
                self.records[result["record_id"]] = deepcopy(result)
            ids = [] if result is None else [result["record_id"]]
        else:
            # Defensive check as well as service-side cutoff filtering.
            for row in result:
                if _date(row["available_at"]) > self.plan.as_of:
                    raise EvidenceValidationError(
                        "Retrieval service returned post-cutoff evidence"
                    )
                if (
                    name == "get_pricing_history"
                    and _date(row["source_available_at"]) > self.plan.as_of
                ):
                    raise EvidenceValidationError(
                        "Pricing source was unavailable at cutoff"
                    )
            ids = [row.get("record_id", row.get("source_record_id")) for row in result]
        self.event(
            tool=name,
            arguments=arguments,
            as_of=self.plan.as_of.isoformat(),
            status="ok",
            record_ids=ids,
            result_count=len(ids),
            tool_calls=self.calls,
            full_record_calls=self.full_calls,
        )
        encoded = _json(result)
        if runtime:
            self.cache[key] = encoded
        return encoded

    def coverage_ledger(self):
        rows = []
        for task in self.plan.tasks:
            if task.priority != "essential":
                continue
            candidates = [
                rid
                for rid, r in self.records.items()
                if r["company_name"] in task.companies
                and r.get("event_category") in task.categories
            ]
            for index, requirement in enumerate(task.evidence_requirements):
                rows.append(
                    {
                        "task_id": task.task_id,
                        "requirement_index": index,
                        "requirement": requirement,
                        "candidate_record_ids": candidates,
                        "status": "needs_semantic_check"
                        if candidates
                        else "missing_full_records",
                    }
                )
        return rows

    def validate_outcome(self, outcome):
        if not isinstance(outcome, GapResearchOutcome):
            raise RequirementAssessmentError(
                "Return a requirement assessment for every essential requirement"
            )
        expected = {
            (row["task_id"], row["requirement_index"]) for row in self.coverage_ledger()
        }
        actual = [
            (row.task_id, row.requirement_index)
            for row in outcome.requirement_assessments
        ]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise RequirementAssessmentError(
                "Requirement assessments must match each essential task/index exactly once"
            )
        for row in outcome.requirement_assessments:
            if not set(row.supporting_record_ids).issubset(self.records):
                raise RequirementAssessmentError(
                    "Requirement support requires a full record retrieved this session"
                )
            if (
                row.status in {"supported", "partial", "conflicting"}
                and not row.supporting_record_ids
            ):
                raise RequirementAssessmentError(
                    "A non-missing assessment requires exact supporting record IDs"
                )
        if outcome.stop_reason == "no_matching_records" and any(
            row.status == "supported" for row in outcome.requirement_assessments
        ):
            raise RequirementAssessmentError(
                "Some supported requirements exist; use a qualified stop rather than claiming no matching records"
            )
        if outcome.stop_reason == "sufficient_for_question" and any(
            row.status != "supported" for row in outcome.requirement_assessments
        ):
            self.event(
                kind="research_stop_normalized",
                reported_stop_reason=outcome.stop_reason,
                recorded_stop_reason="unresolved_requirements",
                note="Preserved all evidence and assessments; unresolved essentials cannot be recorded as complete.",
            )
            outcome = outcome.model_copy(
                update={"stop_reason": "unresolved_requirements"}
            )
        return outcome

    def unassessed_outcome(self, reason):
        self.event(
            kind="requirement_assessment_unavailable",
            reason=reason,
            action="Keep full verified source records; do not fabricate requirement coverage",
        )
        return GapResearchOutcome(
            stop_reason="unresolved_requirements",
            requirement_assessments=[
                RequirementAssessment(
                    task_id=row["task_id"],
                    requirement_index=row["requirement_index"],
                    status="unassessed",
                    supporting_record_ids=[],
                    explanation="No valid Researcher coverage judgment was returned. Inspect the supplied full records; this does not establish that evidence is missing.",
                )
                for row in self.coverage_ledger()
            ],
        )

    def bootstrap(self):
        """Budget-counted, balanced discovery; never reads evaluator inputs."""
        runtime = active_runtime()
        if runtime and runtime.policy.targeted_research:
            return self.targeted_bootstrap()
        tasks = [task for task in self.plan.tasks if task.priority == "essential"]
        companies = sorted({company for task in tasks for company in task.companies})
        price_scope = (
            scope_contract(self.plan.original_question)["kind"]
            == "published_list_price"
        )
        tier = re.search(
            r"\b(starter|growth|enterprise)\b", self.plan.original_question, re.I
        )
        tier = tier.group().title() if tier else None
        chosen = []
        for company in companies:
            if self.calls >= self.limits.max_tool_calls:
                break
            if price_scope:
                rows = json.loads(
                    self.invoke(
                        "get_pricing_history", {"company": company, "plan": tier}
                    )
                )
                if isinstance(rows, list):
                    rows = [
                        r
                        for r in rows
                        if r.get("effective_date")
                        and _date(r["effective_date"]) <= self.plan.as_of
                    ]
                    rows.sort(
                        key=lambda r: (r["effective_date"], r["source_available_at"]),
                        reverse=True,
                    )
            else:
                rows = json.loads(
                    self.invoke(
                        "search_records",
                        {
                            "query": company,
                            "company": company,
                            "limit": self.limits.max_search_results,
                        },
                    )
                )
                if isinstance(rows, list):
                    rows.sort(key=lambda r: r["available_at"], reverse=True)
            if isinstance(rows, list) and rows:
                chosen.append(rows[0].get("record_id", rows[0].get("source_record_id")))
            elif self.calls < self.limits.max_tool_calls:
                # A short lexical fallback, not an unlogged retry or corpus-wide
                # assertion that records do not exist after one narrow search.
                fallback = json.loads(
                    self.invoke(
                        "search_records",
                        {
                            "query": tier or "AI",
                            "company": company,
                            "limit": self.limits.max_search_results,
                        },
                    )
                )
                if isinstance(fallback, list) and fallback:
                    chosen.append(fallback[0]["record_id"])
        for rid in chosen:
            if self.calls >= self.limits.max_tool_calls:
                break
            self.invoke("get_record", {"record_id": rid})
        for row in list(self.records.values()):
            old = row.get("supersedes_record_id")
            if (
                old
                and old not in self.records
                and self.calls < self.limits.max_tool_calls
            ):
                self.invoke("get_record", {"record_id": old})

    def targeted_bootstrap(self):
        """Read evidence after each discovery, before spending all calls on searches."""
        tasks = [task for task in self.plan.tasks if task.priority == "essential"]
        tier = re.search(
            r"\b(starter|growth|enterprise)\b", self.plan.original_question, re.I
        )
        tier = tier.group().title() if tier else None
        raw = list(
            dict.fromkeys(
                (company, category)
                for task in tasks
                for company in task.companies
                for category in task.categories
            )
        )
        by_company = {}
        for company, category in raw:
            by_company.setdefault(company, []).append(category)
        if re.search(
            r"\b(price|prices|pricing|cheapest|cost)\b",
            self.plan.original_question,
            re.I,
        ):
            for categories in by_company.values():
                categories.sort(key=lambda category: category != "Pricing")
        # Cover each company before spending calls on a second category for one.
        pairs = [
            (company, categories[index])
            for index in range(max((len(v) for v in by_company.values()), default=0))
            for company, categories in by_company.items()
            if index < len(categories)
        ]
        for company, category in pairs:
            if self.calls + 2 > self.limits.max_tool_calls:
                break  # Leave room to read what discovery finds.
            if category == "Pricing" and tier:
                rows = json.loads(
                    self.invoke(
                        "get_pricing_history", {"company": company, "plan": tier}
                    )
                )
                if isinstance(rows, list):
                    rows = [
                        r
                        for r in rows
                        if r.get("effective_date")
                        and _date(r["effective_date"]) <= self.plan.as_of
                    ]
                    rows.sort(
                        key=lambda r: (r["effective_date"], r["source_available_at"]),
                        reverse=True,
                    )
            else:
                rows = json.loads(
                    self.invoke(
                        "search_records",
                        {
                            "query": company,
                            "company": company,
                            "category": category,
                            "limit": self.limits.max_search_results,
                        },
                    )
                )
            if isinstance(rows, list) and rows:
                rid = rows[0].get("record_id", rows[0].get("source_record_id"))
                self.invoke("get_record", {"record_id": rid})
        for row in list(self.records.values()):
            old = row.get("supersedes_record_id")
            if (
                old
                and old not in self.records
                and self.calls < self.limits.max_tool_calls
            ):
                self.invoke("get_record", {"record_id": old})

    def assemble(self, outcome):
        """Compile citations and metadata ONLY from successful full retrievals."""
        if not isinstance(outcome, ResearchOutcome):
            raise EvidenceValidationError(
                "Researcher must return a ResearchOutcome, not invented evidence records"
            )
        if not self.log:
            raise EvidenceValidationError(
                "Research cannot abstain without attempting retrieval"
            )
        self.event(
            kind="research_outcome",
            stop_reason=outcome.stop_reason,
            retrieval_limit_reached=self.calls >= self.limits.max_tool_calls,
            allowed_tool_calls=self.calls,
            full_records=len(self.records),
        )
        items = []
        metadata = (
            "record_id",
            "claim_id",
            "company_name",
            "source_type",
            "source_url",
            "published_at",
            "available_at",
            "effective_date",
            "end_date",
            "source_independence_group",
            "supersedes_record_id",
            "verification_status",
            "feature_status",
        )
        for source in self.records.values():
            fields = {key: source.get(key) for key in metadata}
            if active_runtime() and source.get("event_category") == "Pricing":
                fields["pricing_observation"] = pricing_observation(source)
            full = "\n".join(
                str(source[key])
                for key in ["summary", "details", "claim_text"]
                if source.get(key)
            )
            items.append(
                EvidenceItem(
                    **fields,
                    source_title=source["document_title"],
                    finding=source.get("summary")
                    or source.get("claim_text")
                    or source["document_title"],
                    quoted_or_paraphrased_evidence=full,
                    confidence=0.5,
                    limitations=[
                        "Original source assertion, not independently verified. Confidence 0.5 is a neutral structural placeholder, not measured source reliability."
                    ],
                )
            )
        coverage = []
        missing = []
        for task in self.plan.tasks:
            ids = [
                item.record_id
                for item in items
                if item.company_name in task.companies
                and self.records[item.record_id].get("event_category")
                in task.categories
            ]
            notes = [
                "Retrieved record membership does not establish complete coverage of every requirement."
            ]
            if not ids:
                missing.append(
                    f"Task {task.task_id}: no matching full record retrieved; inspect the query log, not a claim that the corpus contains no evidence."
                )
            coverage.append(
                TaskCoverage(
                    task_id=task.task_id,
                    coverage_score=None,
                    supporting_record_ids=ids,
                    notes=notes,
                )
            )
        if not items:
            missing.append(
                "No full records were retrieved after the logged discovery attempts. This is not proof that relevant evidence does not exist."
            )
        if self.calls >= self.limits.max_tool_calls:
            missing.append(
                "Retrieval allowance reached; further searches were not performed."
            )
        if isinstance(outcome, GapResearchOutcome):
            self.event(
                kind="requirement_assessments",
                assessments=[
                    row.model_dump() for row in outcome.requirement_assessments
                ],
                validation="Structural checks only; Researcher self-assessment is not independent ground truth",
            )
            for row in outcome.requirement_assessments:
                if row.status != "supported":
                    missing.append(
                        f"Task {row.task_id}, requirement {row.requirement_index}: {row.status}. {row.explanation}"
                    )
        pack = EvidencePack(
            original_question=self.plan.original_question,
            as_of=self.plan.as_of,
            evidence_items=items,
            coverage_by_task=coverage,
            missing_information=missing,
            contradictions=[],
            retrieval_log=list(self.log),
        )
        return self.validate(pack)

    def exhausted(self, name, budget):
        self.event(tool=name, status="budget_exhausted", budget=budget)
        logger.warning("Researcher retrieval budget exhausted: %s", budget)
        return _json(
            {
                "error": "budget_exhausted",
                "budget": budget,
                "instruction": "Use retrieved evidence and report remaining gaps; do not invent evidence",
            }
        )

    def tools(self):
        @function_tool(failure_error_function=None)
        async def search_records(
            query: str,
            company: str | None = None,
            category: str | None = None,
            limit: int = 10,
        ) -> str:
            """Search public source snippets at the plan cutoff. Fetch full records before citing."""
            return self.invoke(
                "search_records",
                dict(query=query, company=company, category=category, limit=limit),
            )

        @function_tool(failure_error_function=None)
        async def get_record(record_id: str) -> str:
            """Retrieve complete public source metadata and text at the fixed plan cutoff."""
            return self.invoke("get_record", dict(record_id=record_id))

        @function_tool(failure_error_function=None)
        async def get_pricing_history(
            company: str | None = None, plan: str | None = None
        ) -> str:
            """Retrieve historical source observations; full get_record is still required before citing."""
            return self.invoke("get_pricing_history", dict(company=company, plan=plan))

        if active_runtime() and active_runtime().policy.targeted_research:

            @function_tool(failure_error_function=None)
            async def get_coverage_ledger() -> str:
                """Essential requirements and candidate full IDs; membership does not establish support."""
                return _json(self.coverage_ledger())

            return [
                search_records,
                get_record,
                get_pricing_history,
                get_coverage_ledger,
            ]
        return [search_records, get_record, get_pricing_history]

    def validate(self, output):
        if not isinstance(output, EvidencePack):
            raise EvidenceValidationError(
                "Researcher did not return a structured EvidencePack"
            )
        try:
            pack = EvidencePack.model_validate(output.model_dump())
        except ValidationError:
            raise EvidenceValidationError(
                "Invalid EvidencePack, duplicate IDs, or unavailable evidence"
            ) from None
        if (
            pack.original_question != self.plan.original_question
            or pack.as_of != self.plan.as_of
        ):
            raise EvidenceValidationError("EvidencePack changed the question or cutoff")
        if len(pack.evidence_items) > self.limits.max_evidence_items:
            self.exhausted("final_output", "max_evidence_items")
            raise EvidenceValidationError("EvidencePack exceeds max_evidence_items")
        mapping = {"source_title": "document_title"}
        metadata = (
            "claim_id",
            "company_name",
            "source_title",
            "source_type",
            "source_url",
            "published_at",
            "available_at",
            "effective_date",
            "end_date",
            "source_independence_group",
            "supersedes_record_id",
            "verification_status",
            "feature_status",
        )
        for item in pack.evidence_items:
            source = self.records.get(item.record_id)
            if source is None:
                raise EvidenceValidationError(
                    f"Record {item.record_id} was not retrieved in full during this run"
                )
            if _date(source["available_at"]) > self.plan.as_of:
                raise EvidenceValidationError(
                    f"Record {item.record_id} is unavailable at cutoff"
                )
            for field in metadata:
                expected = source.get(mapping.get(field, field))
                if expected is not None and field in {
                    "published_at",
                    "available_at",
                    "effective_date",
                    "end_date",
                }:
                    expected = _date(expected, date_only=True)
                if getattr(item, field) != expected:
                    raise EvidenceValidationError(
                        f"Record {item.record_id}: source metadata mismatch for {field}"
                    )
            try:
                validate_pricing_observation(item.pricing_observation, source)
            except ValueError as exc:
                raise EvidenceValidationError(str(exc)) from None
        expected_tasks = {task.task_id for task in self.plan.tasks}
        if {coverage.task_id for coverage in pack.coverage_by_task} != expected_tasks:
            raise EvidenceValidationError(
                "coverage_by_task must cover exactly the plan task IDs"
            )
        evidence_ids = {item.record_id for item in pack.evidence_items}
        for coverage in pack.coverage_by_task:
            if not set(coverage.supporting_record_ids).issubset(evidence_ids):
                raise EvidenceValidationError(
                    "Task coverage cites IDs absent from evidence_items"
                )
            if not coverage.supporting_record_ids and (
                not coverage.notes or not pack.missing_information
            ):
                raise EvidenceValidationError(
                    "Uncovered tasks require notes and missing_information"
                )
        # Generated log text is untrusted; return the authoritative tool ledger instead.
        pack.retrieval_log = list(self.log)
        return pack


async def researcher(
    plan: ResearchPlan,
    retrieval_service: RetrievalService | None = None,
    *,
    limits: ResearcherLimits | None = None,
) -> EvidencePack:
    """Gather evidence once. Caller-owned retrieval services are never closed here."""
    if not isinstance(plan, ResearchPlan):
        raise ValueError("plan must be a ResearchPlan")
    plan = ResearchPlan.model_validate(plan.model_dump())
    limits = (
        ResearcherLimits()
        if limits is None
        else ResearcherLimits.model_validate(limits.model_dump())
    )
    settings = load_config()
    owned = retrieval_service is None
    service = (
        RetrievalStore(load_dataset(settings.data_dir)) if owned else retrieval_service
    )
    session = ResearchSession(plan, service, limits)
    try:
        try:
            session.bootstrap()
        except BudgetExceeded as exc:
            session.event(kind="research_budget_stop", reason=exc.reason)
            observe_no_model_call(
                "Research stopped by the invocation budget before a model request"
            )
            return session.assemble(ResearchOutcome(stop_reason="budget_exhausted"))
        if not session.records:
            observe_no_model_call(
                "Deterministic no-record result after budget-counted discovery"
            )
            return session.assemble(ResearchOutcome(stop_reason="no_matching_records"))
        prompt = (
            files("ci_agents")
            .joinpath("prompts/researcher.md")
            .read_text(encoding="utf-8")
        )
        targeted = (
            active_runtime() is not None and active_runtime().policy.targeted_research
        )
        if targeted:
            prompt += "\nUse the coverage ledger to target unmet essential requirements. After an empty search, shorten/reformulate it or relax an optional category; avoid repeated identical operations. Fetch full records for every company needed in a comparison. Assess every essential requirement by task_id and zero-based requirement_index, with exact full-record support IDs. Do not equate record count with semantic coverage. Stop early only when all essentials are supported; otherwise report budget exhaustion or no matching records honestly. get_coverage_ledger is a local state view, not a new retrieval."
        async with AsyncOpenAI(
            api_key=settings.openai_api_key,
            max_retries=0,
            default_headers={"Accept-Encoding": OPENAI_ACCEPT_ENCODING},
        ) as client:
            agent = Agent(
                name="Researcher",
                instructions=prompt,
                model=OpenAIResponsesModel(
                    model=settings.openai_model, openai_client=client
                ),
                model_settings=agent_model_settings(
                    settings, parallel_tool_calls=False
                ),
                tools=session.tools(),
                output_type=GapResearchOutcome if targeted else ResearchOutcome,
                handoffs=[],
            )
            try:
                outcome = await run_structured(
                    agent,
                    _json(
                        {
                            "research_plan": plan.model_dump(mode="json"),
                            "retrieval_limits": limits.model_dump(),
                            "already_retrieved_full_records": list(
                                session.records.values()
                            ),
                            "discovery_log": session.log,
                            "remaining_tool_calls": max(
                                0, limits.max_tool_calls - session.calls
                            ),
                            **(
                                {"essential_requirements": session.coverage_ledger()}
                                if targeted
                                else {}
                            ),
                        }
                    ),
                    session.validate_outcome if targeted else lambda value: value,
                    runner=Runner,
                    output_error=RequirementAssessmentError,
                    max_turns=limits.max_model_turns or limits.max_tool_calls + 2,
                )
            except BudgetExceeded as exc:
                session.event(kind="research_budget_stop", reason=exc.reason)
                outcome = ResearchOutcome(stop_reason="budget_exhausted")
            except RequirementAssessmentError:
                if not targeted:
                    raise
                outcome = session.unassessed_outcome(
                    "One structured-output repair exhausted"
                )
            except MaxTurnsExceeded:
                if not targeted:
                    raise ResearcherError(
                        "Researcher model-turn limit reached"
                    ) from None
                outcome = session.unassessed_outcome(
                    "Researcher model-turn limit reached"
                )
            except EvidenceValidationError:
                raise
            except Exception:
                raise ResearcherError(
                    "Researcher SDK execution failed; no retry was attempted"
                ) from None
        return session.assemble(outcome)
    finally:
        if owned:
            service.close()
