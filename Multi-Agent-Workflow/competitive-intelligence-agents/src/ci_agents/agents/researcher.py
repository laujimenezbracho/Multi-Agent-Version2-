"""Bounded Researcher with cutoff-bound local SDK tools and provenance checks."""
from copy import deepcopy
from importlib.resources import files
import json
import logging
from typing import Protocol

from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig, Runner, function_tool
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ..telemetry import observe_sdk_result, observe_retrieval_event
from ..config import OPENAI_ACCEPT_ENCODING, load_config
from ..dataset import _date, load_dataset
from ..retrieval import RetrievalError, RetrievalStore
from ..schemas import EvidencePack, ResearchPlan

logger = logging.getLogger(__name__)


class ResearcherError(RuntimeError):
    """Sanitized SDK execution failure; no retry is attempted."""


class EvidenceValidationError(ResearcherError):
    """Evidence could not be traced to full records retrieved during this run."""


class ResearcherLimits(BaseModel):
    """Explicit per-run configuration, independent of model-generated content."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_tool_calls: int = Field(default=60, ge=1, le=200, strict=True)
    max_search_results: int = Field(default=10, ge=1, le=50, strict=True)
    max_evidence_items: int = Field(default=40, ge=1, le=100, strict=True)
    max_full_records: int = Field(default=40, ge=1, le=100, strict=True)


class RetrievalService(Protocol):
    """Trusted synchronous public retrieval service (normally RetrievalStore)."""
    def search_records(self, query, company=None, category=None, as_of=None, limit=20): ...
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

    def event(self, **fields):
        self.log.append(json.dumps(fields, sort_keys=True))
        observe_retrieval_event(fields)

    def invoke(self, name, arguments):
        if self.calls >= self.limits.max_tool_calls:
            return self.exhausted(name, "max_tool_calls")
        self.calls += 1
        if name == "get_record":
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
            return _json({"error": "Invalid retrieval input; use supported filters and query text"})
        if name == "get_record":
            if result is not None:
                if result.get("record_id") != args["record_id"] or _date(result["available_at"]) > self.plan.as_of:
                    raise EvidenceValidationError("Retrieval service returned an invalid or unavailable source")
                self.records[result["record_id"]] = deepcopy(result)
            ids = [] if result is None else [result["record_id"]]
        else:
            # Defensive check as well as service-side cutoff filtering.
            for row in result:
                if _date(row["available_at"]) > self.plan.as_of:
                    raise EvidenceValidationError("Retrieval service returned post-cutoff evidence")
                if name == "get_pricing_history" and _date(row["source_available_at"]) > self.plan.as_of:
                    raise EvidenceValidationError("Pricing source was unavailable at cutoff")
            ids = [row.get("record_id", row.get("source_record_id")) for row in result]
        self.event(tool=name, arguments=arguments, as_of=self.plan.as_of.isoformat(),
                   status="ok", record_ids=ids, result_count=len(ids),
                   tool_calls=self.calls, full_record_calls=self.full_calls)
        return _json(result)

    def exhausted(self, name, budget):
        self.event(tool=name, status="budget_exhausted", budget=budget)
        logger.warning("Researcher retrieval budget exhausted: %s", budget)
        return _json({"error": "budget_exhausted", "budget": budget,
                      "instruction": "Use retrieved evidence and report remaining gaps; do not invent evidence"})

    def tools(self):
        @function_tool(failure_error_function=None)
        async def search_records(query: str, company: str | None = None,
                                 category: str | None = None, limit: int = 10) -> str:
            """Search public source snippets at the plan cutoff. Fetch full records before citing."""
            return self.invoke("search_records", dict(query=query, company=company, category=category, limit=limit))

        @function_tool(failure_error_function=None)
        async def get_record(record_id: str) -> str:
            """Retrieve complete public source metadata and text at the fixed plan cutoff."""
            return self.invoke("get_record", dict(record_id=record_id))

        @function_tool(failure_error_function=None)
        async def get_pricing_history(company: str | None = None, plan: str | None = None) -> str:
            """Retrieve historical source observations; full get_record is still required before citing."""
            return self.invoke("get_pricing_history", dict(company=company, plan=plan))
        return [search_records, get_record, get_pricing_history]

    def validate(self, output):
        if not isinstance(output, EvidencePack):
            raise EvidenceValidationError("Researcher did not return a structured EvidencePack")
        try:
            pack = EvidencePack.model_validate(output.model_dump())
        except ValidationError:
            raise EvidenceValidationError("Invalid EvidencePack, duplicate IDs, or unavailable evidence") from None
        if pack.original_question != self.plan.original_question or pack.as_of != self.plan.as_of:
            raise EvidenceValidationError("EvidencePack changed the question or cutoff")
        if len(pack.evidence_items) > self.limits.max_evidence_items:
            self.exhausted("final_output", "max_evidence_items")
            raise EvidenceValidationError("EvidencePack exceeds max_evidence_items")
        mapping = {"source_title": "document_title"}
        metadata = ("claim_id", "company_name", "source_title", "source_type", "source_url",
                    "published_at", "available_at", "effective_date", "end_date",
                    "source_independence_group", "supersedes_record_id", "verification_status", "feature_status")
        for item in pack.evidence_items:
            source = self.records.get(item.record_id)
            if source is None:
                raise EvidenceValidationError(f"Record {item.record_id} was not retrieved in full during this run")
            if _date(source["available_at"]) > self.plan.as_of:
                raise EvidenceValidationError(f"Record {item.record_id} is unavailable at cutoff")
            for field in metadata:
                expected = source.get(mapping.get(field, field))
                if expected is not None and field in {"published_at", "available_at", "effective_date", "end_date"}:
                    expected = _date(expected, date_only=True)
                if getattr(item, field) != expected:
                    raise EvidenceValidationError(f"Record {item.record_id}: source metadata mismatch for {field}")
        expected_tasks = {task.task_id for task in self.plan.tasks}
        if {coverage.task_id for coverage in pack.coverage_by_task} != expected_tasks:
            raise EvidenceValidationError("coverage_by_task must cover exactly the plan task IDs")
        evidence_ids = {item.record_id for item in pack.evidence_items}
        for coverage in pack.coverage_by_task:
            if not set(coverage.supporting_record_ids).issubset(evidence_ids):
                raise EvidenceValidationError("Task coverage cites IDs absent from evidence_items")
            if not coverage.supporting_record_ids and (not coverage.notes or not pack.missing_information):
                raise EvidenceValidationError("Uncovered tasks require notes and missing_information")
        # Generated log text is untrusted; return the authoritative tool ledger instead.
        pack.retrieval_log = list(self.log)
        return pack


async def researcher(plan: ResearchPlan, retrieval_service: RetrievalService | None = None,
                     *, limits: ResearcherLimits | None = None) -> EvidencePack:
    """Gather evidence once. Caller-owned retrieval services are never closed here."""
    if not isinstance(plan, ResearchPlan):
        raise ValueError("plan must be a ResearchPlan")
    plan = ResearchPlan.model_validate(plan.model_dump())
    limits = ResearcherLimits() if limits is None else ResearcherLimits.model_validate(limits.model_dump())
    settings = load_config()
    owned = retrieval_service is None
    service = RetrievalStore(load_dataset(settings.data_dir)) if owned else retrieval_service
    session = ResearchSession(plan, service, limits)
    try:
        prompt = files("ci_agents").joinpath("prompts/researcher.md").read_text(encoding="utf-8")
        async with AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0,
                       default_headers={"Accept-Encoding": OPENAI_ACCEPT_ENCODING}) as client:
            agent = Agent(name="Researcher", instructions=prompt,
                          model=OpenAIResponsesModel(model=settings.openai_model, openai_client=client),
                          model_settings=ModelSettings(parallel_tool_calls=False),
                          tools=session.tools(), output_type=EvidencePack, handoffs=[])
            try:
                result = await Runner.run(
                    agent, input=_json({"research_plan": plan.model_dump(mode="json"),
                                        "retrieval_limits": limits.model_dump()}),
                    max_turns=limits.max_tool_calls + 2,
                    run_config=RunConfig(tracing_disabled=True),
                )
                observe_sdk_result(result)
            except EvidenceValidationError:
                raise
            except Exception:
                raise ResearcherError("Researcher SDK execution failed; no retry was attempted") from None
        return session.validate(result.final_output)
    finally:
        if owned:
            service.close()
