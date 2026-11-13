"""Planner only: native SDK structured output, no retrieval tools or handoffs."""
from datetime import datetime, timezone
from importlib.resources import files
import json
import re

from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig, Runner
from openai import AsyncOpenAI
from pydantic import ValidationError

from ..telemetry import observe_sdk_result
from ..diagnostics import log_exception_diagnostic
from ..config import OPENAI_ACCEPT_ENCODING, load_config
from ..dataset import load_dataset
from ..schemas import ResearchPlan


class PlannerError(RuntimeError):
    """Planner execution failed; details exclude credentials and raw API payloads."""


class PlannerInputError(ValueError):
    """Invalid question or cutoff supplied to the Planner."""


class PlannerOutputError(PlannerError):
    """Structured plan failed validation or changed the input contract."""


def _normalized(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def _validate_plan(output, question, cutoff, companies, categories):
    if not isinstance(output, ResearchPlan):
        raise PlannerOutputError("Planner did not return a structured ResearchPlan")
    try:
        plan = ResearchPlan.model_validate(output.model_dump())
    except ValidationError:
        raise PlannerOutputError("Planner returned an invalid ResearchPlan") from None
    if plan.original_question != question or plan.as_of != cutoff:
        raise PlannerOutputError("Planner changed the original question or as_of cutoff")
    signatures = set()
    for task in plan.tasks:
        if not set(task.companies).issubset(companies) or not set(task.categories).issubset(categories):
            raise PlannerOutputError("Planner used companies or categories outside the public metadata")
        signature = (_normalized(task.research_question), tuple(sorted(task.companies)),
                     tuple(sorted(task.categories)))
        if signature in signatures:
            raise PlannerOutputError("Planner returned duplicate research tasks")
        signatures.add(signature)
        if not task.companies or not task.categories or not task.evidence_requirements:
            raise PlannerOutputError("Planner tasks require companies, categories, and evidence requirements")
    if not plan.expected_answer_elements or not plan.comparison_dimensions:
        raise PlannerOutputError("Planner must specify answer elements and comparison dimensions")
    return plan


async def planner(question: str, as_of: datetime) -> ResearchPlan:
    """Plan research using the configured model; callers control live invocation.

    No retries, retrieval, handoffs, or tracing exports. Failures propagate as
    sanitized PlannerError variants; invalid local configuration retains its type.
    """
    if not isinstance(question, str) or not question.strip():
        raise PlannerInputError("question must be non-empty text")
    if len(question) > 10000:
        raise PlannerInputError("question must be at most 10000 characters")
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise PlannerInputError("as_of must be a timezone-aware datetime")
    cutoff = as_of.astimezone(timezone.utc)
    settings = load_config()
    dataset = load_dataset(settings.data_dir)
    # Only metadata leaves this process; no records, pricing rows, or benchmark prompts.
    companies = sorted({row["company_name"] for row in dataset.records})
    categories = sorted({row["event_category"] for row in dataset.records if row.get("event_category")})
    context = json.dumps({"original_question": question, "as_of": cutoff.isoformat(),
                          "companies": companies, "evidence_categories": categories})
    instructions = files("ci_agents").joinpath("prompts/planner.md").read_text(encoding="utf-8")
    # No sampling overrides: configured gpt-5-mini rejects temperature. Unknown
    # future model IDs also retain provider defaults rather than guessed support.
    async with AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0,
                       default_headers={"Accept-Encoding": OPENAI_ACCEPT_ENCODING}) as client:
        agent = Agent(
            name="Planner", instructions=instructions,
            model=OpenAIResponsesModel(model=settings.openai_model, openai_client=client),
            model_settings=ModelSettings(), output_type=ResearchPlan,
            tools=[], handoffs=[],
        )
        phase = "sdk_execution"
        try:
            result = await Runner.run(agent, input=context, max_turns=1,
                                      run_config=RunConfig(tracing_disabled=True))
            phase = "telemetry_processing"
            observe_sdk_result(result)
        except Exception as exc:
            error = PlannerError("Planner SDK execution failed; no retry was attempted")
            error._ci_safe_diagnostic = log_exception_diagnostic(exc, "planner", secrets=(settings.openai_api_key,), phase=phase)
            raise error from None
    try:
        return _validate_plan(result.final_output, question, cutoff, companies, categories)
    except PlannerOutputError as exc:
        exc._ci_safe_diagnostic = log_exception_diagnostic(
            exc, "planner", secrets=(settings.openai_api_key,), category="schema_validation", phase="output_validation")
        raise
    except Exception as exc:
        # Record unanticipated local postprocessing errors without changing their
        # exception type or the pipeline's generic public-facing error message.
        exc._ci_safe_diagnostic = log_exception_diagnostic(
            exc, "planner", secrets=(settings.openai_api_key,), phase="output_validation")
        raise
