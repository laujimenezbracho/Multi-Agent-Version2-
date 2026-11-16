"""Evidence-only Analyst; native SDK output with no tools or regeneration."""

from importlib.resources import files
import json

from agents import Agent, OpenAIResponsesModel, Runner
from agents.exceptions import ModelBehaviorError
from openai import AsyncOpenAI
from pydantic import ValidationError

from ..telemetry import observe_no_model_call
from ..config import OPENAI_ACCEPT_ENCODING, load_config, agent_model_settings
from ..schemas import CompetitiveReport, EvidencePack, ResearchPlan
from ..workflow_runtime import run_structured, active_runtime, BudgetExceeded
from ..evidence_context import evidence_context, numerical_aids


class AnalystError(RuntimeError):
    """Sanitized SDK execution failure; no retry is attempted."""


class AnalystInputError(ValueError):
    """Invalid or inconsistent plan/evidence handoff."""


class AnalystOutputError(AnalystError):
    """Report violates structure, question, cutoff, or citation constraints."""


REPORT_FIELDS = frozenset(CompetitiveReport.model_fields)
SECTION_LISTS = (
    "key_findings",
    "competitor_comparison",
    "strategic_implications",
    "limitations",
)


def _inputs(plan, evidence_pack):
    if not isinstance(plan, ResearchPlan) or not isinstance(
        evidence_pack, EvidencePack
    ):
        raise AnalystInputError("Expected a ResearchPlan and an EvidencePack")
    try:
        # Snapshot and revalidate: in-place mutations may bypass Pydantic validators.
        plan = ResearchPlan.model_validate(plan.model_dump())
        evidence_pack = EvidencePack.model_validate(evidence_pack.model_dump())
    except ValidationError:
        raise AnalystInputError(
            "Invalid plan or evidence pack, including duplicate IDs or post-cutoff evidence"
        ) from None
    if evidence_pack.original_question != plan.original_question:
        raise AnalystInputError("Plan and evidence pack original questions must match")
    if evidence_pack.as_of != plan.as_of:
        raise AnalystInputError("Plan and evidence pack as_of cutoffs must match")
    return plan, evidence_pack


def _validate_report(output, plan, evidence_pack):
    if not isinstance(output, CompetitiveReport):
        raise AnalystOutputError(
            "Analyst did not return a structured CompetitiveReport"
        )
    # Defaults permit partial local construction; generated output must explicitly
    # include every report section, even when claims/sources are empty.
    missing = REPORT_FIELDS - output.model_fields_set
    if missing:
        raise AnalystOutputError(
            "Report omitted required fields: " + ", ".join(sorted(missing))
        )
    try:
        report = CompetitiveReport.model_validate(output.model_dump())
    except ValidationError:
        raise AnalystOutputError(
            "Invalid CompetitiveReport: check claim IDs, factual citations, and field values"
        ) from None
    if report.question != plan.original_question:
        raise AnalystOutputError(
            "Report question must match the original research question"
        )
    if report.as_of != plan.as_of:
        raise AnalystOutputError("Report as_of must match the research plan cutoff")
    for section in SECTION_LISTS:
        if not getattr(report, section):
            raise AnalystOutputError(
                f"Report section {section} must contain content or an explicit evidence limitation"
            )
    evidence = {item.record_id: item for item in evidence_pack.evidence_items}
    cited_ids = set(report.sources)
    for claim in report.claims:
        if claim.claim_type == "factual" and not claim.supporting_record_ids:
            raise AnalystOutputError(
                f"Factual claim {claim.claim_id} requires supporting record IDs"
            )
        cited_ids.update(claim.supporting_record_ids)
    for record_id in sorted(cited_ids):
        if record_id not in evidence:
            raise AnalystOutputError(
                f"Cited record {record_id} is not present in the EvidencePack"
            )
        if evidence[record_id].available_at > plan.as_of:
            raise AnalystOutputError(
                f"Cited record {record_id} is unavailable at the evaluation cutoff"
            )
    # Citation membership is structural provenance, not proof of semantic support.
    return report


async def analyst(plan: ResearchPlan, evidence_pack: EvidencePack) -> CompetitiveReport:
    """Synthesize once using only supplied evidence. Explicit invocation may incur cost.

    No dataset reads, retrieval, tools, handoffs, retries, tracing exports, or recovery.
    """
    plan, evidence_pack = _inputs(plan, evidence_pack)
    if not evidence_pack.evidence_items:
        observe_no_model_call(
            "Explicit abstention for an empty validated evidence pack"
        )
        report = CompetitiveReport(
            question=plan.original_question,
            as_of=plan.as_of,
            executive_summary="Unable to answer from the supplied evidence: no full source records were retrieved. Consult the recorded searches and gaps; this does not mean the corpus contains no relevant information.",
            key_findings=[
                "No source-backed competitive findings can be stated from this evidence pack."
            ],
            competitor_comparison=[
                "No supported comparison is available; no competitor ranking is asserted."
            ],
            strategic_implications=[
                "Collect the missing evidence before making a decision."
            ],
            limitations=evidence_pack.missing_information or ["Empty evidence pack."],
            claims=[],
            sources=[],
        )
        return _validate_report(report, plan, evidence_pack)
    settings = load_config()
    instructions = (
        files("ci_agents").joinpath("prompts/analyst.md").read_text(encoding="utf-8")
    )
    context = json.dumps(
        {
            "research_plan": plan.model_dump(mode="json"),
            "evidence_pack": evidence_context(evidence_pack),
            **(
                {"numerical_aids": numerical_aids(plan, evidence_pack)}
                if active_runtime()
                else {}
            ),
        }
    )
    if active_runtime():
        instructions += "\nFor every strategic inference state supporting claim/source IDs, explicit comparison criteria, assumptions and uncertainty in its text. Numerical aids verify arithmetic on source assertions only; they do not settle conflicting sources or establish all-in cost. Do not infer competitive undercutting without comparable evidence for both companies. An unassessed coverage requirement means no valid Researcher judgment, not that evidence is absent; inspect the full sources to answer supported parts."
    async with AsyncOpenAI(
        api_key=settings.openai_api_key,
        max_retries=0,
        default_headers={"Accept-Encoding": OPENAI_ACCEPT_ENCODING},
    ) as client:
        agent = Agent(
            name="Analyst",
            instructions=instructions,
            model=OpenAIResponsesModel(
                model=settings.openai_model, openai_client=client
            ),
            model_settings=agent_model_settings(settings),
            output_type=CompetitiveReport,
            tools=[],
            handoffs=[],
        )
        try:
            return await run_structured(
                agent,
                context,
                lambda output: _validate_report(output, plan, evidence_pack),
                runner=Runner,
                output_error=AnalystOutputError,
            )
        except (AnalystOutputError, BudgetExceeded):
            raise
        except (ModelBehaviorError, ValidationError):
            raise AnalystOutputError(
                "SDK could not validate the structured CompetitiveReport"
            ) from None
        except Exception:
            raise AnalystError(
                "Analyst SDK execution failed; no retry was attempted"
            ) from None
