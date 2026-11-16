"""Question intent only; no benchmark IDs, answer keys or competitive facts."""

import re
from .schemas import ResearchPlan, ResearchTask


def scope_contract(question):
    text = question.casefold()
    narrow = bool(
        re.search(r"\b(lowest|cheapest)\b", text)
        and re.search(r"\bprice\b", text)
        and not re.search(
            r"\b(value|tco|total|effective|discount|promotion|all.in|cost)\b", text
        )
    )
    return {
        "kind": "published_list_price" if narrow else "question_defined",
        "essential": [
            "Comparable requested-tier prices, currency, price basis, commitment and applicable dates; resolve relevant conflicting observations."
        ]
        if narrow
        else [
            "Answer the original question with supported comparisons and explicit uncertainty."
        ],
        "optional": [
            "Feature-value matrices, SLAs, hypothetical buyer profiles, sales quotes and negotiated discounts are optional unless explicitly requested."
        ]
        if narrow
        else [],
        "rule": "Do not upgrade optional enrichment into an essential requirement or demand evidence unavailable in the permitted corpus.",
    }


def normalize_price_plan(plan, companies):
    """Bound a simple price-comparison plan using intent and public company names.

    No prices, question IDs or private keys are consulted. Explicit extended scope
    remains untouched. Original model output is separately retained by the caller.
    """
    question = plan.original_question
    if scope_contract(question)["kind"] != "published_list_price":
        return plan
    # Additional scope is meaningful even when it names a country outside a fixed
    # allowlist or a historical period. Preserve the model plan in those cases.
    if re.search(
        r"\b(in|for|with|without|including|excluding|except|during|between|before|after)\b|\bas\s+of\b|\b\d{4}\b",
        question,
        re.I,
    ):
        return plan
    if re.search(
        r"\b(region|country|countries|location|uk|usa|us|europe|canada|tax|taxes|vat|fees?|onboarding|checkout|minimums?)\b",
        question,
        re.I,
    ):
        return plan
    mentioned = sorted(
        c for c in companies if re.search(r"\b" + re.escape(c) + r"\b", question, re.I)
    )
    if len(mentioned) == 1:
        return plan  # Do not silently expand a named subset.
    if (
        not re.search(
            r"\b(competitors?|companies|vendors?|providers?)\b", question, re.I
        )
        and len(mentioned) < 2
    ):
        return plan
    selected = mentioned if len(mentioned) >= 2 else sorted(companies)
    result = plan.model_copy(deep=True)
    first = next(task for task in plan.tasks if task.priority == "essential")
    result.objective = "Compare the requested published list prices from available source records. Do not expand this into an all-in-cost, checkout-verification or page-capture investigation."
    result.tasks = [
        ResearchTask(
            task_id=first.task_id,
            research_question=question,
            companies=selected,
            categories=["Pricing"],
            time_scope="Available by "
            + plan.as_of.isoformat()
            + "; preserve the original question's requested effective period.",
            evidence_requirements=[
                "For each requested company, retrieve full sources for the requested tier amount, currency, price basis, billing commitment and applicable dates.",
                "Resolve relevant corrections or conflicting observations, cite exact full records, and disclose missing comparable prices. Unknown fees and negotiated terms are excluded, not assumed zero.",
            ],
        )
    ]
    result.expected_answer_elements = [
        "Answer which comparable published requested-tier list price is lowest, or qualify inability to establish a complete winner.",
        "Compare requested companies using cited amounts, currency, price basis, billing commitment and applicable dates.",
        "Explain relevant source conflicts and missing coverage; keep unknown fees, promotions and negotiated terms separate from list price.",
    ]
    result.comparison_dimensions = [
        "Published requested-tier list price",
        "Units, commitment and effective periods",
        "Source conflicts and uncertainty",
    ]
    result.optional_context = list(plan.optional_context) + [
        "Planner-proposed enrichment, not a condition for answering a simple published-list-price question: "
        + requirement
        for task in plan.tasks
        for requirement in task.evidence_requirements
    ]
    return ResearchPlan.model_validate(result.model_dump())
