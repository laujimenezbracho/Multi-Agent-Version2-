"""Lossless text compaction and Decimal arithmetic, without extra source access."""

from decimal import Decimal
import re

from .workflow_runtime import active_runtime


def evidence_context(pack):
    data = pack.model_dump(mode="json")
    runtime = active_runtime()
    if runtime is None or not runtime.policy.compact_context:
        return data
    # Full logs remain in pack artifacts. No source passage is truncated or selected
    # out: only duplicated text and the operation log are removed from model input.
    data.pop("retrieval_log", None)
    for item in data["evidence_items"]:
        if item["finding"] in item["quoted_or_paraphrased_evidence"]:
            item.pop("finding")
    data["context_note"] = (
        "All original evidence text and metadata retained; duplicate finding text and retrieval audit omitted. Audit remains in artifacts."
    )
    return data


def numerical_aids(plan, pack):
    seats = re.search(
        r"\b(\d+)\s+(?:growth\s+)?(?:seats?|users?)\b", plan.original_question, re.I
    )
    count = int(seats.group(1)) if seats else None
    groups, excluded = {}, []
    superseded = {
        item.supersedes_record_id
        for item in pack.evidence_items
        if item.supersedes_record_id
        and item.available_at <= plan.as_of
        and item.effective_date is not None
        and item.effective_date <= plan.as_of
        and (item.end_date is None or item.end_date >= plan.as_of)
    }
    for item in pack.evidence_items:
        p = item.pricing_observation
        if p is None or p.amount is None:
            continue
        reason = None
        if item.record_id in superseded:
            reason = "Explicitly superseded by another supplied source; source conflict still requires review."
        elif item.effective_date is None or item.effective_date > plan.as_of:
            reason = "Unknown or future effective date."
        elif item.end_date is not None and item.end_date < plan.as_of:
            reason = "Expired source observation."
        elif not all([p.plan, p.currency, p.billing_frequency, p.price_basis]):
            reason = "Unknown comparison units."
        if reason:
            excluded.append({"record_id": item.record_id, "reason": reason})
            continue
        amount = Decimal(p.amount)
        key = (p.plan, p.currency, p.billing_frequency, p.price_basis)
        row = {
            "record_id": item.record_id,
            "company": item.company_name,
            "amount": str(amount),
            "effective_date": item.effective_date.isoformat(),
            "source_type": item.source_type,
            "verification_status": item.verification_status,
        }
        if count is not None and p.price_basis == "USD_per_seat_per_month":
            if p.seat_minimum is not None and count >= p.seat_minimum:
                row.update(
                    seats=count,
                    monthly_subscription_baseline=str(amount * count),
                    annual_subscription_baseline=str(amount * count * 12),
                )
            else:
                row["seat_calculation_withheld"] = "Seat minimum unknown or not met."
        groups.setdefault(key, []).append(row)
    return {
        "comparison_groups": [
            {
                "plan": k[0],
                "currency": k[1],
                "billing_frequency": k[2],
                "price_basis": k[3],
                "observations": sorted(
                    v, key=lambda r: (Decimal(r["amount"]), r["record_id"])
                ),
            }
            for k, v in sorted(groups.items())
        ],
        "excluded_from_arithmetic": excluded,
        "limitations": [
            "Arithmetic on source assertions, not verified current-price selection or an all-in-cost winner.",
            "Conflicting prices and source credibility must be resolved from original passages; official/confirmed labels do not prove truth.",
            "Unknown fees, taxes, add-ons, contract terms and discount eligibility are not zero.",
        ],
    }
