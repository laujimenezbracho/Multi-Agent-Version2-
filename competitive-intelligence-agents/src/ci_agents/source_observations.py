"""Structured observations copied from public sources, never inferred by agents."""

from decimal import Decimal, InvalidOperation

from .schemas import PricingObservation


def pricing_observation(source):
    raw = source.get("metric_value")
    try:
        amount = Decimal(str(raw))
        amount = format(amount, "f") if amount.is_finite() and amount >= 0 else None
    except (InvalidOperation, ValueError):
        amount = None
    return PricingObservation(
        plan=source.get("plan"),
        amount=amount,
        currency=source.get("currency"),
        billing_frequency=source.get("billing_frequency"),
        price_basis=source.get("price_basis"),
        seat_minimum=source.get("seat_minimum"),
    )


def validate_pricing_observation(observation, source):
    """Legacy sources may omit the structure; supplied values must match exactly."""
    if observation is None:
        return
    expected = pricing_observation(source)
    for name in (
        "plan",
        "currency",
        "billing_frequency",
        "price_basis",
        "seat_minimum",
    ):
        if getattr(observation, name) != getattr(expected, name):
            raise ValueError(
                "Structured pricing metadata differs from the public source"
            )
    if (observation.amount is None) != (expected.amount is None) or (
        observation.amount is not None
        and Decimal(observation.amount) != Decimal(expected.amount)
    ):
        raise ValueError("Structured pricing amount differs from the public source")


def validate_evidence_items(items, records, as_of, full_record_ids):
    """Verify source identity and full-retrieval provenance before a handoff."""
    fields = (
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
    for item in items:
        source = records.get(item.record_id)
        if source is None or item.record_id not in full_record_ids:
            raise ValueError(
                "Evidence must reference a public record retrieved in full"
            )
        if source["available_at"] > as_of:
            raise ValueError("Evidence source was unavailable at the cutoff")
        if any(getattr(item, name) != source.get(name) for name in fields):
            raise ValueError("Evidence metadata differs from its public source")
        if item.source_title != source["document_title"]:
            raise ValueError("Evidence source title differs from its public source")
        validate_pricing_observation(item.pricing_observation, source)
