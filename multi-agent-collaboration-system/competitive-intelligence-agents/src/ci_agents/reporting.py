"""Deterministic, offline, two-page presentation of immutable run artifacts.

Independent evaluation is deliberately not loaded. Counts are observations, not
accuracy scores. Long sections are bounded excerpts; JSON remains authoritative.
"""
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import warnings
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, KeepInFrame

from .persistence import verify_manifest


def _read(root, name, default=None):
    path = root / name
    if path.is_symlink():
        raise ValueError("Reporting input must not be a symbolic link")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _value(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:,.3f}"
    return str(value)


def report_sections(directory):
    """Return layout-independent content using identical definitions for V1/V2."""
    root = Path(directory).resolve()
    verify_manifest(root)
    meta = _read(root, "metadata.json")
    if meta.get("execution_status") not in {"completed", "completed_review_failed"}:
        raise ValueError("PDF requires a completed question run")
    result = _read(root, "run_result.json")
    report = result["final_report"]
    review = result.get("review_result") or {}
    metrics = _read(root, "metrics.json", {})
    stages = result.get("stage_metrics", [])
    v2 = result.get("configuration") == "v2"
    pack = _read(root, "merged_evidence_pack.json" if v2 else "evidence_pack.json")
    if pack is None:
        pack = _read(root, "initial_evidence_pack.json", {})
    items = pack.get("evidence_items", [])
    sources = {item["record_id"]: item for item in items}
    claims = report.get("claims", [])
    citations = {rid for claim in claims for rid in claim.get("supporting_record_ids", [])}
    events = []
    for name in (("initial_retrieval_logs.json", "recovery_retrieval_logs.json") if v2 else ("retrieval_logs.json",)):
        events.extend(_read(root, name, []))
    discovered = {rid for event in events for rid in event.get("record_ids", [])}
    full = {rid for event in events if event.get("tool") == "get_record" and event.get("status") == "ok" for rid in event.get("record_ids", [])}
    totals = metrics.get("final_totals" if v2 else "totals", {})
    def summed(field):
        values = [s.get(field) for s in stages]
        return sum(values) if values and all(v is not None for v in values) else None
    def count(label, value):
        return f"{label}: {_value(value)}"
    first = [("Executive answer", [report["executive_summary"]])]
    first += [(title, report.get(field, [])) for title, field in (
        ("Key findings", "key_findings"), ("Competitor comparison", "competitor_comparison"))]
    first.append(("Main evidence sources", [f"{rid} | {sources.get(rid, {}).get('source_title', 'Title unavailable')} | {sources.get(rid, {}).get('source_type', 'Type unavailable')} | available {sources.get(rid, {}).get('available_at', 'unknown')}" for rid in report.get("sources", [])]))
    first += [(title, report.get(field, [])) for title, field in (
        ("Strategic implications", "strategic_implications"), ("Limitations and uncertainty", "limitations"))]
    quality = [count("Internal Reviewer completeness (0-1)", review.get("completeness_score")),
        count("Claims", len(claims)), "Independent answer accuracy / completeness / comparison correctness: Pending"]
    unsupported = [count("Factual claims", sum(c.get("claim_type") == "factual" for c in claims)),
        count("Internal flagged issues", len(review.get("flagged_issues", []))),
        "Independent incorrect / unsupported / fabricated claims and severity: Pending"]
    evidence = [count("Unique cited records", len(citations)), count("Claim-citation links", sum(len(c.get("supporting_record_ids", [])) for c in claims)),
        count("Selected evidence records", len(items)), count("Discovered record IDs", len(discovered)), count("Unique full records retrieved", len(full)),
        count("Internal Reviewer evidence support (0-1)", review.get("evidence_support_score")),
        count("Internal Reviewer temporal consistency (0-1)", review.get("temporal_consistency_score")),
        "Independent citation validity / semantic support / temporal accuracy / leakage: Pending"]
    recovery = [count("Final internal Reviewer decision", review.get("decision")),
        "Independent Reviewer precision / recall / decision correctness: Pending"]
    if v2:
        delta = _read(root, "claim_changes.json", {})
        changes = _read(root, "evidence_changes.json", {})
        incremental = metrics.get("incremental_total", {})
        recovery += [count("Initial Reviewer decision", meta.get("initial_reviewer_decision")),
            count("Recovery attempted", meta.get("recovery_attempted")), count("Recovery state", meta.get("recovery_state")),
            count("Recovery reason", meta.get("recovery_reason")), "Issues targeted: " + (", ".join(meta.get("trigger_issue_ids", [])) or "None"),
            "Claim changes (stable IDs): " + ", ".join(f"{k}={len(delta.get(k, []))}" for k in ("added", "removed", "modified")),
            f"Evidence records initial / final: {len(changes.get('initial_evidence_ids', []))} / {len(changes.get('final_evidence_ids', []))}",
            count("Additional tokens", incremental.get("total_tokens")), count("Additional latency seconds", incremental.get("latency_seconds")),
            count("Additional estimated USD", incremental.get("estimated_cost_usd")),
            "Independent recovery success / regression / quality improvement: Pending"]
    else:
        recovery.append("Recovery attempt / success / regression / incremental cost: N/A (V1)")
    execution = [count(label, totals.get(field, summed(field))) for label, field in (
        ("Model requests", "llm_requests"), ("Input tokens (includes cached)", "input_tokens"),
        ("Cached input tokens (subset)", "cached_input_tokens"), ("Output tokens", "output_tokens"), ("Total tokens", "total_tokens"),
        ("Total latency seconds (initial + recovery for V2)", "latency_seconds"), ("Estimated USD (configured rates)", "estimated_cost_usd"))]
    execution += [count("Retrieval tool calls (attempts)", len(events))]
    execution += [f"{s['stage_name']} latency seconds: {_value(s.get('latency_seconds'))}" for s in stages]
    if v2:
        execution += [count("Initial V1 latency seconds", metrics.get("initial_v1", {}).get("latency_seconds")),
            count("Observed V2 controller seconds", meta.get("observed_controller_wall_clock")),
            count("Invocation mode", meta.get("mode")), "Paired mode totals include inherited baseline usage; additional usage is new work only."]
    second = list(zip(("1. Task quality and completeness", "2. Unsupported or incorrect claims", "3. Evidence and source quality", "4. Error detection and recovery", "5. Execution time and latency"), (quality, unsupported, evidence, recovery, execution)))
    return meta, report, first, second


def _excerpt(values, budget=3600):
    text = " | ".join(str(value) for value in values) or "None reported"
    if len(text) > budget:
        return text[:budget].rsplit(" ", 1)[0] + " ... [excerpt; complete text in run JSON]"
    return text


def generate_pdf(directory):
    """Publish report-0001.pdf etc exclusively; never overwrite run artifacts."""
    root = Path(directory).resolve()
    meta, report, first, second = report_sections(root)
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=A4, invariant=1, pageCompression=1)
    canvas.setTitle("Competitive intelligence | " + meta["run_id"])
    canvas.setAuthor("Competitive Intelligence Agents")
    width, height = A4
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=12, textColor=colors.HexColor("#243447"), spaceAfter=6)
    heading = ParagraphStyle("heading", parent=body, fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#136F85"), spaceBefore=9)
    for page, sections in enumerate((first, second), 1):
        canvas.setFillColor(colors.HexColor("#152E46"))
        canvas.rect(0, height-77, width, 77, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 18)
        canvas.drawString(36, height-34, "Research output" if page == 1 else "Performance and review")
        canvas.setFont("Helvetica", 9)
        canvas.drawString(36, height-55, f"{meta.get('question_id') or 'Custom question'}  |  {meta.get('pipeline_version')}  |  {meta.get('model')}")
        story = [Paragraph(escape(report["question"]), heading),
            Paragraph(escape(f"Cutoff: {meta.get('as_of')} | Run: {meta['run_id']}"), body)]
        for title, values in sections:
            content = escape(_excerpt(values, 3000)) if page == 1 else "<br/>".join(escape(str(value)) for value in values)
            story += [Paragraph(escape(title), heading), Paragraph(content, body)]
        frame = KeepInFrame(width-72, height-130, story, mode="shrink")
        frame.wrapOn(canvas, width-72, height-130)
        frame.drawOn(canvas, 36, height-94-frame.height)
        canvas.setFillColor(colors.HexColor("#526477"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(36, 25, "Internal Reviewer scores are not independent evaluation. JSON artifacts are authoritative.")
        canvas.drawRightString(width-36, 13, f"Reporting v1 | {page} / 2")
        canvas.showPage()
    canvas.save()
    output = root / "reports"
    if output.is_symlink():
        raise ValueError("Reports directory must not be a symbolic link")
    output.mkdir(exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pdf-", dir=output)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(stream.getvalue())
        for index in range(1, 100000):
            destination = output / f"report-{index:04d}.pdf"
            try:
                os.link(temporary, destination)
                return destination
            except FileExistsError:
                continue
        raise FileExistsError("PDF report sequence exhausted")
    finally:
        Path(temporary).unlink(missing_ok=True)


def report_completed_run(directory):
    """Best effort presentation; never change pipeline status or sealed artifacts."""
    root = Path(directory)
    try:
        metadata = _read(root, "metadata.json", {})
        if metadata.get("execution_status") in {"completed", "completed_review_failed"}:
            return generate_pdf(root)
    except Exception as exc:
        warnings.warn(f"PDF reporting failed ({type(exc).__name__}); use ci-agents report to retry.", RuntimeWarning)
    return None
