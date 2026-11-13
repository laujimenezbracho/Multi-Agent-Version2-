"""Offline PDF checks; no SDK execution or private evaluation inputs."""
import json
from hashlib import sha256
from pathlib import Path

import fitz
import pytest

from ci_agents import cli
from ci_agents.persistence import RunStore, verify_manifest
from ci_agents.reporting import generate_pdf, report_sections, report_completed_run


@pytest.fixture
def run(tmp_path):
    def create(v2=False, long=False):
        store = RunStore(tmp_path, "v2" if v2 else "v1")
        report = dict(question="Which competitor? <tag>&", executive_summary="Answer " * (4000 if long else 5),
            key_findings=["Finding"], competitor_comparison=["Comparison"], strategic_implications=["Implication"],
            limitations=["Unknown fees"], claims=[dict(claim_id="C1", claim_type="factual", supporting_record_ids=["R1"])], sources=["R1"])
        review = dict(decision="FAIL", completeness_score=.5, evidence_support_score=.8, temporal_consistency_score=1, flagged_issues=[{}])
        stage = dict(stage_name="planner", latency_seconds=1.25, input_tokens=10, cached_input_tokens=4, output_tokens=2, total_tokens=12, llm_requests=1)
        totals = dict(input_tokens=10, output_tokens=2, total_tokens=12, llm_requests=1, latency_seconds=1.5, estimated_cost_usd=None)
        meta = dict(run_id=store.directory.name, pipeline_version="v2_bounded" if v2 else "v1_fixed", model="test-model",
            question_id="Q01", as_of="2026-11-11T23:59:59Z", execution_status="completed_review_failed",
            initial_reviewer_decision="FAIL", recovery_attempted=True, recovery_state="completed", trigger_issue_ids=["I1"], mode="paired_continuation")
        store.write_json("metadata.json", meta)
        store.write_json("run_result.json", dict(configuration="v2" if v2 else "v1", final_report=report, review_result=review, stage_metrics=[stage]))
        store.write_json("metrics.json", dict(final_totals=totals, initial_v1=totals, incremental_total=totals) if v2 else dict(totals=totals))
        store.write_json("initial_evidence_pack.json" if v2 else "evidence_pack.json", dict(evidence_items=[dict(record_id="R1", source_title="Source", source_type="official", available_at="2026-01-01Z")]))
        store.write_json("initial_retrieval_logs.json" if v2 else "retrieval_logs.json", [dict(tool="get_record", status="ok", record_ids=["R1"])])
        if v2:
            store.write_json("claim_changes.json", dict(added=[], removed=[], modified=[{}]))
            store.write_json("evidence_changes.json", dict(initial_evidence_ids=["R1"], final_evidence_ids=["R1", "R2"]))
        store.finalize()
        return store.directory
    return create


def text(path):
    with fitz.open(path) as doc:
        assert len(doc) == 2
        return "\n".join(page.get_text() for page in doc)


@pytest.mark.parametrize("v2", [False, True])
def test_automatic_pdf_and_shared_definitions(run, v2):
    directory = run(v2)
    content = text(directory / "reports/report-0001.pdf")
    for label in ("Research output", "Performance and review", "Task quality", "Unsupported", "Evidence and source", "Error detection", "Execution time", "Pending", "Cached input", "Unavailable"):
        assert label in content
    if v2:
        assert "Additional tokens: 12" in content
        assert "Issues targeted: I1" in content
        assert "Evidence records initial / final: 1 / 2" in content
    else:
        assert "N/A (V1)" in content


def test_regeneration_is_deterministic_and_preserves_artifacts(run):
    directory = run()
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    second = generate_pdf(directory)
    assert second.name == "report-0002.pdf"
    assert second.read_bytes() == (directory / "reports/report-0001.pdf").read_bytes()
    assert all((directory / name).read_bytes() == data for name, data in before.items())
    verify_manifest(directory)


def test_long_content_exactly_two_pages_and_excerpt(run):
    directory = run(long=True)
    assert "excerpt" in text(directory / "reports/report-0001.pdf")


def test_offline_cli_needs_no_configuration_or_agents(run, monkeypatch, capsys):
    directory = run()
    monkeypatch.setattr(cli, "load_config", lambda **kw: pytest.fail("No config needed"))
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **kw: pytest.fail("No agents"))
    assert cli.main(["report", "--run-dir", str(directory)]) == 0
    assert "report-0002.pdf" in capsys.readouterr().out


def test_tampered_input_rejected(run):
    directory = run()
    (directory / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        generate_pdf(directory)


def test_reporting_failure_does_not_change_run_result(run, monkeypatch):
    directory = run()
    from ci_agents import reporting
    monkeypatch.setattr(reporting, "generate_pdf", lambda *a: (_ for _ in ()).throw(OSError("sensitive message")))
    with pytest.warns(RuntimeWarning, match="OSError") as warnings:
        report_completed_run(directory)
    assert "sensitive message" not in str(warnings[0].message)
    verify_manifest(directory)


def test_missing_unknown_usage_stays_unknown(run):
    directory = run()
    _, _, _, sections = report_sections(directory)
    execution = dict(sections)["5. Execution time and latency"]
    assert "Cached input tokens (subset): 4" in execution
    assert "Estimated USD (configured rates): Unavailable" in execution


def test_failed_run_has_no_pdf(tmp_path):
    store = RunStore(tmp_path, "failed")
    store.write_json("metadata.json", {"execution_status": "execution_failed"})
    store.finalize()
    assert not (store.directory / "reports").exists()
    with pytest.raises(ValueError, match="completed"):
        generate_pdf(store.directory)


def test_missing_reportlab_does_not_break_finalization(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "reporting":
            raise ImportError("Synthetic missing dependency")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    store = RunStore(tmp_path, "dependency_failure")
    store.write_json("metadata.json", {"execution_status": "completed"})
    with pytest.warns(RuntimeWarning, match="ImportError"):
        store.finalize()
    verify_manifest(store.directory)
