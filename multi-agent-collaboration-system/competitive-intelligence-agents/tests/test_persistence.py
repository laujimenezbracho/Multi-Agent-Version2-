import json
from pathlib import Path

import pytest

from ci_agents.evaluation_contracts import IndependentEvaluation, PairedComparison
from ci_agents.persistence import (RunStore, dataset_fingerprints, fingerprint, prompt_fingerprints,
                                  render_report, validate_run_id, verify_manifest)
from ci_agents.schemas import CompetitiveReport, ReportClaim, ReviewIssue, ReviewResult

PUBLIC = Path(__file__).resolve().parents[1] / "data/public"


@pytest.mark.parametrize("run_id", ["abc", "run-01", "A_1", "a" * 80])
def test_valid_run_ids(run_id):
    assert validate_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["", "..", "../escape", "a/b", "a\\b", "CON", "nul", "C:drive", "a.", "a" * 81, None])
def test_invalid_run_ids(run_id):
    with pytest.raises(ValueError):
        validate_run_id(run_id)


def test_no_overwrite_atomic_json_and_manifest(tmp_path):
    store = RunStore(tmp_path, "baseline")
    store.write_json("plan.json", {"question": "Example", "unknown": None})
    original = (store.directory / "plan.json").read_bytes()
    with pytest.raises(FileExistsError):
        store.write_json("plan.json", {"changed": True})
    assert (store.directory / "plan.json").read_bytes() == original
    with pytest.raises(FileExistsError):
        RunStore(tmp_path, "baseline")
    store.finalize()
    assert verify_manifest(store.directory)["artifact_fingerprints"]["plan.json"]
    with pytest.raises(FileExistsError):
        store.write_json("review.json", {})
    assert not list(store.directory.glob(".artifact-*"))
    (store.directory / "plan.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_manifest(store.directory)


def test_artifact_paths_and_redaction(tmp_path):
    store = RunStore(tmp_path, "safe", secrets=["private-test-token"])
    with pytest.raises(ValueError):
        store.write_json("../outside.json", {})
    store.write_json("error.json", {"message": "private-test-token sk-anothersecret"})
    assert "private-test-token" not in (store.directory / "error.json").read_text()
    assert "sk-anothersecret" not in (store.directory / "error.json").read_text()
    json.loads((store.directory / "error.json").read_text())


def test_hashes_stable_and_content_sensitive(tmp_path):
    assert fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})
    first = dataset_fingerprints(PUBLIC)
    assert first == dataset_fingerprints(PUBLIC)
    assert set(first["files"]) == {"public_web_records.jsonl", "pricing_events.csv", "public_questions.csv"}
    prompts = prompt_fingerprints()
    assert set(prompts) == {"planner", "researcher", "analyst", "reviewer"}
    assert all(len(value) == 64 for value in prompts.values())


def test_markdown_escapes_content_and_keeps_claims():
    report = CompetitiveReport(question="<script>alert(1)</script>", as_of="2026-11-11T23:59:59Z",
        executive_summary="[click](https://example.test) ![image](bad)",
        claims=[ReportClaim(claim_id="C1", text="100 USD < 120 USD", claim_type="factual",
                            supporting_record_ids=["WEB-00001"], confidence=0.8)], sources=["WEB-00001"])
    review = ReviewResult(decision="FAIL", completeness_score=0.5, evidence_support_score=0.8, temporal_consistency_score=1,
        flagged_issues=[ReviewIssue(issue_id="I1", severity="major", issue_type="missing_required_information",
                                   explanation="Insufficient comparison", suggested_correction="Add evidence")])
    text = render_report(report, review)
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "\\[click\\]" in text and "\\!\\[image\\]" in text
    assert "C1" in text and "WEB\\-00001" in text
    assert "100 USD &lt; 120 USD" in text and "FAIL" in text and "Insufficient comparison" in text


def test_evaluation_contract_stays_null_until_independently_evaluated():
    evaluation = IndependentEvaluation(run_id="R1")
    values = evaluation.model_dump()
    for name, value in values.items():
        if name not in {"schema_version", "run_id"}:
            assert value is None
    assert IndependentEvaluation.model_validate_json(evaluation.model_dump_json()) == evaluation
    with pytest.raises(ValueError):
        IndependentEvaluation(run_id="R1", factual_accuracy=1.1)
