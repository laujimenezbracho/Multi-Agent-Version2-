"""Evaluator post-processing contracts; no model judgments are invented."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_v2_reliability import (
    paired_summary,
    validate_recovery_grade,
    reviewer_challenge,
    ensure_new_evaluation,
    validate_cached_case,
    validate_cached_challenge,
    public_output,
)
from ci_agents.dataset import load_dataset
from ci_agents.schemas import CompetitiveReport
from datetime import datetime, timezone


def case():
    report = {"claims": [{"claim_id": "C1"}]}
    source = {"record_id": "R1"}
    return {
        "initial": {"report": report, "original_cited_sources": [source]},
        "final": {"report": report, "original_cited_sources": [source]},
    }


def grade(**changes):
    value = {
        "status": "recovery_comparison",
        "corrected_defects": [
            {
                "explanation": "Wrong amount corrected",
                "reference_record_ids": ["R1"],
                "initial_claim_ids": ["C1"],
                "final_claim_ids": ["C1"],
            }
        ],
        "introduced_defects": [],
        "remaining_defects": [],
        "explanation": "Source-grounded comparison",
    }
    value.update(changes)
    return value


def test_recovery_grade_requires_known_sources_and_claims():
    rubric = {"audited_reference_sources": [{"record_id": "R1"}]}
    assert (
        len(validate_recovery_grade(grade(), case(), rubric)["corrected_defects"]) == 1
    )
    for field, value in [
        ("reference_record_ids", ["FAKE"]),
        ("initial_claim_ids", ["FAKE"]),
        ("final_claim_ids", ["FAKE"]),
    ]:
        invalid = grade()
        invalid["corrected_defects"][0][field] = value
        with pytest.raises(ValueError):
            validate_recovery_grade(invalid, case(), rubric)


def test_paired_summary_keeps_failed_grading_missing():
    rows = [
        {
            "question_id": "Q01",
            "repetition": 1,
            "variant": "initial",
            "scores": {
                "status": "assessed",
                "decision_safe_coverage": 50,
                "material_error_count": 1,
            },
        },
        {
            "question_id": "Q01",
            "repetition": 1,
            "variant": "final",
            "scores": {"status": "invalid_grader_output"},
        },
    ]
    out = paired_summary(rows, {})[0]
    assert (
        out["quality_delta"] is None
        and out["comparison_status"] == "incomplete_or_invalid"
    )
    assert out["initial_material_errors"] == 1 and out["final_material_errors"] is None


def test_challenge_is_a_single_controlled_price_mutation():
    root = Path(__file__).resolve().parents[1]
    data = load_dataset(root / "data/public")
    cutoff = datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)
    p, e, valid = reviewer_challenge(data, cutoff, False)
    _, _, wrong = reviewer_challenge(data, cutoff, True)
    assert (
        valid.executive_summary.replace("USD 149", "USD 1") == wrong.executive_summary
    )
    assert valid.sources == wrong.sources == ["WEB-00002"]
    assert (
        valid.question == p.original_question
        and e.evidence_items[0].record_id == "WEB-00002"
    )
    assert CompetitiveReport.model_validate(valid.model_dump()) == valid


def test_completed_evaluation_cannot_be_implicitly_regraded(tmp_path):
    (tmp_path / "evaluation_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="already completed"):
        ensure_new_evaluation(tmp_path)
    ensure_new_evaluation(tmp_path, prepare_only=True)


def test_cached_reviewer_prediction_integrity():
    from ci_agents.persistence import fingerprint

    value = {"valid_output": True, "review": {"decision": "PASS"}}
    value["prediction_sha256"] = fingerprint(value)
    validate_cached_challenge(value)
    value["review"]["decision"] = "FAIL"
    with pytest.raises(ValueError, match="Cached Reviewer prediction"):
        validate_cached_challenge(value)


def test_recovery_grade_requires_actual_material_defect_anchors():
    invalid = grade()
    invalid["corrected_defects"][0]["reference_record_ids"] = []
    with pytest.raises(ValueError, match="source anchor"):
        validate_recovery_grade(
            invalid, case(), {"audited_reference_sources": [{"record_id": "R1"}]}
        )
    invalid = grade()
    invalid["corrected_defects"][0]["initial_claim_ids"] = []
    with pytest.raises(ValueError, match="initial claim"):
        validate_recovery_grade(
            invalid, case(), {"audited_reference_sources": [{"record_id": "R1"}]}
        )


def test_cached_predictions_must_match_the_sealed_run(tmp_path, monkeypatch):
    from ci_agents.persistence import RunStore

    monkeypatch.setattr(
        "ci_agents.reporting.report_completed_run", lambda directory: None
    )
    root = Path(__file__).resolve().parents[1]
    data = load_dataset(root / "data/public")
    cutoff = datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)
    _, _, report = reviewer_challenge(data, cutoff, False)
    store = RunStore(tmp_path, "cache-run")
    store.write_json("initial_report.json", report)
    store.write_json("final_report.json", report)
    store.write_json("metadata.json", {"execution_status": "completed"})
    store.finalize()
    prediction = {
        "initial": public_output(report, data),
        "final": public_output(report, data),
        "resources": None,
        "execution_status": "completed",
    }
    validate_cached_case(prediction, store.directory, data)
    prediction["final"]["report"]["executive_summary"] = "An accidentally edited report"
    with pytest.raises(ValueError, match="Cached prediction"):
        validate_cached_case(prediction, store.directory, data)
