"""Append-only run artifacts with exclusive publication and verifiable manifests."""
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import re
import tempfile

from pydantic import BaseModel
from .telemetry import redact_text

STAGE_NAMES = ("planner", "researcher", "analyst", "reviewer")
PUBLIC_FILES = ("public_web_records.jsonl", "pricing_events.csv", "public_questions.csv")


def canonical_json(value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_run_id(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run_id must contain 1-80 ASCII letters, digits, underscores or hyphens")
    if run_id.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError("run_id is a reserved Windows filename")
    return run_id


def dataset_fingerprints(directory):
    files = {name: sha256((Path(directory) / name).read_bytes()).hexdigest() for name in PUBLIC_FILES}
    return {"algorithm": "sha256", "definition": "canonical sorted JSON mapping filenames to SHA-256 of exact bytes",
            "files": files, "sha256": fingerprint(files)}


def prompt_fingerprints():
    from importlib.resources import files
    return {name: sha256(files("ci_agents").joinpath(f"prompts/{name}.md").read_bytes()).hexdigest()
            for name in STAGE_NAMES}


class RunStore:
    """Never reuse a directory or replace an artifact, including after finalization."""
    def __init__(self, runs_dir, run_id, *, secrets=()):
        validate_run_id(run_id)
        root = Path(runs_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.directory = root / run_id
        self.directory.mkdir(exist_ok=False)
        self.secrets = tuple(secrets)
        self.fingerprints = {}
        self._closed = False

    def _safe(self, value):
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        if isinstance(value, dict):
            return {redact_text(str(key), self.secrets): self._safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._safe(val) for val in value]
        if isinstance(value, str):
            return redact_text(value, self.secrets)
        return value

    def write_json(self, name, value):
        return self.write_text(name, canonical_json(self._safe(value)) + "\n")

    def write_text(self, name, text):
        if self._closed or (self.directory / "manifest.json").exists():
            raise FileExistsError("Run artifacts are finalized and cannot be overwritten")
        if not re.fullmatch(r"[a-z][a-z0-9_]*\.(json|md|csv)", name):
            raise ValueError("Invalid artifact filename")
        destination = self.directory / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Artifact already exists: {name}")
        data = redact_text(text, self.secrets).encode("utf-8")
        handle, temporary = tempfile.mkstemp(prefix=".artifact-", dir=self.directory)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # Hard-link publication is atomic and refuses existing destinations on
            # Windows and POSIX; no replace/overwrite fallback is allowed.
            os.link(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self.fingerprints[name] = sha256(data).hexdigest()
        return destination

    def finalize(self):
        manifest = {"schema_version": "run_manifest.v1", "hash_algorithm": "sha256_exact_file_bytes",
                    "artifact_fingerprints": dict(self.fingerprints)}
        self.write_json("manifest.json", manifest)
        self._closed = True
        # Presentation runs after sealing; it cannot alter experiment artifacts.
        try:
            from .reporting import report_completed_run
            report_completed_run(self.directory)
        except Exception as exc:
            import warnings
            warnings.warn(f"PDF reporting unavailable ({type(exc).__name__}); use ci-agents report to retry.", RuntimeWarning)
        return manifest


def verify_manifest(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Manifest must not be a symbolic link")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "run_manifest.v1":
        raise ValueError("Unsupported manifest version")
    for name, expected in manifest["artifact_fingerprints"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*\.(json|md|csv)", name):
            raise ValueError("Invalid manifest artifact path")
        path = directory / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact fingerprint mismatch: {name}")
    return manifest


def baseline_reference(runs_dir, run_id):
    from .evaluation_contracts import BaselineReference
    validate_run_id(run_id)
    root = Path(runs_dir).resolve()
    directory = root / run_id
    if directory.is_symlink() or directory.resolve().parent != root:
        raise ValueError("Baseline path must stay inside runs directory")
    manifest = verify_manifest(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["execution_status"] not in {"completed", "completed_review_failed"}:
        raise ValueError("Baseline must be a completed initial run")
    required = {"plan.json", "evidence_pack.json", "report.json", "review.json", "metrics.json", "retrieval_logs.json", "metadata.json"}
    if not required.issubset(manifest["artifact_fingerprints"]):
        raise ValueError("Baseline lacks required initial artifacts")
    return BaselineReference(baseline_run_id=run_id,
        manifest_sha256=sha256((directory / "manifest.json").read_bytes()).hexdigest(),
        artifact_fingerprints=manifest["artifact_fingerprints"], question=metadata["question"],
        question_id=metadata["question_id"], as_of=metadata["as_of"], comparison_key=metadata["comparison_key"])


def verify_baseline_reference(runs_dir, reference):
    actual = baseline_reference(runs_dir, reference.baseline_run_id)
    if actual != reference:
        raise ValueError("Baseline reference does not match immutable artifacts")
    return actual


def render_report(report, review=None):
    def safe(text):
        # Escape HTML and Markdown, neutralizing links/images/autolinks without
        # changing the displayed factual text. Source URLs remain inert text.
        text = html.escape(str(text), quote=True)
        text = text.replace("\\", "\\\\")
        for char in "`*_{}[]()#+-.!|":
            text = text.replace(char, "\\" + char)
        return text.replace("\r", "").replace("\n", "  \n")
    lines = ["# Competitive intelligence report", "", safe(report.question), "",
             "Evaluation cutoff: " + safe(report.as_of.isoformat()), "", "## Executive summary", "", safe(report.executive_summary)]
    for title, field in (("Key findings", "key_findings"), ("Competitor comparison", "competitor_comparison"),
                         ("Strategic implications", "strategic_implications"), ("Limitations", "limitations")):
        lines += ["", "## " + title, ""] + ["- " + safe(text) for text in getattr(report, field)]
    lines += ["", "## Claims and source references", ""]
    for claim in report.claims:
        lines += ["- " + safe(claim.claim_id) + " (" + safe(claim.claim_type) + "): " + safe(claim.text)
                  + "; sources: " + safe(", ".join(claim.supporting_record_ids) or "none")]
    lines += ["", "## Reviewer", "", "Decision: " + (review.decision if review else "unavailable (execution did not complete)")]
    if review:
        lines += ["Scores: " + safe(f"completeness={review.completeness_score}; evidence_support={review.evidence_support_score}; temporal={review.temporal_consistency_score}")]
        for issue in review.flagged_issues:
            lines += ["- " + safe(f"{issue.issue_id} [{issue.severity}/{issue.issue_type}]: {issue.explanation}; correction: {issue.suggested_correction}")]
    return "\n".join(lines) + "\n"
