"""No live requests: real SDK exception objects injected through mocked Runner."""
import asyncio
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from agents.exceptions import ModelBehaviorError
from openai import AuthenticationError, RateLimitError, BadRequestError, NotFoundError, APIConnectionError
try:
    import httpx2 as sdk_httpx  # Transport used by the installed OpenAI SDK.
except ImportError:
    sdk_httpx = httpx
from pydantic import ValidationError

from ci_agents.config import load_config
from ci_agents.diagnostics import exception_diagnostic, log_exception_diagnostic
from ci_agents.telemetry import sanitized_error
from ci_agents import pipeline
from ci_agents.schemas import ResearchPlan

planner_module = importlib.import_module("ci_agents.agents.planner")
PUBLIC = Path(__file__).resolve().parents[1] / "data/public"
CUTOFF = datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)
SECRET = "private-test-credential"
PROVIDER_MESSAGE = "Raw provider message contains private-test-credential and confidential payload"


def sdk_exception(case):
    request = sdk_httpx.Request("POST", "https://api.openai.com/v1/responses",
                               headers={"Authorization": "Bearer " + SECRET}, content=b"private request payload")
    if case == "network":
        return APIConnectionError(message=PROVIDER_MESSAGE, request=request)
    if case == "schema":
        return ModelBehaviorError(PROVIDER_MESSAGE)
    definitions = {
        "authentication": (AuthenticationError, 401, "invalid_api_key"),
        "quota": (RateLimitError, 429, "insufficient_quota"),
        "rate_limit": (RateLimitError, 429, "rate_limit_exceeded"),
        "invalid_request": (BadRequestError, 400, "unsupported_parameter"),
        "model_access": (NotFoundError, 404, "model_not_found"),
        "server_schema": (BadRequestError, 400, "invalid_json_schema"),
    }
    cls, status, code = definitions[case]
    response = sdk_httpx.Response(status, request=request, headers={"x-request-id": "req_safe_test_123"},
                                  json={"error": {"message": PROVIDER_MESSAGE}})
    return cls(PROVIDER_MESSAGE, response=response, body={"code": code, "message": PROVIDER_MESSAGE})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    mocks = []
    for library in {httpx, sdk_httpx}:
        for client in (library.AsyncClient, library.Client):
            mock = AsyncMock(side_effect=AssertionError("No HTTP permitted")) if client is library.AsyncClient else lambda *a, **k: pytest.fail("No HTTP permitted")
            monkeypatch.setattr(client, "send", mock)
            if isinstance(mock, AsyncMock):
                mocks.append(mock)
    yield
    for mock in mocks:
        mock.assert_not_called()


@pytest.fixture
def settings(tmp_path):
    return load_config(environ={"OPENAI_API_KEY": SECRET, "OPENAI_MODEL": "gpt-5-mini",
                               "DATA_DIR": str(PUBLIC), "RUNS_DIR": str(tmp_path / "runs")})


@pytest.mark.parametrize("case,category", [("authentication", "authentication"), ("quota", "quota"),
    ("rate_limit", "rate_limit"), ("invalid_request", "invalid_request"), ("network", "network"),
    ("schema", "schema_validation"), ("model_access", "model_access"), ("server_schema", "schema_validation")])
def test_planner_logs_allowlisted_original_metadata(case, category, settings, monkeypatch, caplog):
    original = sdk_exception(case)
    monkeypatch.setattr(planner_module, "load_config", lambda: settings)
    run = AsyncMock(side_effect=original)
    monkeypatch.setattr(planner_module.Runner, "run", run)
    with pytest.raises(planner_module.PlannerError) as caught:
        asyncio.run(planner_module.planner("Compare public pricing", CUTOFF))
    error = sanitized_error(caught.value, "planner", secrets=(SECRET,))
    assert error["error_type"] == "PlannerError"
    assert error["message"] == "Execution failed during planner; provider details omitted"
    diagnostic = error["diagnostic"]
    assert diagnostic["original_exception_class"] == type(original).__name__
    assert diagnostic["category"] == category and diagnostic["stage"] == "planner"
    assert diagnostic["http_status_code"] == getattr(original, "status_code", None)
    assert diagnostic["api_error_code"] == getattr(original, "code", None)
    assert diagnostic["request_id"] == getattr(original, "request_id", None)
    serialized = json.dumps(error) + caplog.text
    for forbidden in [SECRET, PROVIDER_MESSAGE, "confidential payload", "private request payload", "Authorization", "Bearer"]:
        assert forbidden not in serialized
    assert all(record.exc_info is None and record.stack_info is None for record in caplog.records)
    assert str(caught.value) == "Planner SDK execution failed; no retry was attempted"
    run.assert_awaited_once()


def test_safe_metadata_survives_pipeline_persistence(settings, monkeypatch, caplog):
    original = sdk_exception("quota")
    runner = AsyncMock(side_effect=original)
    monkeypatch.setattr(planner_module.Runner, "run", runner)
    downstream = []
    for name in ["researcher", "analyst", "reviewer"]:
        mock = AsyncMock(side_effect=AssertionError("No downstream execution"))
        monkeypatch.setattr(pipeline, name, mock)
        downstream.append(mock)
    result = asyncio.run(pipeline.run_pipeline("Compare public pricing", CUTOFF, "diagnostic-test", settings=settings))
    assert result.execution_status == "execution_failed" and result.final_report is None
    directory = settings.runs_dir / "diagnostic-test"
    error = json.loads((directory / "error.json").read_text())
    assert error["diagnostic"]["original_exception_class"] == "RateLimitError"
    assert error["diagnostic"]["category"] == "quota"
    assert error["diagnostic"]["request_id"] == "req_safe_test_123"
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    assert diagnostics["failures"] == [error["diagnostic"]]
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["error"]["diagnostic"] == error["diagnostic"]
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["failure_diagnostics"] == [error["diagnostic"]]
    assert not (directory / "plan.json").exists()
    for path in directory.iterdir():
        content = path.read_text()
        assert SECRET not in content and PROVIDER_MESSAGE not in content and "private request payload" not in content
    for mock in downstream:
        mock.assert_not_called()
    runner.assert_awaited_once()


def test_suppressed_context_is_inspectable_before_process_exits():
    try:
        try:
            raise sdk_exception("authentication")
        except Exception:
            raise planner_module.PlannerError("Generic") from None
    except planner_module.PlannerError as error:
        diagnostic = exception_diagnostic(error, "planner", secrets=(SECRET,))
    assert diagnostic["original_exception_class"] == "AuthenticationError" and diagnostic["http_status_code"] == 401


def test_local_structured_validation_failure(settings, monkeypatch, caplog):
    monkeypatch.setattr(planner_module, "load_config", lambda: settings)
    run = AsyncMock(return_value=SimpleNamespace(final_output="not a structured plan"))
    monkeypatch.setattr(planner_module.Runner, "run", run)
    with pytest.raises(planner_module.PlannerOutputError) as caught:
        asyncio.run(planner_module.planner("Compare public pricing", CUTOFF))
    diagnostic = exception_diagnostic(caught.value, "planner")
    assert diagnostic["category"] == "schema_validation"
    assert diagnostic["original_exception_class"] == "PlannerOutputError"
    assert diagnostic["http_status_code"] is None and diagnostic["request_id"] is None
    assert "not a structured plan" not in caplog.text


def test_pydantic_errors_do_not_serialize_input_values(caplog):
    try:
        ResearchPlan.model_validate({"original_question": SECRET})
    except ValidationError as exc:
        diagnostic = log_exception_diagnostic(exc, "planner", secrets=(SECRET,))
    assert diagnostic["category"] == "schema_validation"
    assert diagnostic["original_exception_class"] == "ValidationError"
    assert SECRET not in caplog.text


def test_rejects_secrets_and_malformed_scalar_fields(caplog):
    error = sdk_exception("invalid_request")
    error.code = SECRET
    error.request_id = SECRET
    safe = log_exception_diagnostic(error, "planner", secrets=(SECRET,))
    assert safe["api_error_code"] is None and safe["request_id"] is None
    error.code = "sk-exposedcredential"
    error.request_id = "req_header\nAuthorization: Bearer stolen"
    safe = log_exception_diagnostic(error, "planner")
    assert safe["api_error_code"] is None and safe["request_id"] is None
    assert "stolen" not in caplog.text and "sk-exposedcredential" not in caplog.text and SECRET not in caplog.text


def test_unknown_error_has_unknown_not_guessed_diagnosis():
    diagnostic = exception_diagnostic(RuntimeError(PROVIDER_MESSAGE), "planner", secrets=(SECRET,))
    assert diagnostic["category"] == "unknown"
    assert diagnostic["http_status_code"] is None and diagnostic["api_error_code"] is None
    assert diagnostic["request_id"] is None
