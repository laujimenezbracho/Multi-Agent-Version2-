"""Offline replays of real SDK paths; synthetic errors are not claimed as live root causes."""
import asyncio
from datetime import datetime, timezone
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from openai import AsyncOpenAI
from openai.types.responses import Response
try:
    import httpx2 as sdk_httpx
except ImportError:
    sdk_httpx = httpx
import pytest

from ci_agents.config import load_config, use_settings
from ci_agents.diagnostics import exception_diagnostic, local_diagnostics
from ci_agents.telemetry import capture_stage, sanitized_error

_SDK_ASYNC_SEND = sdk_httpx.AsyncClient.send
module = importlib.import_module("ci_agents.agents.planner")
QUESTION = "Compare Growth pricing"
CUTOFF = datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)
SECRET = "test-only-private-credential"


def payload():
    return {"original_question": QUESTION, "as_of": CUTOFF.isoformat(), "objective": "Collect comparable evidence",
        "tasks": [{"task_id": "T1", "research_question": "Collect Growth price observations",
                   "companies": ["NexaFlow"], "categories": ["Pricing"], "time_scope": "At cutoff",
                   "evidence_requirements": ["Comparable official prices and limitations"]}],
        "expected_answer_elements": ["Prices and uncertainty"], "comparison_dimensions": ["Pricing"]}


def response_for(text=None, usage=True):
    return Response.model_validate({"id": "resp_offline", "object": "response", "created_at": 0,
        "model": "gpt-5-mini", "status": "completed", "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        "output": [{"id": "msg_offline", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text or json.dumps(payload()), "annotations": []}]}],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}} if usage else None})


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for library in {httpx, sdk_httpx}:
        monkeypatch.setattr(library.AsyncClient, "send", AsyncMock(side_effect=AssertionError("No network permitted")))
        monkeypatch.setattr(library.Client, "send", lambda *a, **k: pytest.fail("No network permitted"))
    monkeypatch.delenv("CI_DIAGNOSTIC_MODE", raising=False)


@pytest.fixture
def settings():
    return load_config(environ={"OPENAI_API_KEY": SECRET, "OPENAI_MODEL": "gpt-5-mini"})


@pytest.mark.parametrize("case", ["valid", "timezone", "no_usage"])
def test_real_runner_real_adapter_and_parser_with_mocked_api(settings, monkeypatch, case):
    # Mock only the provider boundary, not Runner.run or model._fetch_response.
    data = payload()
    if case == "timezone":
        data["as_of"] = "2026-11-11T18:59:59-05:00"
    async def replay():
        client = AsyncOpenAI(api_key=SECRET, max_retries=0)
        create = AsyncMock(return_value=response_for(json.dumps(data), usage=case != "no_usage"))
        monkeypatch.setattr(client.responses, "create", create)
        monkeypatch.setattr(module, "AsyncOpenAI", lambda **kwargs: client)
        with use_settings(settings), capture_stage() as capture:
            result = await module.planner(QUESTION, CUTOFF)
        create.assert_awaited_once()
        call = create.call_args.kwargs
        assert call["model"] == "gpt-5-mini"
        assert call["text"]["format"]["type"] == "json_schema"
        assert call["text"]["format"]["strict"] is True
        assert result.as_of == CUTOFF and result.tasks[0].task_id == "T1"
        if case != "no_usage":
            assert capture.usage["input_tokens"] == 10
        else:
            # SDK versions may retain a request marker even without provider usage;
            # this case checks parsing success, not inferred token measurements.
            assert capture.usage is None or capture.usage["requests"] == 1
    asyncio.run(replay())


def test_native_sdk_output_parsing_failure_phase(settings, monkeypatch):
    async def replay():
        client = AsyncOpenAI(api_key=SECRET, max_retries=0)
        monkeypatch.setattr(client.responses, "create", AsyncMock(return_value=response_for("not-json")))
        monkeypatch.setattr(module, "AsyncOpenAI", lambda **kwargs: client)
        with use_settings(settings), local_diagnostics(), pytest.raises(module.PlannerError) as caught:
            await module.planner(QUESTION, CUTOFF)
        diagnostic = exception_diagnostic(caught.value, "planner", secrets=(SECRET,))
        assert diagnostic["phase"] == "sdk_execution"  # SDK owns native output parsing.
        assert diagnostic["category"] == "schema_validation"
        assert diagnostic["origin"] is not None
    asyncio.run(replay())


@pytest.mark.parametrize("phase", ["sdk_execution", "telemetry_processing", "output_validation"])
def test_controlled_typeerror_phase_and_origin(settings, monkeypatch, caplog, phase):
    # Deliberately injected errors prove phase/origin diagnostics, not live causality.
    from ci_agents.schemas import ResearchPlan
    plan = ResearchPlan.model_validate(payload())
    def synthetic_type_error(*args, **kwargs):
        raise TypeError(SECRET + " raw payload and prompt data")
    runner = AsyncMock(return_value=SimpleNamespace(final_output=plan))
    monkeypatch.setattr(module.Runner, "run", runner)
    if phase == "sdk_execution":
        async def sdk_failure(*args, **kwargs):
            synthetic_type_error()
        runner.side_effect = sdk_failure
    elif phase == "telemetry_processing":
        monkeypatch.setattr(module, "observe_sdk_result", synthetic_type_error)
    else:
        monkeypatch.setattr(module, "_validate_plan", synthetic_type_error)
    expected = TypeError if phase == "output_validation" else module.PlannerError
    with use_settings(settings), local_diagnostics(), pytest.raises(expected) as caught:
        asyncio.run(module.planner(QUESTION, CUTOFF))
    diagnostic = sanitized_error(caught.value, "planner", secrets=(SECRET,))["diagnostic"]
    assert diagnostic["original_exception_class"] == "TypeError"
    assert diagnostic["phase"] == phase and diagnostic["category"] == "unknown"
    assert diagnostic["origin"]["module"] == __name__
    assert diagnostic["origin"]["function"] == "synthetic_type_error"
    assert diagnostic["origin"]["line"] > 0
    assert "filename" not in diagnostic["origin"]
    assert SECRET not in json.dumps(diagnostic) + caplog.text
    assert "raw payload" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records if record.name.startswith("ci_agents"))
    runner.assert_awaited_once()


def test_optional_mode_default_off_and_environment_toggle(monkeypatch):
    def fail():
        raise TypeError("sensitive message")
    try:
        fail()
    except TypeError as exc:
        assert "origin" not in exception_diagnostic(exc, "planner", phase="sdk_execution")
        monkeypatch.setenv("CI_DIAGNOSTIC_MODE", "1")
        debug = exception_diagnostic(exc, "planner", phase="sdk_execution")
        assert debug["origin"]["function"] == "fail" and debug["phase"] == "sdk_execution"
        with local_diagnostics(False):
            assert "origin" not in exception_diagnostic(exc, "planner", phase="sdk_execution")


def test_local_output_error_diagnostic_is_output_phase(settings, monkeypatch):
    from ci_agents.schemas import ResearchPlan
    data = payload()
    data["tasks"][0]["companies"] = ["Unknown company"]
    monkeypatch.setattr(module.Runner, "run", AsyncMock(return_value=SimpleNamespace(final_output=ResearchPlan.model_validate(data))))
    with use_settings(settings), local_diagnostics(), pytest.raises(module.PlannerOutputError) as caught:
        asyncio.run(module.planner(QUESTION, CUTOFF))
    diagnostic = exception_diagnostic(caught.value, "planner")
    assert diagnostic["phase"] == "output_validation" and diagnostic["category"] == "schema_validation"
    assert diagnostic["origin"]["function"] == "_validate_plan"


def test_real_http_response_deserialization_on_in_memory_transport(settings, monkeypatch):
    import socket
    def no_socket(*args, **kwargs):
        raise AssertionError("No sockets permitted")
    # Allow HTTP client's orchestration only with an in-memory transport, not TCP.
    monkeypatch.setattr(sdk_httpx.AsyncClient, "send", _SDK_ASYNC_SEND)
    exchanges = []
    def reply(request):
        exchanges.append(request.url.path)
        return sdk_httpx.Response(200, json=response_for().model_dump(mode="json"),
                                  headers={"x-request-id": "req_offline_fixture"})
    async def replay():
        # Windows asyncio creates a local self-pipe when the event loop starts;
        # install network guards after that initialization, before the SDK client.
        with monkeypatch.context() as guard:
            guard.setattr(socket.socket, "connect", no_socket)
            guard.setattr(socket, "create_connection", no_socket)
            transport = sdk_httpx.MockTransport(reply)
            client = AsyncOpenAI(api_key=SECRET, base_url="https://offline.invalid/v1", max_retries=0,
                http_client=sdk_httpx.AsyncClient(transport=transport, trust_env=False))
            monkeypatch.setattr(module, "AsyncOpenAI", lambda **kwargs: client)
            with use_settings(settings), capture_stage() as capture:
                plan = await module.planner(QUESTION, CUTOFF)
            assert plan.original_question == QUESTION and plan.as_of == CUTOFF
            assert capture.usage["total_tokens"] == 30
    asyncio.run(replay())
    assert exchanges == ["/v1/responses"]


def test_optional_origin_and_phase_survive_run_persistence(settings, monkeypatch, tmp_path):
    from dataclasses import replace
    from ci_agents import pipeline
    from ci_agents.schemas import ResearchPlan
    settings = replace(settings, runs_dir=tmp_path / "runs")
    monkeypatch.setattr(module.Runner, "run", AsyncMock(return_value=SimpleNamespace(
        final_output=ResearchPlan.model_validate(payload()))))
    def failing_observer(result):
        raise TypeError(SECRET + " local observer payload")
    monkeypatch.setattr(module, "observe_sdk_result", failing_observer)
    with local_diagnostics():
        result = asyncio.run(pipeline.run_pipeline(QUESTION, CUTOFF, "offline-debug", settings=settings))
    assert result.execution_status == "execution_failed"
    path = settings.runs_dir / "offline-debug"
    diagnostic = json.loads((path / "diagnostics.json").read_text())["failures"][0]
    assert diagnostic["phase"] == "telemetry_processing"
    assert diagnostic["origin"]["function"] == "failing_observer"
    assert diagnostic["original_exception_class"] == "TypeError"
    assert not (path / "plan.json").exists()
    assert json.loads((path / "error.json").read_text())["message"] == "Execution failed during planner; provider details omitted"
    for artifact in path.iterdir():
        content = artifact.read_text()
        assert SECRET not in content and "local observer payload" not in content
