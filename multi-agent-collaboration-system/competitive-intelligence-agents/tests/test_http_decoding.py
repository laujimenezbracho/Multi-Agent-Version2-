"""Synthetic compression tests only. No live requests or package modifications."""
import asyncio
import gzip
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import zlib

import httpx
import pytest
from openai import AsyncOpenAI

from ci_agents.config import OPENAI_ACCEPT_ENCODING, load_config, use_settings
from ci_agents.schemas import CompetitiveReport, EvidencePack, ResearchPlan, ResearchTask

sdk_httpx = pytest.importorskip("httpx2")
decoders = importlib.import_module("httpx2._decoders")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    for library in {httpx, sdk_httpx}:
        monkeypatch.setattr(library.AsyncHTTPTransport, "handle_async_request",
                            AsyncMock(side_effect=AssertionError("External HTTP forbidden")))
        monkeypatch.setattr(library.HTTPTransport, "handle_request",
                            lambda *args, **kwargs: pytest.fail("External HTTP forbidden"))


def test_exact_installed_brotli_failure_is_reproduced():
    brotli = pytest.importorskip("brotli")
    raw = b"Synthetic local response for a compression compatibility test." * 20
    encoded = brotli.compress(raw)
    assert isinstance(encoded, bytes)
    assert brotli.Decompressor().process(encoded) == raw
    try:
        brotli.Decompressor().process(encoded, output_buffer_limit=1024)
    except TypeError:
        pass
    else:
        pytest.skip("Installed Brotli now supports bounded output; original incompatibility no longer present")
    with pytest.raises(TypeError) as caught:
        list(decoders.BrotliDecoder().decode(encoded))
    tb = caught.value.__traceback__
    while tb.tb_next:
        tb = tb.tb_next
    assert tb.tb_frame.f_globals["__name__"] == "httpx2._decoders"
    assert tb.tb_frame.f_code.co_name == "decode"
    # No provider message is involved: this error comes from an unsupported kwarg.
    assert "keyword" in str(caught.value)


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
def test_real_sdk_deserializes_synthetic_compressed_response(encoding):
    raw = json.dumps({"id": "resp_offline_compressed", "object": "response", "created_at": 0,
        "model": "gpt-5-mini", "status": "completed", "output": [], "parallel_tool_calls": False,
        "tool_choice": "auto", "tools": []}).encode("utf-8")
    encoded = gzip.compress(raw) if encoding == "gzip" else zlib.compress(raw)
    observed = []
    def respond(request):
        observed.append(request.headers.get("accept-encoding"))
        assert request.headers["accept-encoding"] == "gzip, deflate"
        assert "br" not in request.headers["accept-encoding"].split(", ")
        return sdk_httpx.Response(200, headers={"Content-Encoding": encoding, "Content-Type": "application/json"},
                                 stream=sdk_httpx.ByteStream(encoded))
    async def replay():
        async with AsyncOpenAI(api_key="test-only", base_url="https://offline.invalid/v1", max_retries=0,
            default_headers={"Accept-Encoding": OPENAI_ACCEPT_ENCODING},
            http_client=sdk_httpx.AsyncClient(transport=sdk_httpx.MockTransport(respond), trust_env=False)) as client:
            result = await client.responses.create(model="gpt-5-mini", input="Synthetic offline test")
        assert result.id == "resp_offline_compressed"
    asyncio.run(replay())
    assert observed == ["gzip, deflate"]


def test_old_negotiation_reproduces_failure_and_configuration_avoids_it():
    brotli = pytest.importorskip("brotli")
    raw = json.dumps({"id": "resp_negotiated", "object": "response", "created_at": 0,
        "model": "gpt-5-mini", "status": "completed", "output": [], "parallel_tool_calls": False,
        "tool_choice": "auto", "tools": []}).encode()
    try:
        brotli.Decompressor().process(brotli.compress(raw), output_buffer_limit=1024)
    except TypeError:
        pass
    else:
        pytest.skip("Backend incompatibility has been removed from environment")
    encodings = []
    def respond(request):
        # Synthetic server follows Accept-Encoding; no OpenAI service is contacted.
        selected = "br" if "br" in request.headers["accept-encoding"].split(", ") else "gzip"
        encodings.append(selected)
        compressed = brotli.compress(raw) if selected == "br" else gzip.compress(raw)
        return sdk_httpx.Response(200, headers={"Content-Encoding": selected, "Content-Type": "application/json"},
                                 stream=sdk_httpx.ByteStream(compressed))
    async def call(headers=None):
        async with AsyncOpenAI(api_key="test-only", base_url="https://offline.invalid/v1", max_retries=0,
                default_headers=headers,
                http_client=sdk_httpx.AsyncClient(transport=sdk_httpx.MockTransport(respond), trust_env=False)) as client:
            return await client.responses.create(model="gpt-5-mini", input="Offline test")
    with pytest.raises(TypeError):
        asyncio.run(call())
    assert asyncio.run(call({"Accept-Encoding": OPENAI_ACCEPT_ENCODING})).id == "resp_negotiated"
    assert encodings == ["br", "gzip"]


@pytest.mark.parametrize("name", ["planner", "researcher", "analyst", "reviewer"])
def test_each_agent_configures_only_supported_compression(name, monkeypatch):
    module = importlib.import_module("ci_agents.agents." + name)
    settings = load_config(environ={"OPENAI_API_KEY": "test-only", "OPENAI_MODEL": "gpt-5-mini"})
    captured = []
    def factory(**kwargs):
        captured.append(kwargs)
        return AsyncOpenAI(**kwargs)
    monkeypatch.setattr(module, "AsyncOpenAI", factory)
    monkeypatch.setattr(module.Runner, "run", AsyncMock(side_effect=RuntimeError("Mocked stop before any API")))
    plan = ResearchPlan(original_question="Offline test", as_of="2026-11-11T23:59:59Z", objective="Collect evidence",
                        tasks=[ResearchTask(task_id="T1", research_question="Offline test", time_scope="At cutoff")])
    pack = EvidencePack(original_question=plan.original_question, as_of=plan.as_of)
    report = CompetitiveReport(question=plan.original_question, as_of=plan.as_of, executive_summary="No evidence")
    arguments = {"planner": (plan.original_question, plan.as_of), "researcher": (plan, Mock()),
                 "analyst": (plan, pack), "reviewer": (plan, pack, report)}
    with use_settings(settings), pytest.raises(RuntimeError):
        asyncio.run(getattr(module, name)(*arguments[name]))
    assert captured[0]["default_headers"] == {"Accept-Encoding": "gzip, deflate"}
    assert captured[0]["max_retries"] == 0
    module.Runner.run.assert_awaited_once()
