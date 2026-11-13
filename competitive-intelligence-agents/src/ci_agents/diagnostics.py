"""Allowlisted SDK failure metadata; never serialize messages, payloads or headers."""
from contextlib import contextmanager
from contextvars import ContextVar
import os
import json
import logging
import re

import httpx
from openai import APIError, APIConnectionError, AuthenticationError, RateLimitError
from agents.exceptions import ModelBehaviorError
from pydantic import ValidationError

logger = logging.getLogger(__name__)
STAGES = {"initialization", "planner", "researcher", "analyst", "reviewer", "pilot_question"}
CATEGORIES = {"authentication", "quota", "rate_limit", "invalid_request", "model_access",
              "network", "schema_validation", "server_error", "unknown"}


PHASES = {"sdk_execution", "output_validation", "telemetry_processing"}
_local_mode: ContextVar[bool | None] = ContextVar("ci_local_diagnostics", default=None)


@contextmanager
def local_diagnostics(enabled=True):
    """Opt in locally; never enables SDK/HTTP debug logging or traceback dumps."""
    token = _local_mode.set(bool(enabled))
    try:
        yield
    finally:
        _local_mode.reset(token)


def _local_enabled():
    configured = _local_mode.get()
    return configured if configured is not None else os.environ.get("CI_DIAGNOSTIC_MODE", "").lower() in {"1", "true", "yes"}


def _origin(exc, secrets):
    frame = exc.__traceback__
    if frame is None:
        return None
    while frame.tb_next is not None:
        frame = frame.tb_next
    # Only the terminal Python frame's module/function/line. No filename, source,
    # frame locals/globals, exception message, or other frames are serialized.
    return _clean_origin({"module": frame.tb_frame.f_globals.get("__name__"),
                          "function": frame.tb_frame.f_code.co_name, "line": frame.tb_lineno}, secrets)


def _clean_origin(origin, secrets):
    if not isinstance(origin, dict):
        return None
    def identifier(value, pattern):
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            return None
        return None if any(secret and secret in value for secret in secrets) else value
    line = origin.get("line")
    return {"module": identifier(origin.get("module"), r"[A-Za-z_][A-Za-z0-9_.]{0,199}"),
            "function": identifier(origin.get("function"), r"[A-Za-z_][A-Za-z0-9_]{0,199}"),
            "line": line if isinstance(line, int) and not isinstance(line, bool) and line > 0 else None}


def _token(value, secrets=(), *, request_id=False):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        return None
    if any(secret and secret in value for secret in secrets) or value.lower().startswith(("sk-", "bearer", "authorization")):
        return None
    # Request IDs are opaque tokens, never values copied from generic headers.
    return value


def _name(exc, secrets):
    value = _token(type(exc).__name__, secrets)
    return value if value is not None else "Exception"


def _clean(diagnostic, stage, secrets):
    status = diagnostic.get("http_status_code")
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        status = None
    category = diagnostic.get("category")
    result = {"schema_version": "sdk_failure.v1",
            "original_exception_class": _token(diagnostic.get("original_exception_class"), secrets) or "Exception",
            "http_status_code": status,
            "api_error_code": _token(diagnostic.get("api_error_code"), secrets),
            "request_id": _token(diagnostic.get("request_id"), secrets, request_id=True),
            "stage": stage if stage in STAGES else "unknown",
            "category": category if category in CATEGORIES else "unknown"}
    if "phase" in diagnostic or "origin" in diagnostic:
        result["phase"] = diagnostic.get("phase") if diagnostic.get("phase") in PHASES else "unknown"
        result["origin"] = _clean_origin(diagnostic.get("origin"), secrets)
    return result


def exception_diagnostic(exc, stage, *, secrets=(), category=None, phase=None):
    """Inspect only allowlisted scalar attributes and exception chains, never str(exc).

    Suppressed contexts (raise ... from None) are still inspectable in memory; they
    cannot be reconstructed from old artifacts after the original process exits.
    """
    chain, seen = [], set()
    current = exc
    while isinstance(current, BaseException) and id(current) not in seen and len(chain) < 12:
        seen.add(id(current))
        attached = getattr(current, "_ci_safe_diagnostic", None)
        if isinstance(attached, dict):
            return _clean(attached, stage, secrets)
        chain.append(current)
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    original = next((error for error in chain if isinstance(error, (APIError, ModelBehaviorError, ValidationError))),
                    chain[-1] if chain else exc)
    status = getattr(original, "status_code", None)
    code = _token(getattr(original, "code", None), secrets)
    request = _token(getattr(original, "request_id", None), secrets, request_id=True)
    # No response/request/header/body reads. SDK scalar code/request_id attributes
    # are sufficient when available; unsupported/malformed values remain unknown.
    if category is None:
        if code in {"insufficient_quota", "billing_hard_limit_reached", "usage_limit_reached", "exceeded_quota", "billing_not_active"}:
            category = "quota"
        elif code in {"invalid_json_schema", "invalid_function_parameters", "invalid_response_format", "invalid_schema"}:
            category = "schema_validation"
        elif isinstance(original, AuthenticationError) or status == 401 or code == "invalid_api_key":
            category = "authentication"
        elif isinstance(original, RateLimitError) or status == 429:
            category = "rate_limit"
        elif code in {"model_not_found", "model_access_denied", "model_permission_denied"} or status == 403:
            category = "model_access"
        elif isinstance(original, (APIConnectionError, httpx.TransportError, ConnectionError, TimeoutError)) or status == 408:
            category = "network"
        elif isinstance(original, (ModelBehaviorError, ValidationError)) or type(exc).__name__.endswith("OutputError"):
            category = "schema_validation"
        elif status in {400, 404, 409, 422}:
            category = "invalid_request"
        elif isinstance(status, int) and status >= 500:
            category = "server_error"
        else:
            category = "unknown"
    diagnostic = {"original_exception_class": _name(original, secrets), "http_status_code": status,
                  "api_error_code": code, "request_id": request, "category": category}
    if _local_enabled():
        diagnostic.update(phase=phase, origin=_origin(original, secrets))
    return _clean(diagnostic, stage, secrets)


def log_exception_diagnostic(exc, stage, *, secrets=(), category=None, phase=None):
    diagnostic = exception_diagnostic(exc, stage, secrets=secrets, category=category, phase=phase)
    # Never pass exception objects, exc_info, stack_info, repr, messages, or headers.
    logger.warning("Stage failure diagnostic: %s", json.dumps(diagnostic, sort_keys=True))
    return diagnostic
