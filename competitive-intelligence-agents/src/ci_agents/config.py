"""Explicit configuration loading; importing this module has no side effects."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import os

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUTOFF = "2026-11-11T23:59:59Z"
# Avoid negotiating Brotli: installed httpx2 requires the optional Brotli >=1.2
# API, while independently installed older backends can still be auto-detected.
OPENAI_ACCEPT_ENCODING = "gzip, deflate"


class ConfigurationError(ValueError):
    """Configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(repr=False)
    openai_model: str
    data_dir: Path
    runs_dir: Path
    evaluation_cutoff: datetime
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None


def agent_model_settings(settings, **kwargs):
    """Optional, frozen controls; absent settings preserve production defaults."""
    from agents import ModelSettings
    from openai.types.shared import Reasoning
    if settings.reasoning_effort is not None:
        kwargs['reasoning'] = Reasoning(effort=settings.reasoning_effort)
    if settings.max_output_tokens is not None:
        kwargs['max_tokens'] = settings.max_output_tokens
    from .workflow_runtime import active_runtime
    runtime = active_runtime()
    if runtime is not None:
        kwargs['max_tokens'] = min(kwargs.get('max_tokens', runtime.policy.max_output_tokens), runtime.policy.max_output_tokens)
    return ModelSettings(**kwargs)


_runtime_settings: ContextVar[Settings | None] = ContextVar("ci_runtime_settings", default=None)


@contextmanager
def use_settings(settings: Settings):
    """Freeze validated run settings across sequential agent calls without global mutation."""
    token = _runtime_settings.set(settings)
    try:
        yield
    finally:
        _runtime_settings.reset(token)


def load_config(
    *,
    project_root: Path | None = None,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    require_api_key: bool = True,
) -> Settings:
    """Load .env then environment overrides; resolve relative paths at project root.

    An explicit environ mapping replaces the process environment for offline tests.
    No directories, databases, API clients, or network requests are created.
    """
    active = _runtime_settings.get()
    if active is not None and project_root is None and env_file is None and environ is None:
        return active
    root = (project_root or PROJECT_ROOT).expanduser().resolve()
    dotenv_path = Path(env_file) if env_file is not None else root / ".env"
    if not dotenv_path.is_absolute():
        dotenv_path = root / dotenv_path
    values = dict(dotenv_values(dotenv_path, interpolate=False))
    values.update(os.environ if environ is None else environ)

    def required(name: str) -> str:
        value = values.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(
                f"{name} is required. Set it in the environment or {dotenv_path}."
            )
        return value.strip()

    key = required("OPENAI_API_KEY") if require_api_key else (values.get("OPENAI_API_KEY") or "").strip()
    model = required("OPENAI_MODEL")

    def directory(name: str, default: str, *, must_exist: bool) -> Path:
        raw = values.get(name, default)
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigurationError(f"{name} must be a non-empty directory path.")
        path = Path(raw.strip()).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if must_exist and not path.is_dir():
            raise ConfigurationError(f"{name} must point to an existing directory: {path}")
        for candidate in (path, *path.parents):
            if candidate.exists():
                if not candidate.is_dir():
                    raise ConfigurationError(f"{name} path is blocked by a file: {candidate}")
                break
        return path

    data_dir = directory("DATA_DIR", "data/public", must_exist=True)
    runs_dir = directory("RUNS_DIR", "runs", must_exist=False)
    raw_cutoff = values.get("EVALUATION_CUTOFF", DEFAULT_CUTOFF)
    try:
        cutoff = datetime.fromisoformat(raw_cutoff.strip().replace("Z", "+00:00"))
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("timezone required")
    except (AttributeError, TypeError, ValueError):
        raise ConfigurationError(
            "EVALUATION_CUTOFF must be an ISO 8601 timestamp with a timezone, "
            "for example 2026-11-11T23:59:59Z."
        ) from None
    return Settings(key, model, data_dir, runs_dir, cutoff.astimezone(timezone.utc))
