from datetime import datetime, timezone

import pytest

from ci_agents.config import ConfigurationError, load_config


@pytest.fixture
def configured(tmp_path):
    (tmp_path / "data/public").mkdir(parents=True)
    return tmp_path, {"OPENAI_API_KEY": "test-only-secret", "OPENAI_MODEL": "test-model"}


def test_defaults_and_secret_redaction(configured):
    root, env = configured
    settings = load_config(project_root=root, environ=env)
    assert settings.data_dir == root / "data/public"
    assert settings.runs_dir == root / "runs"
    assert settings.evaluation_cutoff == datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)
    assert "test-only-secret" not in repr(settings)
    assert not settings.runs_dir.exists()


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "OPENAI_MODEL"])
def test_missing_required_configuration(configured, name):
    root, env = configured
    env.pop(name)
    with pytest.raises(ConfigurationError, match=name):
        load_config(project_root=root, environ=env)


@pytest.mark.parametrize("cutoff", ["invalid", "2026-11-11T23:59:59", ""])
def test_invalid_cutoff(configured, cutoff):
    root, env = configured
    env["EVALUATION_CUTOFF"] = cutoff
    with pytest.raises(ConfigurationError, match="EVALUATION_CUTOFF"):
        load_config(project_root=root, environ=env)


def test_dotenv_overrides_paths_and_timezone(configured):
    root, env = configured
    (root / ".env").write_text(
        "OPENAI_MODEL=file-model\nDATA_DIR=data/public\nRUNS_DIR=artifacts\n"
        "EVALUATION_CUTOFF=2026-11-11T18:59:59-05:00\n", encoding="utf-8"
    )
    settings = load_config(project_root=root, environ=env)
    assert settings.openai_model == "test-model"
    assert settings.runs_dir == root / "artifacts"
    assert settings.evaluation_cutoff.hour == 23


@pytest.mark.parametrize("name,value", [("DATA_DIR", "missing"), ("DATA_DIR", ""), ("RUNS_DIR", "blocked/child")])
def test_invalid_paths(configured, name, value):
    root, env = configured
    (root / "blocked").write_text("file", encoding="utf-8")
    env[name] = value
    with pytest.raises(ConfigurationError, match=name):
        load_config(project_root=root, environ=env)
