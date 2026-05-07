import pytest

from oci_modelcar.config import Config, ConfigError


def test_config_from_env_minimal(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    cfg = Config.from_env_and_args([])
    assert cfg.hf_repo == "foo/bar"
    assert cfg.hf_revision == "main"
    assert cfg.hf_endpoint == "https://huggingface.co"
    assert cfg.registry == "registry.example.com"
    assert cfg.target_repo == "models/x"
    assert cfg.target_tag is None  # derived later
    assert cfg.workers == 1
    assert cfg.chunk_mib == 8


def test_config_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    monkeypatch.setenv("WORKERS", "4")
    cfg = Config.from_env_and_args(["--workers", "2"])
    assert cfg.workers == 2


def test_config_missing_required_raises(monkeypatch):
    monkeypatch.delenv("HF_REPO", raising=False)
    monkeypatch.delenv("REGISTRY", raising=False)
    monkeypatch.delenv("TARGET_REPO", raising=False)
    with pytest.raises(ConfigError, match="hf_repo"):
        Config.from_env_and_args([])


def test_config_workers_cap(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(ConfigError, match="workers"):
        Config.from_env_and_args(["--workers", "9"])


def test_config_invalid_target_tag(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(ConfigError, match="target_tag"):
        Config.from_env_and_args(["--target-tag", "bad/tag"])


def test_config_workers_zero_raises(monkeypatch):
    """--workers 0 must reach validate() and be rejected, not silently default to 1."""
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(ConfigError, match="workers"):
        Config.from_env_and_args(["--workers", "0"])


def test_config_chunk_mib_zero_raises(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(ConfigError, match="chunk_mib"):
        Config.from_env_and_args(["--chunk-mib", "0"])


def test_verbose_and_quiet_env_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    monkeypatch.setenv("LOG_VERBOSE", "1")
    monkeypatch.setenv("LOG_QUIET", "1")
    with pytest.raises(ConfigError, match="mutually exclusive"):
        Config.from_env_and_args([])


def test_empty_env_string_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    monkeypatch.setenv("HF_REVISION", "")  # empty -> use "main"
    cfg = Config.from_env_and_args([])
    assert cfg.hf_revision == "main"


def test_hf_max_retries_negative_raises(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(ConfigError, match="hf_max_retries"):
        Config.from_env_and_args(["--hf-max-retries", "-1"])
