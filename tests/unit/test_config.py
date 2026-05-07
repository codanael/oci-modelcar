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
