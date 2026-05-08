import base64
import json
import ssl

import pytest
import urllib3.exceptions

from oci_modelcar.http import (
    _SmartRetry,
    build_session,
    docker_config_auth,
    huggingface_token,
    oci_auth_header,
)


def test_oci_auth_header_from_env(monkeypatch):
    monkeypatch.setenv("OCI_USERNAME", "alice")
    monkeypatch.setenv("OCI_PASSWORD", "s3cr3t")
    hdr = oci_auth_header("registry.example.com")
    expected = "Basic " + base64.b64encode(b"alice:s3cr3t").decode()
    assert hdr == {"Authorization": expected}


def test_oci_auth_header_from_docker_config(monkeypatch, tmp_path):
    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".docker"
    cfg_dir.mkdir()
    raw = base64.b64encode(b"bob:hunter2").decode()
    (cfg_dir / "config.json").write_text(
        json.dumps({"auths": {"registry.example.com": {"auth": raw}}})
    )
    hdr = oci_auth_header("registry.example.com")
    assert hdr == {"Authorization": f"Basic {raw}"}


def test_oci_auth_header_no_creds(monkeypatch, tmp_path):
    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert oci_auth_header("registry.example.com") == {}


def test_huggingface_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert huggingface_token() == "hf_secret"


def test_huggingface_token_from_cache_file(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "token").write_text("hf_from_cache\n")
    assert huggingface_token() == "hf_from_cache"


def test_huggingface_token_none(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert huggingface_token() is None


def test_build_session_has_user_agent():
    s = build_session()
    assert "oci-modelcar/" in s.headers["User-Agent"]


def test_docker_config_auth_handles_missing(tmp_path):
    assert docker_config_auth(tmp_path / "missing.json", "x") is None


def test_smart_retry_raises_immediately_on_ssl_error():
    """SSL errors must NOT be retried — they never recover from a CA misconfig.
    increment() must surface the original exception so urllib3 stops retrying."""
    retry = _SmartRetry(total=8, backoff_factor=2)
    err = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
    with pytest.raises(ssl.SSLError):
        retry.increment(method="GET", url="https://example/", error=err)


def test_smart_retry_raises_immediately_on_urllib3_ssl_error():
    retry = _SmartRetry(total=8, backoff_factor=2)
    err = urllib3.exceptions.SSLError("handshake failure")
    with pytest.raises(urllib3.exceptions.SSLError):
        retry.increment(method="GET", url="https://example/", error=err)


def test_smart_retry_raises_immediately_on_proxy_error():
    retry = _SmartRetry(total=8, backoff_factor=2)
    err = urllib3.exceptions.ProxyError("bad proxy", OSError("nope"))
    with pytest.raises(urllib3.exceptions.ProxyError):
        retry.increment(method="GET", url="https://example/", error=err)


def test_smart_retry_passes_through_other_errors():
    """Non-fatal transport errors still consume retries normally."""
    retry = _SmartRetry(total=8, backoff_factor=2)
    err = urllib3.exceptions.ProtocolError("connection reset")
    new_retry = retry.increment(method="GET", url="https://example/", error=err)
    assert isinstance(new_retry, _SmartRetry)


def test_build_session_uses_smart_retry():
    """The session adapter must carry the SSL-aware retry policy."""
    s = build_session()
    adapter = s.get_adapter("https://example/")
    assert isinstance(adapter.max_retries, _SmartRetry)
