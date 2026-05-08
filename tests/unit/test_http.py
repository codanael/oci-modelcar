import base64
import json
import ssl

import pytest
import requests
import urllib3.exceptions

from oci_modelcar.http import (
    _SmartRetry,
    build_session,
    docker_config_auth,
    huggingface_token,
    is_transient_ssl,
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


def test_build_session_user_agent_override_via_env(monkeypatch):
    """OCI_MODELCAR_USER_AGENT lets users mimic wget/curl when a proxy/AV
    treats the python-requests UA differently. Diagnostic-only knob."""
    monkeypatch.setenv("OCI_MODELCAR_USER_AGENT", "Wget/1.21.4")
    s = build_session()
    assert s.headers["User-Agent"] == "Wget/1.21.4"


def test_build_session_no_connection_close_by_default(monkeypatch):
    """requests.Session ships Connection: keep-alive by default; we leave that
    untouched unless explicitly opted in."""
    monkeypatch.delenv("OCI_MODELCAR_FORCE_CONNECTION_CLOSE", raising=False)
    s = build_session()
    assert s.headers.get("Connection", "").lower() != "close"


def test_build_session_force_connection_close_via_env(monkeypatch):
    """OCI_MODELCAR_FORCE_CONNECTION_CLOSE=1 disables HTTP keep-alive on every
    request. Diagnostic-only: useful when a proxy mishandles long-lived TLS
    connections (mid-stream EOF after AV pass-through, etc.)."""
    monkeypatch.setenv("OCI_MODELCAR_FORCE_CONNECTION_CLOSE", "1")
    s = build_session()
    assert s.headers["Connection"] == "close"


def test_build_session_force_connection_close_falsy_values_ignored(monkeypatch):
    monkeypatch.setenv("OCI_MODELCAR_FORCE_CONNECTION_CLOSE", "0")
    s = build_session()
    assert s.headers.get("Connection", "").lower() != "close"


def test_build_session_debug_http_via_env(monkeypatch):
    """OCI_MODELCAR_DEBUG_HTTP=1 turns on urllib3 + http.client wire-level
    debug logging — invaluable when diagnosing proxy/AV behavior in an
    airgapped environment where you can't easily run tcpdump."""
    import http.client
    import logging as _logging

    import oci_modelcar.http as _http_mod

    monkeypatch.setenv("OCI_MODELCAR_DEBUG_HTTP", "1")
    monkeypatch.setattr(_http_mod, "_HTTP_DEBUG_ENABLED", False, raising=False)
    original_debuglevel = http.client.HTTPConnection.debuglevel
    original_urllib3_level = _logging.getLogger("urllib3").level
    try:
        build_session()
        assert http.client.HTTPConnection.debuglevel == 1
        assert _logging.getLogger("urllib3").level == _logging.DEBUG
    finally:
        http.client.HTTPConnection.debuglevel = original_debuglevel
        _logging.getLogger("urllib3").setLevel(original_urllib3_level)


def test_build_session_debug_http_disabled_by_default(monkeypatch):
    """No env var: leave http.client and urllib3 untouched."""
    import http.client

    import oci_modelcar.http as _http_mod

    monkeypatch.delenv("OCI_MODELCAR_DEBUG_HTTP", raising=False)
    monkeypatch.setattr(_http_mod, "_HTTP_DEBUG_ENABLED", False, raising=False)
    original = http.client.HTTPConnection.debuglevel
    http.client.HTTPConnection.debuglevel = 0
    try:
        build_session()
        assert http.client.HTTPConnection.debuglevel == 0
    finally:
        http.client.HTTPConnection.debuglevel = original


def test_docker_config_auth_handles_missing(tmp_path):
    assert docker_config_auth(tmp_path / "missing.json", "x") is None


def test_docker_config_auth_longest_prefix_match(tmp_path):
    """auths['host/repo'] matches target 'host/repo' AND deeper paths."""
    cfg = tmp_path / "auth.json"
    raw = base64.b64encode(b"u:p").decode()
    cfg.write_text(json.dumps({"auths": {"artifactory.example/repo": {"auth": raw}}}))
    # Exact match
    assert docker_config_auth(cfg, "artifactory.example/repo") == raw
    # Sub-path: target = host/repo/something — must match the host/repo entry
    assert docker_config_auth(cfg, "artifactory.example/repo/sub") == raw
    # Bare host: target = host alone — must NOT match a more specific entry
    assert docker_config_auth(cfg, "artifactory.example") is None
    # Different repo prefix: must not cross-match
    assert docker_config_auth(cfg, "artifactory.example/other") is None


def test_docker_config_auth_picks_most_specific_match(tmp_path):
    """When several keys match, the longest wins."""
    cfg = tmp_path / "auth.json"
    bare = base64.b64encode(b"bare:x").decode()
    deep = base64.b64encode(b"deep:y").decode()
    cfg.write_text(
        json.dumps(
            {
                "auths": {
                    "artifactory.example": {"auth": bare},
                    "artifactory.example/repo": {"auth": deep},
                }
            }
        )
    )
    assert docker_config_auth(cfg, "artifactory.example/repo") == deep
    assert docker_config_auth(cfg, "artifactory.example/other") == bare


def test_docker_config_auth_normalizes_legacy_keys(tmp_path):
    """Legacy docker keys with https:// prefix or /v2/ suffix still match."""
    cfg = tmp_path / "auth.json"
    a = base64.b64encode(b"a:1").decode()
    b = base64.b64encode(b"b:2").decode()
    cfg.write_text(
        json.dumps(
            {
                "auths": {
                    "https://legacy.example/v2/": {"auth": a},
                    "https://other.example/": {"auth": b},
                }
            }
        )
    )
    assert docker_config_auth(cfg, "legacy.example") == a
    assert docker_config_auth(cfg, "other.example/repo") == b


def test_oci_auth_header_uses_target_repo(monkeypatch, tmp_path):
    """oci_auth_header passes target_repo so path-keyed auths match."""
    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    cfg_dir = tmp_path / ".docker"
    cfg_dir.mkdir()
    raw = base64.b64encode(b"who:cares").decode()
    (cfg_dir / "config.json").write_text(
        json.dumps({"auths": {"artifactory.example/myproject": {"auth": raw}}})
    )
    # Without target_repo: bare host has no entry → no auth
    assert oci_auth_header("artifactory.example") == {}
    # With target_repo: longest-prefix match resolves
    hdr = oci_auth_header("artifactory.example", target_repo="myproject/model")
    assert hdr == {"Authorization": f"Basic {raw}"}


def test_oci_auth_header_searches_xdg_config_home(monkeypatch, tmp_path):
    """Podman default $HOME/.config/containers/auth.json must be searched too."""
    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    cfg_dir = tmp_path / ".config" / "containers"
    cfg_dir.mkdir(parents=True)
    raw = base64.b64encode(b"pod:man").decode()
    (cfg_dir / "auth.json").write_text(
        json.dumps({"auths": {"registry.example.com": {"auth": raw}}})
    )
    assert oci_auth_header("registry.example.com") == {"Authorization": f"Basic {raw}"}


def test_oci_auth_header_logs_anonymous_when_no_creds(monkeypatch, tmp_path, caplog):
    """When no source matches, a WARNING surfaces — silent anonymous fallback hides 401s."""
    import logging

    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    with caplog.at_level(logging.WARNING, logger="oci_modelcar.http"):
        assert oci_auth_header("registry.example.com") == {}
    assert any(
        "anonymous" in r.getMessage().lower() or "no oci" in r.getMessage().lower()
        for r in caplog.records
    ), f"expected anonymous-fallback warning, got {[r.getMessage() for r in caplog.records]}"


def test_oci_auth_header_logs_resolution_source(monkeypatch, tmp_path, caplog):
    import logging

    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    cfg_dir = tmp_path / ".docker"
    cfg_dir.mkdir()
    raw = base64.b64encode(b"x:y").decode()
    (cfg_dir / "config.json").write_text(
        json.dumps({"auths": {"registry.example.com": {"auth": raw}}})
    )
    with caplog.at_level(logging.INFO, logger="oci_modelcar.http"):
        oci_auth_header("registry.example.com")
    assert any("config.json" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]


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


def test_is_transient_ssl_message_string():
    """The canonical OpenSSL message is enough on its own (no chain required)."""
    e = requests.exceptions.SSLError("EOF occurred in violation of protocol (_ssl.c:2437)")
    assert is_transient_ssl(e) is True


def test_is_transient_ssl_via_explicit_cause():
    """`raise SSLError(...) from ssl.SSLEOFError(...)` exposes the inner type
    via __cause__; isinstance walk must catch it even when the outer message
    doesn't carry the marker string."""
    inner = ssl.SSLEOFError("inner eof")
    outer: requests.exceptions.SSLError
    try:
        raise requests.exceptions.SSLError("connection lost") from inner
    except requests.exceptions.SSLError as e:
        outer = e
    assert is_transient_ssl(outer) is True


def test_is_transient_ssl_via_implicit_context():
    """Common urllib3/requests pattern: raise inside an except without `from`.
    __cause__ stays None but __context__ holds the original SSLEOFError."""
    outer: requests.exceptions.SSLError
    try:
        try:
            raise ssl.SSLEOFError("inner eof")
        except ssl.SSLEOFError:
            raise requests.exceptions.SSLError("wrapped, no from clause")  # noqa: B904
    except requests.exceptions.SSLError as e:
        outer = e
    assert is_transient_ssl(outer) is True


def test_is_transient_ssl_handshake_error_is_not_transient():
    """A pure CA/cert SSLError stays fatal."""
    e = requests.exceptions.SSLError("CERTIFICATE_VERIFY_FAILED")
    assert is_transient_ssl(e) is False


def test_is_transient_ssl_unrelated_exception_is_not_transient():
    e = RuntimeError("not even an SSL error")
    assert is_transient_ssl(e) is False
