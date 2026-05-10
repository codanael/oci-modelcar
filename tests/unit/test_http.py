import json
import ssl
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
import urllib3.exceptions

from oci_modelcar import __version__
from oci_modelcar.http import _SmartRetry, build_session, is_transient_ssl


def test_session_has_default_user_agent(monkeypatch):
    monkeypatch.delenv("OCI_MODELCAR_USER_AGENT", raising=False)
    s = build_session()
    assert s.headers["User-Agent"] == f"oci-modelcar/{__version__}"


def test_session_user_agent_overridable(monkeypatch):
    monkeypatch.setenv("OCI_MODELCAR_USER_AGENT", "custom/1.0")
    s = build_session()
    assert s.headers["User-Agent"] == "custom/1.0"


def test_session_force_connection_close(monkeypatch):
    monkeypatch.setenv("OCI_MODELCAR_FORCE_CONNECTION_CLOSE", "1")
    s = build_session()
    assert s.headers.get("Connection") == "close"


def test_session_default_no_connection_close(monkeypatch):
    monkeypatch.delenv("OCI_MODELCAR_FORCE_CONNECTION_CLOSE", raising=False)
    s = build_session()
    assert s.headers.get("Connection") != "close"


def test_is_transient_ssl_true_for_eof():
    e = ssl.SSLEOFError("EOF occurred in violation of protocol")
    wrapped = requests.exceptions.SSLError(str(e))
    wrapped.__cause__ = e
    assert is_transient_ssl(wrapped) is True


def test_is_transient_ssl_false_for_handshake():
    e = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
    wrapped = requests.exceptions.SSLError(str(e))
    wrapped.__cause__ = e
    assert is_transient_ssl(wrapped) is False


def test_is_transient_ssl_via_message_match():
    """Some wrappers don't preserve __cause__; fall back to message match."""
    e = requests.exceptions.SSLError("EOF occurred in violation of protocol (_ssl.c:2437)")
    assert is_transient_ssl(e) is True


def test_smart_retry_reraises_ssl():
    r = _SmartRetry(total=5)
    with pytest.raises(ssl.SSLError):
        r.increment(error=ssl.SSLError("CERTIFICATE_VERIFY_FAILED"))


def test_smart_retry_reraises_proxy():
    r = _SmartRetry(total=5)
    with pytest.raises(urllib3.exceptions.ProxyError):
        r.increment(error=urllib3.exceptions.ProxyError("bad proxy", OSError()))


def _build_redirect_pair(original_url: str, redirect_url: str, auth: str | None):
    """Construct (response, prepared_request) inputs to Session.rebuild_auth
    that mirror what `requests` produces while following a redirect.

    `response` represents the prior (3xx) response — its `.request.url` is
    the URL we came from. `prepared_request` is the about-to-be-sent next
    request whose Authorization header we want stripped or preserved."""
    import requests as _requests

    response = MagicMock()
    response.request = MagicMock(url=original_url)
    prepared = _requests.PreparedRequest()
    prepared.prepare(method="GET", url=redirect_url)
    if auth is not None:
        prepared.headers["Authorization"] = auth
    return response, prepared


def test_authorization_dropped_on_cross_origin_redirect():
    """`_SafeSession.rebuild_auth` strips Authorization when the redirect
    target's netloc differs from the original. Direct unit test on the
    method — bypasses pytest-httpserver entirely, which has known
    instability on small responses against the GitHub runner."""
    from oci_modelcar.http import _SafeSession

    s = _SafeSession()
    response, prepared = _build_redirect_pair(
        original_url="https://huggingface.co/repo/file",
        redirect_url="https://cdn-lfs.huggingface.co/repo/file",
        auth="Bearer hf_secret",
    )
    s.rebuild_auth(prepared, response)
    assert "Authorization" not in prepared.headers


def test_authorization_preserved_on_same_origin_redirect():
    """Same-netloc redirects preserve Authorization."""
    from oci_modelcar.http import _SafeSession

    s = _SafeSession()
    response, prepared = _build_redirect_pair(
        original_url="https://registry.example.com/v2/models/x/manifests/v1",
        redirect_url="https://registry.example.com/v2/models/x/blobs/sha256:abc",
        auth="Bearer hf_secret",
    )
    s.rebuild_auth(prepared, response)
    assert prepared.headers.get("Authorization") == "Bearer hf_secret"


def test_authorization_dropped_on_port_change():
    """Different ports on the same host → different netloc → strip."""
    from oci_modelcar.http import _SafeSession

    s = _SafeSession()
    response, prepared = _build_redirect_pair(
        original_url="https://example.com:443/a",
        redirect_url="https://example.com:8443/b",
        auth="Bearer hf_secret",
    )
    s.rebuild_auth(prepared, response)
    assert "Authorization" not in prepared.headers


def test_huggingface_token_from_hf_token_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok_a")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    from oci_modelcar.http import huggingface_token

    assert huggingface_token() == "tok_a"


def test_huggingface_token_from_hub_token_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok_b")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    from oci_modelcar.http import huggingface_token

    assert huggingface_token() == "tok_b"


def test_huggingface_token_priority(monkeypatch):
    """HF_TOKEN wins over HUGGING_FACE_HUB_TOKEN."""
    monkeypatch.setenv("HF_TOKEN", "tok_a")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok_b")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    from oci_modelcar.http import huggingface_token

    assert huggingface_token() == "tok_a"


def test_huggingface_token_disabled_by_env(monkeypatch):
    """HF_HUB_DISABLE_IMPLICIT_TOKEN=1 returns None even if a token is set."""
    monkeypatch.setenv("HF_TOKEN", "tok_a")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    from oci_modelcar.http import huggingface_token

    assert huggingface_token() is None


def test_huggingface_token_from_cache_file(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = tmp_path / ".cache" / "huggingface" / "token"
    cache.parent.mkdir(parents=True)
    cache.write_text("tok_from_file\n")
    from oci_modelcar.http import huggingface_token

    assert huggingface_token() == "tok_from_file"


def test_oci_auth_header_from_env(monkeypatch):
    monkeypatch.setenv("OCI_USERNAME", "alice")
    monkeypatch.setenv("OCI_PASSWORD", "s3cret")
    from oci_modelcar.http import oci_auth_header

    h = oci_auth_header("registry.example.com")
    assert h["Authorization"].startswith("Basic ")


def test_oci_auth_header_from_docker_config(monkeypatch, tmp_path):
    import base64

    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    cfg = tmp_path / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True)
    auth = base64.b64encode(b"alice:s3cret").decode()
    cfg.write_text(json.dumps({"auths": {"registry.example.com": {"auth": auth}}}))
    from oci_modelcar.http import oci_auth_header

    h = oci_auth_header("registry.example.com")
    assert h["Authorization"] == f"Basic {auth}"


def test_oci_auth_anonymous_when_no_credentials(monkeypatch, tmp_path, caplog):
    import logging

    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    from oci_modelcar.http import oci_auth_header

    with caplog.at_level(logging.WARNING):
        h = oci_auth_header("registry.example.com")
    assert h == {}
    assert any("anonymously" in r.message for r in caplog.records)
