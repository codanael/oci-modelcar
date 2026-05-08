import ssl

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
