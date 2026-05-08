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


def test_authorization_dropped_on_cross_origin_redirect(httpserver):
    """When HF redirects to S3 or any other host, the Bearer token MUST be
    stripped before the second GET.

    We simulate cross-origin by starting a second HTTPServer bound to
    127.0.0.1 while the primary httpserver fixture uses localhost — the two
    netlocs differ even on the same machine, which is what triggers the auth
    stripping logic.
    """
    from pytest_httpserver import HTTPServer
    from werkzeug.wrappers import Response

    seen_auth_on_s3: list[str | None] = []

    s3_server = HTTPServer(host="127.0.0.1", port=0)
    s3_server.start()
    try:

        def s3_handler(request):
            seen_auth_on_s3.append(request.headers.get("Authorization"))
            return Response(b"data", status=200)

        s3_server.expect_request("/s3-mock/file").respond_with_handler(s3_handler)

        def origin_handler(request):
            return Response(
                "",
                status=302,
                headers={"Location": s3_server.url_for("/s3-mock/file")},
            )

        httpserver.expect_request("/api/redirect-me").respond_with_handler(origin_handler)

        s = build_session()
        r = s.get(
            httpserver.url_for("/api/redirect-me"),
            headers={"Authorization": "Bearer hf_secret"},
        )
        r.raise_for_status()
        assert seen_auth_on_s3 == [None], (
            f"Authorization must be stripped on cross-origin redirect; got {seen_auth_on_s3!r}"
        )
    finally:
        s3_server.clear()
        if s3_server.is_running():
            s3_server.stop()


def test_authorization_preserved_on_same_origin_redirect(httpserver):
    """Same-host redirects should keep the Bearer token."""
    from werkzeug.wrappers import Response

    seen_auth_on_target: list[str | None] = []

    httpserver.expect_request("/redirect").respond_with_data(
        "", status=302, headers={"Location": httpserver.url_for("/target")}
    )

    def target_handler(request):
        seen_auth_on_target.append(request.headers.get("Authorization"))
        return Response("", status=200)

    httpserver.expect_request("/target").respond_with_handler(target_handler)

    s = build_session()
    r = s.get(httpserver.url_for("/redirect"), headers={"Authorization": "Bearer hf_secret"})
    r.raise_for_status()
    assert seen_auth_on_target == ["Bearer hf_secret"]
