from oci_modelcar import __version__
from oci_modelcar.http import build_session


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
