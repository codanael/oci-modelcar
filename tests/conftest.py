"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest

# CI fixture: force Connection: close on every session created via
# `oci_modelcar.http.build_session()`. Without this, pytest-httpserver +
# werkzeug + ubuntu GitHub runners exhibit intermittent recv_into hangs
# on small HTTP responses (the kernel/socket layer buffers the response
# but the client never observes EOF without an explicit close-per-request
# signal). The env var is a documented diagnostic feature of build_session;
# enabling it for tests makes wire behavior deterministic across versions.
os.environ.setdefault("OCI_MODELCAR_FORCE_CONNECTION_CLOSE", "1")


@pytest.fixture(scope="session")
def httpserver_listen_address() -> tuple[str, int]:
    """Force pytest-httpserver to bind on IPv4 only.

    Default binding is to "localhost", which on dual-stack hosts can
    resolve to ::1 and create a routing race where the server accepts
    on IPv6 but the client connects via IPv4 (or vice-versa) — observed
    as intermittent `recv_into` hangs on GitHub Actions ubuntu runners
    across multiple Python versions. Pinning to 127.0.0.1 sidesteps the
    dual-stack ambiguity entirely.
    """
    return ("127.0.0.1", 0)
