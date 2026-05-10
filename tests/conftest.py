"""Shared pytest fixtures."""

from __future__ import annotations

import os

# CI fixture: force Connection: close on every session created via
# `oci_modelcar.http.build_session()`. Without this, pytest-httpserver +
# werkzeug + ubuntu GitHub runners exhibit intermittent recv_into hangs
# on small HTTP responses (the kernel/socket layer buffers the response
# but the client never observes EOF without an explicit close-per-request
# signal). The env var is a documented diagnostic feature of build_session;
# enabling it for tests makes wire behavior deterministic across versions.
os.environ.setdefault("OCI_MODELCAR_FORCE_CONNECTION_CLOSE", "1")
