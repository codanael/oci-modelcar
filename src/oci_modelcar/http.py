"""Shared requests.Session, auth resolution, redirect hooks."""

from __future__ import annotations

import os
import ssl
from typing import Any
from urllib.parse import urlparse

import requests
import urllib3.exceptions
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oci_modelcar import __version__


def _envbool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


_NEVER_RETRY_EXC: tuple[type[Exception], ...] = (
    ssl.SSLError,
    urllib3.exceptions.SSLError,
    urllib3.exceptions.ProxyError,
)


def is_transient_ssl(exc: BaseException) -> bool:
    """True if `exc` is a mid-stream SSL EOF (recoverable via Range/replay).

    Walks both __cause__ and __context__ to find a wrapped SSLEOFError.
    Falls back to message match because some wrappers don't preserve the
    chain (urllib3 → requests roundtrip).
    """
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ssl.SSLEOFError):
            return True
        if "EOF occurred in violation of protocol" in str(cur):
            return True
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    return False


class _SmartRetry(Retry):
    """urllib3 Retry that re-raises SSL/Proxy errors instead of looping on them."""

    def increment(  # type: ignore[override]
        self,
        method: str | None = None,
        url: str | None = None,
        response: Any = None,
        error: Exception | None = None,
        _pool: Any = None,
        _stacktrace: Any = None,
    ) -> Retry:
        if error is not None and isinstance(error, _NEVER_RETRY_EXC):
            raise error
        return super().increment(method, url, response, error, _pool, _stacktrace)


class _SafeSession(requests.Session):
    """Session that strips Authorization on cross-origin redirects regardless
    of whether the header was set per-request or via session.headers.

    requests >=2.32 already strips session-level auth; this also handles
    per-request Authorization headers, which is the common case for
    HuggingFace LFS file pulls (HF redirects to signed S3/CloudFront URLs
    that we must not give the Bearer token to)."""

    def rebuild_auth(
        self,
        prepared_request: requests.PreparedRequest,
        response: requests.Response,
    ) -> None:
        super().rebuild_auth(prepared_request, response)  # type: ignore[no-untyped-call]
        if "Authorization" not in prepared_request.headers:
            return
        original_url = response.request.url
        if original_url is None:
            return
        original_netloc = urlparse(original_url).netloc
        new_netloc = urlparse(prepared_request.url or "").netloc
        if new_netloc and new_netloc != original_netloc:
            del prepared_request.headers["Authorization"]


def build_session() -> requests.Session:
    """Single source of truth for HTTP sessions across the codebase."""
    s = _SafeSession()
    retry = _SmartRetry(
        total=8,
        backoff_factor=2,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers["User-Agent"] = (
        os.environ.get("OCI_MODELCAR_USER_AGENT") or f"oci-modelcar/{__version__}"
    )
    if _envbool("OCI_MODELCAR_FORCE_CONNECTION_CLOSE"):
        s.headers["Connection"] = "close"
    return s
