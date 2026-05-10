"""Shared requests.Session, auth resolution, redirect hooks."""

from __future__ import annotations

import base64
import json
import logging
import os
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import urllib3.exceptions
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oci_modelcar import __version__

log = logging.getLogger(__name__)


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


def huggingface_token() -> str | None:
    """Resolve HF token. Priority: HF_TOKEN > HUGGING_FACE_HUB_TOKEN > cache file.
    Returns None if HF_HUB_DISABLE_IMPLICIT_TOKEN is set."""
    if _envbool("HF_HUB_DISABLE_IMPLICIT_TOKEN"):
        return None
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = os.environ.get(env_name)
        if tok:
            return tok
    cache = Path.home() / ".cache" / "huggingface" / "token"
    if cache.is_file():
        try:
            content = cache.read_text().strip()
            return content or None
        except OSError:
            return None
    return None


def huggingface_auth_header() -> dict[str, str]:
    tok = huggingface_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def oci_auth_header(registry_host: str, target_repo: str | None = None) -> dict[str, str]:
    """Resolve OCI registry auth: env > ~/.docker/config.json > podman auth.json."""
    target = f"{registry_host}/{target_repo}" if target_repo else registry_host

    user = os.environ.get("OCI_USERNAME")
    pwd = os.environ.get("OCI_PASSWORD")
    if user and pwd is not None:
        log.info("OCI auth resolved from OCI_USERNAME/OCI_PASSWORD env")
        token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    for path in _auth_search_paths():
        auth = _docker_config_auth(path, target)
        if auth:
            log.info("OCI auth resolved from %s", path)
            return {"Authorization": f"Basic {auth}"}

    log.warning(
        "no OCI credentials found for %s — pushing anonymously "
        "(set OCI_USERNAME/OCI_PASSWORD or run `podman login`/`docker login`)",
        target,
    )
    return {}


def _auth_search_paths() -> list[Path]:
    paths = [Path.home() / ".docker" / "config.json"]
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        paths.append(Path(xdg_runtime) / "containers" / "auth.json")
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    config_root = Path(xdg_config) if xdg_config else Path.home() / ".config"
    paths.append(config_root / "containers" / "auth.json")
    return paths


def _normalize_auth_key(key: str) -> str:
    for prefix in ("https://", "http://"):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    key = key.rstrip("/")
    if key.endswith("/v2"):
        key = key[: -len("/v2")].rstrip("/")
    return key


def _docker_config_auth(path: Path, target: str) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    auths = data.get("auths", {})
    if not isinstance(auths, dict):
        return None
    best_key, best_len = None, -1
    for raw_key in auths:
        norm = _normalize_auth_key(raw_key)
        if (norm == target or target.startswith(norm + "/")) and len(norm) > best_len:
            best_key, best_len = raw_key, len(norm)
    if best_key is None:
        return None
    entry = auths[best_key]
    if not isinstance(entry, dict):
        return None
    raw = entry.get("auth")
    return raw if isinstance(raw, str) and raw else None


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
