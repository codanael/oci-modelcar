"""Shared HTTP session + auth resolution."""

from __future__ import annotations

import base64
import json
import logging
import os
import ssl
from pathlib import Path
from typing import Any

import requests
import urllib3.exceptions
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oci_modelcar import __version__

log = logging.getLogger(__name__)

_NEVER_RETRY_EXC: tuple[type[Exception], ...] = (
    ssl.SSLError,
    urllib3.exceptions.SSLError,
    urllib3.exceptions.ProxyError,
)


def is_transient_ssl(exc: BaseException) -> bool:
    """True if `exc` is an SSL error whose root cause is a mid-stream EOF
    (the connection got cut after the handshake succeeded), as opposed to a
    handshake-time misconfig (CA invalid, hostname mismatch, expired cert).

    Mid-stream EOFs happen on long transfers when an idle proxy or firewall
    rotates or times out the TCP connection — fully recoverable via Range
    resume / OCI session resync, so they must be retried like any other
    transient cut. Public to the package: consumed by `HfStream._next_chunk`
    and `ChunkedBlobUpload._patch_with_retry` to override the default
    fatal-on-SSLError verdict for this specific case.

    Walks both `__cause__` (explicit `raise ... from ...`) and `__context__`
    (implicit re-raise inside an except block, the common urllib3/requests
    wrapping pattern) to find the underlying `ssl.SSLEOFError`. Falls back
    to a string match on the canonical SSL error message because some
    wrappers re-raise without preserving the chain.
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
    """Retry policy that surfaces non-recoverable errors immediately.

    SSL handshake failures and proxy misconfig don't get better with retry —
    silently looping on them just hides the real issue from the user. Any
    error in `_NEVER_RETRY_EXC` re-raises out of `increment()` so urllib3
    stops dead instead of burning the full backoff schedule.
    """

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


def build_session() -> requests.Session:
    """Session with retries on idempotent methods only.

    Non-idempotent methods (PATCH, PUT) are NOT retried automatically;
    those have their own resync-aware retry logic in oci.py.

    Two diagnostic env vars (opt-in, defaults unchanged) help isolate
    proxy/AV behavior that treats specific clients differently:

    - ``OCI_MODELCAR_USER_AGENT``: override the default User-Agent. Useful
      when a proxy whitelists wget but mangles ``python-requests``.
    - ``OCI_MODELCAR_FORCE_CONNECTION_CLOSE=1``: send ``Connection: close``
      on every request, disabling keep-alive. Useful when a proxy
      mishandles long-lived TLS connections (mid-stream EOF after AV
      pass-through threshold, idle eviction, etc.).
    """
    s = requests.Session()
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
    if os.environ.get("OCI_MODELCAR_FORCE_CONNECTION_CLOSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        s.headers["Connection"] = "close"
    return s


def oci_auth_header(
    registry_host: str,
    target_repo: str | None = None,
) -> dict[str, str]:
    """Resolve registry auth in priority order.

    1. OCI_USERNAME + OCI_PASSWORD env
    2. ~/.docker/config.json
    3. $XDG_RUNTIME_DIR/containers/auth.json (rootless podman)
    4. $XDG_CONFIG_HOME/containers/auth.json (default $HOME/.config/...)

    When `target_repo` is provided, sources are queried with the full
    `host/repo` path so that auths keyed at a sub-path (e.g.
    `artifactory.example/myproject`) win via longest-prefix match.

    Logs a one-line INFO marker for the resolved source, or a WARNING when
    no source matches and the push falls back to anonymous.
    """
    target = f"{registry_host}/{target_repo}" if target_repo else registry_host

    user = os.environ.get("OCI_USERNAME")
    pwd = os.environ.get("OCI_PASSWORD")
    if user and pwd is not None:
        log.info("OCI auth resolved from OCI_USERNAME/OCI_PASSWORD env")
        token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    for path in _auth_search_paths():
        auth = docker_config_auth(path, target)
        if auth:
            log.info("OCI auth resolved from %s", path)
            return {"Authorization": f"Basic {auth}"}

    log.warning(
        "no OCI credentials found for %s — pushing anonymously (set OCI_USERNAME/OCI_PASSWORD "
        "or run `podman login`/`docker login`)",
        target,
    )
    return {}


def _auth_search_paths() -> list[Path]:
    """Ordered list of auth.json paths to consult (most specific first)."""
    paths = [Path.home() / ".docker" / "config.json"]
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        paths.append(Path(xdg_runtime) / "containers" / "auth.json")
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    config_root = Path(xdg_config) if xdg_config else Path.home() / ".config"
    paths.append(config_root / "containers" / "auth.json")
    return paths


def _normalize_auth_key(key: str) -> str:
    """Strip http(s):// prefix, /v2/ trailing path, and trailing slashes."""
    for prefix in ("https://", "http://"):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    key = key.rstrip("/")
    if key.endswith("/v2"):
        key = key[: -len("/v2")].rstrip("/")
    return key


def docker_config_auth(path: Path, target: str) -> str | None:
    """Read a docker/podman config.json and return the base64 auth blob if any.

    `target` may be a bare host or `host/repo`. Auths keys are normalized
    (`https://`, `/v2/`, trailing `/` stripped) and the longest normalized
    key that is a prefix of `target` wins.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except OSError:
        return None
    except json.JSONDecodeError:
        return None
    auths = data.get("auths", {})
    if not isinstance(auths, dict):
        return None

    best_key: str | None = None
    best_len = -1
    for raw_key in auths:
        norm = _normalize_auth_key(raw_key)
        matches = norm == target or target.startswith(norm + "/")
        if matches and len(norm) > best_len:
            best_len = len(norm)
            best_key = raw_key
    if best_key is None:
        return None
    entry = auths[best_key]
    if not isinstance(entry, dict):
        return None
    raw = entry.get("auth")
    return raw if isinstance(raw, str) and raw else None


def huggingface_token() -> str | None:
    """Resolve HF token: HF_TOKEN env > ~/.cache/huggingface/token."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    cache = Path.home() / ".cache" / "huggingface" / "token"
    if cache.is_file():
        try:
            return cache.read_text().strip() or None
        except OSError:
            return None
    return None


def huggingface_auth_header() -> dict[str, str]:
    tok = huggingface_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}
