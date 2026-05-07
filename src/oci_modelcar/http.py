"""Shared HTTP session + auth resolution."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oci_modelcar import __version__


def build_session() -> requests.Session:
    """Session with retries on idempotent methods only.

    Non-idempotent methods (PATCH, PUT) are NOT retried automatically;
    those have their own resync-aware retry logic in oci.py.
    """
    s = requests.Session()
    retry = Retry(
        total=8,
        backoff_factor=2,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers["User-Agent"] = f"oci-modelcar/{__version__}"
    return s


def oci_auth_header(registry_host: str) -> dict[str, str]:
    """Resolve registry auth in priority order.

    1. OCI_USERNAME + OCI_PASSWORD env
    2. ~/.docker/config.json
    3. $XDG_RUNTIME_DIR/containers/auth.json
    """
    user = os.environ.get("OCI_USERNAME")
    pwd = os.environ.get("OCI_PASSWORD")
    if user and pwd is not None:
        token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    docker_cfg = Path.home() / ".docker" / "config.json"
    auth = docker_config_auth(docker_cfg, registry_host)
    if auth:
        return {"Authorization": f"Basic {auth}"}

    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        containers_cfg = Path(xdg) / "containers" / "auth.json"
        auth = docker_config_auth(containers_cfg, registry_host)
        if auth:
            return {"Authorization": f"Basic {auth}"}

    return {}


def docker_config_auth(path: Path, registry_host: str) -> str | None:
    """Read a docker/podman config.json and return the base64 auth blob if any."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except OSError:
        return None
    except json.JSONDecodeError:
        return None
    auths = data.get("auths", {})
    entry = auths.get(registry_host)
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
