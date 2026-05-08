"""Shared requests.Session, auth resolution, redirect hooks."""

from __future__ import annotations

import os

import requests

from oci_modelcar import __version__


def _envbool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def build_session() -> requests.Session:
    """Single source of truth for HTTP sessions across the codebase."""
    s = requests.Session()
    s.headers["User-Agent"] = (
        os.environ.get("OCI_MODELCAR_USER_AGENT") or f"oci-modelcar/{__version__}"
    )
    if _envbool("OCI_MODELCAR_FORCE_CONNECTION_CLOSE"):
        s.headers["Connection"] = "close"
    return s
