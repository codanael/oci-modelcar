"""Tag derivation from HF revision."""

from __future__ import annotations

import re

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_VALID = re.compile(r"[a-zA-Z0-9._-]")


def derive_tag(revision_resolved: str, explicit: str | None) -> str:
    """Compute the OCI image tag.

    - explicit wins
    - 40-char SHA -> first 12 chars
    - else: name sanitized ([^a-zA-Z0-9._-] -> _) and truncated to 128
    """
    if explicit:
        return explicit
    if _FULL_SHA.match(revision_resolved):
        return revision_resolved[:12]
    sanitized = "".join(c if _VALID.match(c) else "_" for c in revision_resolved)
    if not sanitized or sanitized[0] in ".-":
        sanitized = "_" + sanitized
    return sanitized[:128]
