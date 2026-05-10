"""OCI image config + manifest building. Reproducible: no `created` field,
deterministic JSON serialization, layers ordered by caller."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

ML_TAR = "application/vnd.oci.image.layer.v1.tar"
ML_CFG = "application/vnd.oci.image.config.v1+json"
ML_MAN = "application/vnd.oci.image.manifest.v1+json"


@dataclass(frozen=True, slots=True)
class BlobDescriptor:
    media_type: str
    digest: str
    size: int
    hf_path: str  # not serialized; used by runner for ordering and logging

    def to_dict(self) -> dict[str, object]:
        return {
            "mediaType": self.media_type,
            "digest": self.digest,
            "size": self.size,
        }


def build_config_bytes(diff_ids: list[str]) -> bytes:
    """OCI image config without `created` (so config digest is stable)."""
    cfg = {
        "architecture": "amd64",
        "os": "linux",
        "rootfs": {
            "type": "layers",
            "diff_ids": diff_ids,
        },
        "config": {},
    }
    return json.dumps(cfg, separators=(",", ":"), sort_keys=True).encode()


def build_manifest_bytes(
    config_digest: str, config_size: int, layers: Iterable[BlobDescriptor]
) -> bytes:
    """Build manifest. Caller is responsible for ordering `layers`
    deterministically (e.g. sorted by hf_path)."""
    layer_list = list(layers)
    manifest = {
        "schemaVersion": 2,
        "mediaType": ML_MAN,
        "config": {
            "mediaType": ML_CFG,
            "digest": config_digest,
            "size": config_size,
        },
        "layers": [d.to_dict() for d in layer_list],
    }
    return json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def derive_tag(revision_resolved: str, explicit: str | None) -> str:
    """Derive the OCI tag from a resolved HF revision.

    - If `explicit` is given, return it as-is (Config validates the format).
    - If `revision_resolved` is a 40-char SHA, take the first 12 chars
      (matches `git rev-parse --short=12`).
    - Otherwise sanitize: lowercase, replace [/] with -, strip trailing -.
    """
    if explicit is not None:
        return explicit
    if _SHA_RE.match(revision_resolved):
        return revision_resolved[:12]
    out = revision_resolved.lower().replace("/", "-").replace(" ", "-")
    return out.rstrip("-")
