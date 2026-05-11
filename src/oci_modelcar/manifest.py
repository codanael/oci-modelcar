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

# Reverse-DNS scoped annotations for layer reuse across re-pushes.
# A future run can match (hf_path, hf_sha256) against an existing manifest's
# annotations and skip HF download + tar build + push for unchanged files.
ANN_HF_PATH = "io.github.codanael.modelcar.hf-path"
ANN_HF_SHA256 = "io.github.codanael.modelcar.hf-sha256"


@dataclass(frozen=True, slots=True)
class BlobDescriptor:
    media_type: str
    digest: str
    size: int
    hf_path: str
    hf_sha256: str | None = None  # 64-hex when the HF file is LFS-backed

    def to_dict(self) -> dict[str, object]:
        annotations: dict[str, str] = {ANN_HF_PATH: self.hf_path}
        if self.hf_sha256 is not None:
            annotations[ANN_HF_SHA256] = self.hf_sha256
        return {
            "mediaType": self.media_type,
            "digest": self.digest,
            "size": self.size,
            "annotations": annotations,
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
