"""OCI image manifest + config builder.

Reproducible: same inputs always yield the same bytes (no `created` field).
"""

from __future__ import annotations

import json

from oci_modelcar.oci import ML_CFG, ML_MAN, BlobDescriptor


def build_config_bytes(diff_ids: list[str]) -> bytes:
    """Minimal OCI image config (compliant with image-spec v1.1).

    Required: architecture, os, rootfs.type, rootfs.diff_ids.
    Optional fields omitted on purpose (deterministic across runs).
    """
    cfg = {
        "architecture": "amd64",
        "os": "linux",
        "rootfs": {"type": "layers", "diff_ids": list(diff_ids)},
        "config": {},
    }
    return json.dumps(cfg, separators=(",", ":"), sort_keys=True).encode()


def build_manifest_bytes(layers: list[BlobDescriptor], config_descriptor: BlobDescriptor) -> bytes:
    manifest = {
        "schemaVersion": 2,
        "mediaType": ML_MAN,
        "config": {
            "mediaType": ML_CFG,
            "digest": config_descriptor.digest,
            "size": config_descriptor.size,
        },
        "layers": [layer.to_dict() for layer in layers],
    }
    return json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
