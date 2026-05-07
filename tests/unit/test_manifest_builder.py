import hashlib
import json

from oci_modelcar.manifest import build_config_bytes, build_manifest_bytes
from oci_modelcar.oci import ML_CFG, ML_MAN, ML_TAR, BlobDescriptor


def test_config_is_minimal_no_created():
    diff_ids = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
    cfg = build_config_bytes(diff_ids)
    parsed = json.loads(cfg)
    assert parsed == {
        "architecture": "amd64",
        "os": "linux",
        "rootfs": {"type": "layers", "diff_ids": diff_ids},
        "config": {},
    }
    assert "created" not in parsed
    assert "history" not in parsed


def test_config_is_deterministic():
    diff_ids = ["sha256:" + "a" * 64]
    a = build_config_bytes(diff_ids)
    b = build_config_bytes(diff_ids)
    assert a == b


def test_manifest_schema_and_media_type():
    layers = [BlobDescriptor(media_type=ML_TAR, digest="sha256:" + "a" * 64, size=10)]
    cfg_desc = BlobDescriptor(media_type=ML_CFG, digest="sha256:" + "c" * 64, size=42)
    raw = build_manifest_bytes(layers, cfg_desc)
    m = json.loads(raw)
    assert m["schemaVersion"] == 2
    assert m["mediaType"] == ML_MAN
    assert m["config"]["mediaType"] == ML_CFG
    assert m["config"]["digest"] == cfg_desc.digest
    assert m["config"]["size"] == cfg_desc.size
    assert m["layers"][0]["mediaType"] == ML_TAR
    assert m["layers"][0]["digest"] == layers[0].digest


def test_manifest_is_deterministic_for_same_inputs():
    layers = [BlobDescriptor(media_type=ML_TAR, digest="sha256:" + "a" * 64, size=10)]
    cfg_desc = BlobDescriptor(media_type=ML_CFG, digest="sha256:" + "c" * 64, size=42)
    a = build_manifest_bytes(layers, cfg_desc)
    b = build_manifest_bytes(layers, cfg_desc)
    assert a == b
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
