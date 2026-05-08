import hashlib
import json

import pytest

from oci_modelcar.manifest import (
    BlobDescriptor,
    build_config_bytes,
    build_manifest_bytes,
    derive_tag,
)


def test_blob_descriptor_to_dict():
    d = BlobDescriptor(
        media_type="application/vnd.oci.image.layer.v1.tar",
        digest="sha256:" + "a" * 64,
        size=12345,
        hf_path="model.safetensors",
    )
    assert d.to_dict() == {
        "mediaType": "application/vnd.oci.image.layer.v1.tar",
        "digest": "sha256:" + "a" * 64,
        "size": 12345,
    }


def test_config_bytes_no_created_field():
    """v0.x design lock-in: NO `created` field, so config bytes are
    deterministic across runs and config digest is stable."""
    diff_ids = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
    cfg = build_config_bytes(diff_ids)
    parsed = json.loads(cfg)
    assert "created" not in parsed
    assert parsed["rootfs"]["diff_ids"] == diff_ids
    assert parsed["rootfs"]["type"] == "layers"
    assert parsed["architecture"] == "amd64"
    assert parsed["os"] == "linux"


def test_config_bytes_reproducible():
    diff_ids = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
    a = build_config_bytes(diff_ids)
    b = build_config_bytes(diff_ids)
    assert a == b


def test_manifest_bytes_layers_in_provided_order():
    """The runner is responsible for sorting layers alphabetically by
    hf_path. build_manifest_bytes preserves the order it's given."""
    layers = [
        BlobDescriptor(
            "application/vnd.oci.image.layer.v1.tar", "sha256:" + "a" * 64, 100, "a.bin"
        ),
        BlobDescriptor(
            "application/vnd.oci.image.layer.v1.tar", "sha256:" + "b" * 64, 200, "b.bin"
        ),
    ]
    config_digest = "sha256:" + "c" * 64
    config_size = 50
    manifest = build_manifest_bytes(config_digest, config_size, layers)
    parsed = json.loads(manifest)
    assert parsed["mediaType"] == "application/vnd.oci.image.manifest.v1+json"
    assert parsed["config"]["digest"] == config_digest
    assert parsed["config"]["size"] == config_size
    assert [layer["digest"] for layer in parsed["layers"]] == [
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
    ]


def test_manifest_bytes_reproducible():
    layers = [
        BlobDescriptor(
            "application/vnd.oci.image.layer.v1.tar", "sha256:" + "a" * 64, 100, "a.bin"
        ),
    ]
    a = build_manifest_bytes("sha256:" + "c" * 64, 50, layers)
    b = build_manifest_bytes("sha256:" + "c" * 64, 50, layers)
    assert a == b
    assert hashlib.sha256(a).digest() == hashlib.sha256(b).digest()


def test_derive_tag_from_40char_sha():
    sha = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    assert derive_tag(sha, explicit=None) == "9fb191250dd5"


def test_derive_tag_explicit_overrides():
    assert derive_tag("ignored", explicit="v1.0") == "v1.0"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("main", "main"),
        ("feature/x", "feature-x"),
        ("v1.0.0", "v1.0.0"),
        ("Hello World", "hello-world"),
        ("trailing/", "trailing"),
    ],
)
def test_derive_tag_sanitizes_non_sha(raw, expected):
    assert derive_tag(raw, explicit=None) == expected


def test_derive_tag_explicit_validated():
    """Explicit tag is taken as-is; the caller (Config.validate) enforces
    OCI tag rules. derive_tag does not re-validate."""
    assert derive_tag("any", explicit="raw_input") == "raw_input"
