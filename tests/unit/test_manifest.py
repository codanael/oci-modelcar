import hashlib
import json

import pytest

from oci_modelcar.manifest import (
    ANN_HF_PATH,
    ANN_HF_SHA256,
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
        "annotations": {ANN_HF_PATH: "model.safetensors"},
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


def test_blob_descriptor_emits_hf_annotations():
    """Layer descriptors carry modelcar hf-path / hf-sha256 annotations so
    that future runs can reuse the layer when the source file is unchanged."""
    d = BlobDescriptor(
        media_type="application/vnd.oci.image.layer.v1.tar",
        digest="sha256:" + "a" * 64,
        size=100,
        hf_path="weights/model.safetensors",
        hf_sha256="b" * 64,
    )
    out = d.to_dict()
    assert out["annotations"] == {
        ANN_HF_PATH: "weights/model.safetensors",
        ANN_HF_SHA256: "b" * 64,
    }


def test_blob_descriptor_emits_path_only_when_no_lfs_sha():
    """Non-LFS files (small configs) have no sha256; emit just the path annotation."""
    d = BlobDescriptor(
        media_type="application/vnd.oci.image.layer.v1.tar",
        digest="sha256:" + "a" * 64,
        size=100,
        hf_path="config.json",
        hf_sha256=None,
    )
    out = d.to_dict()
    assert out["annotations"] == {ANN_HF_PATH: "config.json"}


def test_manifest_layers_carry_annotations():
    layers = [
        BlobDescriptor(
            "application/vnd.oci.image.layer.v1.tar",
            "sha256:" + "a" * 64,
            100,
            "model.safetensors",
            "f" * 64,
        ),
    ]
    manifest = json.loads(build_manifest_bytes("sha256:" + "c" * 64, 50, layers))
    layer = manifest["layers"][0]
    assert layer["annotations"][ANN_HF_PATH] == "model.safetensors"
    assert layer["annotations"][ANN_HF_SHA256] == "f" * 64


def test_manifest_digest_stable_across_runs_with_annotations():
    """Adding annotations must NOT introduce non-determinism."""
    layers = [
        BlobDescriptor(
            "application/vnd.oci.image.layer.v1.tar",
            "sha256:" + "a" * 64,
            100,
            "f1.bin",
            "1" * 64,
        ),
        BlobDescriptor(
            "application/vnd.oci.image.layer.v1.tar",
            "sha256:" + "b" * 64,
            200,
            "f2.bin",
            "2" * 64,
        ),
    ]
    a = build_manifest_bytes("sha256:" + "c" * 64, 50, layers)
    b = build_manifest_bytes("sha256:" + "c" * 64, 50, layers)
    assert a == b


def test_derive_tag_explicit_validated():
    """Explicit tag is taken as-is; the caller (Config.validate) enforces
    OCI tag rules. derive_tag does not re-validate."""
    assert derive_tag("any", explicit="raw_input") == "raw_input"
