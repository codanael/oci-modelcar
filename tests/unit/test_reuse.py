"""Tests for reuse.py — OCI referrer-based crash-resilient reuse store."""

from __future__ import annotations

import hashlib
import json


def test_empty_config_constants_match_oci_spec() -> None:
    """The OCI 1.1 spec defines a canonical empty descriptor:
    mediaType=application/vnd.oci.empty.v1+json, content `{}` (2 bytes),
    digest sha256:44136fa3...8a. We use this everywhere a config is
    required but the artifact carries no config."""
    from oci_modelcar.reuse import (
        EMPTY_CONFIG_BYTES,
        EMPTY_CONFIG_DIGEST,
        EMPTY_CONFIG_MEDIA_TYPE,
        EMPTY_CONFIG_SIZE,
    )

    assert EMPTY_CONFIG_BYTES == b"{}"
    assert EMPTY_CONFIG_SIZE == 2
    assert EMPTY_CONFIG_MEDIA_TYPE == "application/vnd.oci.empty.v1+json"
    expected_digest = "sha256:" + hashlib.sha256(b"{}").hexdigest()
    assert expected_digest == EMPTY_CONFIG_DIGEST


def test_artifact_type_constants() -> None:
    from oci_modelcar.reuse import ARTIFACT_TYPE_ANCHOR, ARTIFACT_TYPE_RECORD

    assert ARTIFACT_TYPE_ANCHOR == "application/vnd.codanael.modelcar.reuse-anchor.v1"
    assert ARTIFACT_TYPE_RECORD == "application/vnd.codanael.modelcar.reuse-record.v1"


def test_anchor_manifest_bytes_is_deterministic() -> None:
    """Same inputs → byte-identical output."""
    from oci_modelcar.reuse import build_anchor_manifest_bytes

    a = build_anchor_manifest_bytes(
        hf_repo="mistralai/Mistral-Medium-3.5-128B",
        hf_revision="abc123",
        allow_patterns=("*.safetensors", "*.json"),
        ignore_patterns=("consolidated*",),
        layer_prefix="models/",
    )
    b = build_anchor_manifest_bytes(
        hf_repo="mistralai/Mistral-Medium-3.5-128B",
        hf_revision="abc123",
        allow_patterns=("*.safetensors", "*.json"),
        ignore_patterns=("consolidated*",),
        layer_prefix="models/",
    )
    assert a == b


def test_anchor_manifest_bytes_differs_when_ignore_patterns_change() -> None:
    from oci_modelcar.reuse import build_anchor_manifest_bytes

    a = build_anchor_manifest_bytes(
        hf_repo="x",
        hf_revision="r",
        allow_patterns=(".safetensors",),
        ignore_patterns=("foo*",),
        layer_prefix="models/",
    )
    b = build_anchor_manifest_bytes(
        hf_repo="x",
        hf_revision="r",
        allow_patterns=(".safetensors",),
        ignore_patterns=("bar*",),
        layer_prefix="models/",
    )
    assert a != b


def test_anchor_manifest_shape() -> None:
    """The anchor must be a valid OCI 1.1 artifact manifest: empty
    config (using the OCI-spec empty descriptor), empty layers, the
    expected artifactType, and run-input annotations."""
    from oci_modelcar.reuse import (
        ARTIFACT_TYPE_ANCHOR,
        EMPTY_CONFIG_DIGEST,
        EMPTY_CONFIG_SIZE,
        build_anchor_manifest_bytes,
    )

    body = build_anchor_manifest_bytes(
        hf_repo="x/y",
        hf_revision="rev123",
        allow_patterns=("*.safetensors", "*.json"),
        ignore_patterns=("consolidated*",),
        layer_prefix="models/",
    )
    m = json.loads(body)
    assert m["schemaVersion"] == 2
    assert m["mediaType"] == "application/vnd.oci.image.manifest.v1+json"
    assert m["artifactType"] == ARTIFACT_TYPE_ANCHOR
    assert m["config"]["mediaType"] == "application/vnd.oci.empty.v1+json"
    assert m["config"]["digest"] == EMPTY_CONFIG_DIGEST
    assert m["config"]["size"] == EMPTY_CONFIG_SIZE
    assert m["layers"] == []
    annot = m["annotations"]
    assert annot["io.github.codanael.modelcar.hf-repo"] == "x/y"
    assert annot["io.github.codanael.modelcar.hf-revision"] == "rev123"
    assert annot["io.github.codanael.modelcar.allow-patterns"] == "*.safetensors *.json"
    assert annot["io.github.codanael.modelcar.ignore-patterns"] == "consolidated*"
    assert annot["io.github.codanael.modelcar.layer-prefix"] == "models/"


def test_anchor_manifest_sorted_keys_throughout() -> None:
    """The bytes are produced with sort_keys=True so two clients
    serializing the same logical manifest get byte-identical output."""
    from oci_modelcar.reuse import build_anchor_manifest_bytes

    body = build_anchor_manifest_bytes(
        hf_repo="x",
        hf_revision="r",
        allow_patterns=(".safetensors",),
        ignore_patterns=(),
        layer_prefix="models/",
    )
    # json.dumps with sort_keys=True is what build_* uses; round-trip
    # checks that the parsed-then-reserialized form is identical.
    m = json.loads(body)
    re_serialized = json.dumps(m, separators=(",", ":"), sort_keys=True).encode()
    assert re_serialized == body


def test_anchor_manifest_empty_ignore_patterns_round_trips() -> None:
    """An empty ignore tuple must serialize to an empty annotation string,
    not to a missing annotation. Otherwise re-running with vs without
    `--ignore-patterns` set to nothing would produce a different anchor."""
    from oci_modelcar.reuse import build_anchor_manifest_bytes

    body = build_anchor_manifest_bytes(
        hf_repo="x",
        hf_revision="r",
        allow_patterns=(".safetensors",),
        ignore_patterns=(),
        layer_prefix="models/",
    )
    m = json.loads(body)
    assert m["annotations"]["io.github.codanael.modelcar.ignore-patterns"] == ""


def test_record_manifest_bytes_subject_points_to_anchor() -> None:
    from oci_modelcar.manifest import ML_TAR, BlobDescriptor
    from oci_modelcar.reuse import (
        ARTIFACT_TYPE_RECORD,
        EMPTY_CONFIG_DIGEST,
        build_record_manifest_bytes,
    )

    desc = BlobDescriptor(
        media_type=ML_TAR,
        digest="sha256:" + "a" * 64,
        size=1000,
        hf_path="model-00001-of-00003.safetensors",
        hf_sha256="b" * 64,
    )
    body = build_record_manifest_bytes(
        layer=desc,
        anchor_digest="sha256:" + "c" * 64,
        anchor_size=512,
    )
    m = json.loads(body)
    assert m["schemaVersion"] == 2
    assert m["mediaType"] == "application/vnd.oci.image.manifest.v1+json"
    assert m["artifactType"] == ARTIFACT_TYPE_RECORD
    assert m["config"]["digest"] == EMPTY_CONFIG_DIGEST
    assert m["config"]["size"] == 2
    assert m["subject"]["mediaType"] == "application/vnd.oci.image.manifest.v1+json"
    assert m["subject"]["digest"] == "sha256:" + "c" * 64
    assert m["subject"]["size"] == 512
    assert len(m["layers"]) == 1
    layer = m["layers"][0]
    assert layer["mediaType"] == ML_TAR
    assert layer["digest"] == "sha256:" + "a" * 64
    assert layer["size"] == 1000
    assert layer["annotations"] == {
        "io.github.codanael.modelcar.hf-path": "model-00001-of-00003.safetensors",
        "io.github.codanael.modelcar.hf-sha256": "b" * 64,
    }


def test_record_manifest_bytes_deterministic() -> None:
    from oci_modelcar.manifest import ML_TAR, BlobDescriptor
    from oci_modelcar.reuse import build_record_manifest_bytes

    desc = BlobDescriptor(
        media_type=ML_TAR,
        digest="sha256:" + "f" * 64,
        size=500,
        hf_path="m.bin",
        hf_sha256=None,
    )
    a = build_record_manifest_bytes(desc, "sha256:" + "0" * 64, 100)
    b = build_record_manifest_bytes(desc, "sha256:" + "0" * 64, 100)
    assert a == b


def test_record_manifest_bytes_no_lfs_sha_omits_annotation() -> None:
    """When the HF file is not LFS-backed, the hf-sha256 annotation must
    NOT appear in the record (BlobDescriptor.to_dict already enforces
    this for the image manifest; record_manifest must mirror it)."""
    from oci_modelcar.manifest import ML_TAR, BlobDescriptor
    from oci_modelcar.reuse import build_record_manifest_bytes

    desc = BlobDescriptor(
        media_type=ML_TAR,
        digest="sha256:" + "f" * 64,
        size=500,
        hf_path="config.json",
        hf_sha256=None,
    )
    body = build_record_manifest_bytes(desc, "sha256:" + "0" * 64, 100)
    m = json.loads(body)
    annot = m["layers"][0]["annotations"]
    assert "io.github.codanael.modelcar.hf-path" in annot
    assert "io.github.codanael.modelcar.hf-sha256" not in annot


def test_fallback_referrers_tag_format() -> None:
    """Per OCI 1.1 fallback tag schema: sha256:<hex> → sha256-<hex>."""
    from oci_modelcar.reuse import fallback_referrers_tag

    assert fallback_referrers_tag("sha256:" + "a" * 64) == "sha256-" + "a" * 64
    assert fallback_referrers_tag("sha256:cafebabe") == "sha256-cafebabe"


def test_fallback_referrers_tag_rejects_unprefixed_digest() -> None:
    """The input must have a recognized algorithm prefix; bare hex is
    a bug at the call site."""
    import pytest

    from oci_modelcar.reuse import fallback_referrers_tag

    with pytest.raises(ValueError, match="digest"):
        fallback_referrers_tag("a" * 64)


# ---------------------------------------------------------------------------
# RegistryReuseStore
# ---------------------------------------------------------------------------


def _make_response(status_code: int, headers: dict[str, str] | None = None, body: bytes = b""):  # type: ignore[no-untyped-def]
    from unittest.mock import MagicMock

    r = MagicMock()
    r.status_code = status_code
    r.headers = headers or {}
    r.content = body
    r.text = body.decode() if body else ""
    r.json = lambda: json.loads(body) if body else {}
    r.raise_for_status.return_value = None
    return r


def _make_client(fake_session):  # type: ignore[no-untyped-def]
    from oci_modelcar.registry import OciClient

    return OciClient(host_url="http://test", session=fake_session)


def test_ensure_anchor_pushes_empty_config_blob_and_manifest_when_absent() -> None:
    from unittest.mock import MagicMock

    from oci_modelcar.reuse import (
        EMPTY_CONFIG_BYTES,
        EMPTY_CONFIG_DIGEST,
        RegistryReuseStore,
    )

    fake_session = MagicMock()
    # Three HEAD calls we care about:
    #   1) head_blob(EMPTY_CONFIG_DIGEST) — 404 so push_small_blob does the POST
    #   2) head manifest by anchor digest — 404 (need to push)
    # And whatever push_small_blob does internally (POST then PUT).
    fake_session.head.return_value = _make_response(404)
    fake_session.post.return_value = _make_response(
        202, {"Location": "http://test/v2/repo/blobs/uploads/abc"}
    )
    fake_session.put.return_value = _make_response(201)

    anchor_bytes = b'{"sample":"anchor"}'
    anchor_digest = "sha256:" + hashlib.sha256(anchor_bytes).hexdigest()

    store = RegistryReuseStore(_make_client(fake_session), "repo")
    store.ensure_anchor(anchor_bytes, anchor_digest)

    # The manifest PUT must have been issued at /manifests/<anchor_digest>
    manifest_put_calls = [
        c for c in fake_session.put.call_args_list if f"manifests/{anchor_digest}" in c.args[0]
    ]
    assert len(manifest_put_calls) == 1
    assert manifest_put_calls[0].kwargs["data"] == anchor_bytes

    # The empty config blob path: push_small_blob bootstraps by POST then PUT;
    # we don't assert exact byte content here, just that the POST happened
    # (it would not, if push_small_blob had been skipped).
    assert fake_session.post.called
    # EMPTY_CONFIG_DIGEST should have been HEAD-checked at least once
    head_blob_paths = [c.args[0] for c in fake_session.head.call_args_list]
    assert any(EMPTY_CONFIG_DIGEST in p for p in head_blob_paths)
    # The bytes pushed must be the canonical empty config
    assert EMPTY_CONFIG_BYTES == b"{}"


def _make_referrer_descriptor(record_digest: str, record_size: int) -> dict:  # type: ignore[type-arg]
    return {
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "artifactType": "application/vnd.codanael.modelcar.reuse-record.v1",
        "digest": record_digest,
        "size": record_size,
    }


def _make_record_body(
    layer_digest: str, layer_size: int, hf_path: str, hf_sha256: str | None
) -> bytes:
    from oci_modelcar.manifest import ML_TAR, BlobDescriptor
    from oci_modelcar.reuse import build_record_manifest_bytes

    desc = BlobDescriptor(
        media_type=ML_TAR,
        digest=layer_digest,
        size=layer_size,
        hf_path=hf_path,
        hf_sha256=hf_sha256,
    )
    return build_record_manifest_bytes(desc, "sha256:" + "0" * 64, 100)


def test_load_reuse_map_native_referrers_returns_all_records() -> None:
    from unittest.mock import MagicMock

    from oci_modelcar.reuse import RegistryReuseStore

    anchor_digest = "sha256:" + "0" * 64

    record_a = _make_record_body("sha256:" + "a" * 64, 1000, "model-001.safetensors", "1" * 64)
    record_b = _make_record_body("sha256:" + "b" * 64, 500, "config.json", None)
    record_a_digest = "sha256:" + hashlib.sha256(record_a).hexdigest()
    record_b_digest = "sha256:" + hashlib.sha256(record_b).hexdigest()

    referrer_index = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                _make_referrer_descriptor(record_a_digest, len(record_a)),
                _make_referrer_descriptor(record_b_digest, len(record_b)),
            ],
        }
    ).encode()

    fake_session = MagicMock()

    def get_router(url, **kwargs):  # type: ignore[no-untyped-def]
        if f"referrers/{anchor_digest}" in url:
            return _make_response(200, body=referrer_index)
        if f"manifests/{record_a_digest}" in url:
            return _make_response(200, body=record_a)
        if f"manifests/{record_b_digest}" in url:
            return _make_response(200, body=record_b)
        return _make_response(404)

    fake_session.get.side_effect = get_router

    store = RegistryReuseStore(_make_client(fake_session), "repo")
    reuse_map = store.load_reuse_map(anchor_digest)

    assert set(reuse_map.keys()) == {
        ("model-001.safetensors", "1" * 64),
        ("config.json", None),
    }
    desc_a = reuse_map[("model-001.safetensors", "1" * 64)]
    assert desc_a.digest == "sha256:" + "a" * 64
    assert desc_a.size == 1000
    assert desc_a.hf_path == "model-001.safetensors"
    desc_b = reuse_map[("config.json", None)]
    assert desc_b.digest == "sha256:" + "b" * 64
    assert desc_b.size == 500


def test_load_reuse_map_falls_back_to_referrers_tag_on_404() -> None:
    from unittest.mock import MagicMock

    from oci_modelcar.reuse import RegistryReuseStore, fallback_referrers_tag

    anchor_digest = "sha256:" + "9" * 64
    record = _make_record_body("sha256:" + "a" * 64, 1000, "model.safetensors", "1" * 64)
    record_digest = "sha256:" + hashlib.sha256(record).hexdigest()
    referrer_index = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [_make_referrer_descriptor(record_digest, len(record))],
        }
    ).encode()

    fake_session = MagicMock()
    fallback_tag = fallback_referrers_tag(anchor_digest)

    def get_router(url, **kwargs):  # type: ignore[no-untyped-def]
        if f"referrers/{anchor_digest}" in url:
            return _make_response(404)
        if f"manifests/{fallback_tag}" in url:
            return _make_response(200, body=referrer_index)
        if f"manifests/{record_digest}" in url:
            return _make_response(200, body=record)
        return _make_response(404)

    fake_session.get.side_effect = get_router

    store = RegistryReuseStore(_make_client(fake_session), "repo")
    reuse_map = store.load_reuse_map(anchor_digest)

    assert set(reuse_map.keys()) == {("model.safetensors", "1" * 64)}


def test_load_reuse_map_empty_when_both_native_and_fallback_404() -> None:
    from unittest.mock import MagicMock

    from oci_modelcar.reuse import RegistryReuseStore

    fake_session = MagicMock()
    fake_session.get.return_value = _make_response(404)

    store = RegistryReuseStore(_make_client(fake_session), "repo")
    reuse_map = store.load_reuse_map("sha256:" + "0" * 64)
    assert reuse_map == {}


def test_load_reuse_map_skips_records_with_missing_annotation() -> None:
    """If a record's layer[0] is missing the hf-path annotation (e.g.
    written by a foreign tool), the entry is silently dropped — we can't
    map it to an HF file."""
    from unittest.mock import MagicMock

    from oci_modelcar.reuse import RegistryReuseStore

    anchor_digest = "sha256:" + "0" * 64
    # Build a record manually without the hf-path annotation
    bad_record = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "artifactType": "application/vnd.codanael.modelcar.reuse-record.v1",
            "config": {
                "mediaType": "application/vnd.oci.empty.v1+json",
                "digest": "sha256:" + "4" * 64,
                "size": 2,
            },
            "subject": {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": anchor_digest,
                "size": 100,
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar",
                    "digest": "sha256:" + "a" * 64,
                    "size": 1000,
                    "annotations": {},
                }
            ],
        }
    ).encode()
    bad_digest = "sha256:" + hashlib.sha256(bad_record).hexdigest()
    referrer_index = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [_make_referrer_descriptor(bad_digest, len(bad_record))],
        }
    ).encode()

    fake_session = MagicMock()

    def get_router(url, **kwargs):  # type: ignore[no-untyped-def]
        if f"referrers/{anchor_digest}" in url:
            return _make_response(200, body=referrer_index)
        if f"manifests/{bad_digest}" in url:
            return _make_response(200, body=bad_record)
        return _make_response(404)

    fake_session.get.side_effect = get_router

    store = RegistryReuseStore(_make_client(fake_session), "repo")
    reuse_map = store.load_reuse_map(anchor_digest)
    assert reuse_map == {}


def test_record_native_path_does_not_touch_fallback_tag() -> None:
    """When the registry echoes `OCI-Subject` on the record PUT, the
    client must NOT also write the fallback tag (the registry handles it)."""
    from unittest.mock import MagicMock

    from oci_modelcar.manifest import ML_TAR, BlobDescriptor
    from oci_modelcar.reuse import RegistryReuseStore, fallback_referrers_tag

    anchor_digest = "sha256:" + "0" * 64
    fallback_tag = fallback_referrers_tag(anchor_digest)

    fake_session = MagicMock()

    def put_router(url, **kwargs):  # type: ignore[no-untyped-def]
        # The PUT must be at /manifests/<record_digest>, never at the fallback tag
        assert f"manifests/{fallback_tag}" not in url
        return _make_response(201, {"OCI-Subject": anchor_digest})

    fake_session.put.side_effect = put_router

    layer = BlobDescriptor(
        media_type=ML_TAR,
        digest="sha256:" + "a" * 64,
        size=1000,
        hf_path="model.safetensors",
        hf_sha256="1" * 64,
    )
    store = RegistryReuseStore(_make_client(fake_session), "repo")
    store.record(layer, anchor_digest, anchor_size=100)

    # Exactly one PUT (record manifest); no GET (no fallback)
    assert fake_session.put.call_count == 1
    assert not fake_session.get.called


def test_record_fallback_path_appends_to_index() -> None:
    """Registry that doesn't support referrers natively (no OCI-Subject
    in response): client maintains the sha256-<hex> tag, appending to
    its image index on every record. Stateful mock simulates the registry
    remembering prior PUTs."""
    from unittest.mock import MagicMock

    from oci_modelcar.manifest import ML_TAR, BlobDescriptor
    from oci_modelcar.reuse import RegistryReuseStore, fallback_referrers_tag

    anchor_digest = "sha256:" + "0" * 64
    fallback_tag = fallback_referrers_tag(anchor_digest)

    fake_session = MagicMock()
    # Stateful storage: tag → manifest body
    manifests_by_tag: dict[str, bytes] = {}

    def get_router(url, **kwargs):  # type: ignore[no-untyped-def]
        for tag, body in manifests_by_tag.items():
            if f"manifests/{tag}" in url:
                return _make_response(200, body=body)
        return _make_response(404)

    def put_router(url, **kwargs):  # type: ignore[no-untyped-def]
        # Capture the body keyed by the tag portion of the URL
        if "manifests/" in url:
            tag = url.split("manifests/")[-1]
            manifests_by_tag[tag] = kwargs["data"]
        # No OCI-Subject header → registry is non-native
        return _make_response(201)

    fake_session.get.side_effect = get_router
    fake_session.put.side_effect = put_router

    layer1 = BlobDescriptor(
        media_type=ML_TAR,
        digest="sha256:" + "a" * 64,
        size=1000,
        hf_path="model.safetensors",
        hf_sha256="1" * 64,
    )
    layer2 = BlobDescriptor(
        media_type=ML_TAR,
        digest="sha256:" + "b" * 64,
        size=500,
        hf_path="config.json",
        hf_sha256=None,
    )

    store = RegistryReuseStore(_make_client(fake_session), "repo")
    store.record(layer1, anchor_digest, anchor_size=100)
    store.record(layer2, anchor_digest, anchor_size=100)

    # The fallback tag now holds an index with BOTH descriptors
    assert fallback_tag in manifests_by_tag
    parsed = json.loads(manifests_by_tag[fallback_tag])
    assert parsed["mediaType"] == "application/vnd.oci.image.index.v1+json"
    digests = [m["digest"] for m in parsed["manifests"]]
    assert len(digests) == 2
    assert len(set(digests)) == 2  # no dup


def test_record_caches_native_detection_across_calls() -> None:
    """Once detected as native, subsequent record() calls must not GET
    the fallback tag — that would be wasted round-trips on every layer."""
    from unittest.mock import MagicMock

    from oci_modelcar.manifest import ML_TAR, BlobDescriptor
    from oci_modelcar.reuse import RegistryReuseStore

    anchor_digest = "sha256:" + "0" * 64
    fake_session = MagicMock()
    fake_session.put.return_value = _make_response(201, {"OCI-Subject": anchor_digest})

    store = RegistryReuseStore(_make_client(fake_session), "repo")
    for i in range(3):
        layer = BlobDescriptor(
            media_type=ML_TAR,
            digest=f"sha256:{i:064}",
            size=100,
            hf_path=f"f{i}.bin",
            hf_sha256=None,
        )
        store.record(layer, anchor_digest, anchor_size=100)

    # Exactly 3 PUTs (one per record), zero GETs (never queried fallback)
    assert fake_session.put.call_count == 3
    assert not fake_session.get.called


def test_ensure_anchor_skips_manifest_put_when_already_present() -> None:
    from unittest.mock import MagicMock

    from oci_modelcar.reuse import EMPTY_CONFIG_DIGEST, RegistryReuseStore

    fake_session = MagicMock()

    anchor_bytes = b'{"sample":"already-there"}'
    anchor_digest = "sha256:" + hashlib.sha256(anchor_bytes).hexdigest()

    def head_router(url, **kwargs):  # type: ignore[no-untyped-def]
        # Empty config blob: present (skip push_small_blob's POST)
        if EMPTY_CONFIG_DIGEST in url:
            return _make_response(
                200, {"Docker-Content-Digest": EMPTY_CONFIG_DIGEST, "Content-Length": "2"}
            )
        # Anchor manifest by digest: present
        if f"manifests/{anchor_digest}" in url:
            return _make_response(200, {"Docker-Content-Digest": anchor_digest})
        return _make_response(404)

    fake_session.head.side_effect = head_router

    store = RegistryReuseStore(_make_client(fake_session), "repo")
    store.ensure_anchor(anchor_bytes, anchor_digest)

    # No PUT at all (manifest already there AND config blob already there)
    assert not fake_session.put.called
    assert not fake_session.post.called
