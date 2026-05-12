"""Integration test for re-push reuse: layer annotations + reuse-map +
worker phase-0 skip. End-to-end without HTTP — registry round-trips and
HF traffic are mocked, but the pipeline data path is exercised."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

from oci_modelcar.config import Config
from oci_modelcar.download import HfFile
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.manifest import ANN_HF_PATH, ANN_HF_SHA256, ML_TAR
from oci_modelcar.pipeline import Pipeline


def _make_cfg(tmp_path: Path, **overrides: object) -> Config:
    base: dict[str, object] = dict(
        hf_repo="foo/bar",
        registry="registry.example.com",
        target_repo="models/x",
        hf_revision="main",
        hf_endpoint="https://huggingface.co",
        target_tag=None,
        also_tags=[],
        allow_patterns=(".bin",),
        layer_prefix="models/",
        workers=1,
        spool_dir=tmp_path / "spool",
        clean_hf_after_push=False,
        hf_max_retries=3,
        oci_max_retries=3,
        fail_fast=True,
        force=False,
        log_style="text",
        verbose=False,
        quiet=True,
        dry_run=False,
        sub_command="push",
    )
    base.update(overrides)
    return Config(**base)  # type: ignore[arg-type]


class _StatefulRegistry:
    """Minimal in-memory OCI registry. Records blobs and manifests by tag."""

    def __init__(self) -> None:
        self.blobs: dict[str, int] = {}  # digest → size
        self.manifests: dict[str, bytes] = {}  # tag → bytes
        self.manifest_digests: dict[str, str] = {}  # tag → digest

    def put_blob(self, digest: str, size: int) -> None:
        self.blobs[digest] = size

    def put_manifest(self, tag: str, body: bytes) -> str:
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        self.manifests[tag] = body
        self.manifest_digests[tag] = digest
        return digest


def test_repush_unchanged_model_reuses_all_layers(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """
    First push:  downloads each file, builds tar, pushes blob.
    Second push: same revision, same files — downloads ZERO, pushes ZERO.
    """
    revision = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    files = [
        HfFile("model.safetensors", 1024, lfs_sha256="1" * 64),
        HfFile("tokenizer.bin", 256, lfs_sha256="2" * 64),
    ]
    registry = _StatefulRegistry()

    # The downloader writes deterministic bytes so each run produces the same
    # tar digest.
    def fake_download(repo: str, rev: str, hf_file: HfFile, progress_cb=None) -> Path:  # type: ignore[no-untyped-def]
        sources = tmp_path / "spool" / "sources"
        sources.mkdir(parents=True, exist_ok=True)
        p = sources / hf_file.path
        p.write_bytes(hf_file.path.encode() * (hf_file.size // len(hf_file.path) + 1))
        # Truncate so size matches exactly
        with open(p, "r+b") as fh:
            fh.truncate(hf_file.size)
        return p

    downloader = MagicMock()
    downloader.resolve_revision.return_value = revision
    downloader.list_files.return_value = files
    downloader.download.side_effect = fake_download

    # Patch HEAD/get_manifest helpers + push_small_blob + push_manifest +
    # validate_manifest_tag + head_blob. We thread state through `registry`.

    def fake_head_blob(client, repo: str, digest: str):  # type: ignore[no-untyped-def]
        if digest in registry.blobs:
            return {"digest": digest, "size": registry.blobs[digest]}
        return None

    def fake_get_manifest_digest(client, repo: str, tag: str) -> str | None:  # type: ignore[no-untyped-def]
        return registry.manifest_digests.get(tag)

    def fake_fetch_manifest(client, repo: str, tag: str):  # type: ignore[no-untyped-def]
        body = registry.manifests.get(tag)
        if body is None:
            return None
        return json.loads(body)

    pushed_blobs: list[str] = []

    class FakeStreaming:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def push_from_file(self, path: Path, size: int, digest: str, progress_cb=None) -> None:  # type: ignore[no-untyped-def]
            registry.put_blob(digest, size)
            pushed_blobs.append(digest)

    def fake_push_small_blob(client, repo: str, data: bytes) -> str:  # type: ignore[no-untyped-def]
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        registry.put_blob(digest, len(data))
        return digest

    def fake_push_manifest(client, repo: str, tag: str, body: bytes) -> str:  # type: ignore[no-untyped-def]
        return registry.put_manifest(tag, body)

    def fake_validate(client, repo: str, tag: str, expected: str) -> None:  # type: ignore[no-untyped-def]
        assert registry.manifest_digests.get(tag) == expected

    monkeypatch.setattr("oci_modelcar.pipeline.head_blob", fake_head_blob)
    monkeypatch.setattr(
        "oci_modelcar.pipeline.get_manifest_digest_at_tag", fake_get_manifest_digest
    )
    monkeypatch.setattr("oci_modelcar.pipeline.fetch_manifest_at_tag", fake_fetch_manifest)
    monkeypatch.setattr("oci_modelcar.pipeline.StreamingBlobUpload", FakeStreaming)
    monkeypatch.setattr("oci_modelcar.pipeline.push_small_blob", fake_push_small_blob)
    monkeypatch.setattr("oci_modelcar.pipeline.push_manifest", fake_push_manifest)
    monkeypatch.setattr("oci_modelcar.pipeline.validate_manifest_tag", fake_validate)
    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    cfg = _make_cfg(tmp_path)
    plog = PipelineLogger(quiet=True)
    fake_registry_client = MagicMock(target_repo="models/x")
    fake_registry_client.host = "registry.example.com"

    # --- FIRST RUN ---
    pipeline_1 = Pipeline(cfg, plog, downloader=downloader, registry_client=fake_registry_client)
    result_1 = pipeline_1.run()
    assert downloader.download.call_count == 2
    assert len(pushed_blobs) == 2
    assert result_1.manifest_digest in registry.manifest_digests.values()

    # Manifest must carry per-layer annotations so the next run can reuse them.
    manifest_body = next(iter(registry.manifests.values()))
    manifest_parsed = json.loads(manifest_body)
    for layer in manifest_parsed["layers"]:
        assert ANN_HF_PATH in layer["annotations"]
        assert ANN_HF_SHA256 in layer["annotations"]
        assert layer["mediaType"] == ML_TAR

    # --- SECOND RUN (same files, same revision) ---
    downloader.download.reset_mock()
    pushed_blobs.clear()

    pipeline_2 = Pipeline(cfg, plog, downloader=downloader, registry_client=fake_registry_client)
    result_2 = pipeline_2.run()

    # The whole point: no HF traffic, no blob PUT.
    assert downloader.download.call_count == 0, "second push must reuse cached layers"
    assert pushed_blobs == [], "second push must not PUT any new blob"

    # And the manifest digest matches the first run's (byte-identical manifest).
    assert result_2.manifest_digest == result_1.manifest_digest
