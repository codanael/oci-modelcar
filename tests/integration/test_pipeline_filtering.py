"""Integration test for --ignore-patterns / glob --allow-patterns.

The pipeline is exercised end-to-end against an in-memory registry; the
HF API tree and bytes are mocked, but ``HfDownloader.list_files``
itself runs for real so the filter wiring (CLI → Config → Pipeline →
fnmatch) is covered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from huggingface_hub.hf_api import ModelInfo, RepoFile

from oci_modelcar.config import Config
from oci_modelcar.download import HfDownloader, HfFile
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.pipeline import Pipeline


def _make_cfg(tmp_path: Path, **overrides: object) -> Config:
    base: dict[str, object] = dict(
        hf_repo="mistralai/Mistral-Medium-3.5-128B",
        registry="registry.example.com",
        target_repo="models/mistral",
        hf_revision="main",
        hf_endpoint="https://huggingface.co",
        target_tag=None,
        also_tags=[],
        allow_patterns=(".safetensors", ".json"),
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


def _repo_file(path: str, size: int, lfs_sha256: str | None = None) -> Any:
    """RepoFile-shaped mock that passes ``isinstance(entry, RepoFile)``."""
    m = MagicMock(spec=RepoFile)
    m.path = path
    m.size = size
    m.lfs = MagicMock(sha256=lfs_sha256) if lfs_sha256 else None
    return m


class _StatefulRegistry:
    def __init__(self) -> None:
        self.blobs: dict[str, int] = {}
        self.manifests: dict[str, bytes] = {}
        self.manifest_digests: dict[str, str] = {}

    def put_blob(self, digest: str, size: int) -> None:
        self.blobs[digest] = size

    def put_manifest(self, tag: str, body: bytes) -> str:
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        self.manifests[tag] = body
        self.manifest_digests[tag] = digest
        return digest


def test_ignore_patterns_drops_one_weight_set(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Mistral-Medium-3.5-style repo: two parallel weight sets distinguished
    by filename prefix. With ``--ignore-patterns 'consolidated-*'`` we keep
    only the HF-layout files."""
    revision = "abc123456789abc123456789abc123456789abcd"
    tree = [
        _repo_file("model-00001-of-00002.safetensors", 1000, "1" * 64),
        _repo_file("model-00002-of-00002.safetensors", 1000, "2" * 64),
        _repo_file("model.safetensors.index.json", 200),
        _repo_file("consolidated-00001-of-00002.safetensors", 1000, "3" * 64),
        _repo_file("consolidated-00002-of-00002.safetensors", 1000, "4" * 64),
        _repo_file("consolidated.safetensors.index.json", 200),
        _repo_file("config.json", 100),
        _repo_file("params.json", 50),
        _repo_file("tokenizer.json", 17_000),
    ]

    api = MagicMock()
    api.endpoint = "https://hf-mock"
    info = MagicMock(spec=ModelInfo)
    info.sha = revision
    api.repo_info.return_value = info
    api.list_repo_tree.return_value = tree

    downloader = HfDownloader(
        api=api, session=MagicMock(), spool_dir=tmp_path / "spool", stop_event=None
    )

    def fake_download(repo: str, rev: str, hf_file: HfFile, progress_cb=None) -> Path:  # type: ignore[no-untyped-def]
        sources = tmp_path / "spool" / "sources"
        sources.mkdir(parents=True, exist_ok=True)
        p = sources / hf_file.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(hf_file.path.encode() * (hf_file.size // len(hf_file.path) + 1))
        with open(p, "r+b") as fh:
            fh.truncate(hf_file.size)
        return p

    downloader.download = fake_download  # type: ignore[method-assign]

    registry = _StatefulRegistry()
    pushed_blobs: list[str] = []

    def fake_head_blob(client, repo: str, digest: str):  # type: ignore[no-untyped-def]
        if digest in registry.blobs:
            return {"digest": digest, "size": registry.blobs[digest]}
        return None

    class FakeStreaming:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def push_from_file(self, path: Path, size: int, digest: str) -> None:
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
    monkeypatch.setattr("oci_modelcar.pipeline.get_manifest_digest_at_tag", lambda *a, **k: None)
    monkeypatch.setattr("oci_modelcar.pipeline.fetch_manifest_at_tag", lambda *a, **k: None)
    monkeypatch.setattr("oci_modelcar.pipeline.StreamingBlobUpload", FakeStreaming)
    monkeypatch.setattr("oci_modelcar.pipeline.push_small_blob", fake_push_small_blob)
    monkeypatch.setattr("oci_modelcar.pipeline.push_manifest", fake_push_manifest)
    monkeypatch.setattr("oci_modelcar.pipeline.validate_manifest_tag", fake_validate)
    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    cfg = _make_cfg(tmp_path, ignore_patterns=("consolidated*", "params.json"))
    plog = PipelineLogger(quiet=True)
    fake_registry_client = MagicMock(target_repo="models/mistral")
    fake_registry_client.host = "registry.example.com"

    pipeline = Pipeline(cfg, plog, downloader=downloader, registry_client=fake_registry_client)
    result = pipeline.run()

    manifest_body = next(iter(registry.manifests.values()))
    manifest = json.loads(manifest_body)
    pushed_paths = {
        layer["annotations"]["io.github.codanael.modelcar.hf-path"] for layer in manifest["layers"]
    }

    assert pushed_paths == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
    }
    assert "consolidated-00001-of-00002.safetensors" not in pushed_paths
    assert "consolidated-00002-of-00002.safetensors" not in pushed_paths
    assert "consolidated.safetensors.index.json" not in pushed_paths
    assert "params.json" not in pushed_paths
    # And the hyphen-anchored variant is *also* tested so we don't regress
    # the more precise glob that some users might prefer.
    assert all("consolidated" not in p for p in pushed_paths)
    assert result.manifest_digest


def test_filtering_manifest_digest_is_deterministic(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The same filter applied twice → byte-identical manifest digest.
    Filtering does not perturb manifest determinism."""
    revision = "abc123456789abc123456789abc123456789abcd"
    tree = [
        _repo_file("model-00001-of-00001.safetensors", 1000, "1" * 64),
        _repo_file("consolidated-00001-of-00001.safetensors", 1000, "2" * 64),
        _repo_file("config.json", 100),
    ]

    def make_pipeline(target_dir: Path) -> tuple[Pipeline, _StatefulRegistry]:
        api = MagicMock()
        api.endpoint = "https://hf-mock"
        info = MagicMock(spec=ModelInfo)
        info.sha = revision
        api.repo_info.return_value = info
        api.list_repo_tree.return_value = tree

        downloader = HfDownloader(
            api=api, session=MagicMock(), spool_dir=target_dir / "spool", stop_event=None
        )

        def fake_download(repo: str, rev: str, hf_file: HfFile, progress_cb=None) -> Path:  # type: ignore[no-untyped-def]
            sources = target_dir / "spool" / "sources"
            sources.mkdir(parents=True, exist_ok=True)
            p = sources / hf_file.path
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = hf_file.path.encode() * (hf_file.size // len(hf_file.path) + 1)
            p.write_bytes(payload)
            with open(p, "r+b") as fh:
                fh.truncate(hf_file.size)
            return p

        downloader.download = fake_download  # type: ignore[method-assign]

        registry = _StatefulRegistry()

        class FakeStreaming:
            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                pass

            def push_from_file(self, path: Path, size: int, digest: str) -> None:
                registry.put_blob(digest, size)

        def fake_head_blob(client, repo, digest):  # type: ignore[no-untyped-def]
            if digest in registry.blobs:
                return {"digest": digest, "size": registry.blobs[digest]}
            return None

        def fake_push_small_blob(client, repo, data):  # type: ignore[no-untyped-def]
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            registry.put_blob(digest, len(data))
            return digest

        def fake_push_manifest(client, repo, tag, body):  # type: ignore[no-untyped-def]
            return registry.put_manifest(tag, body)

        def fake_validate(client, repo, tag, expected):  # type: ignore[no-untyped-def]
            assert registry.manifest_digests.get(tag) == expected

        monkeypatch.setattr("oci_modelcar.pipeline.head_blob", fake_head_blob)
        monkeypatch.setattr(
            "oci_modelcar.pipeline.get_manifest_digest_at_tag", lambda *a, **k: None
        )
        monkeypatch.setattr("oci_modelcar.pipeline.fetch_manifest_at_tag", lambda *a, **k: None)
        monkeypatch.setattr("oci_modelcar.pipeline.StreamingBlobUpload", FakeStreaming)
        monkeypatch.setattr("oci_modelcar.pipeline.push_small_blob", fake_push_small_blob)
        monkeypatch.setattr("oci_modelcar.pipeline.push_manifest", fake_push_manifest)
        monkeypatch.setattr("oci_modelcar.pipeline.validate_manifest_tag", fake_validate)
        monkeypatch.setattr(
            "oci_modelcar.pipeline.shutil.disk_usage",
            lambda p: type("DU", (), {"free": 100 * 1024**3})(),
        )

        cfg = _make_cfg(target_dir, ignore_patterns=("consolidated*",))
        plog = PipelineLogger(quiet=True)
        fake_registry_client = MagicMock(target_repo="models/mistral")
        fake_registry_client.host = "registry.example.com"
        return (
            Pipeline(cfg, plog, downloader=downloader, registry_client=fake_registry_client),
            registry,
        )

    run_a = tmp_path / "a"
    run_a.mkdir()
    pipeline_a, _ = make_pipeline(run_a)
    result_a = pipeline_a.run()

    run_b = tmp_path / "b"
    run_b.mkdir()
    pipeline_b, _ = make_pipeline(run_b)
    result_b = pipeline_b.run()

    assert result_a.manifest_digest == result_b.manifest_digest
