"""Integration tests for Pipeline failure modes: fail-fast cancellation and
continue-on-error partial failure collection."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from oci_modelcar.config import Config
from oci_modelcar.download import HfFile
from oci_modelcar.errors import PartialFailureError
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.manifest import ML_TAR, BlobDescriptor
from oci_modelcar.pipeline import FileWorker, Pipeline


def _make_cfg(tmp_path, **overrides):  # type: ignore[no-untyped-def]
    base: dict = dict(
        hf_repo="foo/bar",
        registry="registry.example.com",
        target_repo="models/x",
        hf_revision="main",
        hf_endpoint="https://huggingface.co",
        target_tag=None,
        also_tags=[],
        allow_patterns=(".bin",),
        layer_prefix="models/",
        workers=2,
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
    return Config(**base)


def test_fail_fast_cancellation_within_seconds(tmp_path, monkeypatch):
    """Worker f0 raises immediately; f1..f7 simulate long downloads but
    must abort within ~5s of stop_event being set."""
    cfg = _make_cfg(tmp_path, workers=2)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [HfFile(f"f{i}.bin", 100, None) for i in range(8)]
    fake_registry = MagicMock(target_repo="models/x")

    # Patch get_manifest_digest_at_tag so no real HTTP is made
    monkeypatch.setattr(
        "oci_modelcar.pipeline.get_manifest_digest_at_tag",
        lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    def fake_process(
        self: FileWorker,
        repo: str,
        revision: str,
        hf_file: HfFile,
        progress_cb=None,  # type: ignore[no-untyped-def]
    ) -> BlobDescriptor:
        if hf_file.path == "f0.bin":
            raise RuntimeError("boom")
        for _ in range(50):
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError("stop_event")
            time.sleep(0.05)
        return BlobDescriptor(media_type=ML_TAR, digest="sha256:x", size=100, hf_path=hf_file.path)

    monkeypatch.setattr(FileWorker, "process", fake_process)

    plog = PipelineLogger(quiet=True)
    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run()
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"fail-fast took too long: {elapsed:.1f}s"


def test_continue_on_error_collects_and_raises_partial(tmp_path, monkeypatch):
    """With fail_fast=False, all files are attempted; a PartialFailureError
    is raised summarising the failures."""
    cfg = _make_cfg(tmp_path, workers=2, fail_fast=False)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [
        HfFile("ok.bin", 100, None),
        HfFile("bad.bin", 100, None),
    ]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.get_manifest_digest_at_tag",
        lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    def fake_process(
        self: FileWorker,
        repo: str,
        revision: str,
        hf_file: HfFile,
        progress_cb=None,  # type: ignore[no-untyped-def]
    ) -> BlobDescriptor:
        if hf_file.path == "bad.bin":
            raise RuntimeError("bad")
        return BlobDescriptor(media_type=ML_TAR, digest="sha256:x", size=100, hf_path=hf_file.path)

    monkeypatch.setattr(FileWorker, "process", fake_process)

    plog = PipelineLogger(quiet=True)
    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)
    with pytest.raises(PartialFailureError):
        pipeline.run()
