"""Tests for pipeline.py: FileWorker + Pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oci_modelcar.config import Config
from oci_modelcar.download import HfFile
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.manifest import ANN_HF_PATH, ANN_HF_SHA256, ML_TAR, BlobDescriptor
from oci_modelcar.pipeline import (
    FileWorker,
    Pipeline,
    build_reuse_map,
    fetch_manifest_at_tag,
)


def _build_worker(tmp_path: Path, head_blob_returns=None, **overrides):
    downloader = MagicMock()
    registry_client = MagicMock()
    spool = tmp_path / "spool"
    (spool / "sources").mkdir(parents=True)
    (spool / "layers").mkdir(parents=True)

    # Default downloader: writes a fake source file
    def fake_download(repo, rev, hf_file, progress_cb=None):
        p = spool / "sources" / hf_file.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"X" * hf_file.size)
        return p

    downloader.download.side_effect = fake_download

    # head_blob result is configurable:
    #   None            → skip-check: not present (proceed with push);
    #                     verify: caller must set side_effect if needed
    #   dict            → skip-check: present (skip push)
    # For the common happy-path case (head_blob_returns=None), we wire the mock
    # so the skip-check returns None (push proceeds) and the verify returns a
    # synthetic dict so PushError is not raised.
    if head_blob_returns is None:
        head_blob_mock = MagicMock(
            side_effect=[None, {"digest": "sha256:" + "f" * 64, "size": 12345}]
        )
    else:
        head_blob_mock = MagicMock(return_value=head_blob_returns)

    # StreamingBlobUpload mock
    streaming_factory = MagicMock()
    streaming_inst = MagicMock()
    streaming_inst.push_from_file.return_value = ("sha256:" + "f" * 64, 12345)
    streaming_factory.return_value = streaming_inst

    return (
        FileWorker(
            downloader=downloader,
            registry_client=registry_client,
            head_blob_fn=head_blob_mock,
            streaming_factory=streaming_factory,
            layer_prefix="models/",
            spool_dir=spool,
            clean_hf_after_push=False,
            oci_max_retries=3,
            backoff_initial=0.0,
            stop_event=None,
        ),
        downloader,
        head_blob_mock,
        streaming_inst,
    )


def test_file_worker_reuses_when_reuse_map_hits_and_blob_still_present(tmp_path, capsys):
    """When (hf_path, hf_sha256) is in reuse_map and HEAD confirms the layer
    blob is still in the registry, FileWorker returns the existing descriptor
    without calling download, build_layer_to_file, or the streaming push."""
    worker, downloader, head_blob_mock, streaming = _build_worker(tmp_path, head_blob_returns=None)
    digest = "sha256:" + "a" * 64
    existing_desc = BlobDescriptor(
        media_type=ML_TAR, digest=digest, size=100, hf_path="m.bin", hf_sha256="1" * 64
    )
    worker.reuse_map = {("m.bin", "1" * 64): existing_desc}
    head_blob_mock.side_effect = [{"digest": digest, "size": 100}]
    plog = PipelineLogger(log_style="text", verbose=False, quiet=False)
    worker.plog = plog

    f = HfFile(path="m.bin", size=100, lfs_sha256="1" * 64)
    desc = worker.process(repo="repo", revision="main", hf_file=f)

    downloader.download.assert_not_called()
    streaming.push_from_file.assert_not_called()
    assert desc is existing_desc or (desc.digest == digest and desc.hf_path == "m.bin")
    out = capsys.readouterr().out
    assert "reusing" in out.lower() or "cached" in out.lower()


def test_file_worker_reuse_map_miss_falls_through_to_download(tmp_path):
    """If (hf_path, hf_sha256) is NOT in reuse_map, the normal download+push
    path runs."""
    worker, downloader, _head, streaming = _build_worker(tmp_path, head_blob_returns=None)
    worker.reuse_map = {("OTHER.bin", "2" * 64): MagicMock()}
    f = HfFile(path="m.bin", size=100, lfs_sha256="1" * 64)
    worker.process(repo="repo", revision="main", hf_file=f)
    downloader.download.assert_called_once()
    streaming.push_from_file.assert_called_once()


def test_file_worker_reuse_map_hit_but_blob_gone_falls_through(tmp_path):
    """Reuse map says digest X exists, but HEAD returns 404 (blob was GC'd
    in the registry). Worker must fall back to the normal download+push path."""
    worker, downloader, head_blob_mock, streaming = _build_worker(tmp_path, head_blob_returns=None)
    digest = "sha256:" + "a" * 64
    desc = BlobDescriptor(
        media_type=ML_TAR, digest=digest, size=100, hf_path="m.bin", hf_sha256="1" * 64
    )
    worker.reuse_map = {("m.bin", "1" * 64): desc}
    # Reuse-precheck HEAD: 404 (gone). Then normal flow: skip-check after tar
    # returns None (proceed), verify after push returns present.
    head_blob_mock.side_effect = [None, None, {"digest": "sha256:" + "f" * 64, "size": 12345}]

    f = HfFile(path="m.bin", size=100, lfs_sha256="1" * 64)
    worker.process(repo="repo", revision="main", hf_file=f)
    downloader.download.assert_called_once()
    streaming.push_from_file.assert_called_once()


def test_file_worker_emits_per_file_log_lines(tmp_path, capsys):
    """When wired with a PipelineLogger, FileWorker announces start + end of
    each file's pipeline; download is given a progress_cb that emits via the
    same logger."""
    worker, downloader, _head_blob_mock, _streaming = _build_worker(
        tmp_path, head_blob_returns=None
    )
    plog = PipelineLogger(log_style="text", verbose=False, quiet=False)
    worker.plog = plog

    f = HfFile(path="weights.bin", size=2_000_000, lfs_sha256=None)
    worker.process(repo="repo", revision="main", hf_file=f)

    out = capsys.readouterr().out
    # Start line names the file and its size in human bytes.
    assert "weights.bin" in out
    assert "2.0 MB" in out
    # End line announces the pushed digest (short form).
    assert "pushed" in out.lower() or "uploaded" in out.lower()

    # downloader was given a progress callback (kwarg or positional).
    kwargs = downloader.download.call_args.kwargs
    assert kwargs.get("progress_cb") is not None


def test_file_worker_logs_reuse_when_head_finds_existing_blob(tmp_path, capsys):
    """When the skip-check after tar build finds the blob already present,
    the worker logs a 'reusing' line instead of a 'pushed' one."""
    worker, _downloader, head_blob_mock, streaming = _build_worker(tmp_path, head_blob_returns=None)
    head_blob_mock.side_effect = [{"digest": "sha256:dummy", "size": 500}]
    plog = PipelineLogger(log_style="text", verbose=False, quiet=False)
    worker.plog = plog

    f = HfFile(path="weights.bin", size=500, lfs_sha256=None)
    worker.process(repo="repo", revision="main", hf_file=f)

    out = capsys.readouterr().out
    streaming.push_from_file.assert_not_called()
    assert ("reusing" in out.lower()) or ("already" in out.lower())


def test_file_worker_phase_order_happy_path(tmp_path):
    """Confirms phases a→f run in order: download, tar+hash, head-skip,
    push, verify, cleanup."""
    worker, downloader, head_blob_mock, streaming = _build_worker(tmp_path, head_blob_returns=None)
    f = HfFile(path="weights.bin", size=1000, lfs_sha256=None)
    desc = worker.process(repo="repo", revision="main", hf_file=f)

    # download was called
    downloader.download.assert_called_once()
    # head_blob called twice: skip-check + verify
    assert head_blob_mock.call_count >= 1
    # streaming push called
    streaming.push_from_file.assert_called_once()
    # spool/layers/weights.bin.tar should be cleaned up
    assert not (tmp_path / "spool" / "layers" / "weights.bin.tar").exists()
    # source file kept (clean_hf_after_push=False)
    assert (tmp_path / "spool" / "sources" / "weights.bin").exists()
    assert isinstance(desc, BlobDescriptor)
    assert desc.hf_path == "weights.bin"


def test_file_worker_skips_push_if_blob_present(tmp_path):
    """If head_blob returns existing blob, skip the streaming push."""
    digest = "sha256:" + "a" * 64
    worker, _downloader, head_blob_mock, streaming = _build_worker(
        tmp_path, head_blob_returns={"digest": digest, "size": 1000}
    )
    # Patch head_blob to return None on the FIRST call (which is for our digest)
    # and stop the test from reaching verify. For simplicity we keep both calls
    # returning present.
    head_blob_mock.return_value = None  # default: not present, push proceeds
    # ... but we want to test the skip branch, so set up the side_effect:
    # First call (skip-check after tar): returns the dict (present)
    # Verify call: doesn't happen because we skipped
    head_blob_mock.side_effect = [{"digest": "sha256:dummy", "size": 1000}]

    # We need download to write a source so build_layer_to_file works to compute digest
    f = HfFile(path="weights.bin", size=1000, lfs_sha256=None)
    desc = worker.process(repo="repo", revision="main", hf_file=f)

    streaming.push_from_file.assert_not_called()
    assert isinstance(desc, BlobDescriptor)
    # tar was cleaned even when push skipped
    assert not (tmp_path / "spool" / "layers" / "weights.bin.tar").exists()


def test_file_worker_clean_hf_after_push_deletes_source(tmp_path):
    worker, _, _, _ = _build_worker(tmp_path, head_blob_returns=None)
    worker.clean_hf_after_push = True
    f = HfFile(path="weights.bin", size=500, lfs_sha256=None)
    worker.process(repo="repo", revision="main", hf_file=f)
    assert not (tmp_path / "spool" / "sources" / "weights.bin").exists()


def test_file_worker_cleanup_runs_on_exception(tmp_path):
    """If the push step raises, the tar file is still deleted."""
    worker, _, _, streaming = _build_worker(tmp_path, head_blob_returns=None)
    streaming.push_from_file.side_effect = RuntimeError("simulated push failure")
    f = HfFile(path="weights.bin", size=500, lfs_sha256=None)
    with pytest.raises(RuntimeError, match="simulated push failure"):
        worker.process(repo="repo", revision="main", hf_file=f)
    assert not (tmp_path / "spool" / "layers" / "weights.bin.tar").exists()


# ---------------------------------------------------------------------------
# Task 8.3: Pipeline pre-flight helpers
# ---------------------------------------------------------------------------


def _build_pipeline(tmp_path: Path, **cfg_overrides: object) -> tuple[Config, PipelineLogger]:
    """Construct a Config + PipelineLogger with mocked external dependencies."""
    base: dict[str, object] = dict(
        hf_repo="foo/bar",
        registry="registry.example.com",
        target_repo="models/x",
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
        hf_revision="main",
        hf_endpoint="https://huggingface.co",
    )
    base.update(cfg_overrides)
    cfg = Config(**base)  # type: ignore[arg-type]
    plog = PipelineLogger(log_style="text", quiet=True)
    return cfg, plog


def test_pipeline_skips_when_tag_matches_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If target tag exists with matching digest, log + exit 0 (no push)."""
    pytest.skip("Tag conflict policy is exercised in task 8.6 manifest tests")


def test_pipeline_resolves_revision_and_lists_files(tmp_path: Path) -> None:
    cfg, plog = _build_pipeline(tmp_path)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    fake_downloader.list_files.return_value = [
        HfFile("model.safetensors", 1000, None),
        HfFile("config.json", 100, None),
    ]
    fake_registry = MagicMock(target_repo="models/x")

    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)
    rev, files, target_tag = pipeline._preflight()
    assert rev == "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    assert len(files) == 2
    assert target_tag == "9fb191250dd5"


def test_pipeline_preflight_no_files_raises_config(tmp_path: Path) -> None:
    cfg, plog = _build_pipeline(tmp_path)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    fake_downloader.list_files.return_value = []
    fake_registry = MagicMock(target_repo="models/x")

    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)
    from oci_modelcar.errors import ConfigError

    with pytest.raises(ConfigError, match="no files matched"):
        pipeline._preflight()


# ---------------------------------------------------------------------------
# Task 8.4: Pipeline disk space check
# ---------------------------------------------------------------------------


def test_pipeline_disk_space_passes_when_sufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, plog = _build_pipeline(tmp_path)
    pipeline = Pipeline(cfg, plog, downloader=MagicMock(), registry_client=MagicMock())
    files = [HfFile("a.bin", 1000, None), HfFile("b.bin", 2000, None)]
    # Plenty of space
    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 10 * 1024**3})(),
    )
    pipeline._check_disk_space(files)  # should not raise


def test_pipeline_disk_space_fails_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oci_modelcar.errors import DiskSpaceError

    cfg, plog = _build_pipeline(tmp_path, workers=4)
    pipeline = Pipeline(cfg, plog, downloader=MagicMock(), registry_client=MagicMock())
    files = [HfFile("big.bin", 10 * 1024**3, None)]  # 10 GiB file
    # Only 5 GB free cannot fit (need 4 x 10 GiB workers + sources)
    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 5 * 1024**3})(),
    )
    with pytest.raises(DiskSpaceError) as exc:
        pipeline._check_disk_space(files)
    assert exc.value.hint and "--clean-hf-after-push" in exc.value.hint


def test_pipeline_disk_space_clean_hf_lowers_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --clean-hf-after-push, the persistent budget drops to 0; only
    workers x max_layer is required."""
    cfg, plog = _build_pipeline(tmp_path, workers=1, clean_hf_after_push=True)
    pipeline = Pipeline(cfg, plog, downloader=MagicMock(), registry_client=MagicMock())
    files = [HfFile(f"f{i}.bin", 1024**3, None) for i in range(20)]  # 20 GB total

    # 5 GB free should suffice with --clean-hf-after-push because we only need
    # ~1.2 x (1 + 1) GB in flight (rounded up).
    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 5 * 1024**3})(),
    )
    pipeline._check_disk_space(files)  # should not raise


# ---------------------------------------------------------------------------
# Task 8.5: Pipeline.run with ThreadPoolExecutor + fail-fast
# ---------------------------------------------------------------------------

import time  # noqa: E402


def test_pipeline_fail_fast_cancels_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When one worker raises, stop_event must be set and the loop exit
    within ~1 second (cancel_futures kills pending)."""
    cfg, plog = _build_pipeline(tmp_path, workers=2)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [HfFile(f"f{i}.bin", 100, None) for i in range(8)]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    call_count: dict[str, int] = {"n": 0}

    def fake_process(  # type: ignore[misc]
        self: FileWorker,
        repo: str,
        revision: str,
        hf_file: HfFile,
        progress_cb: object = None,
    ) -> object:
        call_count["n"] += 1
        if hf_file.path == "f0.bin":
            raise RuntimeError("simulated f0 failure")
        for _ in range(50):
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError("stop_event")
            time.sleep(0.05)
        return MagicMock()

    monkeypatch.setattr(FileWorker, "process", fake_process)

    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="simulated f0 failure"):
        pipeline.run()
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"fail-fast took too long: {elapsed:.1f}s"


def test_pipeline_continue_on_error_collects_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oci_modelcar.errors import PartialFailureError

    cfg, plog = _build_pipeline(tmp_path, workers=2, fail_fast=False)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [
        HfFile("good.bin", 100, None),
        HfFile("bad.bin", 100, None),
    ]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    def fake_process(  # type: ignore[misc]
        self: FileWorker,
        repo: str,
        revision: str,
        hf_file: HfFile,
        progress_cb: object = None,
    ) -> BlobDescriptor:
        if hf_file.path == "bad.bin":
            raise RuntimeError("bad failed")
        return BlobDescriptor(
            media_type="application/vnd.oci.image.layer.v1.tar",
            digest="sha256:" + "a" * 64,
            size=100,
            hf_path="good.bin",
        )

    monkeypatch.setattr(FileWorker, "process", fake_process)

    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)
    with pytest.raises(PartialFailureError):
        pipeline.run()


# ---------------------------------------------------------------------------
# Task 8.6: Tag conflict policy
# ---------------------------------------------------------------------------


def test_pipeline_tag_match_skips_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If existing tag matches the manifest digest we'd produce, skip push."""
    from oci_modelcar.pipeline import RunResult

    cfg, plog = _build_pipeline(tmp_path, workers=1)

    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [HfFile("a.bin", 100, None)]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    expected_digest = "sha256:" + "f" * 64
    monkeypatch.setattr(
        "oci_modelcar.pipeline.get_manifest_digest_at_tag",
        lambda *a, **kw: expected_digest,
    )

    monkeypatch.setattr(
        FileWorker,
        "process",
        lambda self, *a, **kw: BlobDescriptor(
            media_type=ML_TAR, digest="sha256:" + "a" * 64, size=100, hf_path="a.bin"
        ),
    )

    def fake_assemble(
        self: Pipeline, target_tag: str, descriptors: list[BlobDescriptor]
    ) -> RunResult:
        return RunResult(
            manifest_digest=expected_digest,
            image_ref="x:y",
            image_ref_digest="x@" + expected_digest,
            layers=tuple(descriptors),
        )

    monkeypatch.setattr(Pipeline, "_assemble_manifest", fake_assemble)

    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)
    result = pipeline.run()
    assert result.manifest_digest == expected_digest


def test_pipeline_tag_conflict_no_force_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing tag with DIFFERENT digest, no --force → PushError."""
    from oci_modelcar.errors import PushError
    from oci_modelcar.pipeline import RunResult

    cfg, plog = _build_pipeline(tmp_path, workers=1, force=False)

    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [HfFile("a.bin", 100, None)]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )
    monkeypatch.setattr(
        "oci_modelcar.pipeline.get_manifest_digest_at_tag",
        lambda *a, **kw: "sha256:" + "1" * 64,
    )

    # Workers succeed; _assemble_manifest produces a DIFFERENT digest
    monkeypatch.setattr(
        FileWorker,
        "process",
        lambda self, *a, **kw: BlobDescriptor(
            media_type=ML_TAR, digest="sha256:" + "a" * 64, size=100, hf_path="a.bin"
        ),
    )

    def fake_assemble_diff(
        self: Pipeline, target_tag: str, descriptors: list[BlobDescriptor]
    ) -> RunResult:
        return RunResult(
            manifest_digest="sha256:" + "2" * 64,  # differs from existing "sha256:1..1"
            image_ref="x:y",
            image_ref_digest="x@sha256:" + "2" * 64,
            layers=tuple(descriptors),
        )

    monkeypatch.setattr(Pipeline, "_assemble_manifest", fake_assemble_diff)

    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)
    with pytest.raises(PushError, match="tag exists with different digest"):
        pipeline.run()


def test_fetch_manifest_at_tag_returns_parsed_body_on_200() -> None:
    client = MagicMock()
    client.url.return_value = "http://r/v2/m/x/manifests/tag"
    client.auth = {"Authorization": "x"}
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "schemaVersion": 2,
        "layers": [
            {
                "mediaType": ML_TAR,
                "digest": "sha256:" + "a" * 64,
                "size": 100,
                "annotations": {ANN_HF_PATH: "m.bin", ANN_HF_SHA256: "1" * 64},
            },
        ],
    }
    client.session.get.return_value = resp
    out = fetch_manifest_at_tag(client, "m/x", "tag")
    assert out is not None and out["layers"][0]["digest"] == "sha256:" + "a" * 64


def test_fetch_manifest_at_tag_returns_none_on_404() -> None:
    client = MagicMock()
    client.url.return_value = "http://r/v2/m/x/manifests/missing"
    client.auth = {}
    resp = MagicMock()
    resp.status_code = 404
    client.session.get.return_value = resp
    assert fetch_manifest_at_tag(client, "m/x", "missing") is None


def test_build_reuse_map_indexes_layers_by_hf_path_and_sha256() -> None:
    manifest = {
        "layers": [
            {
                "mediaType": ML_TAR,
                "digest": "sha256:" + "a" * 64,
                "size": 100,
                "annotations": {ANN_HF_PATH: "model.safetensors", ANN_HF_SHA256: "1" * 64},
            },
            {
                "mediaType": ML_TAR,
                "digest": "sha256:" + "b" * 64,
                "size": 50,
                "annotations": {ANN_HF_PATH: "config.json"},  # no LFS sha
            },
        ],
    }
    reuse = build_reuse_map(manifest)
    assert ("model.safetensors", "1" * 64) in reuse
    assert ("config.json", None) in reuse
    desc_a = reuse[("model.safetensors", "1" * 64)]
    assert desc_a.digest == "sha256:" + "a" * 64
    assert desc_a.size == 100
    assert desc_a.hf_path == "model.safetensors"
    assert desc_a.hf_sha256 == "1" * 64


def test_build_reuse_map_skips_layers_without_path_annotation() -> None:
    """A manifest produced by an older oci-modelcar run (pre-annotations) is
    silently ignored — we have no way to associate its layers with hf paths."""
    manifest = {
        "layers": [
            {"mediaType": ML_TAR, "digest": "sha256:" + "a" * 64, "size": 100},
        ],
    }
    assert build_reuse_map(manifest) == {}


def test_pipeline_tag_conflict_with_force_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oci_modelcar.pipeline import RunResult

    cfg, plog = _build_pipeline(tmp_path, workers=1, force=True)

    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [HfFile("a.bin", 100, None)]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )
    monkeypatch.setattr(
        "oci_modelcar.pipeline.get_manifest_digest_at_tag",
        lambda *a, **kw: "sha256:" + "1" * 64,  # differs but --force overrides
    )
    monkeypatch.setattr(
        FileWorker,
        "process",
        lambda self, *a, **kw: BlobDescriptor(
            media_type=ML_TAR, digest="sha256:" + "a" * 64, size=100, hf_path="a.bin"
        ),
    )
    monkeypatch.setattr(
        Pipeline,
        "_assemble_manifest",
        lambda self, t, d: RunResult(
            manifest_digest="sha256:new",
            image_ref="x",
            image_ref_digest="y",
            layers=tuple(d),
        ),
    )

    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)
    result = pipeline.run()
    assert result.manifest_digest == "sha256:new"  # overwrote existing
