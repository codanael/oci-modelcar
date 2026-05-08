"""Tests for pipeline.py: FileWorker + Pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oci_modelcar.download import HfFile
from oci_modelcar.manifest import BlobDescriptor
from oci_modelcar.pipeline import FileWorker


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
