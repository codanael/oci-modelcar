"""Tests for run_push per-file logging behavior (mono and multi worker)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from oci_modelcar import runner
from oci_modelcar.config import Config
from oci_modelcar.hf import HfClient, HfFile
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.oci import ML_TAR, BlobDescriptor
from oci_modelcar.runner import FileTelemetry

_SHA = "a" * 40


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, files: list[HfFile]) -> None:
    """Bypass network: mock HF + OCI calls so run_push only exercises orchestration."""
    monkeypatch.setattr(HfClient, "resolve_revision", lambda self, rev: _SHA)
    monkeypatch.setattr(HfClient, "list_files", lambda self, rev, allow: list(files))

    def fake_process_one_file(*args, **kwargs):  # type: ignore[no-untyped-def]
        hf_file = kwargs["hf_file"]
        progress_cb = kwargs.get("progress_cb")
        if progress_cb is not None:
            progress_cb(hf_file.size // 2)
            progress_cb(hf_file.size)
        digest = f"sha256:{'b' * 60}{ord(hf_file.path[0]):04x}"
        descriptor = BlobDescriptor(media_type=ML_TAR, digest=digest, size=hf_file.size + 1024)
        telemetry = FileTelemetry(
            bytes_through=hf_file.size,
            producer_wait_s=0.0,
            consumer_wait_s=0.0,
            elapsed_s=0.1,
        )
        return descriptor, digest, telemetry

    monkeypatch.setattr(runner, "process_one_file", fake_process_one_file)
    monkeypatch.setattr(runner, "push_small_blob", lambda *a, **kw: "sha256:" + "c" * 64)
    monkeypatch.setattr(runner, "push_manifest", lambda *a, **kw: "sha256:" + "d" * 64)
    monkeypatch.setattr(runner, "head_blob", lambda *a, **kw: {"digest": "x", "size": 0})
    monkeypatch.setattr(runner, "validate_manifest_tag", lambda *a, **kw: None)


def _run(cfg: Config) -> str:
    buf = io.StringIO()
    plog = PipelineLogger(stream=buf, style="text", use_color=False)
    runner.run_push(cfg, plog)
    return buf.getvalue()


def test_multi_worker_emits_per_file_header_and_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [
        HfFile(path="config.json", size=200),
        HfFile(path="weights/model.safetensors", size=1_000_000_000),
    ]
    _patch_pipeline(monkeypatch, files)

    cfg = Config(
        hf_repo="foo/bar",
        registry="http://reg.example.com",
        target_repo="m/x",
        hf_revision=_SHA,
        state_file=tmp_path / "state.json",
        workers=2,
    )
    out = _run(cfg)

    # Headers (sorted alphabetically by path)
    assert "[  1/2] config.json" in out
    assert "[  2/2] weights/model.safetensors" in out
    # Per-file completion line, prefixed by path
    assert "config.json: -> sha256:" in out
    assert "weights/model.safetensors: -> sha256:" in out


def test_multi_worker_emits_intra_file_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ProgressEmitter is wired so progress_cb calls emit '<path>: NN%' lines."""
    files = [HfFile(path="big.bin", size=2_000_000_000)]  # 2 GB
    _patch_pipeline(monkeypatch, files)

    # Force interval=0 so every progress_cb call emits, regardless of clock
    real_emitter_cls = runner.ProgressEmitter

    def make_emitter(*, emit, path, total, interval, clock=None):  # type: ignore[no-untyped-def]
        return real_emitter_cls(emit=emit, path=path, total=total, interval=0.0)

    monkeypatch.setattr(runner, "ProgressEmitter", make_emitter)

    cfg = Config(
        hf_repo="foo/bar",
        registry="http://reg.example.com",
        target_repo="m/x",
        hf_revision=_SHA,
        state_file=tmp_path / "state.json",
        workers=2,
    )
    out = _run(cfg)

    # Fake process_one_file calls progress_cb(size/2) then progress_cb(size)
    # → 50% line and 100% line
    assert "big.bin: 50%" in out
    assert "big.bin: 100%" in out
    assert "1.00 GB / 2.00 GB" in out


def test_keyboard_interrupt_sets_stop_event_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the main thread takes a KeyboardInterrupt during as_completed,
    run_push sets the shared stop_event so in-flight workers can short-circuit,
    and re-raises so the CLI handler exits 130."""
    files = [HfFile(path=f"f{i}.bin", size=100) for i in range(4)]
    monkeypatch.setattr(HfClient, "resolve_revision", lambda self, rev: _SHA)
    monkeypatch.setattr(HfClient, "list_files", lambda self, rev, allow: list(files))

    seen_stop_events: list[object] = []

    def fake_process_one_file(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen_stop_events.append(kwargs.get("stop_event"))
        digest = "sha256:" + "0" * 64
        descriptor = BlobDescriptor(media_type=ML_TAR, digest=digest, size=100)
        telemetry = FileTelemetry(
            bytes_through=100,
            producer_wait_s=0.0,
            consumer_wait_s=0.0,
            elapsed_s=0.01,
        )
        return descriptor, digest, telemetry

    monkeypatch.setattr(runner, "process_one_file", fake_process_one_file)
    monkeypatch.setattr(runner, "push_small_blob", lambda *a, **kw: "sha256:" + "c" * 64)
    monkeypatch.setattr(runner, "push_manifest", lambda *a, **kw: "sha256:" + "d" * 64)
    monkeypatch.setattr(runner, "head_blob", lambda *a, **kw: {"digest": "x", "size": 0})
    monkeypatch.setattr(runner, "validate_manifest_tag", lambda *a, **kw: None)

    # Force KeyboardInterrupt the first time as_completed is iterated.
    def fake_as_completed(futs):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "as_completed", fake_as_completed)

    cfg = Config(
        hf_repo="foo/bar",
        registry="http://reg.example.com",
        target_repo="m/x",
        hf_revision=_SHA,
        state_file=tmp_path / "state.json",
        workers=4,
    )
    with pytest.raises(KeyboardInterrupt):
        _run(cfg)

    # Workers received a non-None stop_event, and after KeyboardInterrupt it is set.
    non_none = [s for s in seen_stop_events if s is not None]
    assert non_none, "no worker received a stop_event"
    assert all(s.is_set() for s in non_none), "stop_event not set after KeyboardInterrupt"


def test_mono_worker_emits_per_file_header_and_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [
        HfFile(path="a.bin", size=100),
        HfFile(path="b.bin", size=200),
    ]
    _patch_pipeline(monkeypatch, files)

    cfg = Config(
        hf_repo="foo/bar",
        registry="http://reg.example.com",
        target_repo="m/x",
        hf_revision=_SHA,
        state_file=tmp_path / "state.json",
        workers=1,
    )
    out = _run(cfg)

    assert "[  1/2] a.bin" in out
    assert "[  2/2] b.bin" in out
    assert "a.bin: -> sha256:" in out
    assert "b.bin: -> sha256:" in out
