"""Tests for ProgressEmitter and _fmt_bytes."""

from __future__ import annotations

from oci_modelcar.logging import ProgressEmitter, _fmt_bytes


def test_fmt_bytes_scales_to_kb_mb_gb() -> None:
    assert _fmt_bytes(0) == "0 B"
    assert _fmt_bytes(999) == "999 B"
    assert _fmt_bytes(1500) == "1.5 KB"
    assert _fmt_bytes(2_500_000) == "2.5 MB"
    assert _fmt_bytes(3_250_000_000) == "3.25 GB"


def test_progress_throttles_to_interval() -> None:
    """Update calls within `interval` since last emit are dropped."""
    emitted: list[str] = []
    fake_clock = [0.0]
    emitter = ProgressEmitter(
        emit=emitted.append,
        path="weights.safetensors",
        total=1_000_000,
        interval=5.0,
        clock=lambda: fake_clock[0],
    )

    fake_clock[0] = 1.0
    emitter.update(100_000)
    assert emitted == [], "should not emit within interval"

    fake_clock[0] = 4.9
    emitter.update(490_000)
    assert emitted == [], "still within first interval"

    fake_clock[0] = 5.5
    emitter.update(550_000)
    assert len(emitted) == 1, "should emit after interval elapsed"

    fake_clock[0] = 7.0
    emitter.update(700_000)
    assert len(emitted) == 1, "second emit too soon after first"

    fake_clock[0] = 11.0
    emitter.update(990_000)
    assert len(emitted) == 2, "second emit after another interval"


def test_progress_line_format() -> None:
    """Emitted line carries path, percentage, transferred, total."""
    emitted: list[str] = []
    fake_clock = [0.0]
    emitter = ProgressEmitter(
        emit=emitted.append,
        path="model.safetensors",
        total=2_000_000_000,  # 2 GB
        interval=1.0,
        clock=lambda: fake_clock[0],
    )

    fake_clock[0] = 2.0
    emitter.update(500_000_000)  # 500 MB → 25%

    assert len(emitted) == 1
    line = emitted[0]
    assert "model.safetensors" in line
    assert "25%" in line
    assert "500.0 MB" in line
    assert "2.00 GB" in line


def test_progress_zero_total_does_not_divide_by_zero() -> None:
    emitted: list[str] = []
    fake_clock = [0.0]
    emitter = ProgressEmitter(
        emit=emitted.append,
        path="empty.bin",
        total=0,
        interval=1.0,
        clock=lambda: fake_clock[0],
    )
    fake_clock[0] = 2.0
    emitter.update(0)
    assert len(emitted) == 1
    assert "0%" in emitted[0]
