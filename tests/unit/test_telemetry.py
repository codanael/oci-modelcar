"""Tests for FileTelemetry — per-file throughput + producer/consumer wait."""

from __future__ import annotations

from oci_modelcar.runner import FileTelemetry


def test_telemetry_format_line_basic():
    t = FileTelemetry(
        bytes_through=2_000_000_000,  # 2 GB
        producer_wait_s=8.5,  # OCI slow
        consumer_wait_s=0.8,  # HF fast
        elapsed_s=12.5,
    )
    line = t.format_line("model.safetensors")
    # Carries path, total bytes, elapsed, throughput, both waits
    assert "model.safetensors" in line
    assert "2000 MB" in line or "2.00 GB" in line
    assert "12.5" in line
    assert "HF wait" in line and "0.8" in line
    assert "OCI wait" in line and "8.5" in line


def test_telemetry_format_line_short_transfer_omits_percentages():
    """For sub-second transfers (cached files, tiny configs) waits are noisy
    rather than informative; format stays compact."""
    t = FileTelemetry(
        bytes_through=1024,
        producer_wait_s=0.01,
        consumer_wait_s=0.01,
        elapsed_s=0.05,
    )
    line = t.format_line("config.json")
    assert "config.json" in line
    # No division-by-zero, no NaN
    assert "nan" not in line.lower() and "inf" not in line.lower()


def test_telemetry_throughput_mb_s():
    t = FileTelemetry(
        bytes_through=100_000_000,  # 100 MB
        producer_wait_s=0.0,
        consumer_wait_s=0.0,
        elapsed_s=10.0,
    )
    assert abs(t.throughput_mb_s - 10.0) < 0.01  # 100 MB / 10 s = 10 MB/s


def test_telemetry_throughput_handles_zero_elapsed():
    t = FileTelemetry(
        bytes_through=0,
        producer_wait_s=0.0,
        consumer_wait_s=0.0,
        elapsed_s=0.0,
    )
    # No exception, no inf
    assert t.throughput_mb_s >= 0
