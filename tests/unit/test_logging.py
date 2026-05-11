import logging
import threading

from oci_modelcar.logging import (
    AzureFormatter,
    PipelineLogger,
    ProgressEmitter,
    TextFormatter,
    fmt_bytes,
)


def test_pipeline_logger_emits_to_stdout(capsys):
    plog = PipelineLogger(stream=None, log_style="text", verbose=False, quiet=False)
    plog.info("hello")
    out = capsys.readouterr().out
    assert "hello" in out


def test_pipeline_logger_section(capsys):
    plog = PipelineLogger(stream=None, log_style="text", verbose=False, quiet=False)
    plog.section("Phase 1")
    out = capsys.readouterr().out
    assert "Phase 1" in out


def test_pipeline_logger_quiet_suppresses_info(capsys):
    plog = PipelineLogger(stream=None, log_style="text", verbose=False, quiet=True)
    plog.info("hidden")
    out = capsys.readouterr().out
    assert "hidden" not in out


def test_pipeline_logger_verbose_includes_debug(capsys):
    plog = PipelineLogger(stream=None, log_style="text", verbose=True, quiet=False)
    plog.debug("verbose-detail")
    out = capsys.readouterr().out
    assert "verbose-detail" in out


def test_azure_format_uses_logging_command():
    fmt = AzureFormatter()
    record = logging.LogRecord(
        name="x",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="warn-text",
        args=(),
        exc_info=None,
    )
    out = fmt.format(record)
    assert "##[warning]" in out


def test_pipeline_logger_output_variable_azure(capsys):
    plog = PipelineLogger(stream=None, log_style="azure", verbose=False, quiet=False)
    plog.output_variable("manifestDigest", "sha256:abc")
    out = capsys.readouterr().out
    assert "task.setvariable" in out and "manifestDigest" in out


def test_text_formatter_returns_message():
    fmt = TextFormatter()
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    assert fmt.format(record) == "hello world"


def test_pipeline_logger_quiet_does_not_suppress_warning(capsys):
    plog = PipelineLogger(stream=None, log_style="text", verbose=False, quiet=True)
    plog.warning("important")
    out = capsys.readouterr().out
    assert "important" in out


def test_pipeline_logger_output_variable_text(capsys):
    plog = PipelineLogger(stream=None, log_style="text", verbose=False, quiet=False)
    plog.output_variable("digest", "sha256:abc")
    out = capsys.readouterr().out
    assert "digest=sha256:abc" in out


def test_pipeline_logger_section_azure(capsys):
    plog = PipelineLogger(stream=None, log_style="azure", verbose=False, quiet=False)
    plog.section("Resolve")
    out = capsys.readouterr().out
    assert "##[section]Resolve" in out


def test_fmt_bytes_scales_units():
    assert fmt_bytes(0) == "0 B"
    assert fmt_bytes(512) == "512 B"
    assert fmt_bytes(1500) == "1.5 KB"
    assert fmt_bytes(2_500_000) == "2.5 MB"
    assert fmt_bytes(3_200_000_000) == "3.20 GB"


def test_progress_emitter_throttles_to_interval():
    """Emits only when at least `interval` seconds elapsed since the last emit."""
    out: list[str] = []
    fake_now = [0.0]
    emit = ProgressEmitter(
        emit=out.append,
        path="model.safetensors",
        total=1000,
        interval=5.0,
        clock=lambda: fake_now[0],
    )
    # First call at t=0 — primes _last; should not emit yet.
    emit.update(100)
    assert out == []
    # Still within interval — no emit.
    fake_now[0] = 4.0
    emit.update(400)
    assert out == []
    # Past interval — one emit.
    fake_now[0] = 6.0
    emit.update(600)
    assert len(out) == 1
    assert "model.safetensors" in out[0]
    assert "60" in out[0]  # percent
    # Next call right after — throttled.
    fake_now[0] = 7.0
    emit.update(700)
    assert len(out) == 1


def test_progress_emitter_formats_human_bytes():
    out: list[str] = []
    fake_now = [0.0]
    emit = ProgressEmitter(
        emit=out.append,
        path="a.bin",
        total=2_000_000_000,
        interval=1.0,
        clock=lambda: fake_now[0],
    )
    emit.update(0)
    fake_now[0] = 2.0
    emit.update(1_000_000_000)
    assert len(out) == 1
    assert "1.00 GB" in out[0]
    assert "2.00 GB" in out[0]
    assert "50%" in out[0]


def test_progress_emitter_zero_total_does_not_divide():
    out: list[str] = []
    fake_now = [0.0]
    emit = ProgressEmitter(
        emit=out.append, path="x", total=0, interval=0.1, clock=lambda: fake_now[0]
    )
    fake_now[0] = 1.0
    emit.update(0)
    assert len(out) == 1
    assert "0%" in out[0]


def test_pipeline_logger_info_is_thread_safe():
    """Concurrent .info() writes never interleave within a line."""
    import io

    buf = io.StringIO()
    plog = PipelineLogger(stream=buf, log_style="text", verbose=False, quiet=False)

    def worker(tag: str) -> None:
        for i in range(50):
            plog.info(f"{tag}-{i}-XXXXXXXXXXXXXXXX")

    threads = [threading.Thread(target=worker, args=(c,)) for c in "ABCDE"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = buf.getvalue().splitlines()
    assert len(lines) == 5 * 50
    for line in lines:
        # Each line must be one tag's full message, never split.
        assert line.startswith(("A-", "B-", "C-", "D-", "E-"))
