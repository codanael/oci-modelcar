import logging

from oci_modelcar.logging import (
    AzureFormatter,
    PipelineLogger,
    TextFormatter,
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
