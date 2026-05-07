import io

from oci_modelcar.logging import (
    PipelineLogger,
    detect_log_style,
)


def test_detect_log_style_azure(monkeypatch):
    monkeypatch.setenv("TF_BUILD", "True")
    assert detect_log_style(None) == "azure"


def test_detect_log_style_text(monkeypatch):
    monkeypatch.delenv("TF_BUILD", raising=False)
    assert detect_log_style(None) == "text"


def test_detect_log_style_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("TF_BUILD", "True")
    assert detect_log_style("text") == "text"


def test_text_formatter_section():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="text", use_color=False)
    log.section("Resolving HuggingFace revision")
    rendered = out.getvalue()
    assert "Resolving HuggingFace revision" in rendered
    assert "##[" not in rendered


def test_azure_formatter_section():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="azure", use_color=False)
    log.section("Resolving HuggingFace revision")
    assert "##[section]Resolving HuggingFace revision" in out.getvalue()


def test_azure_formatter_group():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="azure", use_color=False)
    log.group_start("file1.safetensors")
    log.info("uploading...")
    log.group_end("done")
    rendered = out.getvalue()
    assert "##[group]file1.safetensors" in rendered
    assert "##[endgroup]" in rendered
    assert "uploading..." in rendered


def test_azure_formatter_warning_and_error():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="azure", use_color=False)
    log.warning("retry")
    log.error("fatal")
    rendered = out.getvalue()
    assert "##[warning]retry" in rendered
    assert "##[error]fatal" in rendered


def test_text_formatter_warning_prefix():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="text", use_color=False)
    log.warning("retry")
    log.error("fatal")
    rendered = out.getvalue()
    assert "WARN" in rendered
    assert "ERROR" in rendered
    assert "##[" not in rendered


def test_set_output_variable_azure():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="azure", use_color=False)
    log.output_variable("manifestDigest", "sha256:abc")
    rendered = out.getvalue()
    assert "##vso[task.setvariable variable=manifestDigest;isOutput=true]sha256:abc" in rendered
    # Plain KEY=VALUE line emitted in both styles
    assert "MANIFESTDIGEST=sha256:abc" in rendered


def test_set_output_variable_text():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="text", use_color=False)
    log.output_variable("manifestDigest", "sha256:abc")
    rendered = out.getvalue()
    assert "MANIFESTDIGEST=sha256:abc" in rendered
    assert "##vso[" not in rendered
