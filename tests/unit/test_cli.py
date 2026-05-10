from unittest.mock import MagicMock

import pytest

from oci_modelcar.cli import main


def test_cli_no_args_shows_usage(capsys):
    rc = main(["oci-modelcar"])
    out = capsys.readouterr()
    assert rc == 1
    assert "usage:" in out.err
    assert out.out == ""


def test_cli_unknown_subcommand(capsys):
    rc = main(["oci-modelcar", "nope"])
    out = capsys.readouterr()
    assert rc == 1
    assert "unknown sub-command" in out.err


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_cli_top_level_help_exits_zero(flag, capsys):
    rc = main(["oci-modelcar", flag])
    out = capsys.readouterr()
    assert rc == 0
    assert "usage:" in out.out
    assert "{push,status,validate}" in out.out
    assert "push --help" in out.out
    assert out.err == ""


def test_cli_push_help_argparse(capsys):
    rc = main(["oci-modelcar", "push", "--help"])
    out = capsys.readouterr()
    assert rc == 0
    assert "usage:" in out.out
    assert "--hf-repo" in out.out


def test_cli_push_dispatches_to_pipeline(monkeypatch):
    fake_pipeline_inst = MagicMock()
    fake_pipeline_inst.run = MagicMock(return_value=MagicMock(manifest_digest="sha256:x"))
    fake_pipeline_cls = MagicMock(return_value=fake_pipeline_inst)
    monkeypatch.setattr("oci_modelcar.cli.Pipeline", fake_pipeline_cls)
    monkeypatch.setattr("oci_modelcar.cli.HfApi", MagicMock())
    monkeypatch.setattr("oci_modelcar.cli.HfDownloader", MagicMock())
    monkeypatch.setattr("oci_modelcar.cli.OciClient", MagicMock())
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")

    rc = main(["oci-modelcar", "push"])
    assert rc == 0
    fake_pipeline_inst.run.assert_called_once()


def test_cli_push_config_error_exits_2(monkeypatch):
    monkeypatch.delenv("HF_REPO", raising=False)
    monkeypatch.delenv("REGISTRY", raising=False)
    monkeypatch.delenv("TARGET_REPO", raising=False)
    rc = main(["oci-modelcar", "push"])
    assert rc == 2


def test_cli_push_gated_repo_exits_3(monkeypatch):
    from oci_modelcar.errors import GatedRepoError

    fake_pipe = MagicMock()
    fake_pipe.run.side_effect = GatedRepoError("gated", hint="accept terms")
    monkeypatch.setattr("oci_modelcar.cli.Pipeline", MagicMock(return_value=fake_pipe))
    monkeypatch.setattr("oci_modelcar.cli.HfApi", MagicMock())
    monkeypatch.setattr("oci_modelcar.cli.HfDownloader", MagicMock())
    monkeypatch.setattr("oci_modelcar.cli.OciClient", MagicMock())
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    rc = main(["oci-modelcar", "push"])
    assert rc == 3


def test_cli_push_disk_space_error_exits_4(monkeypatch):
    from oci_modelcar.errors import DiskSpaceError

    fake_pipe = MagicMock()
    fake_pipe.run.side_effect = DiskSpaceError("no space", hint="more disk")
    monkeypatch.setattr("oci_modelcar.cli.Pipeline", MagicMock(return_value=fake_pipe))
    monkeypatch.setattr("oci_modelcar.cli.HfApi", MagicMock())
    monkeypatch.setattr("oci_modelcar.cli.HfDownloader", MagicMock())
    monkeypatch.setattr("oci_modelcar.cli.OciClient", MagicMock())
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    rc = main(["oci-modelcar", "push"])
    assert rc == 4


def test_cli_push_partial_failure_exits_7(monkeypatch):
    from oci_modelcar.errors import PartialFailureError

    fake_pipe = MagicMock()
    fake_pipe.run.side_effect = PartialFailureError("2/5 failed")
    monkeypatch.setattr("oci_modelcar.cli.Pipeline", MagicMock(return_value=fake_pipe))
    monkeypatch.setattr("oci_modelcar.cli.HfApi", MagicMock())
    monkeypatch.setattr("oci_modelcar.cli.HfDownloader", MagicMock())
    monkeypatch.setattr("oci_modelcar.cli.OciClient", MagicMock())
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    rc = main(["oci-modelcar", "push"])
    assert rc == 7
