import subprocess
import sys


def test_cli_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "oci_modelcar", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "oci-modelcar" in proc.stdout


def test_cli_version_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "oci_modelcar", "--version"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0


def test_cli_push_missing_required_returns_64():
    proc = subprocess.run(
        [sys.executable, "-m", "oci_modelcar", "push"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 64
