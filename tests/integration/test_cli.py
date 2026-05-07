import json
import subprocess
import sys
from pathlib import Path


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


def _write_state(tmp_path: Path, with_image_ref_digest: bool) -> Path:
    state_path = tmp_path / "state.json"
    job: dict = {
        "source": {
            "hf_repo": "foo/bar",
            "hf_revision_input": "main",
            "hf_revision_resolved": "a" * 40,
        },
        "target": {
            "registry": "r.example",
            "repo": "m/x",
            "tag": "abcd1234",
            "also_tags": [],
        },
        "started_at": "2026-05-07T12:00:00Z",
        "updated_at": "2026-05-07T12:00:00Z",
        "completed_at": "2026-05-07T12:01:00Z",
        "manifest_digest": "sha256:" + "f" * 64,
        "files": {},
    }
    if with_image_ref_digest:
        job["image_ref_digest"] = "r.example/m/x@sha256:" + "f" * 64
    state_path.write_text(json.dumps({"version": 1, "jobs": {"abc123def456": job}}))
    return state_path


def test_status_shows_image_ref_digest_when_present(tmp_path: Path):
    state_path = _write_state(tmp_path, with_image_ref_digest=True)
    proc = subprocess.run(
        [sys.executable, "-m", "oci_modelcar", "status", "--state-file", str(state_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ref=r.example/m/x@sha256:" in proc.stdout


def test_status_graceful_when_image_ref_digest_missing(tmp_path: Path):
    state_path = _write_state(tmp_path, with_image_ref_digest=False)
    proc = subprocess.run(
        [sys.executable, "-m", "oci_modelcar", "status", "--state-file", str(state_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "job=abc123def456" in proc.stdout
    assert "ref=" not in proc.stdout
