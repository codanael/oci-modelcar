"""Gap C: --clean-hf-after-push.

After a successful push + HEAD-confirm, source files under <spool>/sources/
must be deleted. Tar layers under <spool>/layers/ are always cleaned.

Verifies:
- spool/layers/ contains no .tar files after push
- spool/sources/ contains no source files after push (--clean-hf-after-push)
- Without --clean-hf-after-push, source files are retained

Requires Docker + network access to huggingface.co.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

HF_REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HF_REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_registry(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{port}/v2/", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"registry on port {port} did not become healthy within {timeout}s")


@pytest.fixture(scope="module")
def clean_registry():  # type: ignore[no-untyped-def]
    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        pytest.skip("docker not available")

    port = _free_port()
    name = f"oci-modelcar-clean-reg-{port}"
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name, "-p", f"{port}:5000", "registry:2"],
        check=True,
        capture_output=True,
    )
    _wait_registry(port)
    yield port
    subprocess.run(["docker", "stop", name], check=False, capture_output=True)


def _find_files(directory: Path) -> list[Path]:
    """Return all regular files under directory (recursive)."""
    if not directory.exists():
        return []
    return [p for p in directory.rglob("*") if p.is_file()]


@pytest.mark.e2e
def test_clean_hf_after_push_removes_sources(clean_registry: int, tmp_path: Path) -> None:
    """With --clean-hf-after-push: sources/ must be empty after successful push.

    Tar layers are always cleaned (even without the flag).
    """
    spool = tmp_path / "spool_clean"
    env = os.environ.copy()
    cmd = [
        "oci-modelcar",
        "push",
        "--hf-repo",
        HF_REPO,
        "--hf-revision",
        HF_REVISION,
        "--registry",
        f"localhost:{clean_registry}",
        "--target-repo",
        "e2e/clean-llama",
        "--spool-dir",
        str(spool),
        "--clean-hf-after-push",
        "--quiet",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (
        f"push with --clean-hf-after-push failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # spool/layers/ must have no .tar files
    layers_dir = spool / "layers"
    leftover_tars = _find_files(layers_dir) if layers_dir.exists() else []
    assert leftover_tars == [], (
        "tar files left in spool/layers after push (layers always cleaned):\n"
        + "\n".join(str(p) for p in leftover_tars)
    )

    # spool/sources/ must have no source files (--clean-hf-after-push)
    sources_dir = spool / "sources"
    leftover_sources = _find_files(sources_dir) if sources_dir.exists() else []
    assert leftover_sources == [], (
        "source files left in spool/sources after --clean-hf-after-push:\n"
        + "\n".join(str(p) for p in leftover_sources)
        + "\nExpected all sources to be deleted after successful push."
    )


@pytest.mark.e2e
def test_without_clean_sources_are_retained(clean_registry: int, tmp_path: Path) -> None:
    """Without --clean-hf-after-push: source files must be retained in spool/sources/.

    This is a control test confirming the flag's effect is observable.
    """
    spool = tmp_path / "spool_retain"
    env = os.environ.copy()
    cmd = [
        "oci-modelcar",
        "push",
        "--hf-repo",
        HF_REPO,
        "--hf-revision",
        HF_REVISION,
        "--registry",
        f"localhost:{clean_registry}",
        "--target-repo",
        "e2e/retain-llama",
        "--spool-dir",
        str(spool),
        # Note: NO --clean-hf-after-push
        "--quiet",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (
        f"push without --clean-hf-after-push failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # spool/layers/ must have no .tar files (always cleaned)
    layers_dir = spool / "layers"
    leftover_tars = _find_files(layers_dir) if layers_dir.exists() else []
    assert leftover_tars == [], (
        "tar files left in spool/layers (always cleaned, flag not needed):\n"
        + "\n".join(str(p) for p in leftover_tars)
    )

    # spool/sources/ must retain the downloaded files (flag not set)
    sources_dir = spool / "sources"
    retained_sources = _find_files(sources_dir) if sources_dir.exists() else []
    assert retained_sources != [], (
        "spool/sources/ is empty despite NOT using --clean-hf-after-push. "
        "Either the download didn't happen or sources are being cleaned unconditionally. "
        "This is a bug."
    )
