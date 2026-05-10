"""Gap E: status and validate sub-commands against real registry:2.

Tests:
1. status: lists tags with correct digests after push
2. validate happy path: exit 0, "coherent (N layers)" in output
3. validate detects missing blob: exit non-zero when a layer blob is deleted
4. status against non-existent repo: exit 0, friendly message

Special attention to:
- status JSON shape: registry:2 returns {"tags": [...]} — verify impl uses .get("tags", [])
- validate's head_blob returns None on 404 (not raise) — verify the discriminant works
- registry:2 caches blobs in memory; must restart container to flush after file deletion

Requires Docker + network access to huggingface.co.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

HF_REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HF_REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
EXPECTED_TAG = "9fb191250dd5"
MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"


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
def status_validate_setup(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    """Start a registry and push the pinned tiny llama once.

    Returns (port, reg_name) for use by all tests in this module.
    """
    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        pytest.skip("docker not available")

    port = _free_port()
    reg_name = f"oci-modelcar-status-reg-{port}"
    subprocess.run(["docker", "rm", "-f", reg_name], check=False, capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", reg_name, "-p", f"{port}:5000", "registry:2"],
        check=True,
        capture_output=True,
    )
    _wait_registry(port)

    # Initial push
    spool = tmp_path_factory.mktemp("spool_status")
    env = os.environ.copy()
    push_cmd = [
        "oci-modelcar",
        "push",
        "--hf-repo",
        HF_REPO,
        "--hf-revision",
        HF_REVISION,
        "--registry",
        f"localhost:{port}",
        "--target-repo",
        "e2e/status-llama",
        "--spool-dir",
        str(spool),
        "--quiet",
    ]
    result = subprocess.run(push_cmd, env=env, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        subprocess.run(["docker", "stop", reg_name], check=False, capture_output=True)
        pytest.fail(
            f"initial push failed (can't run status/validate tests):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    yield {"port": port, "reg_name": reg_name}

    subprocess.run(["docker", "stop", reg_name], check=False, capture_output=True)


@pytest.mark.e2e
def test_status_lists_pushed_tag(status_validate_setup: dict) -> None:  # type: ignore[type-arg]
    """oci-modelcar status lists the pushed tag with its manifest digest."""
    port = status_validate_setup["port"]
    result = subprocess.run(
        [
            "oci-modelcar",
            "status",
            "--registry",
            f"localhost:{port}",
            "--target-repo",
            "e2e/status-llama",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"status failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert EXPECTED_TAG in combined, (
        f"status output does not mention expected tag {EXPECTED_TAG!r}:\n{combined}"
    )
    # The digest should appear — starts with sha256:
    assert "sha256:" in combined, f"status output contains no sha256 digest:\n{combined}"


@pytest.mark.e2e
def test_validate_happy_path(status_validate_setup: dict) -> None:  # type: ignore[type-arg]
    """validate exits 0 with 'coherent (N layers)' for a valid push."""
    port = status_validate_setup["port"]
    result = subprocess.run(
        [
            "oci-modelcar",
            "validate",
            "--registry",
            f"localhost:{port}",
            "--target-repo",
            "e2e/status-llama",
            "--target-tag",
            EXPECTED_TAG,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"validate happy path failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    combined = result.stdout + result.stderr
    # Expect "coherent (N layers)" in the output
    assert re.search(r"coherent\s*\(\d+\s*layers\)", combined), (
        f"validate output does not contain 'coherent (N layers)':\n{combined}"
    )


@pytest.mark.e2e
def test_validate_detects_missing_blob(
    status_validate_setup: dict,  # type: ignore[type-arg]
    tmp_path: Path,
) -> None:
    """validate exits non-zero when a layer blob is manually deleted from registry.

    registry:2 caches blobs in memory — a file deletion alone does not cause
    HEAD to return 404. The container must be restarted to flush the cache.

    Sequence:
    1. GET the manifest; find a layer digest.
    2. Delete the blob data file inside the container.
    3. Restart the container to flush registry:2's in-memory cache.
    4. Confirm HEAD of the deleted blob returns 404.
    5. Run validate — must exit non-zero with an error about the missing layer.
    """
    port = status_validate_setup["port"]
    reg_name = status_validate_setup["reg_name"]

    # GET the manifest to find layer digests
    manifest_url = f"http://localhost:{port}/v2/e2e/status-llama/manifests/{EXPECTED_TAG}"
    r = requests.get(manifest_url, headers={"Accept": MANIFEST_TYPE}, timeout=30)
    assert r.status_code == 200, f"manifest GET failed: {r.status_code}"
    manifest = r.json()
    layers = manifest.get("layers", [])
    assert len(layers) > 0, "manifest has no layers"

    # Pick the first layer to delete
    victim_digest = layers[0]["digest"]
    # digest format: sha256:<64hex>
    algo, hex_digest = victim_digest.split(":", 1)
    assert algo == "sha256" and len(hex_digest) == 64, (
        f"unexpected digest format: {victim_digest!r}"
    )
    prefix = hex_digest[:2]

    # Delete the blob data file from inside the container.
    # The actual content is at: .../blobs/<algo>/<2char>/<hex>/data
    blob_data_path = f"/var/lib/registry/docker/registry/v2/blobs/{algo}/{prefix}/{hex_digest}/data"
    del_result = subprocess.run(
        ["docker", "exec", reg_name, "sh", "-c", f"rm -f {blob_data_path}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert del_result.returncode == 0, (
        f"failed to delete blob data file from registry container:\n{del_result.stderr}"
    )

    # registry:2 serves from its in-memory index — restart to flush.
    restart_result = subprocess.run(
        ["docker", "restart", reg_name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert restart_result.returncode == 0, (
        f"failed to restart registry container: {restart_result.stderr}"
    )
    _wait_registry(port, timeout=30)

    # Confirm the blob is now missing (HEAD returns 404)
    blob_url = f"http://localhost:{port}/v2/e2e/status-llama/blobs/{victim_digest}"
    hr = requests.head(blob_url, timeout=10)
    assert hr.status_code == 404, (
        f"Expected blob HEAD to return 404 after file deletion + registry restart, "
        f"got {hr.status_code}. Test infrastructure assumption failed."
    )

    # validate must now exit non-zero
    val_result = subprocess.run(
        [
            "oci-modelcar",
            "validate",
            "--registry",
            f"localhost:{port}",
            "--target-repo",
            "e2e/status-llama",
            "--target-tag",
            EXPECTED_TAG,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert val_result.returncode != 0, (
        f"validate returned 0 despite missing blob {victim_digest}!\n"
        f"stdout: {val_result.stdout}\nstderr: {val_result.stderr}\n"
        "Expected non-zero exit to signal the missing layer."
    )
    combined = val_result.stdout + val_result.stderr
    # Should mention missing layer
    assert "missing" in combined.lower() or "layer" in combined.lower(), (
        f"validate error output does not mention missing layer:\n{combined}"
    )


@pytest.mark.e2e
def test_status_nonexistent_repo(status_validate_setup: dict) -> None:  # type: ignore[type-arg]
    """status against a non-existent repo exits 0 with a friendly message (not a crash)."""
    port = status_validate_setup["port"]
    result = subprocess.run(
        [
            "oci-modelcar",
            "status",
            "--registry",
            f"localhost:{port}",
            "--target-repo",
            "e2e/does-not-exist",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Should exit 0 (not crash) — the impl returns 0 on 404 with a friendly message
    assert result.returncode == 0, (
        f"status for non-existent repo should exit 0, got {result.returncode}:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
