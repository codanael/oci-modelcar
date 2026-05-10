"""E2E: re-push of an already-present blob short-circuits via head_blob.

Verifies the skip-already-present invariant (FileWorker step c) using real
wire traffic against registry:2:

1. Before first push: head_blob returns None (blob absent).
2. First push: StreamingBlobUpload.push_from_file succeeds.
3. After first push: head_blob returns the descriptor with matching digest
   and size — this is the discriminant that FileWorker.process uses at step c
   to skip the upload entirely on re-push.
4. The descriptor is idempotent: a second push of the same blob leaves
   head_blob returning the same descriptor.

Requires Docker.
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

from oci_modelcar.layer import build_layer_to_file
from oci_modelcar.registry import OciClient, StreamingBlobUpload, head_blob


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
def fresh_registry() -> int:  # type: ignore[return]
    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        pytest.skip("docker not available")

    port = _free_port()
    name = f"oci-modelcar-skip-reg-{port}"
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-p",
            f"{port}:5000",
            "registry:2",
        ],
        check=True,
        capture_output=True,
    )
    _wait_registry(port)

    yield port  # type: ignore[misc]

    subprocess.run(["docker", "stop", name], check=False, capture_output=True)


@pytest.mark.e2e
def test_skip_already_present_on_repush(fresh_registry: int, tmp_path: Path) -> None:
    """After a successful push, head_blob returns the descriptor for the same
    digest, enabling FileWorker step c to skip the upload entirely on re-push.

    Phase 1: blob absent.
    Phase 2: first push succeeds.
    Phase 3: head_blob returns descriptor with correct digest + size.
    Phase 4: second push of same digest → head_blob still returns same descriptor
             (registry is idempotent for duplicate content).
    """
    source = tmp_path / "payload.bin"
    source.write_bytes(b"S" * 4096)
    tar_path = tmp_path / "payload.bin.tar"
    digest, layer_size = build_layer_to_file(source, "models/", "payload.bin", tar_path)

    client = OciClient(host_url=f"http://localhost:{fresh_registry}", target_repo="e2e/skip")

    # Phase 1: blob must be absent before first push
    before = head_blob(client, "e2e/skip", digest)
    assert before is None, f"blob should be absent before first push; got {before!r}"

    # Phase 2: first push
    upload = StreamingBlobUpload(client=client, repo="e2e/skip", max_retries=2)
    out_digest, out_size = upload.push_from_file(tar_path, layer_size, digest)
    assert out_digest == digest
    assert out_size == layer_size

    # Phase 3: head_blob returns descriptor — this is the discriminant for
    # FileWorker.process step c (skip check before every push attempt)
    after_first = head_blob(client, "e2e/skip", digest)
    assert after_first is not None, "blob must be present after first push"
    assert after_first["digest"] == digest, (
        f"descriptor digest mismatch: {after_first['digest']!r} != {digest!r}"
    )
    assert after_first["size"] == layer_size, (
        f"descriptor size mismatch: {after_first['size']!r} != {layer_size!r}"
    )

    # Phase 4: second push of the same blob — registry tolerates duplicate
    # uploads (idempotent). head_blob result must be unchanged.
    out_digest_2, out_size_2 = upload.push_from_file(tar_path, layer_size, digest)
    assert out_digest_2 == digest
    assert out_size_2 == layer_size

    after_second = head_blob(client, "e2e/skip", digest)
    assert after_second == after_first, (
        f"registry state must be idempotent across pushes of same digest; "
        f"first={after_first!r} second={after_second!r}"
    )
