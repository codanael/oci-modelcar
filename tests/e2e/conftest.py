"""E2E fixtures: local registry:2, skopeo path discovery."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest


def _wait_for_registry(host: str, timeout: float = 30.0) -> None:
    h, p = host.split(":")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((h, int(p)), timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"registry at {host} did not come up within {timeout}s")


@pytest.fixture(scope="session")
def local_registry() -> Iterator[SimpleNamespace]:
    if "OCI_MODELCAR_E2E_REGISTRY" in os.environ:
        yield SimpleNamespace(host=os.environ["OCI_MODELCAR_E2E_REGISTRY"])
        return
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    cid = (
        subprocess.check_output(["docker", "run", "-d", "--rm", "-p", "5000:5000", "registry:2"])
        .decode()
        .strip()
    )
    try:
        _wait_for_registry("localhost:5000")
        yield SimpleNamespace(host="localhost:5000")
    finally:
        subprocess.run(["docker", "kill", cid], check=False, capture_output=True)


@pytest.fixture(scope="session")
def skopeo_bin() -> str:
    if shutil.which("skopeo") is None:
        pytest.skip("skopeo not available")
    return "skopeo"


@pytest.fixture(scope="session")
def hf_endpoint() -> str:
    return os.environ.get("OCI_MODELCAR_E2E_HF_ENDPOINT", "https://huggingface.co")
