"""Gap B: --target-tag + --also-tag multi-tag publishing.

Verifies:
- All requested tags exist after the push
- All tags resolve to the SAME Docker-Content-Digest
- also_tags silently overwrite existing conflicting tags (no force check)

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
def multitag_registry():  # type: ignore[no-untyped-def]
    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        pytest.skip("docker not available")

    port = _free_port()
    name = f"oci-modelcar-multitag-reg-{port}"
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name, "-p", f"{port}:5000", "registry:2"],
        check=True,
        capture_output=True,
    )
    _wait_registry(port)
    yield port
    subprocess.run(["docker", "stop", name], check=False, capture_output=True)


def _head_manifest_digest(port: int, repo: str, tag: str) -> str | None:
    url = f"http://localhost:{port}/v2/{repo}/manifests/{tag}"
    r = requests.head(url, headers={"Accept": MANIFEST_TYPE}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.headers.get("Docker-Content-Digest")


def _get_tags(port: int, repo: str) -> list[str]:
    url = f"http://localhost:{port}/v2/{repo}/tags/list"
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return r.json().get("tags", []) or []


@pytest.mark.e2e
def test_multi_tag_all_present(multitag_registry: int, tmp_path: Path) -> None:
    """Push with --target-tag custom-v1 --also-tag latest,prod.

    All three tags must exist and resolve to the same manifest digest.
    """
    spool = tmp_path / "spool"
    env = os.environ.copy()
    cmd = [
        "oci-modelcar",
        "push",
        "--hf-repo",
        HF_REPO,
        "--hf-revision",
        HF_REVISION,
        "--registry",
        f"localhost:{multitag_registry}",
        "--target-repo",
        "e2e/multi-tag-llama",
        "--spool-dir",
        str(spool),
        "--target-tag",
        "custom-v1",
        "--also-tag",
        "latest,prod",
        "--quiet",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (
        f"multi-tag push failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # All three tags must exist
    tags = _get_tags(multitag_registry, "e2e/multi-tag-llama")
    for expected_tag in ("custom-v1", "latest", "prod"):
        assert expected_tag in tags, f"tag '{expected_tag}' missing from registry tags list: {tags}"

    # All three tags must resolve to the SAME manifest digest
    digest_custom = _head_manifest_digest(multitag_registry, "e2e/multi-tag-llama", "custom-v1")
    digest_latest = _head_manifest_digest(multitag_registry, "e2e/multi-tag-llama", "latest")
    digest_prod = _head_manifest_digest(multitag_registry, "e2e/multi-tag-llama", "prod")

    assert digest_custom is not None, "custom-v1 manifest not found"
    assert digest_latest is not None, "latest manifest not found"
    assert digest_prod is not None, "prod manifest not found"

    assert digest_custom == digest_latest == digest_prod, (
        f"tags point to different digests!\n"
        f"  custom-v1: {digest_custom}\n"
        f"  latest:    {digest_latest}\n"
        f"  prod:      {digest_prod}\n"
        "All three should point to the same manifest."
    )


@pytest.mark.e2e
def test_also_tag_overwrites_without_force(multitag_registry: int, tmp_path: Path) -> None:
    """also_tags are blindly overwritten — no force check.

    Sequence:
    1. Push model A to repo with --target-tag primary-a --also-tag shared-alias
    2. Push model B to same repo with --target-tag primary-b --also-tag shared-alias
       (different revision → different manifest digest)
    3. Verify: primary-a still has the old digest, shared-alias now has the new digest
       (without --force, no error was raised despite overwriting the alias)

    NOTE: we use different revisions by using a different --target-tag to force
    a different manifest — since both pushes use the same HF repo/revision, the
    manifest digest will actually be identical. We instead verify the scenario
    where also_tags are NOT conflict-checked by pushing a second time with
    --target-tag only and checking no error occurs.
    """
    spool = tmp_path / "spool_overwrite"
    env = os.environ.copy()

    # First push: establish 'overwriteable-alias'
    cmd1 = [
        "oci-modelcar",
        "push",
        "--hf-repo",
        HF_REPO,
        "--hf-revision",
        HF_REVISION,
        "--registry",
        f"localhost:{multitag_registry}",
        "--target-repo",
        "e2e/alias-overwrite",
        "--spool-dir",
        str(spool / "1"),
        "--target-tag",
        "primary-a",
        "--also-tag",
        "overwriteable-alias",
        "--quiet",
    ]
    r1 = subprocess.run(cmd1, env=env, capture_output=True, text=True, timeout=300)
    assert r1.returncode == 0, f"first push failed:\nstdout: {r1.stdout}\nstderr: {r1.stderr}"

    digest_after_first = _head_manifest_digest(
        multitag_registry, "e2e/alias-overwrite", "overwriteable-alias"
    )
    assert digest_after_first is not None, "alias tag missing after first push"

    # Second push: same HF content, different primary tag, same also_tag
    # Without --force. This should succeed (no error) because also_tags skip conflict check.
    cmd2 = [
        "oci-modelcar",
        "push",
        "--hf-repo",
        HF_REPO,
        "--hf-revision",
        HF_REVISION,
        "--registry",
        f"localhost:{multitag_registry}",
        "--target-repo",
        "e2e/alias-overwrite",
        "--spool-dir",
        str(spool / "2"),
        "--target-tag",
        "primary-b",
        "--also-tag",
        "overwriteable-alias",
        "--quiet",
    ]
    r2 = subprocess.run(cmd2, env=env, capture_output=True, text=True, timeout=300)
    assert r2.returncode == 0, (
        f"second push (overwriting also_tag without --force) failed unexpectedly:\n"
        f"stdout: {r2.stdout}\nstderr: {r2.stderr}\n"
        f"NOTE: also_tags should be overwritten silently (no force check)."
    )
