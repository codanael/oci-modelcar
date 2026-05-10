"""End-to-end against real HuggingFace and a local docker registry:2.

Requires:
- Docker daemon running
- Network access to huggingface.co

Run with: .venv/bin/python -m pytest tests/e2e/ -m e2e -v
"""

from __future__ import annotations

import os
import subprocess

import pytest

HF_REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HF_REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
EXPECTED_TAG = "9fb191250dd5"


@pytest.mark.e2e
def test_e2e_push_tiny_llama(local_registry, tmp_path):
    """Push the pinned tiny llama to the local registry and validate."""
    env = os.environ.copy()
    spool = tmp_path / "spool"
    cmd = [
        "oci-modelcar",
        "push",
        "--hf-repo",
        HF_REPO,
        "--hf-revision",
        HF_REVISION,
        "--registry",
        local_registry.host,
        "--target-repo",
        "e2e/tiny-llama",
        "--spool-dir",
        str(spool),
        "--workers",
        "2",
        "--quiet",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"push failed:\n{result.stdout}\n{result.stderr}"

    # Validate
    cmd = [
        "oci-modelcar",
        "validate",
        "--registry",
        local_registry.host,
        "--target-repo",
        "e2e/tiny-llama",
        "--target-tag",
        EXPECTED_TAG,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"validate failed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.e2e
def test_e2e_push_idempotent(local_registry, tmp_path):
    """Re-running the push against an already-pushed tag → exit 0 (skipped)."""
    env = os.environ.copy()
    cmd = [
        "oci-modelcar",
        "push",
        "--hf-repo",
        HF_REPO,
        "--hf-revision",
        HF_REVISION,
        "--registry",
        local_registry.host,
        "--target-repo",
        "e2e/tiny-llama",
        "--spool-dir",
        str(tmp_path / "spool2"),
        "--quiet",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
