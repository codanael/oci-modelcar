"""E2E tests against real HuggingFace + local registry:2."""

from __future__ import annotations

import json
import re
import subprocess
import sys

import pytest

# Pin to a specific commit SHA. Resolve a fresh one if needed:
#   curl -s https://huggingface.co/api/models/hf-internal-testing/tiny-random-LlamaForCausalLM
HF_TEST_REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HF_TEST_REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"


@pytest.mark.e2e
def test_push_tiny_llama(local_registry, skopeo_bin, hf_endpoint, tmp_path):
    state = tmp_path / "state.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "oci_modelcar",
            "push",
            "--hf-repo",
            HF_TEST_REPO,
            "--hf-revision",
            HF_TEST_REVISION,
            "--hf-endpoint",
            hf_endpoint,
            "--registry",
            local_registry.host,
            "--target-repo",
            "test/tiny-llama",
            "--state-file",
            str(state),
            "--workers",
            "2",
            "--allow-patterns",
            ".safetensors .json",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
    assert proc.returncode == 0, proc.stderr
    expected_tag = HF_TEST_REVISION[:12]
    assert f"IMAGEREF={local_registry.host}/test/tiny-llama:{expected_tag}" in proc.stdout
    m = re.search(r"^MANIFESTDIGEST=(sha256:[0-9a-f]{64})$", proc.stdout, re.MULTILINE)
    assert m, f"no MANIFEST= in stdout:\n{proc.stdout}"

    # skopeo inspect
    raw = subprocess.check_output(
        [
            skopeo_bin,
            "inspect",
            "--raw",
            "--tls-verify=false",
            f"docker://{local_registry.host}/test/tiny-llama:{expected_tag}",
        ],
        text=True,
    )
    manifest = json.loads(raw)
    assert manifest["schemaVersion"] == 2
    assert manifest["mediaType"] == "application/vnd.oci.image.manifest.v1+json"
    assert manifest["config"]["mediaType"] == "application/vnd.oci.image.config.v1+json"
    for layer in manifest["layers"]:
        assert layer["mediaType"] == "application/vnd.oci.image.layer.v1.tar"


@pytest.mark.e2e
def test_idempotent_rerun(local_registry, skopeo_bin, hf_endpoint, tmp_path):
    state = tmp_path / "state.json"
    args = [
        sys.executable,
        "-m",
        "oci_modelcar",
        "push",
        "--hf-repo",
        HF_TEST_REPO,
        "--hf-revision",
        HF_TEST_REVISION,
        "--hf-endpoint",
        hf_endpoint,
        "--registry",
        local_registry.host,
        "--target-repo",
        "test/tiny-llama-idem",
        "--state-file",
        str(state),
        "--allow-patterns",
        ".safetensors .json",
    ]
    p1 = subprocess.run(args, capture_output=True, text=True)
    if p1.returncode != 0:
        print("p1 STDOUT:", p1.stdout)
        print("p1 STDERR:", p1.stderr)
    assert p1.returncode == 0

    p2 = subprocess.run(args, capture_output=True, text=True)
    if p2.returncode != 0:
        print("p2 STDOUT:", p2.stdout)
        print("p2 STDERR:", p2.stderr)
    assert p2.returncode == 0
    assert "already completed" in p2.stdout

    d1 = re.search(r"^MANIFESTDIGEST=(sha256:\w+)$", p1.stdout, re.MULTILINE)
    d2 = re.search(r"^MANIFESTDIGEST=(sha256:\w+)$", p2.stdout, re.MULTILINE)
    if d1 and d2:
        assert d1.group(1) == d2.group(1)
