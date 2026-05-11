"""E2E: v1.3 OCI 1.1 referrer-based crash-resilient reuse.

Parametrized over three real registries:

* ``registry:2`` v2.8.3 — no referrers API (route never existed in the v2
  series). Client uses the spec fallback tag schema.
* ``registry:3`` v3.1.1 — also no referrers API; the v3 series ships with
  the same Base/Manifest/Catalog/Tags/Blob/BlobUpload routes as v2, plus
  fixes. The OCI 1.1 referrers feature is still an open proposal
  (https://github.com/distribution/distribution/issues/3716) and was
  not implemented in any released distribution/distribution v3.x as of
  May 2026. Client uses the fallback tag schema.
* ``ghcr.io/project-zot/zot-linux-amd64`` — implements the native OCI
  1.1 referrers API (``GET /v2/<name>/referrers/<digest>`` returns an
  image-index of subjects). Client uses the native code path.

Each scenario is run against each registry. Both code paths are
spec-compliant and produce the same observable end state: records
reachable via *some* path, resume picks them up, zero HF re-download
on crash recovery.

1. Anchor manifest + per-layer reuse records are written to the registry
   on every push. Inspect via direct HTTP.
2. Resume after a simulated mid-push crash: delete the target-tag
   manifest (``REGISTRY_STORAGE_DELETE_ENABLED``), wipe sources,
   re-push. The reuse-map is reconstructed from the records, no HF
   download happens.
3. ``--no-reuse-records`` opt-out skips both the anchor and the records.
4. Different ``--ignore-patterns`` produce different anchor digests —
   no cross-contamination of reuse records between filter configs.

Requires Docker + network access to huggingface.co.
"""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

from oci_modelcar.reuse import (
    ARTIFACT_TYPE_ANCHOR,
    ARTIFACT_TYPE_RECORD,
    build_anchor_manifest_bytes,
    fallback_referrers_tag,
)

# Same tiny HF model the existing e2e tests pin (and CLAUDE.md documents).
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


_ZOT_IMAGE = "ghcr.io/project-zot/zot-linux-amd64:latest"


@pytest.fixture(scope="function", params=["registry:2", "registry:3", _ZOT_IMAGE])
def deletable_registry(request):  # type: ignore[no-untyped-def]
    """A registry that supports DELETE manifests so the resume test can
    simulate a crash by removing the target-tag manifest.

    Parametrized over distribution/distribution v2 and v3 (both fall
    back to the sha256-<hex> tag schema — no referrers API implemented
    upstream as of May 2026) and zot (native referrers support).

    **Function-scoped** rather than module-scoped: zot's manifest
    storage is content-addressed and deduplicated across repos in the
    same registry instance. When our deterministic record artifact
    (same `subject`, same layer, same annotations) is PUT to repo A by
    one test and then to repo B by the next, zot recognizes the
    content as already present, returns 201 idempotently, echoes
    ``Oci-Subject`` — but its per-repo native referrers index for
    repo B's anchor remains empty (no fresh PUT against B triggered an
    index update). The subsequent GET
    ``/v2/repo_B/referrers/<anchor>`` returns 200 with an empty index.
    Tests against a different repo per test must therefore not share
    a registry instance. Function scope spins a fresh registry per
    test (~140 s total for 12 tests, acceptable).
    """
    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        pytest.skip("docker not available")

    image = request.param
    port = _free_port()
    safe = image.split("/")[-1].replace(":", "-")
    name = f"oci-modelcar-v13-reg-{safe}-{port}"
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    # distribution/distribution honors REGISTRY_STORAGE_DELETE_ENABLED.
    # zot enables manifest DELETE by default; its config schema differs.
    extra_args: list[str] = []
    if image in ("registry:2", "registry:3"):
        extra_args = ["-e", "REGISTRY_STORAGE_DELETE_ENABLED=true"]
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
            *extra_args,
            image,
        ],
        check=True,
        capture_output=True,
    )
    _wait_registry(port)
    yield {"port": port, "image": image}
    subprocess.run(["docker", "stop", name], check=False, capture_output=True)


def _is_native_referrers_registry(image: str) -> bool:
    """True if the registry image implements the OCI 1.1 referrers API.

    distribution/distribution v2/v3 do not (proposal #3716 still open).
    zot does, since v2.0.0+.
    """
    return "zot" in image


def _read_records(base: str, anchor_digest: str) -> tuple[list[dict[str, object]], str]:
    """Return (record_descriptors, mode) where mode is 'native' or 'fallback'.

    Tries the native referrers API first; on 404 falls back to GET on the
    sha256-<hex> tag (per OCI Distribution 1.1 spec)."""
    r = requests.get(
        f"{base}/referrers/{anchor_digest}",
        params={"artifactType": ARTIFACT_TYPE_RECORD},
        headers={"Accept": "application/vnd.oci.image.index.v1+json"},
        timeout=10,
    )
    if r.status_code == 200:
        return list(r.json().get("manifests", []) or []), "native"
    tag = fallback_referrers_tag(anchor_digest)
    rf = requests.get(
        f"{base}/manifests/{tag}",
        headers={"Accept": "application/vnd.oci.image.index.v1+json"},
        timeout=10,
    )
    if rf.status_code == 404:
        return [], "neither"
    rf.raise_for_status()
    return list(rf.json().get("manifests", []) or []), "fallback"


def _push(
    *,
    port: int,
    target_repo: str,
    spool: Path,
    clean: bool = False,
    no_reuse_records: bool = False,
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "oci-modelcar",
        "push",
        "--hf-repo",
        HF_REPO,
        "--hf-revision",
        HF_REVISION,
        "--registry",
        f"localhost:{port}",
        "--target-repo",
        target_repo,
        "--spool-dir",
        str(spool),
    ]
    if clean:
        cmd.append("--clean-hf-after-push")
    if no_reuse_records:
        cmd.append("--no-reuse-records")
    if extra:
        cmd.extend(extra)
    return subprocess.run(cmd, env=os.environ.copy(), capture_output=True, text=True, timeout=300)


def _expected_anchor_digest(allow: tuple[str, ...], ignore: tuple[str, ...]) -> str:
    body = build_anchor_manifest_bytes(
        hf_repo=HF_REPO,
        hf_revision=HF_REVISION,
        allow_patterns=allow,
        ignore_patterns=ignore,
        layer_prefix="models/",
    )
    return "sha256:" + hashlib.sha256(body).hexdigest()


_DEFAULT_ALLOW = (".safetensors", ".json", ".txt", ".md", ".model")


@pytest.mark.e2e
def test_anchor_and_records_written_on_push(deletable_registry: dict, tmp_path: Path) -> None:
    """Verify the v1.3 wire-format artifacts land in the registry.

    Both code paths must produce the same observable end state:
      * registry:2 → records reachable via the sha256-<hex> fallback tag
      * registry:3 → records reachable via the native referrers API
    """
    port = deletable_registry["port"]
    image = deletable_registry["image"]
    repo = "e2e/v13-anchor"
    spool = tmp_path / "spool"
    result = _push(port=port, target_repo=repo, spool=spool)
    assert result.returncode == 0, f"push failed:\n{result.stdout}\n{result.stderr}"

    anchor_digest = _expected_anchor_digest(_DEFAULT_ALLOW, ())
    base = f"http://localhost:{port}/v2/{repo}"

    # 1) Anchor manifest is at the expected content-addressed digest.
    h = requests.head(
        f"{base}/manifests/{anchor_digest}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    assert h.status_code == 200, (
        f"anchor manifest missing at {anchor_digest}: status={h.status_code}"
    )

    # 2) The anchor's content matches our deterministic build.
    g = requests.get(
        f"{base}/manifests/{anchor_digest}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    g.raise_for_status()
    anchor = g.json()
    assert anchor["artifactType"] == ARTIFACT_TYPE_ANCHOR
    assert anchor["annotations"]["io.github.codanael.modelcar.hf-repo"] == HF_REPO
    assert anchor["annotations"]["io.github.codanael.modelcar.hf-revision"] == HF_REVISION

    # 3) Records reachable via the path appropriate for this registry.
    record_descriptors, mode = _read_records(base, anchor_digest)
    expected_mode = "native" if _is_native_referrers_registry(image) else "fallback"
    assert mode == expected_mode, (
        f"on {image}, expected mode={expected_mode!r} but got mode={mode!r}; "
        "either the registry's referrers support changed (update the helper) "
        "or the client's OCI-Subject detection regressed"
    )
    assert len(record_descriptors) >= 1, f"no reuse records found ({mode} mode)"

    # 4) Each record manifest carries hf-path + (when LFS) hf-sha256 annotations.
    paths_seen: set[str] = set()
    for desc in record_descriptors:
        record_digest = desc["digest"]
        rg = requests.get(
            f"{base}/manifests/{record_digest}",
            headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
            timeout=10,
        )
        rg.raise_for_status()
        rec = rg.json()
        assert rec["artifactType"] == ARTIFACT_TYPE_RECORD
        assert rec["subject"]["digest"] == anchor_digest
        layer = rec["layers"][0]
        annot = layer["annotations"]
        assert "io.github.codanael.modelcar.hf-path" in annot
        paths_seen.add(annot["io.github.codanael.modelcar.hf-path"])

    assert "config.json" in paths_seen


@pytest.mark.e2e
def test_resume_after_simulated_crash_skips_hf_download(
    deletable_registry: dict, tmp_path: Path
) -> None:
    """Simulate the user's reported failure mode end-to-end:
    1) push with --clean-hf-after-push (sources deleted from spool)
    2) DELETE the target tag manifest (simulates a crash AFTER blob pushes
       but BEFORE the manifest was committed; blobs + records remain)
    3) re-push from a fresh spool — must reuse all layers via referrer
       records, no HF download.
    """
    port = deletable_registry["port"]
    repo = "e2e/v13-resume"
    spool_1 = tmp_path / "spool1"

    # ---- Run 1: successful push, sources cleaned ----
    r1 = _push(port=port, target_repo=repo, spool=spool_1, clean=True)
    assert r1.returncode == 0, f"first push failed:\n{r1.stdout}\n{r1.stderr}"
    assert not any((spool_1 / "sources").rglob("*")) if (spool_1 / "sources").exists() else True, (
        "spool/sources not empty after --clean-hf-after-push"
    )

    # Determine the target tag (default = sha[:12])
    target_tag = HF_REVISION[:12]
    base = f"http://localhost:{port}/v2/{repo}"

    # Fetch and verify the manifest exists, then DELETE it by digest
    head = requests.head(
        f"{base}/manifests/{target_tag}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    assert head.status_code == 200, "target manifest missing after first push?"
    manifest_digest = head.headers["Docker-Content-Digest"]
    d = requests.delete(f"{base}/manifests/{manifest_digest}", timeout=10)
    assert d.status_code in (202, 200), (
        f"DELETE manifest returned {d.status_code} — registry:2 must be started "
        f"with REGISTRY_STORAGE_DELETE_ENABLED=true (fixture sets this)."
    )

    # Confirm the tag is gone (simulates the post-crash state)
    head2 = requests.head(
        f"{base}/manifests/{target_tag}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    assert head2.status_code == 404

    # ---- Run 2: fresh spool, verbose logging to inspect reuse path ----
    spool_2 = tmp_path / "spool2"
    r2 = _push(port=port, target_repo=repo, spool=spool_2, clean=True, extra=["--verbose"])
    assert r2.returncode == 0, f"resume push failed:\n{r2.stdout}\n{r2.stderr}"

    combined = r2.stdout + r2.stderr

    # The pipeline must announce that it picked records up via the referrer API.
    assert "found via referrer records" in combined, (
        f"expected 'found via referrer records' log line on resume; full output:\n{combined}"
    )

    # No HF download lines must appear — every layer must have hit phase 0 reuse.
    download_lines = [line for line in combined.splitlines() if ": downloading (" in line]
    assert download_lines == [], (
        "resume run downloaded from HF — referrer reuse failed to short-circuit:\n"
        + "\n".join(download_lines)
    )

    # And the manifest is back at the target tag with the same digest as run 1.
    head3 = requests.head(
        f"{base}/manifests/{target_tag}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    assert head3.status_code == 200
    assert head3.headers["Docker-Content-Digest"] == manifest_digest, (
        "resume manifest digest differs from pre-crash digest — non-deterministic?"
    )


@pytest.mark.e2e
def test_no_reuse_records_skips_anchor_and_records(
    deletable_registry: dict, tmp_path: Path
) -> None:
    """With ``--no-reuse-records``: no anchor manifest at the expected
    deterministic digest, no records returned by the referrers API."""
    port = deletable_registry["port"]
    repo = "e2e/v13-no-records"
    spool = tmp_path / "spool"
    result = _push(port=port, target_repo=repo, spool=spool, no_reuse_records=True)
    assert result.returncode == 0, f"push failed:\n{result.stdout}\n{result.stderr}"

    anchor_digest = _expected_anchor_digest(_DEFAULT_ALLOW, ())
    base = f"http://localhost:{port}/v2/{repo}"

    h = requests.head(
        f"{base}/manifests/{anchor_digest}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    assert h.status_code == 404, (
        f"anchor manifest present at {anchor_digest} despite --no-reuse-records "
        f"(status={h.status_code})"
    )

    r = requests.get(
        f"{base}/referrers/{anchor_digest}",
        params={"artifactType": ARTIFACT_TYPE_RECORD},
        headers={"Accept": "application/vnd.oci.image.index.v1+json"},
        timeout=10,
    )
    # Native referrers returns 200 + empty index even for unknown subjects.
    # Fallback registries return 404. Either way: no record descriptors.
    if r.status_code == 200:
        index = r.json()
        assert index.get("manifests", []) == [], "records present despite --no-reuse-records"
    else:
        assert r.status_code == 404, (
            f"unexpected referrers status {r.status_code} with --no-reuse-records"
        )

    # Sanity: the actual image manifest IS there (the feature opt-out
    # doesn't disable the regular push).
    target_tag = HF_REVISION[:12]
    h2 = requests.head(
        f"{base}/manifests/{target_tag}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    assert h2.status_code == 200, "regular image manifest missing after push"


@pytest.mark.e2e
def test_anchor_digest_differs_by_filter_inputs(deletable_registry: dict, tmp_path: Path) -> None:
    """The anchor digest is keyed on the run inputs. Two pushes with different
    --ignore-patterns to different target repos must produce different
    anchor digests (no cross-contamination of reuse records)."""
    port = deletable_registry["port"]
    spool_a = tmp_path / "a"
    spool_b = tmp_path / "b"

    r_a = _push(
        port=port,
        target_repo="e2e/v13-fa",
        spool=spool_a,
        extra=["--ignore-patterns", "*.md"],
    )
    assert r_a.returncode == 0, r_a.stderr
    r_b = _push(
        port=port,
        target_repo="e2e/v13-fb",
        spool=spool_b,
        extra=["--ignore-patterns", "*.txt"],
    )
    assert r_b.returncode == 0, r_b.stderr

    digest_a = _expected_anchor_digest(_DEFAULT_ALLOW, ("*.md",))
    digest_b = _expected_anchor_digest(_DEFAULT_ALLOW, ("*.txt",))
    assert digest_a != digest_b

    for repo, digest in (("e2e/v13-fa", digest_a), ("e2e/v13-fb", digest_b)):
        h = requests.head(
            f"http://localhost:{port}/v2/{repo}/manifests/{digest}",
            headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
            timeout=10,
        )
        assert h.status_code == 200, f"expected anchor {digest} in {repo}, got {h.status_code}"

    # Cross-check: the wrong digest is NOT in the other repo
    h_cross = requests.head(
        f"http://localhost:{port}/v2/e2e/v13-fa/manifests/{digest_b}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    assert h_cross.status_code == 404, (
        "cross-repo anchor leakage — anchor digests must be per-config"
    )
