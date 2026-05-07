"""Tests for the idempotent re-run path of run_push.

These tests pre-populate state to skip the heavy push pipeline and exercise
only the early-return branch that re-emits IMAGEREFDIGEST/IMAGEREF.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from oci_modelcar.config import Config
from oci_modelcar.hf import HfClient
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.runner import run_push
from oci_modelcar.state import JsonStateStore

_SHA = "a" * 40
_TAG = _SHA[:12]
_DIGEST = "sha256:" + "f" * 64


def _seed_legacy_state(state_path: Path, registry: str) -> str:
    """Write a v0.1.0-shaped state file (no image_ref_digest) for one completed job.

    Returns the matching job_key.
    """
    job_key = JsonStateStore.compute_job_key(
        hf_repo="foo/bar",
        revision_resolved=_SHA,
        registry=registry,
        target_repo="m/x",
        target_tag=_TAG,
    )
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    job_key: {
                        "source": {
                            "hf_repo": "foo/bar",
                            "hf_revision_input": _SHA,
                            "hf_revision_resolved": _SHA,
                        },
                        "target": {
                            "registry": registry,
                            "repo": "m/x",
                            "tag": _TAG,
                            "also_tags": [],
                        },
                        "started_at": "2026-05-07T12:00:00Z",
                        "updated_at": "2026-05-07T12:00:00Z",
                        "completed_at": "2026-05-07T12:01:00Z",
                        "manifest_digest": _DIGEST,
                        "files": {},
                    }
                },
            }
        )
    )
    return job_key


def _run(cfg: Config) -> str:
    buf = io.StringIO()
    plog = PipelineLogger(stream=buf, style="text", use_color=False)
    run_push(cfg, plog)
    return buf.getvalue()


@pytest.fixture
def patched_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HfClient, "resolve_revision", lambda self, rev: _SHA)


def test_idempotent_legacy_state_reconstructs_image_ref_digest(tmp_path: Path, patched_hf: None):
    """v0.1.0 state file (no image_ref_digest) is re-emitted by reconstruction
    from manifest_digest. Output must use the bare host (no scheme) so it can
    feed directly into `cosign sign`.
    """
    state_path = tmp_path / "state.json"
    _seed_legacy_state(state_path, "http://reg.example.com")

    cfg = Config(
        hf_repo="foo/bar",
        registry="http://reg.example.com",
        target_repo="m/x",
        hf_revision=_SHA,
        state_file=state_path,
    )
    out = _run(cfg)

    assert f"IMAGEREFDIGEST=reg.example.com/m/x@{_DIGEST}" in out, out
    assert "IMAGEREFDIGEST=http://" not in out, out
    assert "IMAGEREF=reg.example.com/m/x:" in out, out
    assert "IMAGEREF=http://" not in out, out
