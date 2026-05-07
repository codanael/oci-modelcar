"""JSON state store with atomic writes and threading.Lock."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FileState:
    size: int  # raw HF file size (used to validate against current tree)
    layer_size: int  # tar-wrapped layer bytes (used in the manifest)
    digest: str
    diff_id: str
    pushed_at: str = field(default_factory=_now_iso)


@dataclass
class JobState:
    hf_repo: str
    hf_revision_input: str
    hf_revision_resolved: str
    registry: str
    target_repo: str
    target_tag: str
    also_tags: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    completed_at: str | None = None
    manifest_digest: str | None = None
    files: dict[str, FileState] = field(default_factory=dict)


class JsonStateStore:
    """File-backed JSON state. Atomic writes, thread-safe."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"version": 1, "jobs": {}}
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):  # fmt: skip
            return {"version": 1, "jobs": {}}
        if raw.get("version") != 1:
            raise RuntimeError(f"unsupported state file version: {raw.get('version')}")
        return cast(dict[str, Any], raw)

    @staticmethod
    def compute_job_key(
        hf_repo: str,
        revision_resolved: str,
        registry: str,
        target_repo: str,
        target_tag: str,
    ) -> str:
        material = f"{hf_repo}:{revision_resolved}→{registry}/{target_repo}:{target_tag}"
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def list_jobs(self) -> list[str]:
        return list(self._data.get("jobs", {}).keys())

    def get_job(self, job_key: str) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self._data["jobs"].get(job_key))

    def upsert_job(self, job_key: str, job: JobState) -> None:
        with self._lock:
            jobs = self._data.setdefault("jobs", {})
            existing = jobs.get(job_key)
            if existing is None:
                jobs[job_key] = self._job_to_dict(job)
            else:
                # preserve files{} on subsequent runs
                existing["source"] = self._source(job)
                existing["target"] = self._target(job)
                existing["updated_at"] = _now_iso()

    def has_pushed(self, job_key: str, hf_path: str, expected_size: int) -> bool:
        with self._lock:
            job = self._data["jobs"].get(job_key)
            if job is None:
                return False
            entry = job["files"].get(hf_path)
            if entry is None:
                return False
            if entry.get("size") != expected_size:
                return False
            return entry.get("pushed_at") is not None

    def get_pushed(self, job_key: str, hf_path: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._data["jobs"].get(job_key)
            if job is None:
                return None
            return cast(dict[str, Any] | None, job["files"].get(hf_path))

    def mark_pushed(
        self,
        job_key: str,
        hf_path: str,
        digest: str,
        diff_id: str,
        size: int,
        layer_size: int,
    ) -> None:
        with self._lock:
            job = self._data["jobs"][job_key]
            job["files"][hf_path] = {
                "size": size,
                "layer_size": layer_size,
                "digest": digest,
                "diff_id": diff_id,
                "pushed_at": _now_iso(),
            }
            job["updated_at"] = _now_iso()

    def is_completed(self, job_key: str) -> bool:
        job = self._data["jobs"].get(job_key)
        return bool(job and job.get("manifest_digest"))

    def mark_completed(self, job_key: str, manifest_digest: str) -> None:
        with self._lock:
            job = self._data["jobs"][job_key]
            job["manifest_digest"] = manifest_digest
            job["completed_at"] = _now_iso()
            job["updated_at"] = _now_iso()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmpname = tempfile.mkstemp(prefix=".state-", suffix=".json", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self._data, f, indent=2, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())
                os.chmod(tmpname, 0o600)
                os.replace(tmpname, self.path)
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(tmpname)
                raise

    @staticmethod
    def _job_to_dict(job: JobState) -> dict[str, Any]:
        return {
            "source": JsonStateStore._source(job),
            "target": JsonStateStore._target(job),
            "started_at": job.started_at,
            "updated_at": job.updated_at,
            "completed_at": job.completed_at,
            "manifest_digest": job.manifest_digest,
            "files": {k: asdict(v) for k, v in job.files.items()},
        }

    @staticmethod
    def _source(job: JobState) -> dict[str, str]:
        return {
            "hf_repo": job.hf_repo,
            "hf_revision_input": job.hf_revision_input,
            "hf_revision_resolved": job.hf_revision_resolved,
        }

    @staticmethod
    def _target(job: JobState) -> dict[str, Any]:
        return {
            "registry": job.registry,
            "repo": job.target_repo,
            "tag": job.target_tag,
            "also_tags": list(job.also_tags),
        }
