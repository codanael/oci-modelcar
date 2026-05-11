"""HuggingFace metadata via huggingface_hub.HfApi + bytes streamer with
mid-stream cancellation, atomic write, cross-origin auth strip."""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import http.client
import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import urllib3.exceptions
from huggingface_hub import HfApi
from huggingface_hub import errors as hf_errors
from huggingface_hub.hf_api import RepoFile

from oci_modelcar.errors import (
    DownloadError,
    EntryNotFoundError,
    GatedRepoError,
    RevisionNotFoundError,
)
from oci_modelcar.http import huggingface_auth_header, is_transient_ssl

log = logging.getLogger(__name__)

_TRANSIENT_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    requests.RequestException,
    urllib3.exceptions.ProtocolError,
    http.client.IncompleteRead,
    OSError,
)
_FATAL_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    requests.exceptions.SSLError,
    requests.exceptions.ProxyError,
)
_CHUNK_DEFAULT = 1024 * 1024  # 1 MiB stop_event polling granularity
_GLOB_META = "*?["


def _compile_filter(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize user-supplied filter tokens for fnmatch.

    A token containing no glob metacharacters (``*?[``) is treated as a
    suffix and rewritten to ``*<token>``. A token already carrying glob
    metacharacters is kept verbatim. This preserves the historical
    ``endswith``-style behavior of ``--allow-patterns`` while letting
    new patterns use full globs (e.g. ``consolidated-*``).
    """
    return tuple(p if any(c in p for c in _GLOB_META) else f"*{p}" for p in patterns)


def _path_matches_any(path: str, compiled: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in compiled)


@dataclass(frozen=True, slots=True)
class HfFile:
    path: str
    size: int
    lfs_sha256: str | None  # 64-hex from /api/models tree, when LFS-backed


class HfDownloader:
    def __init__(
        self,
        api: HfApi,
        session: requests.Session,
        spool_dir: Path | None,
        stop_event: threading.Event | None,
        max_retries: int = 10,
        backoff_initial: float = 1.0,
        backoff_cap: float = 60.0,
    ) -> None:
        self.api = api
        self.session = session
        self.spool_dir = spool_dir
        self.stop_event = stop_event
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_cap = backoff_cap

    def resolve_revision(self, repo: str, revision: str) -> str:
        try:
            info = self.api.repo_info(repo, revision=revision)
        except hf_errors.GatedRepoError as e:
            raise GatedRepoError(
                f"Repo {repo} is gated: {e}",
                hint=f"Accept terms at https://huggingface.co/{repo}, then re-run.",
            ) from e
        except hf_errors.RevisionNotFoundError as e:
            raise RevisionNotFoundError(
                f"Revision not found: {repo}@{revision}: {e}",
                hint=f"check available revisions at https://huggingface.co/{repo}",
            ) from e
        except hf_errors.RepositoryNotFoundError as e:
            raise DownloadError(
                f"Repository not found: {repo}: {e}",
                hint=f"check the repo name at https://huggingface.co/{repo}",
            ) from e
        # ModelInfo.sha is `str | None` per huggingface_hub typings — the
        # backend may legitimately omit it. Guard against silently passing
        # the literal string "None" downstream into resolve URLs.
        if info.sha is None:
            raise RevisionNotFoundError(
                f"HF returned no SHA for {repo}@{revision}",
                hint="check that the revision exists and the API endpoint responds",
            )
        return info.sha

    def list_files(
        self,
        repo: str,
        revision: str,
        allow: tuple[str, ...],
        ignore: tuple[str, ...] = (),
    ) -> list[HfFile]:
        out: list[HfFile] = []
        try:
            tree = self.api.list_repo_tree(repo, revision=revision, recursive=True)
            entries = list(tree)
        except hf_errors.GatedRepoError as e:
            raise GatedRepoError(
                f"Repo {repo} is gated: {e}",
                hint=f"Accept terms at https://huggingface.co/{repo}, then re-run.",
            ) from e
        except hf_errors.RevisionNotFoundError as e:
            raise RevisionNotFoundError(
                f"Revision not found: {repo}@{revision}: {e}",
                hint=f"check available revisions at https://huggingface.co/{repo}",
            ) from e
        except hf_errors.EntryNotFoundError as e:
            raise EntryNotFoundError(f"Entry not found in {repo}@{revision}: {e}") from e
        except hf_errors.RepositoryNotFoundError as e:
            raise DownloadError(
                f"Repository not found: {repo}: {e}",
                hint=f"check the repo name at https://huggingface.co/{repo}",
            ) from e
        allow_compiled = _compile_filter(allow)
        ignore_compiled = _compile_filter(ignore)
        for _entry in entries:
            # Narrow to files. huggingface_hub yields RepoFile/RepoFolder; the
            # canonical discriminant is isinstance(RepoFile) — RepoFolder has
            # no `.size`/`.lfs`. Tests use MagicMock(spec=RepoFile) so
            # isinstance succeeds there too.
            if not isinstance(_entry, RepoFile):
                continue
            entry = _entry
            if not _path_matches_any(entry.path, allow_compiled):
                continue
            if _path_matches_any(entry.path, ignore_compiled):
                continue
            lfs = entry.lfs
            sha = lfs.sha256 if lfs is not None else None
            out.append(HfFile(path=entry.path, size=int(entry.size), lfs_sha256=sha))
        out.sort(key=lambda f: f.path)
        return out

    def download(
        self,
        repo: str,
        revision: str,
        hf_file: HfFile,
        progress_cb: Callable[[int], None] | None = None,
    ) -> Path:
        """Download to <spool>/sources/<hf_path>.partial, atomic rename to .../<hf_path>.

        If a complete file already exists at the final path with the expected
        size, return it without any HTTP traffic. Atomic-rename guarantees the
        on-disk file is a valid prior download (size + LFS sha256 were already
        verified before the rename); re-hashing on every re-run would defeat
        the optimization.
        """
        if self.spool_dir is None:
            raise RuntimeError("spool_dir required for download()")
        sources = self.spool_dir / "sources"
        final = sources / hf_file.path
        if final.exists() and final.stat().st_size == hf_file.size:
            log.debug("HF download skipped, using cached %s", final)
            return final
        partial = sources / (hf_file.path + ".partial")
        partial.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._stream_to(repo, revision, hf_file, partial, progress_cb)
            partial.replace(final)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                partial.unlink()
            with contextlib.suppress(FileNotFoundError):
                final.unlink()
            raise
        return final

    def _stream_to(
        self,
        repo: str,
        revision: str,
        hf_file: HfFile,
        partial: Path,
        progress_cb: Callable[[int], None] | None,
    ) -> None:
        url = f"{self.api.endpoint.rstrip('/')}/{repo}/resolve/{revision}/{hf_file.path}"
        bytes_done = 0
        h: Any = hashlib.sha256() if hf_file.lfs_sha256 else None

        for attempt in range(self.max_retries):
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError(f"HF download of {hf_file.path} aborted by stop_event")
            try:
                self._stream_one_attempt(url, partial, hf_file, bytes_done, h, progress_cb)
                bytes_done = partial.stat().st_size
                if bytes_done >= hf_file.size:
                    if h is not None and hf_file.lfs_sha256 is not None:
                        got = h.hexdigest()
                        if got != hf_file.lfs_sha256:
                            raise DownloadError(
                                f"sha256 mismatch for {hf_file.path}: "
                                f"expected {hf_file.lfs_sha256}, got {got}",
                                hint="HF cache may be corrupted; delete and re-run.",
                            )
                    return
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status in (403, 404):
                    self._raise_specific_http_error(e, repo, revision, hf_file.path)
                # Transient HTTP error (5xx, 408, 429, etc.) — retry
                log.warning(
                    "HF HTTP %d for %s at %d/%d (attempt %d/%d): %s",
                    status,
                    hf_file.path,
                    bytes_done,
                    hf_file.size,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                bytes_done = partial.stat().st_size if partial.exists() else 0
                self._sleep_backoff(attempt)
                continue
            except _FATAL_TRANSPORT_ERRORS as e:
                if isinstance(e, requests.exceptions.SSLError) and is_transient_ssl(e):
                    log.warning(
                        "HF SSL EOF mid-stream for %s at %d/%d (attempt %d/%d)",
                        hf_file.path,
                        bytes_done,
                        hf_file.size,
                        attempt + 1,
                        self.max_retries,
                    )
                    bytes_done = partial.stat().st_size if partial.exists() else 0
                    self._sleep_backoff(attempt)
                    continue
                raise
            except InterruptedError:
                raise  # stop_event abort — never retry
            except _TRANSIENT_TRANSPORT_ERRORS as e:
                log.warning(
                    "HF read failed for %s at %d/%d (attempt %d/%d): %s",
                    hf_file.path,
                    bytes_done,
                    hf_file.size,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                bytes_done = partial.stat().st_size if partial.exists() else 0
                self._sleep_backoff(attempt)
                continue

        raise DownloadError(
            f"HF retries exhausted for {hf_file.path} at offset {bytes_done}",
            hint=f"--hf-max-retries N (currently {self.max_retries})",
        )

    def _stream_one_attempt(
        self,
        url: str,
        partial: Path,
        hf_file: HfFile,
        start: int,
        h: Any,  # hashlib._Hash | None — typed loosely because hashlib._Hash is private
        progress_cb: Callable[[int], None] | None,
    ) -> None:
        headers = dict(huggingface_auth_header())
        if start > 0:
            headers["Range"] = f"bytes={start}-"
        r = self.session.get(url, headers=headers, stream=True, timeout=(10, 600))
        r.raise_for_status()

        # If we asked for Range but got 200, server ignored it: truncate + restart
        if start > 0 and r.status_code == 200:
            log.info("HF server ignored Range; truncating partial and restarting from 0")
            partial.unlink(missing_ok=True)
            start = 0
            if h is not None:
                h = hashlib.sha256()

        mode = "ab" if start > 0 else "wb"
        with open(partial, mode) as f:
            written = start
            for chunk in r.iter_content(chunk_size=_CHUNK_DEFAULT):
                if self.stop_event is not None and self.stop_event.is_set():
                    raise InterruptedError(f"HF download of {hf_file.path} aborted by stop_event")
                if not chunk:
                    continue
                f.write(chunk)
                if h is not None:
                    h.update(chunk)
                written += len(chunk)
                if progress_cb is not None:
                    progress_cb(written)

    def _raise_specific_http_error(
        self,
        e: requests.exceptions.HTTPError,
        repo: str,
        revision: str,
        path: str,
    ) -> None:
        resp = e.response
        if resp is None:
            raise e
        status = resp.status_code
        if status == 403 and resp.headers.get("X-Error-Code") == "GatedRepo":
            raise GatedRepoError(
                f"Repo {repo} is gated.",
                hint=f"Accept terms at https://huggingface.co/{repo}, then re-run.",
            ) from e
        if status == 404:
            req_url = str(resp.request.url) if resp.request else ""
            if "/resolve/" in req_url and path in req_url:
                raise EntryNotFoundError(f"File not found: {repo}@{revision}/{path}") from e
            raise RevisionNotFoundError(f"Revision not found: {repo}@{revision}") from e
        raise e

    def _sleep_backoff(self, attempt: int) -> None:
        cap_delay = min(self.backoff_cap, self.backoff_initial * (2**attempt))
        if cap_delay > 0:
            time.sleep(random.uniform(0, cap_delay))
