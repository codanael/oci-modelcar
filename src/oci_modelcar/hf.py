"""HuggingFace client: revision resolution + file listing + streaming."""

from __future__ import annotations

import contextlib
import logging
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass

import requests

from oci_modelcar.http import build_session, huggingface_auth_header

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HfFile:
    path: str
    size: int


class HfClient:
    def __init__(
        self,
        endpoint: str,
        repo: str,
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.repo = repo
        self.session = session if session is not None else build_session()
        self.timeout = timeout

    @property
    def auth(self) -> dict[str, str]:
        return huggingface_auth_header()

    def resolve_revision(self, revision: str) -> str:
        """Resolve a revision (branch/tag/SHA/'main') to a 40-char SHA."""
        if revision == "main" or not revision:
            url = f"{self.endpoint}/api/models/{self.repo}"
            r = self.session.get(url, headers=self.auth, timeout=self.timeout)
            r.raise_for_status()
            sha = r.json().get("sha")
            if not sha:
                log.warning("HF /api/models/%s did not return sha", self.repo)
                return revision or "main"
            return str(sha)
        url = f"{self.endpoint}/api/models/{self.repo}/revision/{revision}"
        r = self.session.get(url, headers=self.auth, timeout=self.timeout)
        if r.status_code == 404:
            log.warning(
                "HF revision %r not canonicalizable on %s/%s; using as-is",
                revision,
                self.endpoint,
                self.repo,
            )
            return revision
        r.raise_for_status()
        sha = r.json().get("sha")
        return str(sha) if sha else revision

    def list_files(self, revision: str, allow: tuple[str, ...]) -> list[HfFile]:
        """Return [HfFile, ...] sorted by path, filtered by extension."""
        url = f"{self.endpoint}/api/models/{self.repo}/tree/{revision}"
        r = self.session.get(
            url,
            headers=self.auth,
            params={"recursive": "true"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        out: list[HfFile] = []
        for entry in r.json():
            if entry.get("type") != "file":
                continue
            path = entry["path"]
            if not any(path.endswith(ext) for ext in allow):
                continue
            out.append(HfFile(path=path, size=int(entry["size"])))
        out.sort(key=lambda f: f.path)
        return out


CHUNK_DEFAULT = 1024 * 1024  # 1 MiB iter_content size


class HfStream:
    """File-like, read-only. Honors read(n). Resumes via HTTP Range on errors."""

    def __init__(
        self,
        client: HfClient,
        revision: str,
        path: str,
        size: int,
        chunk_size: int = CHUNK_DEFAULT,
        max_retries: int = 10,
        backoff_initial: float = 1.0,
        backoff_cap: float = 60.0,
    ) -> None:
        self.client = client
        self.revision = revision
        self.path = path
        self.expected_size = size
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_cap = backoff_cap
        self.bytes_buffered = 0
        self.buf = b""
        self._r: requests.Response | None = None
        self._it: Iterator[bytes] | None = None
        self._open(start=0)

    def _open(self, start: int) -> None:
        url = f"{self.client.endpoint}/{self.client.repo}/resolve/{self.revision}/{self.path}"
        headers = dict(self.client.auth)
        if start > 0:
            headers["Range"] = f"bytes={start}-"
        r = self.client.session.get(url, headers=headers, stream=True, timeout=600)
        r.raise_for_status()
        if start == 0:
            cl_header = r.headers.get("content-length")
            if cl_header is not None:
                cl = int(cl_header)
                if cl != self.expected_size:
                    raise RuntimeError(
                        f"size mismatch for {self.path}: tree={self.expected_size} got={cl}"
                    )
        else:
            cr = r.headers.get("content-range", "")
            if not cr.startswith(f"bytes {start}-"):
                raise RuntimeError(
                    f"server did not honor Range for {self.path}: got Content-Range={cr!r}"
                )
        self._r = r
        self._it = r.iter_content(chunk_size=self.chunk_size)

    def _close_response(self) -> None:
        if self._r is not None:
            with contextlib.suppress(Exception):
                self._r.close()
        self._r = None
        self._it = None

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, self.backoff_initial * (2**attempt))
        delay += random.uniform(0, delay * 0.1)
        if delay > 0:
            time.sleep(delay)

    def _next_chunk(self) -> bytes | None:
        for attempt in range(self.max_retries):
            try:
                if self._it is None:
                    self._open(start=self.bytes_buffered)
                assert self._it is not None
                chunk = next(self._it)
                self.bytes_buffered += len(chunk)
                return chunk
            except StopIteration:
                return None
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ReadTimeout,
            ) as e:
                log.warning(
                    "HF read failed at offset %d (attempt %d/%d): %s",
                    self.bytes_buffered,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                self._close_response()
                self._sleep_backoff(attempt)
        raise RuntimeError(f"HF retries exhausted for {self.path} at offset {self.bytes_buffered}")

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            chunks = [self.buf]
            self.buf = b""
            while True:
                c = self._next_chunk()
                if c is None:
                    break
                chunks.append(c)
            data = b"".join(chunks)
            if self.bytes_buffered < self.expected_size:
                # Try to resume
                self._close_response()
                self._open(start=self.bytes_buffered)
                while self.bytes_buffered < self.expected_size:
                    c = self._next_chunk()
                    if c is None:
                        break
                    data += c
                if self.bytes_buffered < self.expected_size:
                    raise RuntimeError(
                        f"truncated read for {self.path}: "
                        f"got {self.bytes_buffered} expected {self.expected_size}"
                    )
            return data
        # Bounded read(n)
        while len(self.buf) < n:
            c = self._next_chunk()
            if c is None:
                # Truncation? Try resume
                if self.bytes_buffered < self.expected_size:
                    self._close_response()
                    self._open(start=self.bytes_buffered)
                    continue
                break
            self.buf += c
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def close(self) -> None:
        self._close_response()
