"""HuggingFace client: revision resolution + file listing + streaming."""

from __future__ import annotations

import contextlib
import http.client
import logging
import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import requests
import urllib3.exceptions

from oci_modelcar.http import build_session, huggingface_auth_header, is_transient_ssl

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
        progress_cb: Callable[[int], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.client = client
        self.revision = revision
        self.path = path
        self.expected_size = size
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_cap = backoff_cap
        self.progress_cb = progress_cb
        self.stop_event = stop_event
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
            pct = 100.0 * start / self.expected_size if self.expected_size > 0 else 0.0
            log.info(
                "resuming %s at offset %d/%d (%.1f%%)",
                self.path,
                start,
                self.expected_size,
                pct,
            )
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
        # Full-jitter pattern (https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/):
        # sleep ~ Uniform(0, min(cap, base * 2^attempt)). Wider spread than
        # exponential-plus-additive-jitter — necessary when many clients all
        # retry against a recovering proxy at once.
        cap_delay = min(self.backoff_cap, self.backoff_initial * (2**attempt))
        if cap_delay > 0:
            time.sleep(random.uniform(0, cap_delay))

    def _next_chunk(self) -> bytes | None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise InterruptedError(f"HF read of {self.path} aborted by stop_event")
        for attempt in range(self.max_retries):
            try:
                if self._it is None:
                    self._open(start=self.bytes_buffered)
                assert self._it is not None
                chunk = next(self._it)
                self.bytes_buffered += len(chunk)
                if self.progress_cb is not None:
                    self.progress_cb(self.bytes_buffered)
                return chunk
            except StopIteration:
                if self.bytes_buffered >= self.expected_size:
                    return None
                log.warning(
                    "HF stream ended early for %s at %d/%d (attempt %d/%d)",
                    self.path,
                    self.bytes_buffered,
                    self.expected_size,
                    attempt + 1,
                    self.max_retries,
                )
                self._close_response()
                self._sleep_backoff(attempt)
            except _FATAL_TRANSPORT_ERRORS as e:
                # SSL handshake / proxy misconfig never recovers — surface the
                # real cause immediately. But an SSL EOF after bytes already
                # flowed is just a connection cut; resume via Range like any
                # other transient transport error.
                if isinstance(e, requests.exceptions.SSLError) and is_transient_ssl(e):
                    log.warning(
                        "HF SSL EOF mid-stream for %s at %d/%d (attempt %d/%d); resuming via Range",
                        self.path,
                        self.bytes_buffered,
                        self.expected_size,
                        attempt + 1,
                        self.max_retries,
                    )
                    self._close_response()
                    self._sleep_backoff(attempt)
                    continue
                self._close_response()
                raise
            except _TRANSIENT_TRANSPORT_ERRORS as e:
                log.warning(
                    "HF read failed for %s at offset %d/%d (attempt %d/%d): %s",
                    self.path,
                    self.bytes_buffered,
                    self.expected_size,
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
            return b"".join(chunks)
        while len(self.buf) < n:
            c = self._next_chunk()
            if c is None:
                break
            self.buf += c
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def close(self) -> None:
        self._close_response()
