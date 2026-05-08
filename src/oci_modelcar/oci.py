"""OCI Distribution v1.1 client: chunked blob upload, blob/manifest validation."""

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from dataclasses import dataclass
from functools import cached_property

import requests

from oci_modelcar.http import build_session, is_transient_ssl, oci_auth_header

log = logging.getLogger(__name__)

ML_TAR = "application/vnd.oci.image.layer.v1.tar"


def _is_loopback(host: str) -> bool:
    """True if host is a loopback/local address (no TLS by default)."""
    # IPv6 bare address (e.g. "::1") — no port stripping needed
    if host == "::1" or host.startswith("["):
        return host in ("::1", "[::1]")
    h = host.split(":", 1)[0]
    if h in ("localhost", "127.0.0.1"):
        return True
    return h.startswith("127.")


ML_CFG = "application/vnd.oci.image.config.v1+json"
ML_MAN = "application/vnd.oci.image.manifest.v1+json"


@dataclass(frozen=True, slots=True)
class BlobDescriptor:
    media_type: str
    digest: str
    size: int

    def to_dict(self) -> dict[str, object]:
        return {"mediaType": self.media_type, "digest": self.digest, "size": self.size}


class OciClient:
    def __init__(
        self,
        host_url: str | None = None,
        registry_host: str | None = None,
        session: requests.Session | None = None,
        target_repo: str | None = None,
    ) -> None:
        if host_url is not None:
            self.base = host_url.rstrip("/")
            self.host = self.base.split("//", 1)[-1]
        else:
            assert registry_host is not None
            self.host = registry_host
            # Allow callers to encode scheme in registry_host (e.g. "http://localhost:5000")
            if registry_host.startswith(("http://", "https://")):
                self.base = registry_host.rstrip("/")
                self.host = registry_host.split("//", 1)[-1]
            elif _is_loopback(registry_host):
                self.base = f"http://{registry_host}"
            else:
                self.base = f"https://{registry_host}"
        self.session = session if session is not None else build_session()
        self.target_repo = target_repo

    @cached_property
    def auth(self) -> dict[str, str]:
        # Resolved once per client; the source-resolution log fires only on
        # first access, not on every PATCH chunk.
        return oci_auth_header(self.host, target_repo=self.target_repo)

    def url(self, *parts: str) -> str:
        return f"{self.base}/v2/" + "/".join(parts)


class ChunkedBlobUpload:
    """Streaming blob upload with PATCH chunks and PUT finalization.

    Memory bound: ~2 * chunk_size.
    Compliant with OCI Distribution v1.1: Content-Range is inclusive 'N-M'.
    """

    def __init__(
        self,
        client: OciClient,
        repo: str,
        chunk_size: int = 8 * 1024 * 1024,
        max_retries: int = 10,
        backoff_initial: float = 1.0,
        backoff_cap: float = 60.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.client = client
        self.repo = repo
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_cap = backoff_cap
        self.stop_event = stop_event
        self.h = hashlib.sha256()
        self.buf = bytearray()
        self.server_offset = 0
        self.total = 0
        self.location = self._begin()

    def _begin(self) -> str:
        url = self.client.url(self.repo, "blobs", "uploads") + "/"
        r = self.client.session.post(url, headers=self.client.auth, timeout=60)
        if r.status_code != 202:
            r.raise_for_status()
            raise RuntimeError(f"unexpected status {r.status_code} on upload init")
        loc = r.headers.get("Location")
        if not loc:
            raise RuntimeError("upload init missing Location header")
        return loc

    def write(self, data: bytes) -> int:
        n = len(data)
        self.h.update(data)
        self.buf.extend(data)
        self.total += n
        while len(self.buf) >= self.chunk_size:
            self._flush(self.chunk_size)
        return n

    def _flush(self, size: int) -> None:
        chunk = bytes(self.buf[:size])
        del self.buf[:size]
        self._patch_with_retry(chunk)

    def _patch_with_retry(self, chunk: bytes) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise InterruptedError(
                f"OCI upload to {self.repo} aborted by stop_event at offset {self.server_offset}"
            )
        # The chunk owns the byte range [initial_offset .. target_end] of the
        # blob, fixed at entry. As the server commits partial bytes (visible
        # via _resync), each retry re-slices the chunk so we never re-send
        # already-acked bytes — otherwise the server returns 416 forever.
        initial_offset = self.server_offset
        target_end = initial_offset + len(chunk) - 1
        # Progress-aware retry budget: a retry that walked the offset forward
        # proves the connection works at least intermittently, so we refresh
        # the budget for the rest of the chunk. A long upload through a flaky
        # proxy can survive an arbitrary number of cuts as long as each one
        # makes some progress. Budget only exhausts on consecutive zero-progress
        # failures.
        attempts_left = self.max_retries
        backoff_idx = 0
        last_offset = self.server_offset
        while True:
            if self.server_offset > last_offset:
                attempts_left = self.max_retries
                backoff_idx = 0
            last_offset = self.server_offset
            if attempts_left <= 0:
                raise RuntimeError(
                    f"PATCH retries exhausted at offset {self.server_offset} "
                    f"(target {target_end}, started at {initial_offset})"
                )
            bytes_done = self.server_offset - initial_offset
            if bytes_done >= len(chunk):
                return
            slice_chunk = chunk[bytes_done:]
            slice_start = self.server_offset
            try:
                hdr = {
                    **self.client.auth,
                    "Content-Type": "application/octet-stream",
                    "Content-Range": f"{slice_start}-{target_end}",  # OCI: inclusive, no "bytes " prefix
                    "Content-Length": str(len(slice_chunk)),
                }
                r = self.client.session.patch(
                    self.location, data=slice_chunk, headers=hdr, timeout=600
                )
                if r.status_code == 202:
                    self.location = r.headers.get("Location", self.location)
                    self.server_offset = target_end + 1
                    return
                if r.status_code == 416:
                    log.warning("PATCH 416 at [%d-%d], resyncing", slice_start, target_end)
                    self._resync()
                    if self.server_offset >= target_end + 1:
                        return
                    attempts_left -= 1
                    backoff_idx += 1
                    continue
                if r.status_code in (408, 429) or 500 <= r.status_code < 600:
                    log.warning(
                        "PATCH transient %d at [%d-%d] attempt %d",
                        r.status_code,
                        slice_start,
                        target_end,
                        backoff_idx + 1,
                    )
                    self._sleep_backoff(backoff_idx)
                    self._resync()
                    if self.server_offset >= target_end + 1:
                        return
                    attempts_left -= 1
                    backoff_idx += 1
                    continue
                r.raise_for_status()
            except (requests.exceptions.SSLError, requests.exceptions.ProxyError) as e:
                # Cert / proxy misconfig never recovers. But an SSL EOF after
                # bytes already flowed is a mid-stream connection cut — treat
                # like any other transient: backoff + resync + retry.
                if isinstance(e, requests.exceptions.SSLError) and is_transient_ssl(e):
                    log.warning(
                        "PATCH SSL EOF [%d-%d] attempt %d, resyncing",
                        slice_start,
                        target_end,
                        backoff_idx + 1,
                    )
                    self._sleep_backoff(backoff_idx)
                    self._resync()
                    if self.server_offset >= target_end + 1:
                        return
                    attempts_left -= 1
                    backoff_idx += 1
                    continue
                raise
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                log.warning(
                    "PATCH failed [%d-%d] attempt %d: %s",
                    slice_start,
                    target_end,
                    backoff_idx + 1,
                    e,
                )
                self._sleep_backoff(backoff_idx)
                self._resync()
                if self.server_offset >= target_end + 1:
                    return
                attempts_left -= 1
                backoff_idx += 1

    def _resync(self) -> None:
        r = self.client.session.get(self.location, headers=self.client.auth, timeout=30)
        if r.status_code != 204:
            r.raise_for_status()
        rng = r.headers.get("Range", "")
        if rng:
            try:
                end = int(rng.split("-")[1])
                self.server_offset = end + 1
            except (ValueError, IndexError):  # fmt: skip
                self.server_offset = 0
        else:
            self.server_offset = 0

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, self.backoff_initial * (2**attempt))
        delay += random.uniform(0, delay * 0.1)
        if delay > 0:
            time.sleep(delay)

    def close(self) -> tuple[str, int]:
        digest = "sha256:" + self.h.hexdigest()
        sep = "&" if "?" in self.location else "?"
        url = f"{self.location}{sep}digest={digest}"
        if self.buf:
            hdr = {
                **self.client.auth,
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(self.buf)),
            }
            r = self.client.session.put(url, data=bytes(self.buf), headers=hdr, timeout=600)
        else:
            r = self.client.session.put(url, headers=self.client.auth, timeout=120)
        if r.status_code != 201:
            r.raise_for_status()
            raise RuntimeError(f"unexpected status {r.status_code} on PUT close")
        return digest, self.total


def push_small_blob(client: OciClient, repo: str, data: bytes) -> str:
    """Monolithic POST + PUT for small blobs (config). Returns digest."""
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    head_url = client.url(repo, "blobs", digest)
    h = client.session.head(head_url, headers=client.auth, timeout=30)
    if h.status_code == 200:
        return digest
    init_url = client.url(repo, "blobs", "uploads") + "/"
    r = client.session.post(init_url, headers=client.auth, timeout=30)
    if r.status_code != 202:
        r.raise_for_status()
    loc = r.headers["Location"]
    sep = "&" if "?" in loc else "?"
    hdr = {
        **client.auth,
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(data)),
    }
    r = client.session.put(f"{loc}{sep}digest={digest}", data=data, headers=hdr, timeout=120)
    if r.status_code != 201:
        r.raise_for_status()
    return digest


def head_blob(client: OciClient, repo: str, digest: str) -> dict[str, object]:
    """HEAD a blob, validate Docker-Content-Digest, return {digest, size}."""
    url = client.url(repo, "blobs", digest)
    r = client.session.head(url, headers=client.auth, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"blob not found in {repo}: {digest}")
    got = r.headers.get("Docker-Content-Digest", "")
    if got != digest:
        raise RuntimeError(f"digest mismatch on HEAD {digest}: server returned {got!r}")
    cl = r.headers.get("Content-Length", "0")
    return {"digest": digest, "size": int(cl)}


def push_manifest(client: OciClient, repo: str, tag: str, manifest_bytes: bytes) -> str:
    url = client.url(repo, "manifests", tag)
    hdr = {**client.auth, "Content-Type": ML_MAN}
    r = client.session.put(url, data=manifest_bytes, headers=hdr, timeout=60)
    if r.status_code not in (200, 201):
        r.raise_for_status()
    return "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()


def validate_manifest_tag(client: OciClient, repo: str, tag: str, expected_digest: str) -> None:
    url = client.url(repo, "manifests", tag)
    r = client.session.get(url, headers={**client.auth, "Accept": ML_MAN}, timeout=30)
    r.raise_for_status()
    got = r.headers.get("Docker-Content-Digest", "")
    if got != expected_digest:
        raise RuntimeError(
            f"manifest digest mismatch on tag {tag}: expected {expected_digest} got {got!r}"
        )
