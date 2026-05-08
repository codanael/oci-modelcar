"""OCI Distribution v1.1 client. v1: single-PATCH streaming upload from
a local file. No chunked mode."""

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from functools import cached_property
from pathlib import Path

import requests

from oci_modelcar.errors import PushError
from oci_modelcar.http import build_session, is_transient_ssl, oci_auth_header
from oci_modelcar.manifest import ML_MAN

log = logging.getLogger(__name__)


def _is_loopback(host: str) -> bool:
    if host == "::1" or host.startswith("["):
        return host in ("::1", "[::1]")
    h = host.split(":", 1)[0]
    if h in ("localhost", "127.0.0.1"):
        return True
    return h.startswith("127.")


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
        return oci_auth_header(self.host, target_repo=self.target_repo)

    def url(self, *parts: str) -> str:
        return f"{self.base}/v2/" + "/".join(parts)


def head_blob(client: OciClient, repo: str, digest: str) -> dict[str, object] | None:
    """HEAD a blob. Returns {digest, size} on 200, None on 404. Raises on
    digest mismatch."""
    url = client.url(repo, "blobs", digest)
    r = client.session.head(url, headers=client.auth, timeout=30)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        r.raise_for_status()
    got = r.headers.get("Docker-Content-Digest", "")
    if got != digest:
        raise RuntimeError(f"digest mismatch on HEAD {digest}: server returned {got!r}")
    cl = int(r.headers.get("Content-Length", "0"))
    return {"digest": digest, "size": cl}


def push_small_blob(client: OciClient, repo: str, data: bytes) -> str:
    """Monolithic POST + PUT for small blobs (config). Returns digest."""
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    if head_blob(client, repo, digest) is not None:
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


def push_manifest(client: OciClient, repo: str, tag: str, manifest_bytes: bytes) -> str:
    url = client.url(repo, "manifests", tag)
    hdr = {**client.auth, "Content-Type": ML_MAN}
    r = client.session.put(url, data=manifest_bytes, headers=hdr, timeout=60)
    if r.status_code not in (200, 201):
        r.raise_for_status()
        raise RuntimeError(f"unexpected status {r.status_code} on manifest PUT")
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


class StreamingBlobUpload:
    """Single-PATCH streaming blob upload, body sourced from a local file.

    Mirrors containers/image (Podman) and Jib: one PATCH per blob means one
    TCP request, one LB routing decision, one node receives the entire blob.
    The local file is the replayable source for retries.
    """

    def __init__(
        self,
        client: OciClient,
        repo: str,
        max_retries: int = 5,
        backoff_initial: float = 1.0,
        backoff_cap: float = 60.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.client = client
        self.repo = repo
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_cap = backoff_cap
        self.stop_event = stop_event

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

    def push_from_file(self, tar_path: Path, total_size: int, digest: str) -> tuple[str, int]:
        """POST init → PATCH from file (full replay on cut) → PUT close.

        Each retry attempt re-opens tar_path and rewinds to offset 0.
        Backoff: full jitter Uniform(0, min(cap, base * 2^attempt)).
        Accepts {200, 201, 202, 204} on PATCH (Artifactory + Harbor quirks).
        """
        if self.stop_event is not None and self.stop_event.is_set():
            raise InterruptedError(f"OCI upload to {self.repo} aborted before start")
        location = self._begin()
        last_exc: BaseException | None = None

        success = False
        for attempt in range(self.max_retries):
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError(
                    f"OCI upload to {self.repo} aborted before attempt {attempt + 1}"
                )
            if attempt > 0:
                self._sleep_backoff(attempt - 1)
            try:
                with open(tar_path, "rb") as body:
                    hdr = {
                        **self.client.auth,
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(total_size),
                        "Content-Range": f"0-{total_size - 1}",
                    }
                    r = self.client.session.patch(
                        location, data=body, headers=hdr, timeout=(30, 600)
                    )
                if r.status_code in (200, 201, 202, 204):
                    location = r.headers.get("Location", location)
                    success = True
                    break
                if r.status_code in (408, 429) or 500 <= r.status_code < 600:
                    log.warning(
                        "PATCH transient %d for %s attempt %d/%d",
                        r.status_code,
                        self.repo,
                        attempt + 1,
                        self.max_retries,
                    )
                    last_exc = RuntimeError(f"transient {r.status_code}")
                    continue
                r.raise_for_status()
                raise RuntimeError(
                    f"unexpected PATCH status {r.status_code} for streaming upload to {self.repo}"
                )
            except requests.exceptions.SSLError as e:
                if not is_transient_ssl(e):
                    raise
                log.warning(
                    "PATCH SSL EOF for %s attempt %d/%d, will retry from offset 0",
                    self.repo,
                    attempt + 1,
                    self.max_retries,
                )
                last_exc = e
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                log.warning(
                    "PATCH transient transport for %s attempt %d/%d: %s",
                    self.repo,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                last_exc = e

        if not success:
            raise PushError(
                f"OCI PATCH retries exhausted ({self.max_retries}) for {self.repo}: "
                f"last error {last_exc!r}",
                hint=f"--oci-max-retries N (currently {self.max_retries}), or check registry health.",
            )

        sep = "&" if "?" in location else "?"
        url = f"{location}{sep}digest={digest}"
        rp = self.client.session.put(url, headers=self.client.auth, timeout=120)
        if rp.status_code != 201:
            rp.raise_for_status()
            raise RuntimeError(f"unexpected status {rp.status_code} on PUT close")
        return digest, total_size

    def _sleep_backoff(self, attempt: int) -> None:
        cap_delay = min(self.backoff_cap, self.backoff_initial * (2**attempt))
        if cap_delay > 0:
            time.sleep(random.uniform(0, cap_delay))
