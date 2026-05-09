"""E2E: OCI PATCH sabotage → upload succeeds via file replay.

Verifies that StreamingBlobUpload.push_from_file retries cleanly when the
first PATCH attempt receives a transient HTTP 503. The chaos proxy returns
503 without touching the registry on the first large PATCH so the upload
session Location remains valid. The retry sends the full tar file again
from disk (Jib-style full-PATCH replay) and completes successfully.

Design note on TCP-RST chaos: abruptly resetting the TCP connection on PATCH
destroys registry:2's upload session (it responds BLOB_UPLOAD_INVALID on the
next PATCH). The current StreamingBlobUpload implementation reuses the same
Location across retries without re-POSTing, so TCP-RST does not work as a
recovery scenario with registry:2 — the spec §3.2 d "retry from POST" intent
is not yet reflected in the code. This test therefore targets HTTP-level 503
which leaves the session intact.

Requires Docker.
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import requests

from oci_modelcar.layer import build_layer_to_file
from oci_modelcar.registry import OciClient, StreamingBlobUpload, head_blob

_DROP_AFTER_BYTES = 500_000


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


def _make_chaos_handler(  # type: ignore[no-untyped-def]
    reg_port: int,
    sabotage_done: threading.Event,
) -> type[http.server.BaseHTTPRequestHandler]:
    """Return a request handler class. First large PATCH → 503 (body discarded,
    registry not touched). All other requests → transparent relay to registry."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        # Silence the default access log
        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _relay(self, method: str, body: bytes | None = None) -> None:
            """Forward request to real registry and pipe response back."""
            reg_url = f"http://127.0.0.1:{reg_port}{self.path}"
            # Forward relevant headers
            fwd_headers = {}
            for key in (
                "Content-Type",
                "Content-Length",
                "Content-Range",
                "Authorization",
                "Accept",
            ):
                val = self.headers.get(key)
                if val is not None:
                    fwd_headers[key] = val

            resp = requests.request(
                method,
                reg_url,
                headers=fwd_headers,
                data=body,
                timeout=(30, 120),
                allow_redirects=False,
                stream=True,
            )
            self.send_response(resp.status_code)
            for hdr in (
                "Content-Type",
                "Content-Length",
                "Location",
                "Docker-Upload-Uuid",
                "Docker-Distribution-Api-Version",
                "Range",
                "Docker-Content-Digest",
            ):
                val = resp.headers.get(hdr)
                if val is not None:
                    # Location URLs from the registry use the internal port;
                    # rewrite to proxy port so the client keeps routing through us.
                    if hdr == "Location":
                        val = val.replace(
                            f"127.0.0.1:{reg_port}", f"localhost:{self.server.server_address[1]}"
                        ).replace(
                            f"localhost:{reg_port}", f"localhost:{self.server.server_address[1]}"
                        )
                    self.send_header(hdr, val)
            self.end_headers()
            for chunk in resp.iter_content(8192):
                if chunk:
                    self.wfile.write(chunk)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return b""
            return self.rfile.read(length)

        def do_GET(self) -> None:
            self._relay("GET")

        def do_HEAD(self) -> None:
            self._relay("HEAD")

        def do_POST(self) -> None:
            body = self._read_body()
            self._relay("POST", body)

        def do_PUT(self) -> None:
            body = self._read_body()
            self._relay("PUT", body)

        def do_PATCH(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            is_large = content_length >= _DROP_AFTER_BYTES

            if is_large and not sabotage_done.is_set():
                sabotage_done.set()
                # Drain the request body so the client connection stays clean
                remaining = content_length
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 65536))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                # Return 503 without touching the registry
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            # For all other PATCHes, relay to the registry streaming
            body = self._read_body()
            self._relay("PATCH", body)

    return _Handler


@pytest.fixture(scope="module")
def chaos_oci_setup():  # type: ignore[no-untyped-def]
    """Start registry:2 + chaos proxy that returns 503 on first large PATCH."""
    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        pytest.skip("docker not available")

    reg_port = _free_port()
    proxy_port = _free_port()

    reg_name = f"oci-modelcar-chaos-reg-{reg_port}"
    subprocess.run(["docker", "rm", "-f", reg_name], check=False, capture_output=True)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            reg_name,
            "-p",
            f"{reg_port}:5000",
            "registry:2",
        ],
        check=True,
        capture_output=True,
    )
    _wait_registry(reg_port)

    sabotage_done = threading.Event()
    handler_cls = _make_chaos_handler(reg_port, sabotage_done)

    class _ThreadedServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    proxy_server = _ThreadedServer(("127.0.0.1", proxy_port), handler_cls)
    proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    proxy_thread.start()

    yield {
        "registry_port": reg_port,
        "proxy_port": proxy_port,
        "sabotage_done": sabotage_done,
    }

    proxy_server.shutdown()
    subprocess.run(["docker", "stop", reg_name], check=False, capture_output=True)


@pytest.mark.e2e
def test_oci_patch_retry_through_chaos_proxy(
    chaos_oci_setup: dict,  # type: ignore[type-arg]
    tmp_path: Path,
) -> None:
    """Push a 1 MiB+ blob through a chaos proxy that returns 503 on the
    first PATCH. The upload must survive by replaying from the spool file
    and land in the real registry.

    This verifies the transient-HTTP-error retry path in
    StreamingBlobUpload.push_from_file under real wire conditions.
    """
    source = tmp_path / "payload.bin"
    source.write_bytes(b"R" * (1024 * 1024))  # 1 MiB
    tar_path = tmp_path / "payload.bin.tar"
    digest, layer_size = build_layer_to_file(source, "models/", "payload.bin", tar_path)

    assert layer_size > _DROP_AFTER_BYTES, (
        f"tar layer ({layer_size}) must be > {_DROP_AFTER_BYTES} to trigger sabotage"
    )

    proxy_url = f"http://localhost:{chaos_oci_setup['proxy_port']}"
    client = OciClient(host_url=proxy_url, target_repo="e2e/chaos")
    upload = StreamingBlobUpload(
        client=client,
        repo="e2e/chaos",
        max_retries=3,
        backoff_initial=0.0,
    )
    out_digest, out_size = upload.push_from_file(tar_path, layer_size, digest)

    assert out_digest == digest
    assert out_size == layer_size
    assert chaos_oci_setup["sabotage_done"].is_set(), (
        "chaos proxy should have returned 503 on the first PATCH"
    )

    # Verify the blob landed in the real registry (bypass proxy)
    real_client = OciClient(
        host_url=f"http://localhost:{chaos_oci_setup['registry_port']}",
        target_repo="e2e/chaos",
    )
    info = head_blob(real_client, "e2e/chaos", digest)
    assert info is not None, "blob must be present in real registry after retry"
    assert info["digest"] == digest
    assert info["size"] == layer_size
