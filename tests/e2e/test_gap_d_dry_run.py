"""Gap D: --dry-run.

Should perform pre-flight (revision resolve + file listing + disk check)
but write NOTHING to the registry and download NOTHING to spool.

Verifies:
- Exit 0
- No POST/PATCH/PUT requests reach the registry (only GET/HEAD pre-flight)
- spool/sources/ is empty (no files downloaded)
- spool/layers/ is empty (no tar files created)

Uses an intercepting HTTP proxy to record every request method seen.

Requires Docker + network access to huggingface.co.
"""

from __future__ import annotations

import http.server
import os
import socket
import socketserver
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import requests

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


def _make_recording_handler(
    reg_port: int,
    seen_methods: list[str],
    lock: threading.Lock,
) -> type[http.server.BaseHTTPRequestHandler]:
    """Proxy that records every HTTP method seen from the client."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _relay(self, method: str, body: bytes | None = None) -> None:
            with lock:
                seen_methods.append(method)
            reg_url = f"http://127.0.0.1:{reg_port}{self.path}"
            fwd_headers = {}
            for key in ("Content-Type", "Content-Length", "Authorization", "Accept"):
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
                    if hdr == "Location":
                        proxy_port = self.server.server_address[1]
                        val = val.replace(
                            f"127.0.0.1:{reg_port}", f"localhost:{proxy_port}"
                        ).replace(f"localhost:{reg_port}", f"localhost:{proxy_port}")
                    self.send_header(hdr, val)
            self.end_headers()
            for chunk in resp.iter_content(8192):
                if chunk:
                    self.wfile.write(chunk)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0"))
            return self.rfile.read(length) if length > 0 else b""

        def do_GET(self) -> None:
            self._relay("GET")

        def do_HEAD(self) -> None:
            self._relay("HEAD")

        def do_POST(self) -> None:
            self._relay("POST", self._read_body())

        def do_PUT(self) -> None:
            self._relay("PUT", self._read_body())

        def do_PATCH(self) -> None:
            self._relay("PATCH", self._read_body())

    return _Handler


@pytest.fixture(scope="module")
def dry_run_setup():  # type: ignore[no-untyped-def]
    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        pytest.skip("docker not available")

    reg_port = _free_port()
    proxy_port = _free_port()
    reg_name = f"oci-modelcar-dry-reg-{reg_port}"
    subprocess.run(["docker", "rm", "-f", reg_name], check=False, capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", reg_name, "-p", f"{reg_port}:5000", "registry:2"],
        check=True,
        capture_output=True,
    )
    _wait_registry(reg_port)

    seen_methods: list[str] = []
    lock = threading.Lock()
    handler_cls = _make_recording_handler(reg_port, seen_methods, lock)

    class _ThreadedServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    proxy = _ThreadedServer(("127.0.0.1", proxy_port), handler_cls)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()

    yield {
        "registry_port": reg_port,
        "proxy_port": proxy_port,
        "seen_methods": seen_methods,
        "lock": lock,
    }

    proxy.shutdown()
    subprocess.run(["docker", "stop", reg_name], check=False, capture_output=True)


@pytest.mark.e2e
def test_dry_run_writes_nothing(dry_run_setup: dict, tmp_path: Path) -> None:  # type: ignore[type-arg]
    """--dry-run must exit 0 with ZERO POST/PATCH/PUT to the registry.

    The recording proxy intercepts every request. After --dry-run, we assert
    that only GET and HEAD methods were used (pre-flight only).
    """
    seen_methods = dry_run_setup["seen_methods"]
    lock = dry_run_setup["lock"]
    proxy_port = dry_run_setup["proxy_port"]

    with lock:
        seen_methods.clear()

    spool = tmp_path / "spool_dry"
    env = os.environ.copy()
    cmd = [
        "oci-modelcar",
        "push",
        "--hf-repo",
        HF_REPO,
        "--hf-revision",
        HF_REVISION,
        "--registry",
        f"localhost:{proxy_port}",
        "--target-repo",
        "e2e/dry-run-llama",
        "--spool-dir",
        str(spool),
        "--dry-run",
        "--quiet",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"--dry-run failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    with lock:
        methods_snapshot = list(seen_methods)

    # Must have NO mutating requests
    mutating = [m for m in methods_snapshot if m in ("POST", "PATCH", "PUT", "DELETE")]
    assert mutating == [], (
        f"--dry-run sent mutating requests to the registry: {mutating}\n"
        f"All methods seen: {methods_snapshot}\n"
        "No POST/PATCH/PUT should reach the registry during --dry-run."
    )

    # Must have made at least some requests (the HEAD for tag existence check)
    assert methods_snapshot, (
        "--dry-run made NO requests to the registry at all — "
        "the pre-flight HEAD (tag existence check) should still happen."
    )

    # Spool must be empty — no sources downloaded
    sources_dir = spool / "sources"
    layers_dir = spool / "layers"

    sources_files = (
        [p for p in sources_dir.rglob("*") if p.is_file()] if sources_dir.exists() else []
    )
    layers_files = [p for p in layers_dir.rglob("*") if p.is_file()] if layers_dir.exists() else []

    assert sources_files == [], (
        "--dry-run downloaded source files to spool/sources/:\n"
        + "\n".join(str(p) for p in sources_files)
    )
    assert layers_files == [], "--dry-run created tar files in spool/layers/:\n" + "\n".join(
        str(p) for p in layers_files
    )
