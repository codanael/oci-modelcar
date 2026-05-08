#!/usr/bin/env python3
"""HF download diagnostic probe.

Compares wget / curl / Python requests against the same HuggingFace
resolve URL (or any URL). Useful when a proxy or AV treats specific
clients differently and oci-modelcar fails where wget succeeds (or vice
versa).

The probe reports for each backend:
  - bytes received
  - wall time
  - throughput (MB/s)
  - error type and message (if any)
  - byte offset at which the failure happened (requests backend only,
    via streaming counter)

It also lets you toggle the levers most likely to interact with proxy
behavior:
  --connection-close      disable HTTP keep-alive
  --user-agent            mimic wget / curl / a custom UA
  --chunk-size            tweak the requests iter_content chunk size
  --range-start/--range-end   bracket the failure (e.g. start past the AV
                              scan threshold to confirm it's the cause)
  --insecure              disable TLS cert validation (suspecting MITM)
  --debug-http            urllib3 + http.client wire logs for the requests
                          backend; -v / --debug for curl / wget

Examples
--------

# Compare all three backends, full file, defaults:
python tools/hf_download_probe.py \\
    --hf-repo Qwen/Qwen2.5-0.5B-Instruct \\
    --file model.safetensors

# Reproduce a specific failure with requests, with verbose levers:
python tools/hf_download_probe.py \\
    --url https://hf.proxy.local/.../model.safetensors \\
    --backend requests \\
    --connection-close \\
    --user-agent 'Wget/1.21.4' \\
    --chunk-size 65536

# Skip the first 1.3 GB to focus on the post-threshold behavior:
python tools/hf_download_probe.py \\
    --url https://... \\
    --backend all \\
    --range-start 1395864371

Auth: HF_TOKEN env var is honored (Bearer header injected into all
backends).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import requests


def enable_http_debug() -> None:
    """urllib3 + http.client wire-level debug logging on stderr.

    Delegates to ``oci_modelcar.http._maybe_enable_http_debug`` so the
    body-truncating print wrapper (which keeps PATCH/PUT bodies out of
    the output) is installed identically to a production run with
    ``OCI_MODELCAR_DEBUG_HTTP=1``.
    """
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s")
    os.environ["OCI_MODELCAR_DEBUG_HTTP"] = "1"
    from oci_modelcar.http import _maybe_enable_http_debug  # type: ignore[import-untyped]

    _maybe_enable_http_debug()


@dataclass
class ProbeResult:
    backend: str
    bytes: int = 0
    duration_s: float = 0.0
    error: str | None = None
    error_at_offset: int | None = None
    chunks: int = 0
    extra: list[str] = field(default_factory=list)

    @property
    def speed_mbps(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return (self.bytes / 1_048_576.0) / self.duration_s

    @property
    def ok(self) -> bool:
        return self.error is None and self.bytes > 0


def _hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    cache = Path.home() / ".cache" / "huggingface" / "token"
    if cache.is_file():
        try:
            return cache.read_text().strip() or None
        except OSError:
            return None
    return None


def build_url(repo: str, revision: str, file: str, endpoint: str) -> str:
    return f"{endpoint.rstrip('/')}/{repo}/resolve/{revision}/{file}"


def build_headers(
    user_agent: str | None,
    connection_close: bool,
    range_start: int,
    range_end: int | None,
) -> dict[str, str]:
    h: dict[str, str] = {}
    tok = _hf_token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    if user_agent:
        h["User-Agent"] = user_agent
    if connection_close:
        h["Connection"] = "close"
    if range_start > 0 or range_end is not None:
        end_str = str(range_end) if range_end is not None else ""
        h["Range"] = f"bytes={range_start}-{end_str}"
    return h


def _redact(headers: dict[str, str]) -> dict[str, str]:
    return {k: ("<redacted>" if k.lower() == "authorization" else v) for k, v in headers.items()}


def probe_requests(
    url: str,
    output: Path,
    headers: dict[str, str],
    chunk_size: int,
    insecure: bool,
    max_bytes: int | None,
    report_every: int,
) -> ProbeResult:
    r = ProbeResult(backend="requests")
    sess = requests.Session()
    sess.headers.update(headers)
    last_report = 0
    start = time.monotonic()
    try:
        resp = sess.get(url, stream=True, timeout=60, verify=not insecure)
        r.extra.append(f"HTTP {resp.status_code}")
        if "Content-Length" in resp.headers:
            r.extra.append(f"Content-Length: {int(resp.headers['Content-Length']):,}")
        if "Content-Range" in resp.headers:
            r.extra.append(f"Content-Range: {resp.headers['Content-Range']}")
        resp.raise_for_status()
        with output.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                r.bytes += len(chunk)
                r.chunks += 1
                if r.bytes - last_report >= report_every:
                    elapsed = time.monotonic() - start
                    print(
                        f"  [requests] {r.bytes:>14,} bytes  "
                        f"{(r.bytes / 1_048_576) / elapsed:>7.2f} MB/s",
                        flush=True,
                    )
                    last_report = r.bytes
                if max_bytes is not None and r.bytes >= max_bytes:
                    r.extra.append(f"stopped at --max-bytes={max_bytes:,}")
                    break
    except Exception as e:
        r.error = f"{type(e).__name__}: {e}"
        r.error_at_offset = r.bytes
        # Drill into __cause__/__context__ for the OpenSSL message:
        cur: BaseException | None = e
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if isinstance(cur, OSError):
                r.extra.append(f"  inner OSError: errno={cur.errno} {cur}")
            cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    finally:
        r.duration_s = time.monotonic() - start
    return r


def probe_subprocess(
    backend: str,
    args: list[str],
    output: Path,
    timeout_s: float,
    echo_stderr: bool = False,
) -> ProbeResult:
    r = ProbeResult(backend=backend)
    start = time.monotonic()
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        r.duration_s = time.monotonic() - start
        if echo_stderr and proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()
            r.error = f"exit {proc.returncode}: {tail[-300:]}"
        else:
            for line in (proc.stderr or "").splitlines()[-30:]:
                low = line.lower().strip()
                if low.startswith("< http/") or "content-length:" in low or "content-range:" in low:
                    r.extra.append(line.strip())
    except subprocess.TimeoutExpired:
        r.duration_s = time.monotonic() - start
        r.error = f"timeout after {timeout_s}s"
    except FileNotFoundError:
        r.duration_s = time.monotonic() - start
        r.error = f"{backend} not installed"
        return r
    if output.exists():
        r.bytes = output.stat().st_size
        if r.error and r.bytes > 0:
            r.error_at_offset = r.bytes
    return r


def probe_curl(
    url: str,
    output: Path,
    headers: dict[str, str],
    insecure: bool,
    max_bytes: int | None,
    timeout_s: float,
    debug: bool,
) -> ProbeResult:
    args = [
        "curl",
        "-sS",
        "-L",
        "--fail",
        "-o",
        str(output),
    ]
    if debug:
        args += ["-v", "--trace-time"]
    else:
        args.append("-v")
    if insecure:
        args.append("--insecure")
    if max_bytes is not None:
        args += ["--max-filesize", str(max_bytes)]
    for k, v in headers.items():
        args += ["-H", f"{k}: {v}"]
    args.append(url)
    return probe_subprocess("curl", args, output, timeout_s, echo_stderr=debug)


def probe_wget(
    url: str,
    output: Path,
    headers: dict[str, str],
    insecure: bool,
    timeout_s: float,
    debug: bool,
) -> ProbeResult:
    args = [
        "wget",
        "--tries=1",
        "-O",
        str(output),
    ]
    if debug:
        args.append("--debug")
    else:
        args.append("-q")
    if insecure:
        args.append("--no-check-certificate")
    for k, v in headers.items():
        args += [f"--header={k}: {v}"]
    args.append(url)
    return probe_subprocess("wget", args, output, timeout_s, echo_stderr=debug)


def _format_result(r: ProbeResult) -> str:
    status = "OK    " if r.ok else "FAILED"
    head = (
        f"  {r.backend:9s} {status} {r.bytes:>14,} bytes  "
        f"{r.duration_s:>6.2f}s  {r.speed_mbps:>7.2f} MB/s"
    )
    parts = [head]
    if r.error:
        offset = f" (at offset {r.error_at_offset:,})" if r.error_at_offset else ""
        parts.append(f"      ERROR{offset}: {r.error}")
    for ln in r.extra:
        parts.append(f"      {ln}")
    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", help="Direct URL to fetch (raw, not HF-resolved).")
    g.add_argument("--hf-repo", help="HF repo id (e.g. owner/name); needs --file.")

    p.add_argument("--file", help="File path inside the HF repo.")
    p.add_argument("--revision", default="main")
    p.add_argument("--hf-endpoint", default="https://huggingface.co")
    p.add_argument(
        "--backend",
        choices=["all", "requests", "curl", "wget"],
        default="all",
    )
    p.add_argument(
        "--connection-close",
        action="store_true",
        help="Add Connection: close to all requests (disable keep-alive).",
    )
    p.add_argument(
        "--user-agent",
        default=None,
        help="Override the User-Agent (e.g. 'Wget/1.21.4').",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="iter_content chunk size for the requests backend (default 8192).",
    )
    p.add_argument(
        "--range-start",
        type=int,
        default=0,
        help="Start byte for Range header (0 = no Range, full download).",
    )
    p.add_argument(
        "--range-end",
        type=int,
        default=None,
        help="End byte for Range header (inclusive). Only valid with --range-start>=0.",
    )
    p.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Stop after receiving N bytes (truncates output).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/hf-probe.bin"),
        help="Output file base path; backend name is suffixed.",
    )
    p.add_argument(
        "--keep",
        action="store_true",
        help="Keep downloaded files instead of deleting them after each run.",
    )
    p.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS cert validation (--no-check-certificate / -k).",
    )
    p.add_argument(
        "--report-every",
        type=int,
        default=100 * 1024 * 1024,
        help="Emit a progress line every N bytes (requests backend, default 100 MiB).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-backend timeout in seconds (default 600).",
    )
    p.add_argument(
        "--debug-http",
        action="store_true",
        help=(
            "urllib3 + http.client wire-level logs (requests backend); "
            "passes -v / --debug to curl / wget and echoes their stderr. "
            "Lots of output — pipe to a file."
        ),
    )
    args = p.parse_args()

    if args.debug_http:
        enable_http_debug()

    if args.url:
        url = args.url
    else:
        if not args.file:
            p.error("--file is required when using --hf-repo")
        url = build_url(args.hf_repo, args.revision, args.file, args.hf_endpoint)

    headers = build_headers(
        args.user_agent,
        args.connection_close,
        args.range_start,
        args.range_end,
    )

    print(f"URL:      {url}")
    print(f"Headers:  {_redact(headers)}")
    print(
        f"Levers:   chunk_size={args.chunk_size}  insecure={args.insecure}  max_bytes={args.max_bytes}"
    )
    print()

    backends = ["wget", "curl", "requests"] if args.backend == "all" else [args.backend]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results: list[ProbeResult] = []
    for b in backends:
        out = args.output.with_suffix(f".{b}")
        if out.exists():
            out.unlink()
        print(f"--- {b} ---", flush=True)
        try:
            if b == "requests":
                r = probe_requests(
                    url,
                    out,
                    headers,
                    args.chunk_size,
                    args.insecure,
                    args.max_bytes,
                    args.report_every,
                )
            elif b == "curl":
                r = probe_curl(
                    url,
                    out,
                    headers,
                    args.insecure,
                    args.max_bytes,
                    args.timeout,
                    args.debug_http,
                )
            else:
                r = probe_wget(url, out, headers, args.insecure, args.timeout, args.debug_http)
        except Exception as e:
            r = ProbeResult(
                backend=b,
                error=f"probe crashed: {type(e).__name__}: {e}",
            )
            r.extra.append(traceback.format_exc().splitlines()[-1])

        print(_format_result(r))
        results.append(r)
        if not args.keep and out.exists():
            out.unlink()
        print()

    print("=== summary ===")
    for r in results:
        print(_format_result(r))

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    if shutil.which("curl") is None and shutil.which("wget") is None:
        print(
            "WARNING: neither curl nor wget is on PATH; only the requests backend will work.",
            file=sys.stderr,
        )
    sys.exit(main())
