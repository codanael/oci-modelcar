import http.client
import logging
from typing import Any
from unittest.mock import patch

import pytest
import requests
import urllib3.exceptions
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.hf import HfClient, HfStream


def _make_client(httpserver: HTTPServer) -> HfClient:
    return HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")


def _make_short_handler(payload: bytes, start: int, deliver: int, total: int) -> Any:
    """Build a handler that delivers `deliver` bytes starting at `start`,
    while claiming the full remaining length in Content-Length."""

    def handler(req: Any) -> Response:
        if start > 0:
            assert req.headers.get("Range") == f"bytes={start}-"

        def gen() -> Any:
            yield payload[start : start + deliver]

        if start == 0:
            return Response(
                gen(),
                status=200,
                headers={"Content-Length": str(total)},
            )
        return Response(
            gen(),
            status=206,
            headers={
                "Content-Length": str(total - start),
                "Content-Range": f"bytes {start}-{total - 1}/{total}",
            },
        )

    return handler


def test_hfstream_reads_full_file(httpserver: HTTPServer):
    payload = b"X" * 1024
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    client = _make_client(httpserver)
    stream = HfStream(client, revision="main", path="file.bin", size=len(payload))
    out = stream.read(-1)
    assert out == payload


def test_hfstream_read_in_chunks(httpserver: HTTPServer):
    payload = bytes(range(256)) * 4  # 1024 bytes
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    client = _make_client(httpserver)
    stream = HfStream(client, revision="main", path="file.bin", size=len(payload))
    out = b""
    while True:
        chunk = stream.read(100)
        if not chunk:
            break
        out += chunk
    assert out == payload


def test_hfstream_invokes_progress_cb_after_each_chunk(httpserver: HTTPServer):
    """progress_cb receives cumulative bytes_buffered after every chunk read."""
    payload = b"X" * 4096
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    client = _make_client(httpserver)
    seen: list[int] = []
    stream = HfStream(
        client,
        revision="main",
        path="file.bin",
        size=len(payload),
        chunk_size=1024,
        progress_cb=seen.append,
    )
    out = stream.read(-1)
    assert out == payload
    # 4096 bytes at 1024-byte chunks → 4 progress calls with cumulative totals
    assert seen == [1024, 2048, 3072, 4096]


def test_hfstream_progress_cb_optional(httpserver: HTTPServer):
    """No progress_cb → no error, no calls."""
    payload = b"Y" * 256
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    client = _make_client(httpserver)
    stream = HfStream(client, revision="main", path="file.bin", size=len(payload))
    assert stream.read(-1) == payload


def test_hfstream_aborts_when_stop_event_is_set(httpserver: HTTPServer):
    """If stop_event is already set, HfStream raises InterruptedError on next read."""
    import threading

    payload = b"X" * 4096
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    client = _make_client(httpserver)
    stop = threading.Event()
    stream = HfStream(
        client,
        revision="main",
        path="file.bin",
        size=len(payload),
        chunk_size=1024,
        stop_event=stop,
    )
    stop.set()
    with pytest.raises(InterruptedError):
        stream.read(-1)


def test_hfstream_does_not_retry_on_ssl_error(httpserver: HTTPServer):
    """SSLError mid-stream surfaces immediately, no retry attempts.

    A misconfigured CA never recovers — silent retries just hide the cause.
    """
    payload = b"X" * 100
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )

    iter_calls = {"n": 0}

    def ssl_iter_content(self: requests.Response, chunk_size: int = 1) -> Any:
        iter_calls["n"] += 1
        raise requests.exceptions.SSLError("CERTIFICATE_VERIFY_FAILED")
        yield  # pragma: no cover  (make this a generator)

    client = _make_client(httpserver)
    with patch.object(requests.Response, "iter_content", ssl_iter_content):
        stream = HfStream(
            client,
            revision="main",
            path="file.bin",
            size=len(payload),
            max_retries=10,
            backoff_initial=0.0,
            chunk_size=10,
        )
        with pytest.raises(requests.exceptions.SSLError):
            stream.read(-1)
    # iter_content was wired during __init__; the read raised SSL on first next().
    # Without the fix, _next_chunk would silently retry up to 10 times, calling
    # _open()→iter_content() each time. With the fix, exactly one iter_content call.
    assert iter_calls["n"] == 1, f"expected 1 iter_content call, got {iter_calls['n']}"


def test_hfstream_retries_on_ssl_eof_mid_stream(httpserver: HTTPServer):
    """An SSL EOF mid-stream (after handshake succeeded and bytes flowed) is a
    network blip, not a CA misconfig — must resume via Range like other transient
    errors. Regression seen on a 1.34 GB shard cut by an idle-proxy timeout.
    """
    payload = b"A" * 100

    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        _make_short_handler(payload, start=30, deliver=70, total=100)
    )

    real_iter_content = requests.Response.iter_content
    call_count = {"n": 0}

    def flaky_iter_content(self: requests.Response, chunk_size: int = 1) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            yield payload[:30]
            raise requests.exceptions.SSLError(
                "EOF occurred in violation of protocol (_ssl.c:2437)"
            )
        yield from real_iter_content(self, chunk_size=chunk_size)

    client = _make_client(httpserver)
    with patch.object(requests.Response, "iter_content", flaky_iter_content):
        stream = HfStream(
            client,
            revision="main",
            path="file.bin",
            size=len(payload),
            max_retries=3,
            backoff_initial=0.0,
            chunk_size=10,
        )
        out = stream.read(-1)
    assert out == payload
    assert call_count["n"] >= 2, "must have retried after SSL EOF"


def test_hfstream_does_not_retry_on_proxy_error(httpserver: HTTPServer):
    payload = b"Y" * 100
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )

    iter_calls = {"n": 0}

    def proxy_iter_content(self: requests.Response, chunk_size: int = 1) -> Any:
        iter_calls["n"] += 1
        raise requests.exceptions.ProxyError("bad proxy")
        yield  # pragma: no cover

    client = _make_client(httpserver)
    with patch.object(requests.Response, "iter_content", proxy_iter_content):
        stream = HfStream(
            client,
            revision="main",
            path="file.bin",
            size=len(payload),
            max_retries=10,
            backoff_initial=0.0,
            chunk_size=10,
        )
        with pytest.raises(requests.exceptions.ProxyError):
            stream.read(-1)
    assert iter_calls["n"] == 1


def test_hfstream_size_mismatch_raises(httpserver: HTTPServer):
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        b"X" * 100, headers={"Content-Length": "100"}
    )
    client = _make_client(httpserver)
    with pytest.raises(RuntimeError, match="size mismatch"):
        HfStream(client, revision="main", path="file.bin", size=200)


def test_hfstream_range_resume(httpserver: HTTPServer):
    """First request delivers truncated bytes; resume with Range honors offset."""
    payload = b"A" * 100
    call_count = {"n": 0}

    def second_handler(request):
        call_count["n"] += 1
        rng = request.headers.get("Range")
        assert rng == "bytes=50-"
        start = 50
        return Response(
            payload[start:],
            status=206,
            headers={
                "Content-Length": str(len(payload) - start),
                "Content-Range": f"bytes {start}-{len(payload) - 1}/{len(payload)}",
            },
        )

    def truncated_gen():
        yield payload[:50]

    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        lambda req: Response(
            truncated_gen(),
            status=200,
            headers={"Content-Length": str(len(payload))},  # claim 100, deliver 50
        )
    )
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        second_handler
    )

    client = _make_client(httpserver)
    stream = HfStream(
        client,
        revision="main",
        path="file.bin",
        size=len(payload),
        max_retries=3,
        backoff_initial=0.0,
        chunk_size=10,
    )
    out = stream.read(-1)
    assert out == payload
    assert call_count["n"] == 1


def test_hfstream_resumes_through_multiple_cuts_read_full(httpserver: HTTPServer):
    """A flaky upstream that truncates repeatedly is fully recovered via Range
    even when read(-1) is used: the resume loop must be unbounded, not single-shot."""
    payload = b"X" * 200
    for offset in (0, 50, 100):
        httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
            _make_short_handler(payload, start=offset, deliver=50, total=200)
        )
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        _make_short_handler(payload, start=150, deliver=50, total=200)
    )

    client = _make_client(httpserver)
    stream = HfStream(
        client,
        revision="main",
        path="file.bin",
        size=len(payload),
        max_retries=2,
        backoff_initial=0.0,
        chunk_size=10,
    )
    out = stream.read(-1)
    assert out == payload


def test_hfstream_resumes_through_multiple_cuts_read_bounded(httpserver: HTTPServer):
    """Same scenario but consumed via small read(n) calls: must drain fully."""
    payload = b"Y" * 200
    for offset in (0, 50, 100):
        httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
            _make_short_handler(payload, start=offset, deliver=50, total=200)
        )
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        _make_short_handler(payload, start=150, deliver=50, total=200)
    )

    client = _make_client(httpserver)
    stream = HfStream(
        client,
        revision="main",
        path="file.bin",
        size=len(payload),
        max_retries=2,
        backoff_initial=0.0,
        chunk_size=10,
    )
    out = b""
    while len(out) < len(payload):
        chunk = stream.read(37)
        if not chunk:
            break
        out += chunk
    assert out == payload


def test_hfstream_logs_progress_on_resume(httpserver: HTTPServer, caplog: pytest.LogCaptureFixture):
    """Each resume after a cut should emit an INFO log carrying the new offset
    and the expected total, so multi-resume runs are observable."""
    payload = b"Z" * 100
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        _make_short_handler(payload, start=0, deliver=40, total=100)
    )
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        _make_short_handler(payload, start=40, deliver=60, total=100)
    )
    client = _make_client(httpserver)
    stream = HfStream(
        client,
        revision="main",
        path="file.bin",
        size=len(payload),
        max_retries=2,
        backoff_initial=0.0,
        chunk_size=10,
    )
    with caplog.at_level(logging.INFO, logger="oci_modelcar.hf"):
        out = stream.read(-1)
    assert out == payload
    resume_logs = [r for r in caplog.records if "resum" in r.getMessage().lower()]
    assert resume_logs, "expected at least one resume log line"
    msg = resume_logs[0].getMessage()
    assert "40" in msg and "100" in msg


@pytest.mark.parametrize(
    "exc",
    [
        urllib3.exceptions.ProtocolError("Connection broken"),
        http.client.IncompleteRead(b""),
        OSError("connection reset by peer"),
    ],
    ids=["ProtocolError", "IncompleteRead", "OSError"],
)
def test_hfstream_retries_on_transport_exceptions(httpserver: HTTPServer, exc: Exception):
    """Transport-layer exceptions raised by iter_content (regardless of which
    specific class urllib3/requests surfaces) must trigger a Range-based resume."""
    payload = b"A" * 100
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        _make_short_handler(payload, start=30, deliver=70, total=100)
    )

    real_iter_content = requests.Response.iter_content
    call_count = {"n": 0}

    def flaky_iter_content(self: requests.Response, chunk_size: int = 1) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            yield payload[:30]
            raise exc
        yield from real_iter_content(self, chunk_size=chunk_size)

    client = _make_client(httpserver)
    with patch.object(requests.Response, "iter_content", flaky_iter_content):
        stream = HfStream(
            client,
            revision="main",
            path="file.bin",
            size=len(payload),
            max_retries=3,
            backoff_initial=0.0,
            chunk_size=10,
        )
        out = stream.read(-1)
    assert out == payload
    assert call_count["n"] >= 2


def test_hfstream_size_mismatch_on_resume_validates_content_range(httpserver: HTTPServer):
    """Resume handshake verifies Content-Range starts with the requested offset."""
    payload = b"A" * 100

    def truncated_gen2():
        yield payload[:50]

    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        lambda req: Response(
            truncated_gen2(),
            status=200,
            headers={"Content-Length": str(len(payload))},
        )
    )
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload[40:],  # server returns wrong start (40 instead of 50)
        status=206,
        headers={"Content-Range": "bytes 40-99/100"},
    )
    client = _make_client(httpserver)
    stream = HfStream(
        client,
        revision="main",
        path="file.bin",
        size=len(payload),
        max_retries=2,
        backoff_initial=0.0,
        chunk_size=10,
    )
    with pytest.raises(RuntimeError, match="did not honor Range"):
        stream.read(-1)
