import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.hf import HfClient, HfStream


def _make_client(httpserver: HTTPServer) -> HfClient:
    return HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")


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
