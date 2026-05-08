import hashlib
import re
from typing import Any
from unittest.mock import patch

import pytest
import requests
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.oci import ChunkedBlobUpload, OciClient


def _client(httpserver: HTTPServer) -> OciClient:
    return OciClient(host_url=httpserver.url_for(""))


def test_chunked_upload_aborts_when_stop_event_is_set(httpserver: HTTPServer):
    """If stop_event is set before a flush, write raises InterruptedError."""
    import threading

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/abort")}
    )
    client = _client(httpserver)
    stop = threading.Event()
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=1024, stop_event=stop)
    stop.set()
    import pytest as _pytest

    with _pytest.raises(InterruptedError):
        upload.write(b"X" * 2048)  # forces _flush, which checks stop_event


def test_chunked_upload_happy_path(httpserver: HTTPServer):
    payload = b"X" * (8 * 1024 * 1024 + 100)  # > 1 chunk
    expected_digest = "sha256:" + hashlib.sha256(payload).hexdigest()

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/123")}
    )

    received: list[bytes] = []

    def patch_handler(request):
        cr = request.headers["Content-Range"]
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m, f"bad Content-Range: {cr!r}"
        received.append(request.data)
        end = int(m.group(2))
        return Response(
            "",
            status=202,
            headers={
                "Location": httpserver.url_for("/upload/123"),
                "Range": f"0-{end}",
            },
        )

    httpserver.expect_request("/upload/123", method="PATCH").respond_with_handler(patch_handler)

    httpserver.expect_request("/upload/123", method="PUT").respond_with_data(
        "",
        status=201,
        headers={"Location": httpserver.url_for(f"/v2/repo/blobs/{expected_digest}")},
    )

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=8 * 1024 * 1024)
    upload.write(payload)
    digest, total = upload.close()
    assert digest == expected_digest
    assert total == len(payload)
    assert b"".join(received) == payload[: 8 * 1024 * 1024]


def test_content_range_format_no_prefix(httpserver: HTTPServer):
    """OCI Content-Range MUST be 'N-M', NEVER 'bytes N-M/total'.

    Small payload (< chunk_size) goes entirely to PUT, no PATCH happens —
    so this test mostly verifies the system doesn't blow up; the more direct
    test for Content-Range format is the next test.
    """
    payload = b"Y" * 10

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/X")}
    )
    httpserver.expect_request("/upload/X", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64)
    upload.write(payload)
    upload.close()


def test_chunked_upload_does_not_retry_on_ssl_error(httpserver: HTTPServer):
    """SSLError on PATCH must surface immediately, no retry."""
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/ssl")}
    )
    # Without an early-raise on SSL, the OCI retry path falls into _resync (GET).
    # Register a GET handler so the no-fix case fails fast on the count assertion
    # rather than hanging on urllib3 status-code retries against an unmatched route.
    httpserver.expect_request("/upload/ssl", method="GET").respond_with_data(
        "", status=204, headers={"Range": "0-0"}
    )
    client = _client(httpserver)
    upload = ChunkedBlobUpload(
        client,
        repo="repo",
        chunk_size=64,
        max_retries=10,
        backoff_initial=0.0,
    )

    calls = {"n": 0}

    def ssl_patch(self: requests.Session, *args: object, **kwargs: object) -> object:
        calls["n"] += 1
        raise requests.exceptions.SSLError("CERTIFICATE_VERIFY_FAILED")

    with (
        patch.object(requests.Session, "patch", ssl_patch),
        pytest.raises(requests.exceptions.SSLError),
    ):
        upload.write(b"Z" * 128)  # forces flush of one 64-byte chunk
    assert calls["n"] == 1, f"expected 1 PATCH attempt, got {calls['n']}"


def test_chunked_upload_retries_on_ssl_eof_mid_stream(httpserver: HTTPServer):
    """SSL EOF on a PATCH after bytes flowed = mid-stream connection cut, must
    retry via _resync — same semantics as ConnectionError. Only handshake-time
    SSL errors are fatal."""
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/eof")}
    )
    # _resync GET — declares we're at offset 64 already (so the next PATCH for
    # the next chunk is fine; the failed one is replayed against a server that
    # behaves correctly).
    httpserver.expect_request("/upload/eof", method="GET").respond_with_data(
        "", status=204, headers={"Range": "0-0"}
    )
    received = bytearray()

    def patch_handler(request: Any) -> Response:
        received.extend(request.data)
        return Response(
            "",
            status=202,
            headers={
                "Location": httpserver.url_for("/upload/eof"),
                "Range": f"0-{len(received) - 1}",
            },
        )

    client = _client(httpserver)
    upload = ChunkedBlobUpload(
        client, repo="repo", chunk_size=64, max_retries=10, backoff_initial=0.0
    )

    calls = {"n": 0}
    real_patch = requests.Session.patch

    def flaky_patch(self: requests.Session, *args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.SSLError(
                "EOF occurred in violation of protocol (_ssl.c:2437)"
            )
        return real_patch(self, *args, **kwargs)

    httpserver.expect_request("/upload/eof", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/upload/eof", method="PUT").respond_with_data("", status=201)

    with patch.object(requests.Session, "patch", flaky_patch):
        upload.write(b"Z" * 128)
        upload.close()
    assert calls["n"] >= 2, "must have retried after SSL EOF"


def test_chunked_upload_does_not_retry_on_proxy_error(httpserver: HTTPServer):
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/px")}
    )
    httpserver.expect_request("/upload/px", method="GET").respond_with_data(
        "", status=204, headers={"Range": "0-0"}
    )
    client = _client(httpserver)
    upload = ChunkedBlobUpload(
        client,
        repo="repo",
        chunk_size=64,
        max_retries=10,
        backoff_initial=0.0,
    )

    calls = {"n": 0}

    def proxy_patch(self: requests.Session, *args: object, **kwargs: object) -> object:
        calls["n"] += 1
        raise requests.exceptions.ProxyError("bad proxy")

    with (
        patch.object(requests.Session, "patch", proxy_patch),
        pytest.raises(requests.exceptions.ProxyError),
    ):
        upload.write(b"Z" * 128)
    assert calls["n"] == 1


def test_patch_content_range_is_inclusive(httpserver: HTTPServer):
    payload = b"Z" * 200
    seen: list[tuple[int, int, int]] = []

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/Y")}
    )

    def patch_handler(request):
        cr = request.headers["Content-Range"]
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m
        start, end = int(m.group(1)), int(m.group(2))
        seen.append((start, end, len(request.data)))
        # Spec: end - start + 1 == len(body)
        assert end - start + 1 == len(request.data)
        return Response(
            "",
            status=202,
            headers={"Location": httpserver.url_for("/upload/Y"), "Range": f"0-{end}"},
        )

    httpserver.expect_request("/upload/Y", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/upload/Y", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64)
    upload.write(payload)
    upload.close()
    assert seen
    for start, end, body_len in seen:
        assert end == start + body_len - 1
