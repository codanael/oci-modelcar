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


def test_retry_budget_resets_when_server_makes_progress(httpserver: HTTPServer):
    """A hostile proxy that drops mid-PATCH but lets the server commit a few
    bytes each time should NOT exhaust the retry budget, because each retry
    is real progress. With max_retries=2 we need 4 PATCH attempts to walk
    the chunk forward in 16-byte hops — only possible if the budget refreshes
    on observed progress (server_offset advances)."""
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/grind")}
    )

    # Server walks forward 16 bytes per resync, mimicking the bytes that were
    # actually committed before each mid-stream cut.
    progress = {"committed": 0}

    def get_handler(request: Any) -> Response:
        progress["committed"] += 16
        end = min(progress["committed"], 64) - 1
        return Response("", status=204, headers={"Range": f"0-{end}"})

    httpserver.expect_request("/upload/grind", method="GET").respond_with_handler(get_handler)
    httpserver.expect_request("/upload/grind", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(
        client, repo="repo", chunk_size=64, max_retries=2, backoff_initial=0.0
    )

    patch_calls = {"n": 0}

    def always_eof_patch(self: requests.Session, *args: object, **kwargs: object) -> object:
        patch_calls["n"] += 1
        raise requests.exceptions.SSLError("EOF occurred in violation of protocol (_ssl.c:2437)")

    with patch.object(requests.Session, "patch", always_eof_patch):
        upload.write(b"Z" * 64)
        upload.close()

    # 4 PATCHes attempted, but with reset-on-progress none ever "exhausted".
    assert patch_calls["n"] == 4, (
        f"expected 4 PATCH attempts (each making progress should refresh "
        f"the budget), got {patch_calls['n']}"
    )


def test_patch_200_treated_as_success_artifactory_quirk(httpserver: HTTPServer):
    """OCI Distribution v1.1 says PATCH must return 202; Artifactory has been
    observed to return 200 on chunk commit. Both must be treated as success.
    Otherwise the retry loop falls through without advancing server_offset
    or decrementing attempts_left, causing infinite re-PATCH of the same
    range (or, when the server re-checks, a 416 storm)."""
    payload = b"A" * 200

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/jfrog")}
    )

    received = bytearray()
    patch_count = {"n": 0}

    def patch_handler(request: Any) -> Response:
        patch_count["n"] += 1
        cr = request.headers["Content-Range"]
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m
        end = int(m.group(2))
        received.extend(request.data)
        # 200 instead of 202 — the Artifactory non-conformance we're testing.
        return Response(
            "",
            status=200,
            headers={
                "Location": httpserver.url_for("/upload/jfrog"),
                "Range": f"0-{end}",
            },
        )

    httpserver.expect_request("/upload/jfrog", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/upload/jfrog", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(
        client, repo="repo", chunk_size=64, max_retries=3, backoff_initial=0.0
    )
    upload.write(payload)
    upload.close()

    # 200 bytes / 64-byte chunks → 3 PATCH (64+64+64) + remainder via PUT.
    # If 200 isn't accepted, we'd see >3 attempts (re-PATCH) or a hang.
    assert patch_count["n"] == 3, (
        f"200 must be treated as success — one PATCH per chunk; got {patch_count['n']}"
    )
    assert bytes(received) == payload[:192]


def test_unhandled_patch_status_raises_instead_of_looping(httpserver: HTTPServer):
    """Belt-and-braces: any status the loop doesn't explicitly handle must
    surface as an error, not silently spin. Before the 200 fix, a 2xx
    non-202/non-200 response would fall through `raise_for_status()` (no-op
    on 2xx) and re-iterate without progress or budget decrement → infinite
    loop. We pick 299 (a 2xx code with no spec meaning) to confirm we no
    longer trust 'anything 2xx that isn't 202'."""
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/weird")}
    )
    # 416 fallback resync would also need handling; register so we fail on
    # the actual error rather than on an unmatched route.
    httpserver.expect_request("/upload/weird", method="GET").respond_with_data(
        "", status=204, headers={"Range": "0-0"}
    )
    httpserver.expect_request("/upload/weird", method="PATCH").respond_with_data("", status=299)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(
        client, repo="repo", chunk_size=64, max_retries=3, backoff_initial=0.0
    )
    with pytest.raises(RuntimeError, match=r"unexpected.*299|status 299"):
        upload.write(b"Z" * 128)


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


def test_backoff_uses_full_jitter(httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch):
    """Full-jitter pattern (AWS Architecture Blog): each retry sleeps for a
    random duration in [0, min(cap, base * 2^attempt)], not the previous
    exponential-plus-10%-jitter. Wider spread = better behavior under
    thundering-herd / proxy-restart scenarios where many clients retry
    simultaneously."""
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/jit")}
    )
    client = _client(httpserver)
    upload = ChunkedBlobUpload(
        client, repo="repo", chunk_size=64, backoff_initial=1.0, backoff_cap=60.0
    )

    uniform_calls: list[tuple[float, float]] = []

    def fake_uniform(a: float, b: float) -> float:
        uniform_calls.append((a, b))
        return (a + b) / 2

    monkeypatch.setattr("oci_modelcar.oci.random.uniform", fake_uniform)
    monkeypatch.setattr("oci_modelcar.oci.time.sleep", lambda d: None)

    upload._sleep_backoff(0)  # cap = 1 * 2^0 = 1
    upload._sleep_backoff(3)  # cap = 1 * 2^3 = 8
    upload._sleep_backoff(10)  # cap = min(60, 1024) = 60

    assert uniform_calls == [(0.0, 1.0), (0.0, 8.0), (0.0, 60.0)], (
        f"full jitter must sample uniformly from 0 to capped exponential; got {uniform_calls}"
    )


def test_backoff_zero_initial_does_not_sleep(
    httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch
):
    """backoff_initial=0 (the test-only fast-path) must not call time.sleep."""
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/zero")}
    )
    client = _client(httpserver)
    upload = ChunkedBlobUpload(
        client, repo="repo", chunk_size=64, backoff_initial=0.0, backoff_cap=60.0
    )

    sleep_calls: list[float] = []
    monkeypatch.setattr("oci_modelcar.oci.time.sleep", lambda d: sleep_calls.append(d))

    upload._sleep_backoff(0)
    upload._sleep_backoff(5)
    assert sleep_calls == [], f"backoff_initial=0 must skip sleep entirely; got {sleep_calls}"


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
