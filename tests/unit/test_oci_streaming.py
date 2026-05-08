"""Tests for StreamingBlobUpload — single PATCH per blob, body=iterator.

This is the upload mode that matches containers/image (Podman, Skopeo) and
Jib: one streaming PATCH per blob instead of N chunked PATCHes. Eliminates
per-PATCH LB routing decisions on registries (Artifactory cluster, Harbor
behind reverse proxies) that lack sticky session affinity.

Tradeoff vs ChunkedBlobUpload: no intra-blob retry. A mid-PATCH cut
surfaces as an error and the runner handles file-level retry across runs
via state.json.
"""

import hashlib
import re
from typing import Any
from unittest.mock import patch

import pytest
import requests
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.oci import OciClient, StreamingBlobUpload


def _client(httpserver: HTTPServer) -> OciClient:
    return OciClient(host_url=httpserver.url_for(""))


def test_streaming_upload_happy_path(httpserver: HTTPServer):
    payload = b"X" * (4 * 1024 * 1024 + 17)  # not block-aligned, ensure no cheating
    expected_digest = "sha256:" + hashlib.sha256(payload).hexdigest()

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/s")}
    )

    received = bytearray()

    def patch_handler(request: Any) -> Response:
        cr = request.headers.get("Content-Range", "")
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m, f"bad Content-Range {cr!r}"
        start, end = int(m.group(1)), int(m.group(2))
        assert start == 0, f"streaming must always be a single PATCH from offset 0; got {start}"
        assert end == len(payload) - 1, (
            f"end must be total-1; got {end}, expected {len(payload) - 1}"
        )
        cl = int(request.headers["Content-Length"])
        assert cl == len(payload), f"Content-Length mismatch: {cl} vs {len(payload)}"
        received.extend(request.data)
        return Response(
            "",
            status=202,
            headers={
                "Location": httpserver.url_for("/upload/s"),
                "Range": f"0-{end}",
            },
        )

    httpserver.expect_request("/upload/s", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/upload/s", method="PUT").respond_with_data(
        "",
        status=201,
        headers={"Location": httpserver.url_for(f"/v2/repo/blobs/{expected_digest}")},
    )

    client = _client(httpserver)
    upload = StreamingBlobUpload(client=client, repo="repo", total_size=len(payload))

    # Producer yields the payload in 1 MiB chunks (typical _PipeBuffer shape).
    def producer() -> Any:
        view = memoryview(payload)
        chunk = 1024 * 1024
        for i in range(0, len(payload), chunk):
            yield bytes(view[i : i + chunk])

    digest, total = upload.upload(producer())
    assert digest == expected_digest
    assert total == len(payload)
    assert bytes(received) == payload


def test_streaming_upload_issues_exactly_one_patch(httpserver: HTTPServer):
    """Whole point of streaming mode: exactly ONE PATCH per blob, regardless
    of payload size. Otherwise it's not different from chunked."""
    payload = b"Y" * (12 * 1024 * 1024)
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/once")}
    )

    patch_calls = {"n": 0}

    def patch_handler(request: Any) -> Response:
        patch_calls["n"] += 1
        return Response(
            "",
            status=202,
            headers={
                "Location": httpserver.url_for("/upload/once"),
                "Range": f"0-{len(payload) - 1}",
            },
        )

    httpserver.expect_request("/upload/once", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/upload/once", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = StreamingBlobUpload(client=client, repo="repo", total_size=len(payload))
    upload.upload(iter([payload]))

    assert patch_calls["n"] == 1, (
        f"streaming must do exactly 1 PATCH per blob; got {patch_calls['n']}"
    )


@pytest.mark.parametrize("success_status", [200, 201, 202, 204])
def test_streaming_upload_accepts_non_spec_success_codes(
    httpserver: HTTPServer, success_status: int
):
    """Same Artifactory/Harbor non-conformance as ChunkedBlobUpload — accept
    {200, 201, 202, 204} as commit success."""
    payload = b"Z" * 1024
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/codes")}
    )
    httpserver.expect_request("/upload/codes", method="PATCH").respond_with_data(
        "", status=success_status, headers={"Location": httpserver.url_for("/upload/codes")}
    )
    httpserver.expect_request("/upload/codes", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = StreamingBlobUpload(client=client, repo="repo", total_size=len(payload))
    upload.upload(iter([payload]))


def test_streaming_upload_unhandled_status_raises(httpserver: HTTPServer):
    """Belt-and-braces: any unhandled status (e.g. 299) must raise rather than
    silently succeed or hang. Same protection as the chunked path."""
    payload = b"A" * 64
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/odd")}
    )
    httpserver.expect_request("/upload/odd", method="PATCH").respond_with_data("", status=299)

    client = _client(httpserver)
    upload = StreamingBlobUpload(client=client, repo="repo", total_size=len(payload))
    with pytest.raises(RuntimeError, match=r"unexpected.*299|status 299"):
        upload.upload(iter([payload]))


def test_streaming_upload_surfaces_ssl_eof(httpserver: HTTPServer):
    """No intra-blob retry by design — a mid-PATCH SSL EOF must surface
    immediately. The runner is responsible for file-level retry across
    runs via state.json."""
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/eof")}
    )
    client = _client(httpserver)
    upload = StreamingBlobUpload(client=client, repo="repo", total_size=128)

    calls = {"n": 0}

    def ssl_patch(self: requests.Session, *args: object, **kwargs: object) -> object:
        calls["n"] += 1
        raise requests.exceptions.SSLError("EOF occurred in violation of protocol (_ssl.c:2437)")

    with (
        patch.object(requests.Session, "patch", ssl_patch),
        pytest.raises(requests.exceptions.SSLError),
    ):
        upload.upload(iter([b"Z" * 128]))
    assert calls["n"] == 1, f"streaming mode must NOT retry; got {calls['n']} attempts"


def test_streaming_upload_passes_content_length_header(httpserver: HTTPServer):
    """Without Content-Length, urllib3 falls back to chunked transfer-encoding,
    which some registries (and proxies) handle differently from a fixed-size
    PATCH. We always set Content-Length so the wire shape matches what
    Podman/Jib send."""
    payload = b"L" * 512
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/len")}
    )

    seen_headers: dict[str, str] = {}

    def patch_handler(request: Any) -> Response:
        seen_headers.update(dict(request.headers))
        return Response("", status=202, headers={"Location": httpserver.url_for("/upload/len")})

    httpserver.expect_request("/upload/len", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/upload/len", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = StreamingBlobUpload(client=client, repo="repo", total_size=len(payload))
    upload.upload(iter([payload]))

    assert seen_headers.get("Content-Length") == str(len(payload))
    # Must NOT use chunked transfer-encoding when Content-Length is known.
    te = seen_headers.get("Transfer-Encoding", "")
    assert "chunked" not in te.lower(), f"unexpected Transfer-Encoding: {te!r}"
