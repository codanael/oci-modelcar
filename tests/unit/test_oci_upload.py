import hashlib
import re

from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.oci import ChunkedBlobUpload, OciClient


def _client(httpserver: HTTPServer) -> OciClient:
    return OciClient(host_url=httpserver.url_for(""))


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
