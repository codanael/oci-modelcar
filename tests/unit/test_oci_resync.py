import re

from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.oci import ChunkedBlobUpload, OciClient


def _client(httpserver: HTTPServer) -> OciClient:
    return OciClient(host_url=httpserver.url_for(""))


def test_resync_no_range_header(httpserver: HTTPServer):
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/1")}
    )
    httpserver.expect_request("/u/1", method="GET").respond_with_data("", status=204)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64)
    upload._resync()
    assert upload.server_offset == 0


def test_resync_with_range_0_0(httpserver: HTTPServer):
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/2")}
    )
    httpserver.expect_request("/u/2", method="GET").respond_with_data(
        "", status=204, headers={"Range": "0-0"}
    )
    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64)
    upload._resync()
    assert upload.server_offset == 1  # 1 byte received


def test_resync_with_range_0_1023(httpserver: HTTPServer):
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/3")}
    )
    httpserver.expect_request("/u/3", method="GET").respond_with_data(
        "", status=204, headers={"Range": "0-1023"}
    )
    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64)
    upload._resync()
    assert upload.server_offset == 1024


def test_416_resyncs_and_skips_if_already_accepted(httpserver: HTTPServer):
    """Server returns 416, but a GET shows the chunk was actually accepted."""
    payload = b"A" * 100
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/4")}
    )

    state = {"first_patch": True}

    def patch_handler(request):
        cr = request.headers["Content-Range"]
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m
        if state["first_patch"]:
            state["first_patch"] = False
            return Response("", status=416)
        return Response(
            "",
            status=202,
            headers={"Location": httpserver.url_for("/u/4"), "Range": f"0-{m.group(2)}"},
        )

    def get_handler(request):
        # Server says: I already have the full payload accepted.
        return Response("", status=204, headers={"Range": f"0-{len(payload) - 1}"})

    httpserver.expect_request("/u/4", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/4", method="GET").respond_with_handler(get_handler)
    httpserver.expect_request("/u/4", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64, backoff_initial=0.0)
    upload.write(payload)
    digest, total = upload.close()
    assert total == 100
    assert digest.startswith("sha256:")


def test_patch_500_then_success_via_resync(httpserver: HTTPServer):
    """PATCH transient 500 -> resync sees no progress -> retry chunk -> success."""
    payload = b"B" * 100
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/5")}
    )

    attempts = {"n": 0}

    def patch_handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return Response("", status=500)
        cr = request.headers["Content-Range"]
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m
        return Response(
            "",
            status=202,
            headers={"Location": httpserver.url_for("/u/5"), "Range": f"0-{m.group(2)}"},
        )

    httpserver.expect_request("/u/5", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/5", method="GET").respond_with_data("", status=204)
    httpserver.expect_request("/u/5", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64, backoff_initial=0.0)
    upload.write(payload)
    _digest, total = upload.close()
    assert total == 100
