import hashlib

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.oci import OciClient, head_blob, push_small_blob


def _client(httpserver: HTTPServer) -> OciClient:
    return OciClient(host_url=httpserver.url_for(""))


def test_push_small_blob_already_exists(httpserver: HTTPServer):
    data = b"{}"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    httpserver.expect_request(f"/v2/repo/blobs/{digest}", method="HEAD").respond_with_data(
        "", status=200, headers={"Docker-Content-Digest": digest}
    )
    client = _client(httpserver)
    out = push_small_blob(client, repo="repo", data=data)
    assert out == digest


def test_push_small_blob_creates(httpserver: HTTPServer):
    data = b'{"x":1}'
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    httpserver.expect_request(f"/v2/repo/blobs/{digest}", method="HEAD").respond_with_data(
        "", status=404
    )
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/cfg")}
    )
    httpserver.expect_request("/u/cfg", method="PUT").respond_with_data("", status=201)
    client = _client(httpserver)
    out = push_small_blob(client, repo="repo", data=data)
    assert out == digest


def test_head_blob_validates_digest(httpserver: HTTPServer):
    digest = "sha256:" + "a" * 64
    resp = Response(status=200)
    resp.headers["Docker-Content-Digest"] = digest
    resp.headers["Content-Length"] = "100"
    httpserver.expect_request(f"/v2/repo/blobs/{digest}", method="HEAD").respond_with_response(resp)
    client = _client(httpserver)
    info = head_blob(client, repo="repo", digest=digest)
    assert info["digest"] == digest
    assert info["size"] == 100


def test_head_blob_digest_mismatch(httpserver: HTTPServer):
    expected = "sha256:" + "a" * 64
    wrong = "sha256:" + "b" * 64
    httpserver.expect_request(f"/v2/repo/blobs/{expected}", method="HEAD").respond_with_data(
        "", status=200, headers={"Docker-Content-Digest": wrong, "Content-Length": "0"}
    )
    client = _client(httpserver)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        head_blob(client, repo="repo", digest=expected)


def test_head_blob_not_found(httpserver: HTTPServer):
    digest = "sha256:" + "c" * 64
    httpserver.expect_request(f"/v2/repo/blobs/{digest}", method="HEAD").respond_with_data(
        "", status=404
    )
    client = _client(httpserver)
    with pytest.raises(RuntimeError, match="not found"):
        head_blob(client, repo="repo", digest=digest)
