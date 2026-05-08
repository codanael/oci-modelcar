import hashlib
import json

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.registry import (
    OciClient,
    head_blob,
    push_manifest,
    push_small_blob,
    validate_manifest_tag,
)


def _client(httpserver: HTTPServer) -> OciClient:
    return OciClient(host_url=httpserver.url_for(""))


def test_oci_client_url_construction():
    c = OciClient(host_url="https://registry.example.com")
    assert c.url("repo", "blobs", "uploads") == "https://registry.example.com/v2/repo/blobs/uploads"


def test_oci_client_loopback_uses_http():
    c = OciClient(registry_host="localhost:5000")
    assert c.base == "http://localhost:5000"


def test_oci_client_remote_uses_https():
    c = OciClient(registry_host="registry.example.com")
    assert c.base == "https://registry.example.com"


def test_oci_client_explicit_scheme_preserved():
    c = OciClient(registry_host="http://custom.example.com:8080")
    assert c.base == "http://custom.example.com:8080"


def test_head_blob_returns_descriptor_when_present(httpserver):
    digest = "sha256:" + hashlib.sha256(b"x").hexdigest()

    def head_handler(request):
        r = Response("", status=200)
        r.headers["Docker-Content-Digest"] = digest
        r.headers["Content-Length"] = "1"
        return r

    httpserver.expect_request(f"/v2/repo/blobs/{digest}", method="HEAD").respond_with_handler(
        head_handler
    )
    info = head_blob(_client(httpserver), "repo", digest)
    assert info == {"digest": digest, "size": 1}


def test_head_blob_returns_none_when_404(httpserver):
    digest = "sha256:" + "a" * 64
    httpserver.expect_request(f"/v2/repo/blobs/{digest}", method="HEAD").respond_with_data(
        "", status=404
    )
    assert head_blob(_client(httpserver), "repo", digest) is None


def test_head_blob_raises_on_digest_mismatch(httpserver):
    digest = "sha256:" + "a" * 64
    other = "sha256:" + "b" * 64

    def head_handler(request):
        r = Response("", status=200)
        r.headers["Docker-Content-Digest"] = other
        r.headers["Content-Length"] = "1"
        return r

    httpserver.expect_request(f"/v2/repo/blobs/{digest}", method="HEAD").respond_with_handler(
        head_handler
    )
    with pytest.raises(RuntimeError, match="digest mismatch"):
        head_blob(_client(httpserver), "repo", digest)


def test_push_small_blob_skips_when_already_present(httpserver):
    data = b"config bytes"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()

    def head_handler(request):
        r = Response("", status=200)
        r.headers["Docker-Content-Digest"] = digest
        r.headers["Content-Length"] = str(len(data))
        return r

    httpserver.expect_request(f"/v2/repo/blobs/{digest}", method="HEAD").respond_with_handler(
        head_handler
    )
    out = push_small_blob(_client(httpserver), "repo", data)
    assert out == digest


def test_push_small_blob_post_then_put(httpserver):
    data = b"config bytes"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()

    httpserver.expect_request(f"/v2/repo/blobs/{digest}", method="HEAD").respond_with_data(
        "", status=404
    )
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/1")}
    )

    received = {"data": b""}

    def put_handler(request):
        received["data"] = request.data
        return Response("", status=201)

    httpserver.expect_request("/u/1", method="PUT").respond_with_handler(put_handler)

    out = push_small_blob(_client(httpserver), "repo", data)
    assert out == digest
    assert received["data"] == data


def test_push_manifest_returns_digest(httpserver):
    body = json.dumps({"schemaVersion": 2, "config": {}, "layers": []}).encode()
    expected_digest = "sha256:" + hashlib.sha256(body).hexdigest()

    received = {"data": b""}

    def put_handler(request):
        received["data"] = request.data
        return Response("", status=201)

    httpserver.expect_request("/v2/repo/manifests/v1", method="PUT").respond_with_handler(
        put_handler
    )
    out = push_manifest(_client(httpserver), "repo", "v1", body)
    assert out == expected_digest
    assert received["data"] == body


def test_validate_manifest_tag_succeeds_on_match(httpserver):
    digest = "sha256:" + "a" * 64
    httpserver.expect_request("/v2/repo/manifests/v1", method="GET").respond_with_data(
        "", status=200, headers={"Docker-Content-Digest": digest}
    )
    validate_manifest_tag(_client(httpserver), "repo", "v1", digest)


def test_validate_manifest_tag_raises_on_mismatch(httpserver):
    digest = "sha256:" + "a" * 64
    other = "sha256:" + "b" * 64
    httpserver.expect_request("/v2/repo/manifests/v1", method="GET").respond_with_data(
        "", status=200, headers={"Docker-Content-Digest": other}
    )
    with pytest.raises(RuntimeError, match="manifest digest mismatch"):
        validate_manifest_tag(_client(httpserver), "repo", "v1", digest)
