import hashlib
import json
import re
from pathlib import Path

import pytest
import requests as _requests  # noqa: F401  # used in Task 5.4 retry tests
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.registry import (
    OciClient,
    StreamingBlobUpload,
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


def test_streaming_push_from_file_happy_path(httpserver, tmp_path: Path):
    payload = b"X" * (4 * 1024 * 1024 + 17)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/1")}
    )

    received = bytearray()

    def patch_handler(request):
        cr = request.headers.get("Content-Range", "")
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m, f"bad Content-Range {cr!r}"
        start, end = int(m.group(1)), int(m.group(2))
        assert start == 0
        assert end == len(payload) - 1
        cl = int(request.headers["Content-Length"])
        assert cl == len(payload)
        received.extend(request.data)
        return Response("", status=202, headers={"Location": httpserver.url_for("/u/1")})

    httpserver.expect_request("/u/1", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/1", method="PUT").respond_with_data("", status=201)

    upload = StreamingBlobUpload(client=_client(httpserver), repo="repo")
    out_digest, out_size = upload.push_from_file(f, len(payload), digest)
    assert out_digest == digest
    assert out_size == len(payload)
    assert bytes(received) == payload


@pytest.mark.parametrize("success_status", [200, 201, 202, 204])
def test_streaming_accepts_non_spec_success_codes(httpserver, tmp_path, success_status):
    """Artifactory returns 200/204; Harbor (some setups) returns 204.
    go-containerregistry accepts {201,202,204}; oras-py accepts {200,201,202}.
    Union: {200,201,202,204}."""
    payload = b"Z" * 1024
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/codes")}
    )
    httpserver.expect_request("/u/codes", method="PATCH").respond_with_data(
        "", status=success_status, headers={"Location": httpserver.url_for("/u/codes")}
    )
    httpserver.expect_request("/u/codes", method="PUT").respond_with_data("", status=201)

    upload = StreamingBlobUpload(client=_client(httpserver), repo="repo")
    out_digest, _ = upload.push_from_file(f, len(payload), digest)
    assert out_digest == digest


def test_streaming_unhandled_status_raises(httpserver, tmp_path):
    """299 (no spec meaning) must raise rather than spin or silently succeed."""
    payload = b"A" * 64
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/odd")}
    )
    httpserver.expect_request("/u/odd", method="PATCH").respond_with_data("", status=299)

    upload = StreamingBlobUpload(client=_client(httpserver), repo="repo")
    with pytest.raises(RuntimeError, match=r"unexpected.*299|status 299"):
        upload.push_from_file(f, len(payload), digest)


def test_streaming_no_chunked_transfer_encoding(httpserver, tmp_path):
    """Content-Length must be set explicitly to avoid chunked TE, which
    some registries handle differently from a fixed-size PATCH."""
    payload = b"L" * 512
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/len")}
    )

    seen_te: list[str | None] = []

    def patch_handler(request):
        seen_te.append(request.headers.get("Transfer-Encoding"))
        return Response("", status=202, headers={"Location": httpserver.url_for("/u/len")})

    httpserver.expect_request("/u/len", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/len", method="PUT").respond_with_data("", status=201)

    upload = StreamingBlobUpload(client=_client(httpserver), repo="repo")
    upload.push_from_file(f, len(payload), digest)

    te = seen_te[0] or ""
    assert "chunked" not in te.lower(), f"Transfer-Encoding leaked chunked: {te!r}"
