import hashlib

import pytest
from pytest_httpserver import HTTPServer

from oci_modelcar.oci import OciClient, push_manifest, validate_manifest_tag


def _client(httpserver: HTTPServer) -> OciClient:
    return OciClient(host_url=httpserver.url_for(""))


def test_push_manifest_returns_digest(httpserver: HTTPServer):
    body = b'{"schemaVersion":2}'
    expected = "sha256:" + hashlib.sha256(body).hexdigest()
    httpserver.expect_request("/v2/repo/manifests/v1", method="PUT").respond_with_data(
        "", status=201, headers={"Docker-Content-Digest": expected}
    )
    client = _client(httpserver)
    digest = push_manifest(client, repo="repo", tag="v1", manifest_bytes=body)
    assert digest == expected


def test_validate_manifest_tag_match(httpserver: HTTPServer):
    body = b'{"schemaVersion":2}'
    expected = "sha256:" + hashlib.sha256(body).hexdigest()
    httpserver.expect_request("/v2/repo/manifests/v1", method="GET").respond_with_data(
        body,
        status=200,
        headers={
            "Docker-Content-Digest": expected,
            "Content-Type": "application/vnd.oci.image.manifest.v1+json",
        },
    )
    client = _client(httpserver)
    validate_manifest_tag(client, repo="repo", tag="v1", expected_digest=expected)


def test_validate_manifest_tag_mismatch(httpserver: HTTPServer):
    body = b'{"schemaVersion":2}'
    expected = "sha256:" + hashlib.sha256(body).hexdigest()
    wrong = "sha256:" + "0" * 64
    httpserver.expect_request("/v2/repo/manifests/v1", method="GET").respond_with_data(
        body, status=200, headers={"Docker-Content-Digest": wrong}
    )
    client = _client(httpserver)
    with pytest.raises(RuntimeError, match="manifest digest mismatch"):
        validate_manifest_tag(client, repo="repo", tag="v1", expected_digest=expected)
