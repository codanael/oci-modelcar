from pytest_httpserver import HTTPServer

from oci_modelcar.hf import HfClient


def test_resolve_revision_main(httpserver: HTTPServer):
    httpserver.expect_request("/api/models/foo/bar").respond_with_json(
        {"sha": "a3f47b09c8d2e6f1a89b7c4d3e8f2a1b5c6d7e8f"}
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    sha = client.resolve_revision("main")
    assert sha == "a3f47b09c8d2e6f1a89b7c4d3e8f2a1b5c6d7e8f"


def test_resolve_revision_explicit_sha(httpserver: HTTPServer):
    full = "0" * 40
    httpserver.expect_request(f"/api/models/foo/bar/revision/{full}").respond_with_json(
        {"sha": full}
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    sha = client.resolve_revision(full)
    assert sha == full


def test_resolve_revision_branch_name(httpserver: HTTPServer):
    httpserver.expect_request("/api/models/foo/bar/revision/release/v1").respond_with_json(
        {"sha": "b" * 40}
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    sha = client.resolve_revision("release/v1")
    assert sha == "b" * 40


def test_resolve_revision_falls_back_on_404(httpserver: HTTPServer):
    httpserver.expect_request("/api/models/foo/bar/revision/unknown").respond_with_data(
        "", status=404
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    sha = client.resolve_revision("unknown")
    # Falls back: returns the input as-is and warns
    assert sha == "unknown"
