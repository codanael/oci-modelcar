import hashlib

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response


def _setup_two_files(httpserver: HTTPServer) -> None:
    # HF tree
    httpserver.expect_request("/api/models/foo/bar").respond_with_json({"sha": "a" * 40})
    httpserver.expect_request("/api/models/foo/bar/tree/main").respond_with_json(
        [
            {"type": "file", "path": "config.json", "size": 12},
            {"type": "file", "path": "model.bin", "size": 100},
        ]
    )
    httpserver.expect_request("/foo/bar/resolve/main/config.json").respond_with_data(
        b'{"x":"v1"}\n\n', headers={"Content-Length": "12"}
    )
    httpserver.expect_request("/foo/bar/resolve/main/model.bin").respond_with_data(
        b"M" * 100, headers={"Content-Length": "100"}
    )

    # OCI: each blob upload init + put. registry:2 normally requires unique upload IDs;
    # we just respond identically and track via path.
    upload_count = {"n": 0}

    def upload_init(request):
        upload_count["n"] += 1
        return Response(
            "",
            status=202,
            headers={"Location": httpserver.url_for(f"/u/{upload_count['n']}")},
        )

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_handler(
        upload_init
    )

    # PUT close for any /u/N
    def put_handler(request):
        return Response(
            "",
            status=201,
            headers={"Docker-Content-Digest": "sha256:" + "0" * 64},
        )

    for i in range(1, 10):
        httpserver.expect_request(f"/u/{i}", method="PUT").respond_with_handler(put_handler)

    # HEAD blob for validation: respond 200 with whatever digest the client expects
    def head_handler(request):
        digest = request.path.split("/")[-1]
        return Response(
            "",
            status=200,
            headers={"Docker-Content-Digest": digest, "Content-Length": "1024"},
        )

    httpserver.expect_request("/v2/repo/blobs/", method="HEAD").respond_with_handler(head_handler)
    # Wildcard digest path:
    httpserver.expect_request(
        regex=r"^/v2/repo/blobs/sha256:[0-9a-f]{64}$",
        method="HEAD",
    ).respond_with_handler(head_handler)

    # Config blob HEAD (push_small_blob does HEAD first)
    httpserver.expect_request(
        regex=r"^/v2/repo/blobs/sha256:[0-9a-f]{64}$",
        method="HEAD",
    ).respond_with_handler(head_handler)

    # Manifest PUT
    def manifest_put(request):
        body = request.data
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        return Response("", status=201, headers={"Docker-Content-Digest": digest})

    httpserver.expect_request("/v2/repo/manifests/aaaaaaaaaaaa", method="PUT").respond_with_handler(
        manifest_put
    )

    # Manifest GET for validation
    def manifest_get(request):
        return Response(
            b"",  # body unused
            status=200,
            headers={"Docker-Content-Digest": "sha256:placeholder"},
        )

    httpserver.expect_request("/v2/repo/manifests/aaaaaaaaaaaa", method="GET").respond_with_handler(
        manifest_get
    )


@pytest.mark.skip(
    reason="Full multi-file integration is asserted via E2E; "
    "this scaffold left as a hook for follow-up."
)
def test_run_push_two_files_writes_state(httpserver: HTTPServer, tmp_path):
    pass
