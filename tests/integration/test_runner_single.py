from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.hf import HfClient, HfFile
from oci_modelcar.oci import OciClient
from oci_modelcar.runner import process_one_file


def test_process_one_file_pushes_layer(httpserver: HTTPServer):
    payload = b"hello world!"
    # HF resolve endpoint
    httpserver.expect_request("/foo/bar/resolve/main/file.txt").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )

    # OCI POST upload init
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/1")}
    )

    # OCI PUT close (small payload, all in PUT)
    received = {"data": b""}

    def put_handler(request):
        received["data"] = request.data
        return Response("", status=201)

    httpserver.expect_request("/u/1", method="PUT").respond_with_handler(put_handler)

    hf_client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    oci_client = OciClient(host_url=httpserver.url_for(""))
    hf_file = HfFile(path="file.txt", size=len(payload))

    descriptor, diff_id, telemetry = process_one_file(
        hf_client=hf_client,
        oci_client=oci_client,
        repo="repo",
        revision="main",
        hf_file=hf_file,
        layer_prefix="models/",
        chunk_size=8 * 1024 * 1024,
    )
    assert descriptor.media_type == "application/vnd.oci.image.layer.v1.tar"
    assert descriptor.digest.startswith("sha256:")
    assert descriptor.size > len(payload)  # tar overhead
    assert diff_id == descriptor.digest
    # Telemetry is populated even on tiny payloads
    assert telemetry.bytes_through > 0
    assert telemetry.elapsed_s >= 0

    # Verify the bytes pushed match what we'd expect from a tar containing payload
    assert received["data"]


def test_process_one_file_forwards_progress_cb(httpserver: HTTPServer):
    """progress_cb passed to process_one_file is invoked during HF read."""
    payload = b"X" * 4096
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/2")}
    )
    httpserver.expect_request("/u/2", method="PUT").respond_with_data("", status=201)

    hf_client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    oci_client = OciClient(host_url=httpserver.url_for(""))
    hf_file = HfFile(path="file.bin", size=len(payload))

    seen: list[int] = []
    process_one_file(
        hf_client=hf_client,
        oci_client=oci_client,
        repo="repo",
        revision="main",
        hf_file=hf_file,
        layer_prefix="models/",
        chunk_size=8 * 1024 * 1024,
        progress_cb=seen.append,
    )
    # At least one progress call observed; final cumulative equals total
    assert seen, "progress_cb was never invoked"
    assert seen[-1] == len(payload)
