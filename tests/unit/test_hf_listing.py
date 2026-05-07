from pytest_httpserver import HTTPServer

from oci_modelcar.hf import HfClient, HfFile


def test_list_files_basic(httpserver: HTTPServer):
    httpserver.expect_request("/api/models/foo/bar/tree/main").respond_with_json(
        [
            {"type": "file", "path": "model.safetensors", "size": 1000},
            {"type": "file", "path": "config.json", "size": 100},
            {"type": "file", "path": ".gitattributes", "size": 50},
            {"type": "directory", "path": "subdir", "size": 0},
        ]
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    files = client.list_files("main", allow=(".safetensors", ".json"))
    assert files == [
        HfFile(path="config.json", size=100),
        HfFile(path="model.safetensors", size=1000),
    ]


def test_list_files_filters_by_extension(httpserver: HTTPServer):
    httpserver.expect_request("/api/models/foo/bar/tree/main").respond_with_json(
        [
            {"type": "file", "path": "a.bin", "size": 1},
            {"type": "file", "path": "b.safetensors", "size": 2},
        ]
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    files = client.list_files("main", allow=(".safetensors",))
    assert [f.path for f in files] == ["b.safetensors"]


def test_list_files_uses_recursive_query_param(httpserver: HTTPServer):
    httpserver.expect_request(
        "/api/models/foo/bar/tree/main", query_string={"recursive": "true"}
    ).respond_with_json([])
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    files = client.list_files("main", allow=(".safetensors",))
    assert files == []
