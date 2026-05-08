from pathlib import Path
from unittest.mock import MagicMock

import pytest
from huggingface_hub import HfApi
from pytest_httpserver import HTTPServer

from oci_modelcar.download import HfDownloader, HfFile
from oci_modelcar.errors import DownloadError
from oci_modelcar.http import build_session


def _make_downloader(httpserver: HTTPServer, spool: Path) -> HfDownloader:
    api = HfApi(endpoint=httpserver.url_for(""))
    # Use a session with no urllib3 retries so test-level max_retries drives the loop
    session = build_session()
    session.adapters.clear()
    from requests.adapters import HTTPAdapter as _Adapter

    session.mount("http://", _Adapter(max_retries=0))
    session.mount("https://", _Adapter(max_retries=0))
    return HfDownloader(
        api=api,
        session=session,
        spool_dir=spool,
        stop_event=None,
        max_retries=3,
        backoff_initial=0.0,
    )


def test_hf_file_carries_metadata():
    f = HfFile(path="model.safetensors", size=1234, lfs_sha256="a" * 64)
    assert f.path == "model.safetensors"
    assert f.size == 1234
    assert f.lfs_sha256 == "a" * 64


def test_hf_file_no_lfs():
    f = HfFile(path="config.json", size=100, lfs_sha256=None)
    assert f.lfs_sha256 is None


def test_resolve_revision_uses_repo_info():
    api = MagicMock()
    api.repo_info.return_value = MagicMock(sha="9fb191250dd56d0ba7ec9785a025ed29c03d5998")
    d = HfDownloader(api=api, session=MagicMock(), spool_dir=None, stop_event=None)
    assert (
        d.resolve_revision("Qwen/Qwen2.5-7B", "main") == "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    )
    api.repo_info.assert_called_once_with("Qwen/Qwen2.5-7B", revision="main")


def test_list_files_filters_by_allow_patterns():
    api = MagicMock()
    api.list_repo_tree.return_value = [
        MagicMock(type="file", path="model.safetensors", size=1000, lfs=MagicMock(sha256="a" * 64)),
        MagicMock(type="file", path="config.json", size=100, lfs=None),
        MagicMock(type="file", path="readme.md", size=50, lfs=None),
        MagicMock(type="file", path="ignored.bin", size=10, lfs=None),
        MagicMock(type="directory", path="subdir", size=0, lfs=None),
    ]
    d = HfDownloader(api=api, session=MagicMock(), spool_dir=None, stop_event=None)
    files = d.list_files("Qwen/Qwen2.5-7B", "main", allow=(".safetensors", ".json", ".md"))
    paths = [f.path for f in files]
    assert paths == sorted(paths)
    assert "model.safetensors" in paths
    assert "config.json" in paths
    assert "readme.md" in paths
    assert "ignored.bin" not in paths
    assert "subdir" not in paths


def test_list_files_extracts_lfs_sha256():
    api = MagicMock()
    lfs_meta = MagicMock(sha256="b" * 64)
    api.list_repo_tree.return_value = [
        MagicMock(type="file", path="model.safetensors", size=1000, lfs=lfs_meta),
        MagicMock(type="file", path="config.json", size=100, lfs=None),
    ]
    d = HfDownloader(api=api, session=MagicMock(), spool_dir=None, stop_event=None)
    files = d.list_files("Qwen/Qwen2.5-7B", "main", allow=(".safetensors", ".json"))
    by_path = {f.path: f for f in files}
    assert by_path["model.safetensors"].lfs_sha256 == "b" * 64
    assert by_path["config.json"].lfs_sha256 is None


def test_download_writes_file_and_returns_path(httpserver: HTTPServer, tmp_path: Path) -> None:
    payload = b"hello world"
    httpserver.expect_request("/repo/resolve/main/file.txt").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )

    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.txt", size=len(payload), lfs_sha256=None)

    result = d.download("repo", "main", f)
    assert result == spool / "sources" / "file.txt"
    assert result.read_bytes() == payload
    assert not (spool / "sources" / "file.txt.partial").exists()


def test_download_preserves_subdirs_in_hf_path(httpserver: HTTPServer, tmp_path: Path) -> None:
    payload = b"sub"
    httpserver.expect_request("/repo/resolve/main/subdir/inner.txt").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )

    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="subdir/inner.txt", size=len(payload), lfs_sha256=None)

    result = d.download("repo", "main", f)
    assert result == spool / "sources" / "subdir" / "inner.txt"
    assert result.read_bytes() == payload


def test_download_calls_progress_cb(httpserver: HTTPServer, tmp_path: Path) -> None:
    payload = b"X" * 4096
    httpserver.expect_request("/repo/resolve/main/big.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="big.bin", size=len(payload), lfs_sha256=None)

    seen: list[int] = []
    d.download("repo", "main", f, progress_cb=seen.append)
    assert seen, "progress_cb was never invoked"
    assert seen[-1] == len(payload)


def test_download_partial_file_cleaned_on_exception(httpserver: HTTPServer, tmp_path: Path) -> None:
    """If the GET fails repeatedly, the .partial file is removed."""
    httpserver.expect_request("/repo/resolve/main/file.bin").respond_with_data("", status=503)
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.bin", size=10, lfs_sha256=None)

    with pytest.raises(DownloadError):
        d.download("repo", "main", f)
    partial = spool / "sources" / "file.bin.partial"
    final = spool / "sources" / "file.bin"
    assert not partial.exists() and not final.exists()


def test_download_aborts_within_two_chunks_of_stop_event(
    httpserver: HTTPServer, tmp_path: Path
) -> None:
    """Regression test for v0.x: 50 GB DL was uncancellable. v1 must cancel
    within ~2 x CHUNK_DEFAULT (~2 MiB) of stop_event.set()."""
    import threading

    big_payload = b"X" * (8 * 1024 * 1024)
    httpserver.expect_request("/repo/resolve/main/big.bin").respond_with_data(
        big_payload, headers={"Content-Length": str(len(big_payload))}
    )
    spool = tmp_path / "spool"
    stop = threading.Event()
    api = HfApi(endpoint=httpserver.url_for(""))
    session = build_session()
    session.adapters.clear()
    from requests.adapters import HTTPAdapter as _Adapter

    session.mount("http://", _Adapter(max_retries=0))
    session.mount("https://", _Adapter(max_retries=0))
    d = HfDownloader(api=api, session=session, spool_dir=spool, stop_event=stop, max_retries=1)
    f = HfFile(path="big.bin", size=len(big_payload), lfs_sha256=None)

    chunks_seen = 0
    raised: list[BaseException] = []

    def progress(_n: int) -> None:
        nonlocal chunks_seen
        chunks_seen += 1
        if chunks_seen == 1:
            stop.set()  # request abort after the first chunk

    def runner() -> None:
        try:
            d.download("repo", "main", f, progress_cb=progress)
        except BaseException as e:
            raised.append(e)

    t = threading.Thread(target=runner)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "download did not abort within timeout"
    assert raised and isinstance(raised[0], InterruptedError)
