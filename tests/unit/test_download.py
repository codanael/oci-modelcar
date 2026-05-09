from pathlib import Path
from unittest.mock import MagicMock

import pytest
from huggingface_hub import HfApi
from pytest_httpserver import HTTPServer

from oci_modelcar.download import HfDownloader, HfFile
from oci_modelcar.errors import DownloadError, EntryNotFoundError, GatedRepoError
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
    from huggingface_hub.hf_api import ModelInfo

    api = MagicMock()
    info = MagicMock(spec=ModelInfo)
    info.sha = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    api.repo_info.return_value = info
    d = HfDownloader(api=api, session=MagicMock(), spool_dir=None, stop_event=None)
    assert (
        d.resolve_revision("Qwen/Qwen2.5-7B", "main") == "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    )
    api.repo_info.assert_called_once_with("Qwen/Qwen2.5-7B", revision="main")


def test_resolve_revision_raises_when_sha_is_none():
    """ModelInfo.sha is `str | None` — guard against silently emitting the
    literal "None" downstream into resolve URLs."""
    from huggingface_hub.hf_api import ModelInfo

    from oci_modelcar.errors import RevisionNotFoundError

    api = MagicMock()
    info = MagicMock(spec=ModelInfo)
    info.sha = None
    api.repo_info.return_value = info
    d = HfDownloader(api=api, session=MagicMock(), spool_dir=None, stop_event=None)
    with pytest.raises(RevisionNotFoundError, match="no SHA"):
        d.resolve_revision("Qwen/Qwen2.5-7B", "main")


def _file_mock(path: str, size: int, lfs=None):
    """RepoFile-shaped mock that passes isinstance(entry, RepoFile)."""
    from huggingface_hub.hf_api import RepoFile

    m = MagicMock(spec=RepoFile)
    m.path = path
    m.size = size
    m.lfs = lfs
    return m


def _folder_mock(path: str):
    from huggingface_hub.hf_api import RepoFolder

    m = MagicMock(spec=RepoFolder)
    m.path = path
    return m


def test_list_files_filters_by_allow_patterns():
    api = MagicMock()
    api.list_repo_tree.return_value = [
        _file_mock("model.safetensors", 1000, lfs=MagicMock(sha256="a" * 64)),
        _file_mock("config.json", 100),
        _file_mock("readme.md", 50),
        _file_mock("ignored.bin", 10),
        _folder_mock("subdir"),
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
        _file_mock("model.safetensors", 1000, lfs=lfs_meta),
        _file_mock("config.json", 100, lfs=None),
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


def test_download_handles_range_200_fallback(
    httpserver: HTTPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If first attempt downloads partial bytes then fails, and the second
    attempt sends Range but server ignores it (returns 200), the partial
    file is truncated and download restarts cleanly."""
    payload = b"Y" * 4096
    state: dict[str, bool] = {"served": False}

    def handler(request):  # type: ignore[no-untyped-def]
        from werkzeug.wrappers import Response

        if not state["served"]:
            state["served"] = True
            # First call: short body (1024 bytes) that closes after Content-Length
            return Response(
                payload[:1024],
                status=200,
                headers={"Content-Length": str(len(payload[:1024]))},
            )
        # Subsequent calls: ignore Range, return full payload with 200
        return Response(payload, status=200, headers={"Content-Length": str(len(payload))})

    httpserver.expect_request("/repo/resolve/main/file.bin").respond_with_handler(handler)
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.bin", size=len(payload), lfs_sha256=None)

    monkeypatch.setattr("oci_modelcar.download.time.sleep", lambda _d: None)
    out = d.download("repo", "main", f)
    assert out.read_bytes() == payload


def test_download_gated_repo_raises_specific_error(httpserver: HTTPServer, tmp_path: Path) -> None:
    httpserver.expect_request("/repo/resolve/main/file.bin").respond_with_data(
        "Gated", status=403, headers={"X-Error-Code": "GatedRepo"}
    )
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.bin", size=10, lfs_sha256=None)

    with pytest.raises(GatedRepoError) as exc:
        d.download("repo", "main", f)
    assert exc.value.hint and "huggingface.co/repo" in exc.value.hint


def test_download_404_on_resolve_raises_entry_not_found(
    httpserver: HTTPServer, tmp_path: Path
) -> None:
    httpserver.expect_request("/repo/resolve/main/missing.bin").respond_with_data(
        "Not found", status=404
    )
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="missing.bin", size=10, lfs_sha256=None)

    with pytest.raises(EntryNotFoundError):
        d.download("repo", "main", f)


def test_download_lfs_sha_verified(httpserver: HTTPServer, tmp_path: Path) -> None:
    import hashlib

    payload = b"hello world"
    correct_sha = hashlib.sha256(payload).hexdigest()
    httpserver.expect_request("/repo/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.bin", size=len(payload), lfs_sha256=correct_sha)

    result = d.download("repo", "main", f)
    assert result.read_bytes() == payload


def test_download_lfs_sha_mismatch_raises(httpserver: HTTPServer, tmp_path: Path) -> None:
    payload = b"hello world"
    wrong_sha = "0" * 64
    httpserver.expect_request("/repo/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.bin", size=len(payload), lfs_sha256=wrong_sha)

    with pytest.raises(DownloadError, match="sha256 mismatch"):
        d.download("repo", "main", f)
