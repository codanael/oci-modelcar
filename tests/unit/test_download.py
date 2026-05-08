from unittest.mock import MagicMock

from oci_modelcar.download import HfDownloader, HfFile


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
