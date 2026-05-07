import json
import threading
from pathlib import Path

from oci_modelcar.state import JobState, JsonStateStore


def test_load_creates_empty_state(tmp_path: Path):
    store = JsonStateStore(tmp_path / "state.json")
    assert store.list_jobs() == []


def test_compute_job_key_is_stable():
    k1 = JsonStateStore.compute_job_key(
        hf_repo="foo/bar",
        revision_resolved="a" * 40,
        registry="r.example",
        target_repo="m/x",
        target_tag="v1",
    )
    k2 = JsonStateStore.compute_job_key(
        hf_repo="foo/bar",
        revision_resolved="a" * 40,
        registry="r.example",
        target_repo="m/x",
        target_tag="v1",
    )
    assert k1 == k2
    assert len(k1) == 16


def test_compute_job_key_differs_on_revision_change():
    k1 = JsonStateStore.compute_job_key(
        hf_repo="foo/bar",
        revision_resolved="a" * 40,
        registry="r",
        target_repo="m",
        target_tag="v1",
    )
    k2 = JsonStateStore.compute_job_key(
        hf_repo="foo/bar",
        revision_resolved="b" * 40,
        registry="r",
        target_repo="m",
        target_tag="v1",
    )
    assert k1 != k2


def test_atomic_write_creates_file(tmp_path: Path):
    store = JsonStateStore(tmp_path / "state.json")
    job = JobState(
        hf_repo="foo/bar",
        hf_revision_input="main",
        hf_revision_resolved="a" * 40,
        registry="r.example",
        target_repo="m/x",
        target_tag="v1",
    )
    store.upsert_job("k1", job)
    store.save()
    assert (tmp_path / "state.json").is_file()
    raw = json.loads((tmp_path / "state.json").read_text())
    assert raw["version"] == 1
    assert "k1" in raw["jobs"]
    assert raw["jobs"]["k1"]["source"]["hf_repo"] == "foo/bar"


def test_mark_pushed_and_has_pushed(tmp_path: Path):
    store = JsonStateStore(tmp_path / "state.json")
    job = JobState(
        hf_repo="foo/bar",
        hf_revision_input="main",
        hf_revision_resolved="a" * 40,
        registry="r",
        target_repo="m",
        target_tag="v1",
    )
    store.upsert_job("k1", job)
    assert not store.has_pushed("k1", "model.safetensors", expected_size=100)
    store.mark_pushed(
        "k1", "model.safetensors", digest="sha256:abc", diff_id="sha256:abc", size=100
    )
    store.save()
    assert store.has_pushed("k1", "model.safetensors", expected_size=100)
    # Wrong size invalidates
    assert not store.has_pushed("k1", "model.safetensors", expected_size=200)


def test_mark_completed(tmp_path: Path):
    store = JsonStateStore(tmp_path / "state.json")
    job = JobState(
        hf_repo="foo/bar",
        hf_revision_input="main",
        hf_revision_resolved="a" * 40,
        registry="r",
        target_repo="m",
        target_tag="v1",
    )
    store.upsert_job("k1", job)
    assert not store.is_completed("k1")
    store.mark_completed("k1", manifest_digest="sha256:abc")
    assert store.is_completed("k1")


def test_concurrent_writes_no_corruption(tmp_path: Path):
    store = JsonStateStore(tmp_path / "state.json")
    job = JobState(
        hf_repo="foo/bar",
        hf_revision_input="main",
        hf_revision_resolved="a" * 40,
        registry="r",
        target_repo="m",
        target_tag="v1",
    )
    store.upsert_job("k1", job)

    def worker(i: int):
        store.mark_pushed(
            "k1",
            f"file{i}.bin",
            digest=f"sha256:{i}",
            diff_id=f"sha256:{i}",
            size=i,
        )
        store.save()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    raw = json.loads((tmp_path / "state.json").read_text())
    assert len(raw["jobs"]["k1"]["files"]) == 20


def test_file_permissions_0600(tmp_path: Path):
    p = tmp_path / "state.json"
    store = JsonStateStore(p)
    store.save()
    mode = p.stat().st_mode & 0o777
    assert mode == 0o600
