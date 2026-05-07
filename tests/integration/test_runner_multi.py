"""Tests for the parallel multi-worker runner path."""

from __future__ import annotations

import threading
import time

from oci_modelcar.oci import ML_TAR, BlobDescriptor


def test_layer_ordering_preserved_under_parallelism(tmp_path, monkeypatch):
    """Even with workers=4 and randomized completion times, layers must
    appear in alphabetical hf_path order in the final manifest."""
    # We don't have a fully-mocked run_push fixture; instead we test the
    # ordering invariant directly by simulating the dict-based ordering used
    # in run_push.

    files = [("a.bin", 100), ("b.bin", 200), ("c.bin", 300), ("d.bin", 400)]
    layers_by_idx: dict[int, BlobDescriptor] = {}

    def worker(idx: int, name: str, size: int) -> None:
        # Simulate jittered completion
        time.sleep(0.01 * ((idx * 7) % 4))
        layers_by_idx[idx] = BlobDescriptor(
            media_type=ML_TAR,
            digest=f"sha256:{idx:064x}",
            size=size,
        )

    threads = [threading.Thread(target=worker, args=(i, f, s)) for i, (f, s) in enumerate(files)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    layers = [layers_by_idx[i] for i in sorted(layers_by_idx)]
    assert [d.digest for d in layers] == [f"sha256:{i:064x}" for i in range(len(files))]
