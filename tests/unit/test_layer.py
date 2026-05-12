import hashlib
import io
import tarfile

import pytest

from oci_modelcar.layer import (
    build_layer_tar_bytes,
    build_layer_to_file,
    make_tar_info,
    tar_layer_size,
)


@pytest.mark.parametrize("file_size", [0, 1, 100, 511, 512, 513, 1024, 1025, 12345, 1048576])
def test_tar_layer_size_matches_actual_bytes(file_size: int):
    """Streaming uploads need to set Content-Length upfront. The formula must
    equal the bytes that build_layer_tar_bytes produces, otherwise the
    registry hangs waiting for missing bytes."""
    actual = len(build_layer_tar_bytes("models/", "weights.bin", b"x" * file_size))
    assert tar_layer_size(file_size) == actual


@pytest.mark.parametrize(
    "file_size",
    [
        2**33 - 1,  # last ustar-fitting size: no PAX
        2**33,  # first PAX-triggering size: alignment puts mismatch at +10240
        2**33 + 1,  # PAX with body not aligned to 512
        2**33 + 511,
        2**33 + 512,
        34_025_419_000,  # Mistral Medium 3.5 shard — real-world regression case
    ],
)
def test_tar_layer_size_accounts_for_pax_extended_size_header(file_size: int):
    """Files >= 2**33 bytes (8 GiB) trigger a PAX extended 'size' header.

    Python tarfile prepends one 512 B 'x'-type header and one 512 B PAX data
    block before the ustar header. The formula must include that 1024 B of
    overhead, otherwise build_layer_to_file raises 'tar size mismatch' on
    real-world >8 GiB shards (e.g., Mistral Medium 3.5's 34 GB safetensors).
    """
    info = make_tar_info("models/", "weights.safetensors", file_size)
    header = info.tobuf(tarfile.PAX_FORMAT, encoding="utf-8", errors="surrogateescape")
    block = 512
    record = 10240
    body_padded = ((file_size + block - 1) // block) * block
    raw = len(header) + body_padded + 2 * block  # header(s) + body + 2-block trailer
    expected = ((raw + record - 1) // record) * record
    assert tar_layer_size(file_size) == expected


def test_build_layer_to_file_writes_tar_and_returns_digest(tmp_path):
    source = tmp_path / "weights.bin"
    payload = b"X" * 12345
    source.write_bytes(payload)

    dest = tmp_path / "weights.bin.tar"
    digest, size = build_layer_to_file(
        source_path=source,
        prefix="models/",
        filename="weights.bin",
        dest_path=dest,
    )

    raw = dest.read_bytes()
    assert size == len(raw)
    assert size == tar_layer_size(len(payload))
    assert digest == "sha256:" + hashlib.sha256(raw).hexdigest()

    # Tar contents must match what build_layer_tar_bytes would produce
    expected = build_layer_tar_bytes("models/", "weights.bin", payload)
    assert raw == expected


def test_build_layer_to_file_streaming_does_not_load_full_payload(tmp_path):
    """Verify streaming works for >1 MiB inputs and output matches in-memory build."""
    source = tmp_path / "big.bin"
    big_payload = b"Y" * (3 * 1024 * 1024)  # 3 MiB > 1 MiB internal chunks
    source.write_bytes(big_payload)
    dest = tmp_path / "big.bin.tar"
    _digest, size = build_layer_to_file(source, "models/", "big.bin", dest)
    assert size == tar_layer_size(len(big_payload))
    expected = build_layer_tar_bytes("models/", "big.bin", big_payload)
    assert dest.read_bytes() == expected


def test_layer_tar_is_reproducible(tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"X" * 12345)
    a = tmp_path / "a.tar"
    b = tmp_path / "b.tar"
    digest_a, _ = build_layer_to_file(source, "models/", "f.bin", a)
    digest_b, _ = build_layer_to_file(source, "models/", "f.bin", b)
    assert digest_a == digest_b
    assert a.read_bytes() == b.read_bytes()


def test_layer_tar_has_zero_mtime_uid_gid(tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"hello")
    dest = tmp_path / "f.tar"
    build_layer_to_file(source, "models/", "f.bin", dest)
    with tarfile.open(fileobj=io.BytesIO(dest.read_bytes()), mode="r") as tf:
        members = tf.getmembers()
        assert len(members) == 1
        m = members[0]
        assert m.name == "models/f.bin"
        assert m.size == 5
        assert m.mtime == 0
        assert m.uid == 0 and m.gid == 0
        assert m.uname == "" and m.gname == ""
        assert m.mode == 0o644
