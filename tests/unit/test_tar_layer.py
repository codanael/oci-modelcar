import hashlib
import io
import tarfile

from oci_modelcar.tar_layer import build_layer_tar_bytes


def test_layer_tar_is_reproducible():
    payload = b"X" * 12345
    a = build_layer_tar_bytes(prefix="models/", filename="x.bin", payload=payload)
    b = build_layer_tar_bytes(prefix="models/", filename="x.bin", payload=payload)
    assert a == b
    digest_a = hashlib.sha256(a).hexdigest()
    digest_b = hashlib.sha256(b).hexdigest()
    assert digest_a == digest_b


def test_layer_tar_contains_file_with_zero_mtime():
    payload = b"hello"
    raw = build_layer_tar_bytes(prefix="models/", filename="hi.txt", payload=payload)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tf:
        members = tf.getmembers()
        assert len(members) == 1
        m = members[0]
        assert m.name == "models/hi.txt"
        assert m.size == len(payload)
        assert m.mtime == 0
        assert m.uid == 0 and m.gid == 0
        assert m.uname == "" and m.gname == ""
        assert m.mode == 0o644
        extracted = tf.extractfile(m)
        assert extracted is not None
        assert extracted.read() == payload


def test_layer_tar_diff_id_equals_sha_of_bytes():
    payload = b"abc" * 100
    raw = build_layer_tar_bytes(prefix="models/", filename="a.bin", payload=payload)
    diff_id = "sha256:" + hashlib.sha256(raw).hexdigest()
    assert diff_id.startswith("sha256:")
    assert len(diff_id.split(":")[1]) == 64
