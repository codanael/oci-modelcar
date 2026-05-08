import pytest

from oci_modelcar.layer import build_layer_tar_bytes, tar_layer_size


@pytest.mark.parametrize("file_size", [0, 1, 100, 511, 512, 513, 1024, 1025, 12345, 1048576])
def test_tar_layer_size_matches_actual_bytes(file_size: int):
    """Streaming uploads need to set Content-Length upfront. The formula must
    equal the bytes that build_layer_tar_bytes produces, otherwise the
    registry hangs waiting for missing bytes."""
    actual = len(build_layer_tar_bytes("models/", "weights.bin", b"x" * file_size))
    assert tar_layer_size(file_size) == actual
