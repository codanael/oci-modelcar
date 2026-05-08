"""Tar layer building. Uncompressed (mediaType vnd.oci.image.layer.v1.tar)
so that layer.digest == diff_id by construction."""

from __future__ import annotations

import io
import tarfile

_TAR_BLOCKSIZE = 512
_TAR_RECORDSIZE = 10240


def tar_layer_size(file_size: int) -> int:
    """Exact bytes produced by build_layer_tar_bytes / build_layer_to_file
    for the given file size. Deterministic given mtime=0/uid=0/gid=0."""
    body_padded = (file_size + _TAR_BLOCKSIZE - 1) // _TAR_BLOCKSIZE * _TAR_BLOCKSIZE
    raw = _TAR_BLOCKSIZE + body_padded + 2 * _TAR_BLOCKSIZE  # header + body + 2-block trailer
    return (raw + _TAR_RECORDSIZE - 1) // _TAR_RECORDSIZE * _TAR_RECORDSIZE


def make_tar_info(prefix: str, filename: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=prefix + filename)
    info.size = size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info


def build_layer_tar_bytes(prefix: str, filename: str, payload: bytes) -> bytes:
    """In-memory tar build. Used by tests; production uses build_layer_to_file."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w|") as tar:
        info = make_tar_info(prefix, filename, len(payload))
        tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()
