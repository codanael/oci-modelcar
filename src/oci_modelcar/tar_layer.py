"""Tar layer streaming wrapper.

Writes one file as an uncompressed tar archive. For uncompressed tar layers
(application/vnd.oci.image.layer.v1.tar), the digest of the tar bytes equals
the diff_id (per OCI image spec).
"""

from __future__ import annotations

import io
import tarfile
from typing import IO

_TAR_BLOCKSIZE = 512  # one tar record (header or data block)
_TAR_RECORDSIZE = 10240  # Python tarfile blocking factor: pads to 20 records


def tar_layer_size(file_size: int) -> int:
    """Exact byte size of a single-file uncompressed tar archive.

    Layout: 512-byte header + file body padded to 512 + 1024-byte trailer
    (two zero blocks), all padded up to RECORDSIZE = 10240. Deterministic
    given mtime=0/uid=0/gid=0 (the headers we always emit). Streaming
    uploads use this to set Content-Length upfront — Content-Length must
    match the bytes actually emitted by ``stream_layer_to`` or the registry
    will hang waiting for missing bytes (or reject as bad framing).
    """
    body_padded = (file_size + _TAR_BLOCKSIZE - 1) // _TAR_BLOCKSIZE * _TAR_BLOCKSIZE
    raw = _TAR_BLOCKSIZE + body_padded + 2 * _TAR_BLOCKSIZE  # header + body + trailer (2 blocks)
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
    """Build a single-file uncompressed tar archive into memory.

    For testing reproducibility. Production uses stream_layer_to.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w|") as tar:
        info = make_tar_info(prefix, filename, len(payload))
        tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def stream_layer_to(
    sink: IO[bytes], prefix: str, filename: str, size: int, source: IO[bytes]
) -> None:
    """Stream a single file as an uncompressed tar layer into `sink`.

    `sink` must implement write(data) -> int (e.g. ChunkedBlobUpload).
    `source` must implement read(n) -> bytes (e.g. HfStream).
    """
    with tarfile.open(fileobj=sink, mode="w|") as tar:
        info = make_tar_info(prefix, filename, size)
        tar.addfile(info, source)
