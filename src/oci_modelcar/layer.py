"""Tar layer building. Uncompressed (mediaType vnd.oci.image.layer.v1.tar)
so that layer.digest == diff_id by construction."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from typing import cast

_TAR_BLOCKSIZE = 512
_TAR_RECORDSIZE = 10240


def tar_layer_size(file_size: int, name: str = "") -> int:
    """Exact bytes produced by build_layer_tar_bytes / build_layer_to_file
    for a member called `name` (prefix + hf_path) of `file_size` bytes.

    The header is measured by serialising the real TarInfo, so PAX extended
    headers that tarfile adds for sizes >= 8 GiB or names > 100 chars are
    included automatically instead of being modelled by hand.
    """
    header = len(
        make_tar_info("", name, file_size).tobuf(
            tarfile.PAX_FORMAT, encoding="utf-8", errors="surrogateescape"
        )
    )
    body_padded = (file_size + _TAR_BLOCKSIZE - 1) // _TAR_BLOCKSIZE * _TAR_BLOCKSIZE
    raw = header + body_padded + 2 * _TAR_BLOCKSIZE  # header(s) + body + 2-block trailer
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


class _HashingWriter:
    """File-like wrapper that hashes every byte written to the inner file."""

    def __init__(self, inner: io.BufferedWriter) -> None:
        self._inner = inner
        self.h = hashlib.sha256()
        self.bytes_written = 0

    def write(self, data: bytes) -> int:
        self.h.update(data)
        n = self._inner.write(data)
        self.bytes_written += n
        return n

    def flush(self) -> None:
        self._inner.flush()


def build_layer_to_file(
    source_path: Path,
    prefix: str,
    filename: str,
    dest_path: Path,
    read_chunk: int = 1024 * 1024,
) -> tuple[str, int]:
    """Build the tar layer at dest_path streaming from source_path.

    Returns (digest, size) where digest is "sha256:<64hex>" and size is the
    total bytes written (== tar_layer_size(source_size)).

    Memory bound: ~read_chunk + tar internal buffering (~64 KiB).
    """
    source_size = source_path.stat().st_size
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as raw:
        writer = _HashingWriter(raw)
        with tarfile.open(fileobj=cast(io.BufferedWriter, writer), mode="w|") as tar:
            info = make_tar_info(prefix, filename, source_size)
            with open(source_path, "rb") as src:
                tar.addfile(info, src)
    digest = "sha256:" + writer.h.hexdigest()
    expected_size = tar_layer_size(source_size, prefix + filename)
    if writer.bytes_written != expected_size:
        raise RuntimeError(
            f"tar size mismatch for {filename}: wrote {writer.bytes_written}, "
            f"formula expected {expected_size}"
        )
    return digest, writer.bytes_written
