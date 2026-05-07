"""Pipeline orchestration: process_one_file + main run loop."""

from __future__ import annotations

import os

from oci_modelcar.hf import HfClient, HfFile, HfStream
from oci_modelcar.oci import (
    ML_TAR,
    BlobDescriptor,
    ChunkedBlobUpload,
    OciClient,
)
from oci_modelcar.tar_layer import stream_layer_to


def process_one_file(
    hf_client: HfClient,
    oci_client: OciClient,
    repo: str,
    revision: str,
    hf_file: HfFile,
    layer_prefix: str,
    chunk_size: int,
    hf_max_retries: int = 10,
    oci_max_retries: int = 10,
    backoff_initial: float = 1.0,
) -> tuple[BlobDescriptor, str]:
    """Stream one HF file as one tar layer; returns (descriptor, diff_id).

    For uncompressed tar layers, diff_id == descriptor.digest.
    """
    hf_stream = HfStream(
        client=hf_client,
        revision=revision,
        path=hf_file.path,
        size=hf_file.size,
        max_retries=hf_max_retries,
        backoff_initial=backoff_initial,
    )
    upload = ChunkedBlobUpload(
        client=oci_client,
        repo=repo,
        chunk_size=chunk_size,
        max_retries=oci_max_retries,
        backoff_initial=backoff_initial,
    )
    try:
        stream_layer_to(
            sink=upload,  # type: ignore[arg-type]
            prefix=layer_prefix,
            filename=os.path.basename(hf_file.path),
            size=hf_file.size,
            source=hf_stream,  # type: ignore[arg-type]
        )
    finally:
        hf_stream.close()
    digest, layer_size = upload.close()
    descriptor = BlobDescriptor(media_type=ML_TAR, digest=digest, size=layer_size)
    return descriptor, digest


__all__ = ["process_one_file"]
