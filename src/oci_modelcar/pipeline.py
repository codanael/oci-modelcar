"""Per-file pipeline orchestration: FileWorker + Pipeline."""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oci_modelcar.download import HfDownloader, HfFile
from oci_modelcar.errors import PushError
from oci_modelcar.layer import build_layer_to_file
from oci_modelcar.manifest import ML_TAR, BlobDescriptor
from oci_modelcar.registry import OciClient, StreamingBlobUpload

log = logging.getLogger(__name__)


class FileWorker:
    """Process one HF file end-to-end (phases a→f from the design spec)."""

    def __init__(
        self,
        downloader: HfDownloader,
        registry_client: OciClient,
        head_blob_fn: Callable[..., Any],
        streaming_factory: Callable[..., StreamingBlobUpload],
        layer_prefix: str,
        spool_dir: Path,
        clean_hf_after_push: bool,
        oci_max_retries: int,
        backoff_initial: float = 1.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.downloader = downloader
        self.registry_client = registry_client
        self.head_blob_fn = head_blob_fn
        self.streaming_factory = streaming_factory
        self.layer_prefix = layer_prefix
        self.spool_dir = spool_dir
        self.clean_hf_after_push = clean_hf_after_push
        self.oci_max_retries = oci_max_retries
        self.backoff_initial = backoff_initial
        self.stop_event = stop_event

    def process(
        self,
        repo: str,
        revision: str,
        hf_file: HfFile,
        progress_cb: Callable[[int], None] | None = None,
    ) -> BlobDescriptor:
        if self.stop_event is not None and self.stop_event.is_set():
            raise InterruptedError(f"worker for {hf_file.path} aborted before start")

        # a. DOWNLOAD
        source_path = self.downloader.download(repo, revision, hf_file, progress_cb=progress_cb)

        # b. TAR + HASH
        tar_path = self.spool_dir / "layers" / (hf_file.path + ".tar")
        try:
            digest, layer_size = build_layer_to_file(
                source_path,
                self.layer_prefix,
                hf_file.path.split("/")[-1],
                tar_path,
            )

            # c. SKIP CHECK — if blob already present, return early
            target_repo = self._target_repo()
            existing = self.head_blob_fn(self.registry_client, target_repo, digest)
            if existing is not None:
                log.info("skip push: blob %s already in registry", digest[:23])
                return BlobDescriptor(
                    media_type=ML_TAR,
                    digest=digest,
                    size=int(existing["size"]),
                    hf_path=hf_file.path,
                )

            # d. PUSH
            streaming = self.streaming_factory(
                client=self.registry_client,
                repo=target_repo,
                max_retries=self.oci_max_retries,
                backoff_initial=self.backoff_initial,
                stop_event=self.stop_event,
            )
            streaming.push_from_file(tar_path, layer_size, digest)

            # e. VERIFY
            verified = self.head_blob_fn(self.registry_client, target_repo, digest)
            if verified is None:
                raise PushError(
                    f"blob {digest} not visible after PUT for {hf_file.path}",
                    hint="registry may not have persisted the upload; retry the run.",
                )

            return BlobDescriptor(
                media_type=ML_TAR, digest=digest, size=layer_size, hf_path=hf_file.path
            )
        finally:
            # f. CLEANUP — always remove tar; remove source if configured
            with contextlib.suppress(FileNotFoundError):
                tar_path.unlink()
            if self.clean_hf_after_push:
                with contextlib.suppress(FileNotFoundError):
                    source_path.unlink()

    def _target_repo(self) -> str:
        repo = self.registry_client.target_repo
        assert repo is not None, "OciClient must have target_repo set for FileWorker"
        return repo
