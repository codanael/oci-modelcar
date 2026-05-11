"""Per-file pipeline orchestration: FileWorker + Pipeline."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oci_modelcar.config import Config
from oci_modelcar.download import HfDownloader, HfFile
from oci_modelcar.errors import ConfigError, DiskSpaceError, PartialFailureError, PushError
from oci_modelcar.layer import build_layer_to_file, tar_layer_size
from oci_modelcar.logging import PipelineLogger, ProgressEmitter, fmt_bytes
from oci_modelcar.manifest import (
    ANN_HF_PATH,
    ANN_HF_SHA256,
    ML_MAN,
    ML_TAR,
    BlobDescriptor,
    build_config_bytes,
    build_manifest_bytes,
    derive_tag,
)
from oci_modelcar.registry import (
    OciClient,
    StreamingBlobUpload,
    head_blob,
    push_manifest,
    push_small_blob,
    validate_manifest_tag,
)

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
        plog: PipelineLogger | None = None,
        progress_interval: float = 5.0,
        reuse_map: dict[tuple[str, str | None], BlobDescriptor] | None = None,
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
        self.plog = plog
        self.progress_interval = progress_interval
        self.reuse_map = reuse_map or {}

    def _say(self, msg: str) -> None:
        if self.plog is not None:
            self.plog.info(msg)

    def process(
        self,
        repo: str,
        revision: str,
        hf_file: HfFile,
        progress_cb: Callable[[int], None] | None = None,
    ) -> BlobDescriptor:
        if self.stop_event is not None and self.stop_event.is_set():
            raise InterruptedError(f"worker for {hf_file.path} aborted before start")

        # 0. REUSE PRE-CHECK — if the previous manifest at the target tag
        # already carries a layer for this (hf_path, hf_sha256) and that
        # blob is still in the registry, skip the whole download+tar+push.
        reuse_hit = self.reuse_map.get((hf_file.path, hf_file.lfs_sha256))
        if reuse_hit is not None:
            target_repo = self._target_repo()
            present = self.head_blob_fn(self.registry_client, target_repo, reuse_hit.digest)
            if present is not None:
                self._say(
                    f"{hf_file.path}: reusing cached layer {reuse_hit.digest[:19]} "
                    f"({fmt_bytes(reuse_hit.size)}) — HF skipped"
                )
                return reuse_hit

        # a. DOWNLOAD
        self._say(f"{hf_file.path}: downloading ({fmt_bytes(hf_file.size)})")
        if progress_cb is None and self.plog is not None and hf_file.size > 0:
            progress_cb = ProgressEmitter(
                emit=self.plog.info,
                path=hf_file.path,
                total=hf_file.size,
                interval=self.progress_interval,
            ).update
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
                self._say(f"{hf_file.path}: reusing existing blob {digest[:19]} (skip push)")
                return BlobDescriptor(
                    media_type=ML_TAR,
                    digest=digest,
                    size=int(existing["size"]),
                    hf_path=hf_file.path,
                    hf_sha256=hf_file.lfs_sha256,
                )

            # d. PUSH
            self._say(f"{hf_file.path}: pushing layer {digest[:19]} ({fmt_bytes(layer_size)})")
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

            self._say(f"{hf_file.path}: pushed {digest[:19]}")
            return BlobDescriptor(
                media_type=ML_TAR,
                digest=digest,
                size=layer_size,
                hf_path=hf_file.path,
                hf_sha256=hf_file.lfs_sha256,
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


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunResult:
    manifest_digest: str
    image_ref: str
    image_ref_digest: str
    layers: tuple[BlobDescriptor, ...]
    skipped_blobs: int = 0


# ---------------------------------------------------------------------------
# Tag helper (HEAD existing manifest)
# ---------------------------------------------------------------------------


def get_manifest_digest_at_tag(client: OciClient, repo: str, tag: str) -> str | None:
    """HEAD the manifest tag; return Docker-Content-Digest or None on 404."""
    url = client.url(repo, "manifests", tag)
    r = client.session.head(
        url,
        headers={**client.auth, "Accept": ML_MAN},
        timeout=30,
    )
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        r.raise_for_status()
    digest = r.headers.get("Docker-Content-Digest")
    return digest if digest else None


def fetch_manifest_at_tag(client: OciClient, repo: str, tag: str) -> dict[str, Any] | None:
    """GET the manifest at `tag`; return the parsed JSON or None on 404."""
    url = client.url(repo, "manifests", tag)
    r = client.session.get(
        url,
        headers={**client.auth, "Accept": ML_MAN},
        timeout=30,
    )
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        r.raise_for_status()
    body: dict[str, Any] = r.json()
    return body


def build_reuse_map(
    manifest: dict[str, Any],
) -> dict[tuple[str, str | None], BlobDescriptor]:
    """Index a manifest's layers by (hf-path, hf-sha256) for reuse on re-push.

    Layers without the hf-path annotation (older oci-modelcar runs, foreign
    images) are silently skipped — without the path we can't map them to an
    HF file in the current run.
    """
    out: dict[tuple[str, str | None], BlobDescriptor] = {}
    for layer in manifest.get("layers", []) or []:
        annotations = layer.get("annotations") or {}
        path = annotations.get(ANN_HF_PATH)
        if not path:
            continue
        sha = annotations.get(ANN_HF_SHA256)
        out[(path, sha)] = BlobDescriptor(
            media_type=layer.get("mediaType", ML_TAR),
            digest=layer["digest"],
            size=int(layer["size"]),
            hf_path=path,
            hf_sha256=sha,
        )
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        plog: PipelineLogger,
        downloader: HfDownloader | None = None,
        registry_client: OciClient | None = None,
    ) -> None:
        self.cfg = cfg
        self.plog = plog
        self._downloader = downloader
        self._registry_client = registry_client

    @property
    def downloader(self) -> HfDownloader:
        if self._downloader is None:
            raise RuntimeError("Pipeline requires a downloader")
        return self._downloader

    @property
    def registry_client(self) -> OciClient:
        if self._registry_client is None:
            raise RuntimeError("Pipeline requires a registry_client")
        return self._registry_client

    def _preflight(self) -> tuple[str, list[HfFile], str]:
        self.plog.section("Resolving HuggingFace revision")
        revision = self.downloader.resolve_revision(self.cfg.hf_repo, self.cfg.hf_revision)
        self.plog.info(f"HF repo  : {self.cfg.hf_repo}")
        self.plog.info(f"Revision : {revision}")

        files = self.downloader.list_files(self.cfg.hf_repo, revision, self.cfg.allow_patterns)
        if not files:
            raise ConfigError(
                f"no files matched allow_patterns {self.cfg.allow_patterns} "
                f"in {self.cfg.hf_repo}@{revision}"
            )
        self.plog.info(f"{len(files)} files matched")

        target_tag = derive_tag(revision, explicit=self.cfg.target_tag)
        return revision, files, target_tag

    def _check_disk_space(self, files: list[HfFile]) -> None:
        if not files:
            return
        max_layer = max(tar_layer_size(f.size) for f in files)
        max_source = max(f.size for f in files)
        total_sources = sum(f.size for f in files)

        in_flight = (max_source + max_layer) * self.cfg.workers * 1.2
        persistent = 0 if self.cfg.clean_hf_after_push else int(total_sources * 1.05)
        needed = int(in_flight + persistent)

        self.cfg.spool_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.cfg.spool_dir).free
        if free < needed:
            raise DiskSpaceError(
                f"Need {needed / 1e9:.1f} GB free in {self.cfg.spool_dir}, "
                f"only {free / 1e9:.1f} GB available.",
                hint=(
                    f"--spool-dir <other-volume>, --clean-hf-after-push, "
                    f"or lower --workers (currently {self.cfg.workers})."
                ),
            )

    def run(self) -> RunResult:
        revision, files, target_tag = self._preflight()
        existing_tag_digest = get_manifest_digest_at_tag(
            self.registry_client, self.cfg.target_repo, target_tag
        )
        reuse_map: dict[tuple[str, str | None], BlobDescriptor] = {}
        if existing_tag_digest is not None and not self.cfg.force:
            existing_manifest = fetch_manifest_at_tag(
                self.registry_client, self.cfg.target_repo, target_tag
            )
            if existing_manifest is not None:
                reuse_map = build_reuse_map(existing_manifest)
                if reuse_map:
                    self.plog.info(
                        f"reuse: {len(reuse_map)} layer(s) annotated in existing manifest "
                        f"at {target_tag!r}"
                    )
        self._check_disk_space(files)

        if self.cfg.dry_run:
            self.plog.info("dry-run: skipping push")
            return RunResult(
                manifest_digest="",
                image_ref="",
                image_ref_digest="",
                layers=(),
                skipped_blobs=0,
            )

        (self.cfg.spool_dir / "sources").mkdir(parents=True, exist_ok=True)
        (self.cfg.spool_dir / "layers").mkdir(parents=True, exist_ok=True)

        stop_event = threading.Event()

        def make_worker() -> FileWorker:
            return FileWorker(
                downloader=self.downloader,
                registry_client=self.registry_client,
                head_blob_fn=head_blob,
                streaming_factory=StreamingBlobUpload,
                layer_prefix=self.cfg.layer_prefix,
                spool_dir=self.cfg.spool_dir,
                clean_hf_after_push=self.cfg.clean_hf_after_push,
                oci_max_retries=self.cfg.oci_max_retries,
                stop_event=stop_event,
                plog=self.plog,
                reuse_map=reuse_map,
            )

        descriptors: list[BlobDescriptor] = []
        failures: list[tuple[str, BaseException]] = []

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            futures: dict[Future[BlobDescriptor], HfFile] = {}
            for hf_file in files:
                worker = make_worker()
                fut = pool.submit(worker.process, self.cfg.hf_repo, revision, hf_file)
                futures[fut] = hf_file

            for fut in list(futures):
                hf_file = futures[fut]
                try:
                    desc = fut.result()
                    descriptors.append(desc)
                except BaseException as e:
                    failures.append((hf_file.path, e))
                    if self.cfg.fail_fast:
                        stop_event.set()
                        for other in futures:
                            if not other.done():
                                other.cancel()
                        raise

        if failures:
            self.plog.error(f"{len(failures)} file(s) failed:")
            for path, exc in failures:
                self.plog.error(f"  {path}: {type(exc).__name__}: {exc}")
            raise PartialFailureError(
                f"{len(failures)}/{len(files)} files failed",
                hint="re-run; succeeded blobs are cached in registry",
            )

        result = self._assemble_manifest(target_tag, descriptors)
        if existing_tag_digest is not None and existing_tag_digest != result.manifest_digest:
            if not self.cfg.force:
                raise PushError(
                    f"tag exists with different digest for {target_tag!r}: "
                    f"registry has {existing_tag_digest}, computed {result.manifest_digest}",
                    hint="use --force to overwrite, or pick a different --target-tag.",
                )
            self.plog.warning(
                f"tag {target_tag!r} existed at {existing_tag_digest} "
                f"but --force overwrote with {result.manifest_digest}"
            )
        return result

    def _assemble_manifest(self, target_tag: str, descriptors: list[BlobDescriptor]) -> RunResult:
        """Assemble and push OCI config + manifest."""
        descriptors.sort(key=lambda d: d.hf_path)
        diff_ids = [d.digest for d in descriptors]
        config_bytes = build_config_bytes(diff_ids)
        config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
        manifest_bytes = build_manifest_bytes(config_digest, len(config_bytes), descriptors)
        new_manifest_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()

        push_small_blob(self.registry_client, self.cfg.target_repo, config_bytes)
        push_manifest(self.registry_client, self.cfg.target_repo, target_tag, manifest_bytes)
        validate_manifest_tag(
            self.registry_client, self.cfg.target_repo, target_tag, new_manifest_digest
        )

        for tag in self.cfg.also_tags:
            push_manifest(self.registry_client, self.cfg.target_repo, tag, manifest_bytes)
            validate_manifest_tag(
                self.registry_client, self.cfg.target_repo, tag, new_manifest_digest
            )

        image_ref = f"{self.registry_client.host}/{self.cfg.target_repo}:{target_tag}"
        image_ref_digest = (
            f"{self.registry_client.host}/{self.cfg.target_repo}@{new_manifest_digest}"
        )
        self.plog.info(f"manifest: {new_manifest_digest}")
        self.plog.info(f"image:    {image_ref}")
        self.plog.output_variable("manifestDigest", new_manifest_digest)
        self.plog.output_variable("imageRef", image_ref)
        self.plog.output_variable("imageRefDigest", image_ref_digest)

        return RunResult(
            manifest_digest=new_manifest_digest,
            image_ref=image_ref,
            image_ref_digest=image_ref_digest,
            layers=tuple(descriptors),
        )
