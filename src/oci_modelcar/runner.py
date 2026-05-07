"""Pipeline orchestration: process_one_file + main run loop."""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from oci_modelcar.config import Config
from oci_modelcar.hf import HfClient, HfFile, HfStream
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.manifest import build_config_bytes, build_manifest_bytes
from oci_modelcar.oci import (
    ML_CFG,
    ML_TAR,
    BlobDescriptor,
    ChunkedBlobUpload,
    OciClient,
    head_blob,
    push_manifest,
    push_small_blob,
    validate_manifest_tag,
)
from oci_modelcar.state import JobState, JsonStateStore
from oci_modelcar.tags import derive_tag
from oci_modelcar.tar_layer import stream_layer_to

log = logging.getLogger(__name__)


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


@dataclass
class RunResult:
    job_key: str
    manifest_digest: str
    image_ref: str
    layers: list[BlobDescriptor]
    skipped: int = 0
    pushed: int = 0
    failed: list[str] = field(default_factory=list)


def run_push(cfg: Config, plog: PipelineLogger) -> RunResult:
    hf_client = HfClient(endpoint=cfg.hf_endpoint, repo=cfg.hf_repo)
    oci_client = OciClient(registry_host=cfg.registry)

    plog.section("Resolving HuggingFace revision")
    revision_resolved = hf_client.resolve_revision(cfg.hf_revision)
    plog.info(f"HF repo     : {cfg.hf_repo}")
    plog.info(f"Revision in : {cfg.hf_revision}")
    plog.info(f"Revision    : {revision_resolved}")

    target_tag = derive_tag(revision_resolved, explicit=cfg.target_tag)
    image_ref = f"{cfg.registry}/{cfg.target_repo}:{target_tag}"
    plog.info(f"Target      : {image_ref}")

    job_key = JsonStateStore.compute_job_key(
        hf_repo=cfg.hf_repo,
        revision_resolved=revision_resolved,
        registry=cfg.registry,
        target_repo=cfg.target_repo,
        target_tag=target_tag,
    )
    state = JsonStateStore(cfg.state_file)
    if state.is_completed(job_key) and not cfg.force:
        existing = state.get_job(job_key)
        assert existing is not None
        plog.info(f"Job already completed: {existing['manifest_digest']}")
        return RunResult(
            job_key=job_key,
            manifest_digest=str(existing["manifest_digest"]),
            image_ref=image_ref,
            layers=[],
        )
    state.upsert_job(
        job_key,
        JobState(
            hf_repo=cfg.hf_repo,
            hf_revision_input=cfg.hf_revision,
            hf_revision_resolved=revision_resolved,
            registry=cfg.registry,
            target_repo=cfg.target_repo,
            target_tag=target_tag,
            also_tags=list(cfg.also_tags),
        ),
    )
    state.save()

    plog.section("Listing files")
    files = hf_client.list_files(revision_resolved, allow=cfg.allow_patterns)
    if not files:
        raise RuntimeError(f"no matching files in {cfg.hf_repo} (allow={cfg.allow_patterns})")
    total_bytes = sum(f.size for f in files)
    plog.info(f"{len(files)} files, {total_bytes / 1e9:.2f} GB total")

    if cfg.dry_run:
        for f in files:
            plog.info(f"  {f.path}  ({f.size / 1e6:.1f} MB)")
        return RunResult(job_key=job_key, manifest_digest="", image_ref=image_ref, layers=[])

    plog.section(f"Pushing {len(files)} layers ({total_bytes / 1e9:.2f} GB)")
    layers_by_idx: dict[int, BlobDescriptor] = {}
    diff_ids_by_idx: dict[int, str] = {}
    skipped = 0
    pushed = 0
    failed: list[str] = []

    def task_for_file(idx: int, hf_file: HfFile) -> tuple[int, BlobDescriptor, str]:
        cached = state.get_pushed(job_key, hf_file.path)
        if cached is not None and cached.get("size") == hf_file.size and cached.get("pushed_at"):
            return (
                idx,
                BlobDescriptor(
                    media_type=ML_TAR,
                    digest=str(cached["digest"]),
                    size=int(cached.get("layer_size", cached["size"])),
                ),
                str(cached["diff_id"]),
            )
        descriptor, diff_id = process_one_file(
            hf_client=hf_client,
            oci_client=oci_client,
            repo=cfg.target_repo,
            revision=revision_resolved,
            hf_file=hf_file,
            layer_prefix=cfg.layer_prefix,
            chunk_size=cfg.chunk_bytes,
            hf_max_retries=cfg.hf_max_retries,
            oci_max_retries=cfg.oci_max_retries,
        )
        state.mark_pushed(
            job_key,
            hf_file.path,
            digest=descriptor.digest,
            diff_id=diff_id,
            size=hf_file.size,
        )
        state.save()
        return idx, descriptor, diff_id

    if cfg.workers == 1:
        for idx, hf_file in enumerate(files):
            with plog.file_scope(
                f"[{idx + 1:>3}/{len(files)}] {hf_file.path} ({hf_file.size / 1e6:.1f} MB)"
            ) as scoped:
                try:
                    _, desc, diff = task_for_file(idx, hf_file)
                    layers_by_idx[idx] = desc
                    diff_ids_by_idx[idx] = diff
                    if state.has_pushed(job_key, hf_file.path, hf_file.size):
                        prev = state.get_pushed(job_key, hf_file.path)
                        if prev and prev.get("pushed_at"):
                            scoped.info(f"-> {desc.digest[:23]}…")
                            pushed += 1  # counts also re-uses
                except Exception as e:
                    scoped.error(f"failed: {e}")
                    failed.append(hf_file.path)
                    if cfg.fail_fast:
                        raise
    else:
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures: list[Future[tuple[int, BlobDescriptor, str]]] = [
                pool.submit(task_for_file, i, f) for i, f in enumerate(files)
            ]
            for fut in as_completed(futures):
                try:
                    idx, desc, diff = fut.result()
                    layers_by_idx[idx] = desc
                    diff_ids_by_idx[idx] = diff
                except Exception as e:
                    failed.append(str(e))
                    if cfg.fail_fast:
                        for other in futures:
                            other.cancel()
                        raise

    if failed and not cfg.fail_fast:
        raise SystemExit(3)
    if failed:
        raise SystemExit(2)

    layers = [layers_by_idx[i] for i in sorted(layers_by_idx)]
    diff_ids = [diff_ids_by_idx[i] for i in sorted(diff_ids_by_idx)]

    plog.section("Building and pushing manifest")
    cfg_bytes = build_config_bytes(diff_ids)
    cfg_digest = push_small_blob(oci_client, repo=cfg.target_repo, data=cfg_bytes)
    cfg_desc = BlobDescriptor(media_type=ML_CFG, digest=cfg_digest, size=len(cfg_bytes))
    manifest_bytes = build_manifest_bytes(layers, cfg_desc)
    manifest_digest = push_manifest(
        oci_client,
        repo=cfg.target_repo,
        tag=target_tag,
        manifest_bytes=manifest_bytes,
    )
    for alias in cfg.also_tags:
        push_manifest(oci_client, repo=cfg.target_repo, tag=alias, manifest_bytes=manifest_bytes)

    plog.section("Validating push")
    for layer in layers:
        head_blob(oci_client, cfg.target_repo, layer.digest)
    head_blob(oci_client, cfg.target_repo, cfg_digest)
    for t in [target_tag, *cfg.also_tags]:
        validate_manifest_tag(
            oci_client,
            repo=cfg.target_repo,
            tag=t,
            expected_digest=manifest_digest,
        )

    state.mark_completed(job_key, manifest_digest=manifest_digest)
    state.save()

    plog.output_variable("manifestDigest", manifest_digest)
    plog.output_variable("imageRef", image_ref)
    return RunResult(
        job_key=job_key,
        manifest_digest=manifest_digest,
        image_ref=image_ref,
        layers=layers,
        pushed=pushed,
        skipped=skipped,
        failed=failed,
    )


__all__ = ["RunResult", "process_one_file", "run_push"]
