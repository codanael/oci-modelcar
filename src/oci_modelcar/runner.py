"""Pipeline orchestration: process_one_file + main run loop."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from oci_modelcar.config import Config
from oci_modelcar.hf import HfClient, HfFile, HfStream
from oci_modelcar.logging import PipelineLogger, ProgressEmitter, _fmt_bytes
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


@dataclass(frozen=True, slots=True)
class FileTelemetry:
    """Per-file pipeline measurements emitted at INFO after each transfer.

    `producer_wait_s` is the time the HF→OCI bridge spent blocked on
    queue.put() because the queue was full — i.e. the OCI consumer couldn't
    keep up. `consumer_wait_s` is the symmetric: queue empty, HF couldn't
    feed fast enough. Comparing the two against `elapsed_s` indicates which
    side is the bottleneck (or whether the pipeline is balanced).
    """

    bytes_through: int
    producer_wait_s: float
    consumer_wait_s: float
    elapsed_s: float

    @property
    def throughput_mb_s(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.bytes_through / 1e6 / self.elapsed_s

    def format_line(self, path: str) -> str:
        size_str = (
            f"{self.bytes_through / 1e9:.2f} GB"
            if self.bytes_through >= 1_000_000_000
            else f"{self.bytes_through / 1e6:.0f} MB"
        )
        if self.elapsed_s < 0.5:
            # Too short to compute meaningful percentages
            return f"{path}: {size_str} in {self.elapsed_s:.2f}s"
        cons_pct = 100 * self.consumer_wait_s / self.elapsed_s
        prod_pct = 100 * self.producer_wait_s / self.elapsed_s
        return (
            f"{path}: {size_str} in {self.elapsed_s:.1f}s "
            f"({self.throughput_mb_s:.0f} MB/s); "
            f"HF wait {self.consumer_wait_s:.1f}s ({cons_pct:.0f}%), "
            f"OCI wait {self.producer_wait_s:.1f}s ({prod_pct:.0f}%)"
        )


class _PipeBuffer:
    """Bounded thread-bridge between an HF-side producer (writable sink) and
    an OCI-side consumer (chunk puller) with telemetry on both blocking sides.

    Producer thread calls .write(bytes) (the sink protocol used by
    stream_layer_to via tarfile). Small writes are accumulated up to
    `coalesce_size` before being put on the queue, which keeps queue traffic
    low even when tarfile writes 10 KiB at a time. The producer signals
    end-of-stream with .close(), and out-of-band errors with
    .report_exception(exc).

    Consumer (main thread) calls .get_chunk() in a loop until None. An
    exception sentinel from the producer is re-raised at consumer side so
    the failure surfaces in the original calling context.

    On consumer-side abort, .drain_and_abort() pops all queued items and
    flips an internal flag so the producer's next .write() raises
    InterruptedError, freeing it from any blocked put().

    Telemetry: producer_wait_s tallies time the producer spent blocked on
    put (queue full ⇒ consumer / OCI is slow); consumer_wait_s tallies time
    the consumer spent blocked on get (queue empty ⇒ producer / HF is slow).
    bytes_through accumulates bytes successfully consumed.
    """

    _EOF = object()  # sentinel; identity-checked

    def __init__(
        self,
        max_chunks: int = 8,
        coalesce_size: int = 1024 * 1024,
        stop_event: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._q: queue.Queue[Any] = queue.Queue(maxsize=max_chunks)
        self._coalesce = coalesce_size
        self._buf = bytearray()
        self._stop = stop_event
        self._clock = clock
        self._aborted = False
        self.producer_wait_s = 0.0
        self.consumer_wait_s = 0.0
        self.bytes_through = 0

    # --- producer (writable sink) ---

    def writable(self) -> bool:
        return True

    def write(self, data: bytes) -> int:
        if self._aborted or (self._stop is not None and self._stop.is_set()):
            raise InterruptedError("pipe write aborted")
        self._buf.extend(data)
        while len(self._buf) >= self._coalesce:
            chunk = bytes(self._buf[: self._coalesce])
            del self._buf[: self._coalesce]
            self._timed_put(chunk)
        return len(data)

    def flush(self) -> None:
        # tarfile may call this; coalescing happens on size threshold + close,
        # not on flush.
        pass

    def close(self) -> None:
        if self._buf:
            chunk = bytes(self._buf)
            self._buf.clear()
            self._timed_put(chunk)
        self._q.put(self._EOF)

    def report_exception(self, exc: BaseException) -> None:
        self._q.put(("exc", exc))

    def _timed_put(self, chunk: bytes) -> None:
        t0 = self._clock()
        self._q.put(chunk)
        self.producer_wait_s += self._clock() - t0

    # --- consumer ---

    def get_chunk(self) -> bytes | None:
        t0 = self._clock()
        item = self._q.get()
        self.consumer_wait_s += self._clock() - t0
        if item is self._EOF:
            return None
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "exc":
            raise item[1]
        assert isinstance(item, bytes), f"unexpected pipe item type: {type(item)!r}"
        self.bytes_through += len(item)
        return item

    def drain_and_abort(self) -> None:
        """Pop everything without raising, and stop accepting writes.

        Used by the consumer when it has decided to abort (its own exception
        or external stop_event), to free a producer that may be blocked in
        a queue.put() because the queue is full.
        """
        self._aborted = True
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return


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
    progress_cb: Callable[[int], None] | None = None,
    stop_event: threading.Event | None = None,
    pipe_max_chunks: int = 8,
    pipe_coalesce_size: int = 1024 * 1024,
) -> tuple[BlobDescriptor, str, FileTelemetry]:
    """Stream one HF file as one tar layer; returns (descriptor, diff_id, telemetry).

    HF download (producer) and OCI push (consumer) run on two threads bridged
    by a bounded `_PipeBuffer`, decoupling the two stages so they no longer
    backpressure each other one-for-one. Telemetry on producer/consumer wait
    times pinpoints which side is the bottleneck.

    For uncompressed tar layers, diff_id == descriptor.digest.
    """
    hf_stream = HfStream(
        client=hf_client,
        revision=revision,
        path=hf_file.path,
        size=hf_file.size,
        max_retries=hf_max_retries,
        backoff_initial=backoff_initial,
        progress_cb=progress_cb,
        stop_event=stop_event,
    )
    pipe = _PipeBuffer(
        max_chunks=pipe_max_chunks,
        coalesce_size=pipe_coalesce_size,
        stop_event=stop_event,
    )
    upload = ChunkedBlobUpload(
        client=oci_client,
        repo=repo,
        chunk_size=chunk_size,
        max_retries=oci_max_retries,
        backoff_initial=backoff_initial,
        stop_event=stop_event,
    )

    def _produce() -> None:
        try:
            try:
                stream_layer_to(
                    sink=pipe,  # type: ignore[arg-type]
                    prefix=layer_prefix,
                    filename=os.path.basename(hf_file.path),
                    size=hf_file.size,
                    source=hf_stream,  # type: ignore[arg-type]
                )
            finally:
                hf_stream.close()
        except BaseException as exc:
            pipe.report_exception(exc)
            return
        pipe.close()

    producer = threading.Thread(
        target=_produce,
        name=f"hfproducer:{hf_file.path}",
        daemon=True,
    )
    started = time.monotonic()
    producer.start()

    try:
        try:
            while True:
                chunk = pipe.get_chunk()
                if chunk is None:
                    break
                upload.write(chunk)
            digest, layer_size = upload.close()
        except BaseException:
            # Unblock any stuck producer put() so producer can exit promptly.
            pipe.drain_and_abort()
            raise
    finally:
        producer.join(timeout=30)

    elapsed = time.monotonic() - started
    telemetry = FileTelemetry(
        bytes_through=pipe.bytes_through,
        producer_wait_s=pipe.producer_wait_s,
        consumer_wait_s=pipe.consumer_wait_s,
        elapsed_s=elapsed,
    )
    descriptor = BlobDescriptor(media_type=ML_TAR, digest=digest, size=layer_size)
    return descriptor, digest, telemetry


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
    oci_client = OciClient(registry_host=cfg.registry, target_repo=cfg.target_repo)

    plog.section("Resolving HuggingFace revision")
    revision_resolved = hf_client.resolve_revision(cfg.hf_revision)
    plog.info(f"HF repo     : {cfg.hf_repo}")
    plog.info(f"Revision in : {cfg.hf_revision}")
    plog.info(f"Revision    : {revision_resolved}")
    if cfg.chunk_mib >= 1024:
        # Per-worker RAM is roughly 2x chunk_size during a flush; called out
        # so users intentionally raising chunk_mib (typically to bypass
        # cluster routing issues — see CHANGELOG) understand the tradeoff.
        plog.info(
            f"OCI upload chunk size: {cfg.chunk_mib} MiB "
            f"(~{2 * cfg.chunk_mib} MiB peak RAM per worker)"
        )

    target_tag = derive_tag(revision_resolved, explicit=cfg.target_tag)
    image_ref = f"{oci_client.host}/{cfg.target_repo}:{target_tag}"
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
        manifest_digest = str(existing["manifest_digest"])
        image_ref_digest = (
            existing.get("image_ref_digest")
            or f"{oci_client.host}/{cfg.target_repo}@{manifest_digest}"
        )
        plog.info(f"Job already completed: {manifest_digest}")
        plog.output_variable("manifestDigest", manifest_digest)
        plog.output_variable("imageRef", image_ref)
        plog.output_variable("imageRefDigest", image_ref_digest)
        return RunResult(
            job_key=job_key,
            manifest_digest=manifest_digest,
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
    for idx, hf_file in enumerate(files):
        plog.info(f"[{idx + 1:>3}/{len(files)}] {hf_file.path} ({_fmt_bytes(hf_file.size)})")

    layers_by_idx: dict[int, BlobDescriptor] = {}
    diff_ids_by_idx: dict[int, str] = {}
    skipped = 0
    pushed = 0
    failed: list[str] = []
    stop_event = threading.Event()

    def task_for_file(
        idx: int, hf_file: HfFile
    ) -> tuple[int, BlobDescriptor, str, bool, FileTelemetry | None]:
        cached = state.get_pushed(job_key, hf_file.path)
        cached_layer_size = cached.get("layer_size") if cached else None
        if (
            cached is not None
            and cached.get("size") == hf_file.size
            and cached.get("pushed_at")
            and cached_layer_size is not None
        ):
            return (
                idx,
                BlobDescriptor(
                    media_type=ML_TAR,
                    digest=str(cached["digest"]),
                    size=int(cached_layer_size),
                ),
                str(cached["diff_id"]),
                True,  # was_cached
                None,  # no transfer happened ⇒ no telemetry
            )
        emitter = ProgressEmitter(
            emit=plog.info,
            path=hf_file.path,
            total=hf_file.size,
            interval=5.0,
        )
        descriptor, diff_id, telemetry = process_one_file(
            hf_client=hf_client,
            oci_client=oci_client,
            repo=cfg.target_repo,
            revision=revision_resolved,
            hf_file=hf_file,
            layer_prefix=cfg.layer_prefix,
            chunk_size=cfg.chunk_bytes,
            hf_max_retries=cfg.hf_max_retries,
            oci_max_retries=cfg.oci_max_retries,
            progress_cb=emitter.update,
            stop_event=stop_event,
        )
        state.mark_pushed(
            job_key,
            hf_file.path,
            digest=descriptor.digest,
            diff_id=diff_id,
            size=hf_file.size,
            layer_size=descriptor.size,
        )
        state.save()
        return idx, descriptor, diff_id, False, telemetry

    def record_result(
        idx: int,
        path: str,
        desc: BlobDescriptor,
        diff: str,
        was_cached: bool,
        telemetry: FileTelemetry | None,
    ) -> None:
        layers_by_idx[idx] = desc
        diff_ids_by_idx[idx] = diff
        suffix = " (cached)" if was_cached else ""
        plog.info(f"{path}: -> {desc.digest[:23]}…{suffix}")
        if telemetry is not None:
            plog.info(telemetry.format_line(path))

    if cfg.workers == 1:
        for idx, hf_file in enumerate(files):
            try:
                _, desc, diff, was_cached, telemetry = task_for_file(idx, hf_file)
                record_result(idx, hf_file.path, desc, diff, was_cached, telemetry)
                if was_cached:
                    skipped += 1
                else:
                    pushed += 1
            except Exception as e:
                plog.error(f"{hf_file.path}: failed: {e}")
                failed.append(hf_file.path)
                if cfg.fail_fast:
                    raise
    else:
        pool = ThreadPoolExecutor(max_workers=cfg.workers)
        future_to_path: dict[
            Future[tuple[int, BlobDescriptor, str, bool, FileTelemetry | None]], str
        ] = {pool.submit(task_for_file, i, f): f.path for i, f in enumerate(files)}
        try:
            for fut in as_completed(future_to_path):
                path = future_to_path[fut]
                try:
                    idx, desc, diff, was_cached, telemetry = fut.result()
                    record_result(idx, path, desc, diff, was_cached, telemetry)
                    if was_cached:
                        skipped += 1
                    else:
                        pushed += 1
                except Exception as e:
                    plog.error(f"{path}: failed: {e}")
                    failed.append(path)
                    if cfg.fail_fast:
                        for other in future_to_path:
                            other.cancel()
                        raise
        except KeyboardInterrupt:
            plog.warning("Interrupted; cancelling pending uploads")
            stop_event.set()
            for other in future_to_path:
                other.cancel()
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

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

    image_ref_digest = f"{oci_client.host}/{cfg.target_repo}@{manifest_digest}"

    state.mark_completed(
        job_key,
        manifest_digest=manifest_digest,
        image_ref_digest=image_ref_digest,
    )
    state.save()

    plog.output_variable("manifestDigest", manifest_digest)
    plog.output_variable("imageRef", image_ref)
    plog.output_variable("imageRefDigest", image_ref_digest)
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
