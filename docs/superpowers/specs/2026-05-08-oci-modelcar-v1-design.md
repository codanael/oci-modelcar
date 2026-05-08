# oci-modelcar v1.0 — Clean rewrite design

**Status**: design.
**Author**: Anael Latassa with Claude.
**Date**: 2026-05-08.
**Supersedes**: `2026-05-07-oci-modelcar-design.md` (v0.x design).

## 1. Why a rewrite

The v0.x line solved the basic shape — stream a HuggingFace model into an
OCI registry as a multi-layer image — but accumulated complexity in service
of a streaming-first pipeline that turned out to be fragile in real
deployment scenarios. Two production observations on Artifactory HA cluster
behind a load balancer drove this rewrite:

1. **Chunked PATCH uploads break on stateless load balancers.** Each PATCH
   of a multi-chunk upload session can be routed to a different cluster
   node. The binary stream isn't replicated server-side, so Node B can't
   continue what Node A started, producing partial-state errors
   (`failed to stream binary to sub provider`,
   `Binary info is only available after successful read of the entire stream`),
   non-spec PATCH responses (200, 204), and intermittent SSL EOFs.

2. **Streaming HF→OCI without a disk buffer means no upload retry on cuts.**
   When the proxy or registry resets a long-running upload (multi-GB layer),
   we can't replay because the source bytes have already been consumed.
   File-level retry across runs (via `state.json`) is too coarse — a 30 GB
   layer cut at 25 GB redoes 25 GB.

The fix is structural, not patch-on-patch. Mature OCI clients
(containers/image, Jib) all share one pattern: **spool the layer to local
disk first, then push it in a single streaming PATCH from the file with
replay-on-cut**. Once you have a replayable source, mid-blob retry becomes
trivial; once you have a single PATCH per blob, cluster routing
unpredictability evaporates.

This rewrite removes the streaming HF-to-OCI pipeline and replaces it with
a per-file pipeline that downloads to disk, builds the tar layer, and
pushes from the file with full-PATCH retry. The change is breaking; v1.0.0
is the right vehicle.

## 2. Goals & non-goals

### Goals

- **Reliability on misconfigured infrastructure**: works against an
  Artifactory HA cluster regardless of LB session affinity (single PATCH
  per blob, no per-PATCH routing decisions).
- **Mid-blob retry on transient cuts**: a multi-GB layer cut mid-upload
  retries cleanly from the local file (Jib-style full-PATCH replay).
- **Mid-stream cancellation**: SIGINT, fail-fast on first error, or stop_event
  propagate to running workers within seconds, not minutes — including
  during multi-GB downloads (current code can be blocked for hours).
- **Cross-origin auth safety**: the HuggingFace token is not sent to S3 /
  CloudFront on redirect.
- **Simpler code base**: remove the `state.json` indirection (registry is
  source of truth), the dual chunked/streaming upload modes, the
  intra-process pipe buffer between download and upload threads. Net
  reduction estimated at 500–700 lines.

### Non-goals

- **No new product features.** Same surface as v0.5: push, status, validate.
- **No cosign integration in this rewrite.** Tracked in
  `2026-05-07-cosign-integration-design.md` as a separate v1.x feature.
- **No support for non-HuggingFace sources.** The single-source design is
  intentional — adding mirroring from S3 / Ollama / etc. would inflate
  scope without serving a user we have today.
- **No GPU / tokenizer awareness.** The tool transports bytes; it does not
  inspect model contents.
- **No backward CLI compatibility.** v1.0.0 is breaking and signals that.
- **No `huggingface_hub` as a hard dependency for downloads.** We use
  `HfApi` for metadata only; bytes are streamed by our own code so we
  retain mid-stream cancellation that the library does not expose.

## 3. Architecture

### 3.1 Job-level flow

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. PRE-FLIGHT (sequential)                                       │
│    - Resolve HF revision → 40-char SHA                           │
│    - List files matching --allow-patterns                        │
│    - Compute target tag (sha[:12] or --target-tag)               │
│    - HEAD manifest tag in registry                               │
│      └─ if exists:                                               │
│         - matching digest + no --force → log + exit 0            │
│         - different digest + no --force → PushError, exit 6      │
│         - --force → continue                                     │
│    - Disk space check (mode-aware)                               │
│      └─ insufficient → DiskSpaceError, exit 4                    │
│                                                                  │
│ 2. PER-FILE PIPELINE (parallel, ThreadPoolExecutor, N workers)  │
│    For each HfFile: see §3.2                                     │
│    - fail-fast (default): first exception → stop_event → cancel  │
│    - continue-on-error: collect failures, abort manifest if any  │
│                                                                  │
│ 3. MANIFEST (sequential, all blobs guaranteed present)           │
│    - Build OCI image config from collected diff_ids              │
│    - push_small_blob(config) → config_digest                     │
│    - Build manifest from layer descriptors                       │
│      (ordered alphabetically by HF path for determinism)         │
│    - push_manifest(target_tag)                                   │
│    - validate_manifest_tag (HEAD with Docker-Content-Digest)     │
│    - For each --also-tag: re-push manifest under that tag        │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Per-file flow (one worker, one file)

```
┌──────────────────────────────────────────────────────────────────┐
│  a. DOWNLOAD                                                     │
│     HfDownloader.download(repo, revision, hf_file)               │
│     - GET <endpoint>/<repo>/resolve/<revision>/<path>            │
│     - Stream to <spool>/sources/<hf_path>.partial                │
│     - Range resume on transient cuts                             │
│     - stop_event polled per chunk (~1 MiB → ~ms latency)         │
│     - Authorization header dropped on cross-origin redirect      │
│     - Atomic rename .partial → <spool>/sources/<hf_path>         │
│     - sha256 verified against hf_file.lfs_sha256 if present      │
│     → source_path                                                │
│                                                                  │
│  b. TAR + HASH                                                   │
│     LayerBuilder.build_to_file(source_path, prefix, dest)        │
│     - Build deterministic tar at <spool>/layers/<hf_path>.tar    │
│       (mtime=0, uid=gid=0, uname=gname="")                       │
│     - sha256 incremental during write                            │
│     → (tar_path, digest, layer_size)                             │
│                                                                  │
│  c. SKIP CHECK                                                   │
│     head_blob(registry, digest)                                  │
│       └─ 200 with Docker-Content-Digest match → cleanup, return  │
│       └─ 404 → proceed to push                                   │
│                                                                  │
│  d. PUSH (single PATCH per blob, full replay on cut)             │
│     StreamingBlobUpload(client, repo).push_from_file(            │
│         tar_path, layer_size, digest)                            │
│     - POST init → location                                       │
│     - for attempt in 1..oci_max_retries:                         │
│         open(tar_path, 'rb') as f                                │
│         PATCH location data=f                                    │
│           Content-Length: layer_size                             │
│           Content-Range: 0-(layer_size-1)                        │
│         Accept {200, 201, 202, 204}                              │
│         break on success                                         │
│         on cut: backoff with full jitter, retry from POST        │
│     - PUT location?digest=<digest>                               │
│     - Exhausted: PushError, exit 6                               │
│                                                                  │
│  e. VERIFY                                                       │
│     head_blob(registry, digest) — confirm Docker-Content-Digest  │
│     2 retries with linear backoff for race conditions            │
│                                                                  │
│  f. CLEANUP                                                      │
│     unlink(tar_path)                                             │
│     if cfg.clean_hf_after_push: unlink(source_path)              │
│     return BlobDescriptor(digest, layer_size, hf_file.path)      │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 Concurrency model

`ThreadPoolExecutor(max_workers=cfg.workers)`. One file = one Future. Each
worker processes phases a→f sequentially for its assigned file before
picking up the next. **Files do not overlap phases across workers** (no
"download file 1 + push file 2" interleaving). This trades some pipeline
parallelism for predictable disk usage and simpler reasoning.

`stop_event` is a `threading.Event` shared across the pool. On set, all
workers detect it at the next checkpoint:

| Phase | Cancellation latency |
|---|---|
| Download | per chunk (~1 MiB) → ~ms |
| Tar+hash | per read/write block → ms |
| Skip-check / verify HEAD | between requests |
| Push PATCH attempt | between attempts; mid-attempt is a network read so it surfaces as ConnectionError when the worker closes its session |
| Inter-phase | instant |

No blocking call exceeds a few seconds after `stop_event.set()`.

## 4. Modules

```
src/oci_modelcar/
  __init__.py        version metadata
  __main__.py        `python -m oci_modelcar` entry
  cli.py             argparse + sub-command dispatch
  config.py          Config dataclass, env+CLI parsing, validation
  http.py            shared requests.Session + auth resolution + redirect hook
  errors.py          custom exceptions hierarchy
  download.py        HfApi wrapper for metadata + bytes streamer
  layer.py           tar build + tar_layer_size + sha256 helpers
  manifest.py        OCI image config + manifest building + derive_tag
  registry.py        OciClient + StreamingBlobUpload + head_blob + push_manifest
  pipeline.py        FileWorker + Pipeline orchestrator
  logging.py         PipelineLogger, text/azure formatters
```

12 files. Dependency graph:

```
cli ──► config
cli ──► pipeline
pipeline ──► download ──► http
pipeline ──► registry ──► http
pipeline ──► layer
pipeline ──► manifest
pipeline ──► logging
download ──► errors
registry ──► errors
http ──► errors
```

Acyclic. Wiring lives only in `cli.py` and `pipeline.py`.

### 4.1 `download.py`

```python
class HfDownloader:
    def __init__(
        self,
        api: HfApi,                       # huggingface_hub.HfApi
        session: requests.Session,        # our build_session()
        spool_dir: Path,
        stop_event: threading.Event,
        max_retries: int = 10,
        backoff_initial: float = 1.0,
        backoff_cap: float = 60.0,
    ): ...

    # Metadata via HfApi (handles auth, follows API drift)
    def resolve_revision(self, repo: str, revision: str) -> str: ...
    def list_files(
        self, repo: str, revision: str, allow: tuple[str, ...]
    ) -> list[HfFile]: ...
    # HfFile carries: path, size, lfs_sha256: str | None

    # Bytes via our streamer (preserves stop_event, retry policy)
    def download(
        self,
        repo: str,
        revision: str,
        hf_file: HfFile,
        progress_cb: Callable[[int], None] | None = None,
    ) -> Path:
        """Download to <spool>/sources/<hf_path>.partial, atomic rename to .../<hf_path>.

        Behavior:
        - per-chunk stop_event polling (~1 MiB granularity)
        - Range resume on transient cuts (ConnectionError, Timeout,
          IncompleteRead, transient SSLEOFError, ProtocolError)
        - Range-200 fallback: server ignored Range → truncate partial,
          restart from offset 0 (matches huggingface_hub behavior)
        - Authorization header stripped on cross-origin redirect
          (urlparse netloc change → security hardening)
        - sha256 verified against hf_file.lfs_sha256 if present
        - retry budget refresh on progress (existing v0.x logic)

        Raises:
        - GatedRepoError on HTTP 403 with X-Error-Code: GatedRepo
        - RevisionNotFoundError, EntryNotFoundError on 404
        - DownloadError on retries exhausted
        - InterruptedError on stop_event
        """
```

### 4.2 `registry.py`

```python
class OciClient:
    """Same shape as v0.x: url(), auth, session, target_repo. No changes."""

class StreamingBlobUpload:
    def __init__(
        self,
        client: OciClient,
        repo: str,
        max_retries: int = 5,
        backoff_initial: float = 1.0,
        backoff_cap: float = 60.0,
        stop_event: threading.Event | None = None,
    ): ...

    def push_from_file(
        self, tar_path: Path, total_size: int, digest: str
    ) -> tuple[str, int]:
        """POST init → PATCH from file (full replay on cut) → PUT close.

        - Each retry attempt re-opens tar_path from offset 0
        - Backoff: full jitter Uniform(0, min(cap, base * 2^attempt))
        - Accepts {200, 201, 202, 204} on PATCH (Artifactory + Harbor quirks)
        - PATCH Content-Length set explicitly to avoid chunked TE
        - POST init has its own short retry budget (3 attempts, transient only)

        Returns (digest, total_size) on success.
        Raises PushError on retries exhausted.
        """

# Free functions (unchanged from v0.x signatures)
def head_blob(client: OciClient, repo: str, digest: str) -> dict | None: ...
def push_small_blob(client: OciClient, repo: str, data: bytes) -> str: ...
def push_manifest(client: OciClient, repo: str, tag: str, body: bytes) -> str: ...
def validate_manifest_tag(
    client: OciClient, repo: str, tag: str, expected_digest: str
) -> None: ...
```

Removed from v0.x: `ChunkedBlobUpload`, `_resync`, `_patch_with_retry`,
`_IteratorReader`, `BufferedTarLayer` (never shipped).

### 4.3 `pipeline.py`

```python
class FileWorker:
    """Process one file: phases a→f. One instance per worker thread."""
    def __init__(
        self,
        downloader: HfDownloader,
        registry_client: OciClient,
        layer_prefix: str,
        spool_dir: Path,
        clean_hf: bool,
        oci_max_retries: int,
        stop_event: threading.Event,
        progress_cb_factory: Callable[[str, int], Callable[[int], None]],
    ): ...

    def process(
        self, repo: str, revision: str, hf_file: HfFile
    ) -> BlobDescriptor: ...


class Pipeline:
    """Job-level orchestration. Owns the ThreadPoolExecutor."""
    def __init__(self, cfg: Config, plog: PipelineLogger): ...

    def run(self) -> RunResult:
        """Phases 1, 2, 3 from §3.1.

        Returns RunResult(manifest_digest, image_ref, image_ref_digest)
        on success. Raises on any unrecoverable error per §7.
        """
```

### 4.4 `errors.py`

```python
class OciModelcarError(Exception):
    """Base. Carries an optional `hint` for actionable guidance."""
    exit_code: int = 1

class ConfigError(OciModelcarError): exit_code = 2

class DownloadError(OciModelcarError): exit_code = 5
class GatedRepoError(DownloadError): exit_code = 3
class RevisionNotFoundError(DownloadError): exit_code = 5
class EntryNotFoundError(DownloadError): exit_code = 5

class DiskSpaceError(OciModelcarError): exit_code = 4

class PushError(OciModelcarError): exit_code = 6

class PartialFailure(OciModelcarError): exit_code = 7
# Raised in continue-on-error mode when some files succeeded and some failed.
```

### 4.5 `http.py`

Carries forward most of v0.x with three additions:

- **Cross-origin redirect hook**: registers a `Session.hooks['response']`
  callback that walks the redirect chain and pops `Authorization` when
  `urlparse(new_url).netloc != urlparse(prev_url).netloc`.
- **Token resolution**: `huggingface_token()` checks in order
  `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `~/.cache/huggingface/token`.
  Honors `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` (returns `None` even if a
  token is present in any of the sources).
- **Split connect/read timeouts** for HF GETs: `(10, 600)` instead of a
  single `600`. Surfaces dead proxies in 10s.

Existing kept: `_SmartRetry`, `is_transient_ssl`, `oci_auth_header`,
`docker_config_auth`, diagnostic env vars (`OCI_MODELCAR_USER_AGENT`,
`OCI_MODELCAR_FORCE_CONNECTION_CLOSE`, `OCI_MODELCAR_DEBUG_HTTP`).

### 4.6 Other modules

- `layer.py`: keep `tar_layer_size`, `make_tar_info`. Add
  `build_layer_to_file(source_path, prefix, filename, dest_path)` that
  writes the tar to `dest_path` and returns
  `(sha256_digest, total_bytes_written)`.
- `manifest.py`: unchanged shape (deterministic, no `created` field).
  `derive_tag` migrates here from `tags.py`.
- `config.py`: see §6 for surface.
- `cli.py`: see §6 for sub-commands.
- `logging.py`: unchanged.
- `__init__.py`, `__main__.py`: unchanged.

## 5. Data flow & contracts

### 5.1 HfFile

```python
@dataclass(frozen=True, slots=True)
class HfFile:
    path: str                # e.g. "model.safetensors"
    size: int                # bytes (from /api/models tree)
    lfs_sha256: str | None   # 64-hex SHA from LFS metadata, when present
```

### 5.2 BlobDescriptor

```python
@dataclass(frozen=True, slots=True)
class BlobDescriptor:
    media_type: str          # "application/vnd.oci.image.layer.v1.tar"
    digest: str              # "sha256:<64hex>"
    size: int                # total tar bytes pushed
    hf_path: str             # for manifest ordering and logs
```

### 5.3 RunResult

```python
@dataclass(frozen=True, slots=True)
class RunResult:
    manifest_digest: str           # "sha256:<64hex>"
    image_ref: str                 # "host/repo:tag"
    image_ref_digest: str          # "host/repo@sha256:..."
    layers: tuple[BlobDescriptor, ...]
    skipped_blobs: int             # number of head_blob hits in step c
    failures: tuple[FailureRecord, ...]  # empty on full success
```

## 6. CLI & configuration

### 6.1 Sub-commands

```
oci-modelcar push      # main entry
oci-modelcar status    # list tags from registry
oci-modelcar validate  # verify manifest:tag coherence
```

### 6.2 `push` flags & env vars

| Flag | Env | Default | Notes |
|---|---|---|---|
| `--hf-repo <org/name>` | `HF_REPO` | required | |
| `--hf-revision <ref>` | `HF_REVISION` | `main` | |
| `--hf-endpoint <url>` | `HF_ENDPOINT` | `https://huggingface.co` | |
| `--allow-patterns <pat...>` | `ALLOW_PATTERNS` | `.safetensors .json .txt .md .model` | space-separated extensions |
| `--registry <host>` | `REGISTRY` | required | scheme inferred (loopback→http, else https) |
| `--target-repo <path>` | `TARGET_REPO` | required | |
| `--target-tag <tag>` | `TARGET_TAG` | sha[:12] | |
| `--also-tag <csv>` | `ALSO_TAGS` | `[]` | |
| `--layer-prefix <path>` | `LAYER_PATH_PREFIX` | `models/` | |
| `--workers <N>` | `WORKERS` | `1` | 1..8 |
| `--spool-dir <path>` | `SPOOL_DIR` | `$TMPDIR/oci-modelcar` | NEW |
| `--clean-hf-after-push` | `CLEAN_HF_AFTER_PUSH` | `false` | NEW |
| `--hf-max-retries <N>` | `HF_MAX_RETRIES` | `10` | |
| `--oci-max-retries <N>` | `OCI_MAX_RETRIES` | `5` | each retry is a full PATCH replay |
| `--fail-fast`/`--continue-on-error` | `FAIL_FAST` | fail-fast | mutually exclusive |
| `--force` | `FORCE` | `false` | overwrite existing tag |
| `--dry-run` | — | `false` | listing + disk check only |
| `--log-style text\|azure` | `LOG_STYLE` | auto-detect | |
| `--verbose`/`--quiet` | `LOG_VERBOSE`, `LOG_QUIET` | normal | |

Removed from v0.5: `--state-file`, `--chunk-mib`, `--upload-mode`.

### 6.3 Auth env vars (no CLI flag)

- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` (NEW: second name)
- `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` (NEW: opt-out)
- `OCI_USERNAME`, `OCI_PASSWORD`, or docker/podman config

### 6.4 `status` and `validate`

```bash
oci-modelcar status --registry ... --target-repo ...
# Lists tags for target_repo with their manifest digests, queried from
# the registry (no local state).

oci-modelcar validate --registry ... --target-repo ... --target-tag ...
# Confirms manifest:tag exists and all referenced blobs are present.
```

### 6.5 Spool layout and disk space pre-flight

Everything lives under `--spool-dir`. Two subdirectories:

```
<spool_dir>/
  sources/
    <hf_path>             # downloaded HF files (e.g. model.safetensors)
    <hf_path>.partial     # atomic-rename target during in-flight download
  layers/
    <hf_path>.tar         # built tar layer, source for PATCH
```

Naming preserves HF directory structure to avoid basename collisions. The
`<hf_path>.partial` suffix mirrors what `huggingface_hub` does — also
makes orphaned partials trivially identifiable by `ls`.

`--clean-hf-after-push` controls only the lifecycle of `sources/` files;
`layers/` tar files are always deleted after a successful HEAD-confirm in
phase e (their persistence has no resume value because they are derived
from `sources/` files).

Disk space pre-flight, mode-aware, before phase 2:

```python
max_layer  = max(tar_layer_size(f.size) for f in files)
max_source = max(f.size for f in files)
total_sources = sum(f.size for f in files)

# Always required: workers × (max source + max tar) in flight, with safety.
needed_in_flight = (max_source + max_layer) * cfg.workers * 1.2

# Persistent if not cleaning: full sum of sources kept until job end.
needed_persistent = 0 if cfg.clean_hf_after_push else int(total_sources * 1.05)

needed = needed_in_flight + needed_persistent

free = shutil.disk_usage(cfg.spool_dir).free
if free < needed:
    raise DiskSpaceError(
        f"Need {needed / 1e9:.1f} GB free in {cfg.spool_dir}, "
        f"only {free / 1e9:.1f} GB available. "
        f"Hints: --spool-dir <other-volume>, --clean-hf-after-push, "
        f"lower --workers (currently {cfg.workers})."
    )
```

Note: this is an estimate, not a hard guarantee. Mid-run ENOSPC during a
write surfaces as `DiskSpaceError` with the same hint text. We do not
attempt to recover by deleting sources mid-flight; the user is expected
to address the root cause (more disk, fewer workers, or `--clean-hf-after-push`).

## 7. Error handling & retries

### 7.1 Retry policy per phase

| Phase | Budget | Backoff | Transient → retry | Fatal → surface |
|---|---|---|---|---|
| Download | `--hf-max-retries` (10) | full-jitter, base 1s, cap 60s | ConnectionError, Timeout, IncompleteRead, ProtocolError, transient SSL EOF | SSL handshake, ProxyError, 401, 403 (gated), 404 |
| Tar+hash | 0 | — | — | IOError, ENOSPC → DiskSpaceError |
| Skip-check HEAD | 3 | linear 1s | 5xx, timeout | other |
| Push PATCH | `--oci-max-retries` (5) | full-jitter, base 1s, cap 60s | mid-stream SSL EOF, ConnectionError, Timeout, ChunkedEncoding, 408/429/5xx | SSL handshake, ProxyError, 401, 403, 404, 413 |
| Verify HEAD | 2 | linear 0.5s, 2s | 404 (race), 5xx | other |
| Manifest PUT | 3 | linear 1s, 2s, 4s | 5xx, 408/429 | 4xx |

Hard cap per file: `max_retries × backoff_cap` ≈ 10 minutes.

### 7.2 Retry budget reset on progress

Download retries refresh their budget when `bytes_buffered` advances
between two errors (carried over from v0.x). Push retries do **not**
reset — each replays from offset 0 anyway, no progress concept.

### 7.3 Fail-fast semantics (default)

```
exception in worker N
  → stop_event.set()
  → other workers exit at next checkpoint within seconds
  → ThreadPoolExecutor.shutdown(wait=True, cancel_futures=True)
  → spool cleanup in worker's finally:
  → Pipeline.run() re-raises the original exception
  → cli.py catches, logs, exits with the appropriate code
```

The first exception raised determines the surface error.

### 7.4 Continue-on-error semantics

```
for each file: collect (file, result-or-exception) tuples
after pool.shutdown():
  if any failures:
    log table of (hf_path, exception type, message)
    SKIP manifest push (we don't write incomplete manifests)
    raise PartialFailure → exit 7
  else:
    proceed to phase 3 (manifest)
```

### 7.5 Tag conflict policy

| State | `--force` | Action |
|---|---|---|
| Tag absent | any | push |
| Tag present, manifest digest matches | any | log "already pushed", exit 0 |
| Tag present, manifest digest differs | absent | refuse (PushError, exit 6) — explicit user action required |
| Tag present, manifest digest differs | present | overwrite |

This prevents accidental clobber of a known-good production tag.

### 7.6 Cleanup contract

`FileWorker.process()` finally block:
- delete `<spool>/sources/<hf_path>.partial` if present
- delete `<spool>/layers/<hf_path>.tar` if present
- `<spool>/sources/<hf_path>` (final source file): leave alone unless
  `--clean-hf-after-push`; this allows next run to skip re-download

`Pipeline.run()` finally:
- `executor.shutdown(wait=False, cancel_futures=True)`
- Server-side OCI upload sessions: not cleaned (registry GC policy
  applies; the v1.1 spec gives registries discretion).

## 8. Testing strategy

### 8.1 Layout

```
tests/
  unit/                # < 1s per file, pytest-httpserver mocks
    test_cli.py
    test_config.py
    test_download.py
    test_errors.py
    test_http.py
    test_layer.py
    test_logging.py
    test_manifest.py
    test_pipeline.py
    test_progress.py
    test_registry.py
    test_telemetry.py
  integration/         # multi-module, mocked HTTP
    test_pipeline_full.py
    test_pipeline_failure_modes.py
    test_cli_dispatch.py
  e2e/                 # real HF + docker registry:2
    test_real_huggingface.py
```

### 8.2 Critical tests (regression-prone behavior)

Highlights — the full list is in §5 of the brainstorm record:

- `test_stop_event_aborts_mid_download` — fixes 50 GB hang
- `test_authorization_stripped_on_cross_origin_redirect` — security
- `test_range_200_fallback_truncates_and_restarts` — robustness
- `test_lfs_sha256_verified_when_provided` — integrity
- `test_atomic_rename_no_partial_left_on_success`
- `test_partial_file_cleaned_on_exception`
- `test_gated_repo_403_raises_specific_class`
- `test_token_resolution_priorities` (parametrized over the four sources)
- `test_push_from_file_retries_on_ssl_eof_with_file_rewound`
- `test_accepts_200_201_202_204_on_patch` (parametrized)
- `test_unhandled_status_raises`
- `test_max_retries_exhausted_raises_PushError`
- `test_skip_entire_job_if_manifest_tag_matches`
- `test_refuses_overwrite_when_tag_exists_with_different_digest`
- `test_fail_fast_cancels_pending_workers_within_seconds`
- `test_continue_on_error_skips_manifest_push`
- `test_continue_on_error_exits_7_on_partial_failure`
- `test_disk_space_preflight_fails_clean`
- `test_disk_space_check_is_mode_aware`
- `test_layers_ordered_alphabetically_by_hf_path_in_manifest`

### 8.3 Coverage targets

- `src/oci_modelcar/`: ≥ 95% line coverage
- Critical phases (download streamer, push retry, fail-fast cancel): 100%
  branch coverage
- E2E: smoke pass against pinned `hf-internal-testing/tiny-random-LlamaForCausalLM`
  at SHA `9fb191250dd56d0ba7ec9785a025ed29c03d5998`, registry:2 on :5000

### 8.4 Pre-commit gates (unchanged from v0.5)

```yaml
- ruff check --fix
- ruff format
- mypy --strict src/
- pytest -m "not e2e" -q
```

## 9. Migration from v0.5

Breaking changes for existing users:

| What | Migration |
|---|---|
| `--state-file` removed | Drop the flag. State is registry-side; idempotency uses HEAD. |
| `--chunk-mib` removed | Drop the flag. There is no chunking anymore. |
| `--upload-mode` removed | Drop the flag. There is one mode. |
| `state.json` orphaned | Safe to delete. The new tool ignores it. |
| `--spool-dir` (NEW) | Defaults to `$TMPDIR/oci-modelcar`, override if your tmp is small. |
| `--clean-hf-after-push` (NEW) | Set when running on ephemeral CI to minimize disk. |
| Retry defaults | OCI default 5 (was 10). Each retry is a full re-upload now; lower default avoids ballooning bandwidth on systematic failures. |

The v0.5.x line is end-of-life upon v1.0.0 release. No back-port branch.

## 10. Out of scope / future work

Tracked elsewhere or deferred:

- **Cosign signing** — separate spec
  (`2026-05-07-cosign-integration-design.md`). Will be wired in v1.1
  after the v1.0 rewrite stabilizes.
- **Distribution as a single binary** — `pipx`/`uv tool` cover most CI
  needs; `shiv` / `pex` packaging is investigated when there is concrete
  demand from a user without Python in their build image.
- **Multi-arch tagging** — current scope is single-architecture per push.
- **GGUF / non-safetensors model packaging** — out of scope; the tool
  transports bytes regardless of format, but model-aware features are
  not planned.
- **HF xet backend** — defers Rust dep. Reconsider when pure-HTTP
  fallback for xet repos becomes lossy.

## Appendix A — Why not huggingface_hub for downloads?

`huggingface_hub.hf_hub_download` is a good library but has one
disqualifying limitation for our workload: **no mid-stream cancellation**.
The download is a synchronous loop over `requests.iter_content` with no
hook between iterations. On a 50 GB file behind a slow proxy, a
`stop_event` set after `KeyboardInterrupt` or fail-fast cannot interrupt
the active download for hours. This bites on the most painful failure
mode of v0.5 (long-running uploads where user wants to abort).

Trade-off: we re-implement the bytes streamer (~250 lines) but keep
`HfApi` for metadata (URL resolution, file listing with LFS sha, auth).
A code-comparison audit (`docs/superpowers/specs/notes/...` if needed)
showed our current streamer is more aggressive on retries (10 vs 5,
full-jitter vs exponential) and offers stop_event by design. The gaps
that the audit identified — cross-origin auth stripping, expanded token
sources, gated-repo error class, Range-200 fallback — are ported in
§4.1, §4.5.

## Appendix B — Why single PATCH per blob (vs chunked)?

Confirmed by reading `containers/image` (Podman, Skopeo) and Jib source:

- `containers/image/docker/docker_image_dest.go:PutBlobWithOptions`
  issues a single PATCH with the full blob streamed (`// FIXME? Chunked
  upload` comment in their code marks this as a deliberate choice).
- Jib's `BlobPusher.java` uses one PATCH per blob, body fed from a
  Blob abstraction that's typically a local file.

Both lack intra-blob `Content-Range` resume; on PATCH failure they retry
the full PATCH. The replayable source is what makes this acceptable — for
them, it's a local file (image cache); for us, it's our spool file.

A single PATCH means one TCP request, one load-balancer routing
decision, one cluster node receiving the entire blob. This is the
property that fails on chunked uploads against a non-sticky LB.

## Appendix C — Estimated impact

| Metric | v0.5 | v1.0 estimate | Δ |
|---|---|---|---|
| `src/` lines of code | ~3000 | ~2300–2500 | −15 to −25% |
| `tests/` lines of code | ~3300 | ~2300–2500 | −25% |
| Total | ~6300 | ~4600–5000 | −20 to −25% |
| External runtime deps | `requests`, `urllib3` | `requests`, `urllib3`, `huggingface_hub` (metadata only) | +1 |
| Test runtime (unit + integration) | ~3s | ~3s expected | — |
