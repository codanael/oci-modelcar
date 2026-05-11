# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.1.0] - 2026-05-11

### Added
- **Per-file progress restored.** `FileWorker` now announces each phase of
  the pipeline via `PipelineLogger`:
  `<path>: downloading (NN MB)`, throttled `NN% (transferred / total)`
  lines, `<path>: pushing layer sha256:abc12 (NN MB)`, and a closing
  `<path>: pushed sha256:abc12` (or `: reusing existing blob …`). The
  v0.x `ProgressEmitter` and human-bytes scaler `fmt_bytes` are back, and
  `PipelineLogger` emits are now mutex-guarded so parallel workers do not
  interleave lines. Restores the live status output that was dropped in
  the v1.0 rewrite.
- **Cross-run layer reuse via manifest annotations.** Each layer descriptor
  now carries `io.github.codanael.modelcar.hf-path` (always) and
  `io.github.codanael.modelcar.hf-sha256` (for LFS-backed files) in its
  `annotations`. On every `push`, the pipeline fetches the existing
  manifest at the target tag (when present and `--force` is off), parses
  these annotations into a reuse-map keyed by `(hf_path, hf_sha256)`, and
  hands it to every worker. When a file's `(path, lfs_sha256)` matches a
  reuse-map entry and the layer blob is still present in the registry,
  the worker skips HF download + tar build + push entirely and reuses the
  existing descriptor as-is. A re-push of an unchanged HuggingFace
  revision now touches HF for zero bytes and the registry for HEAD-only
  traffic.

### Fixed
- **`download.py` now actually honors the cached-source guarantee**
  promised in `CLAUDE.md`. If `<spool>/sources/<hf_path>` already exists
  at the expected size (atomic-rename invariant: it's the completed,
  sha256-verified result of a prior run), `HfDownloader.download()`
  returns the cached path without issuing any HTTP request, instead of
  silently re-downloading on every invocation. Mainly benefits crashed-run
  retries when `--clean-hf-after-push` is off.

## [1.0.1] - 2026-05-10

### Fixed
- Top-level `oci-modelcar --help`/`-h` now prints usage to stdout and exits 0
  instead of treating `--help` as an unknown sub-command (#10).
- Bad `--hf-revision` (and missing repos / gated repos / missing entries)
  no longer escape as raw `huggingface_hub.errors.*` tracebacks. The
  upstream `RevisionNotFoundError`, `RepositoryNotFoundError`,
  `GatedRepoError`, and `EntryNotFoundError` raised by
  `HfApi.repo_info()` and `HfApi.list_repo_tree()` are now remapped to
  our typed exceptions in `download.py`, so the CLI surface produces the
  documented one-line error + actionable hint and the matching exit
  code (3 for gated, 5 for not-found cases) (#11).
- Reword the no-OCI-credentials warning so it makes sense for read-only
  `status`/`validate` calls as well as `push`. Now says "proceeding
  anonymously" and notes that protected registries return 401 on writes
  (#12).

## [1.0.0] - 2026-05-08

### Added
- Per-file pipeline (download → tar → push → cleanup) parallelized via `--workers`.
- `huggingface_hub.HfApi` for metadata (revision resolve, file listing,
  LFS sha256 detection); bytes streamed by our own code so mid-stream
  cancellation works on multi-GB downloads.
- Atomic write semantics for downloaded files (`.partial` → rename).
- Cross-origin Authorization stripping on HF→S3 redirects (security).
- Expanded HF token sources: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
  `~/.cache/huggingface/token`, opt-out via `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.
- Range-200 fallback (server ignores Range → truncate + restart) ported
  from huggingface_hub.
- Specific error classes: `GatedRepoError`, `RevisionNotFoundError`,
  `EntryNotFoundError`, `DiskSpaceError`, `PushError`, `PartialFailureError`.
- Per-class CI exit codes: 0/1/2/3/4/5/6/7.
- `--spool-dir`, `--clean-hf-after-push` flags + matching env vars.
- Tag conflict policy: skip on match, refuse without `--force`, overwrite
  with `--force`.
- Mode-aware disk space pre-flight check (with/without `--clean-hf-after-push`).

### Changed
- **Single PATCH per blob from local file (Jib-style replay-on-cut).**
  Eliminates per-PATCH LB routing decisions on misconfigured Artifactory
  HA clusters. Same wire shape as containers/image and Jib.
- Default `--oci-max-retries` lowered from 10 to 5 (each retry is a full
  PATCH replay; bandwidth ballooning on systematic failures otherwise).
- Tar layer size formula now exposed as `layer.tar_layer_size(file_size)`.

### Removed
- `state.json` and the `state.py` module entirely. Registry HEAD is
  the source of truth for resumability and idempotency.
- `ChunkedBlobUpload` and chunked PATCH mode.
- `--state-file`, `--chunk-mib`, `--upload-mode` flags. Use `--spool-dir`
  and `--clean-hf-after-push` for the new disk model.
- `_PipeBuffer` thread-bridge (per-file pipeline replaces it).
- `tags.py` (`derive_tag` migrated into `manifest.py`).

### Security
- HF Authorization tokens are no longer forwarded on cross-origin
  redirects. Previous versions could leak a Bearer token to S3 /
  CloudFront (HF's redirect target for LFS files), where the request
  was rejected but the token may have been logged.

## [0.5.0] - 2026-05-08

### Added
- **Pipelined HF download / OCI push.** `process_one_file` now runs the HF
  read + tar wrapping on a producer thread and the OCI PATCH stream on
  the main (consumer) thread, bridged by a bounded `_PipeBuffer`. The two
  stages no longer backpressure each other one-for-one — when
  HF ≈ OCI in throughput, the pipelined version effectively doubles
  total wall-clock throughput vs. the serial coupling. Memory cost:
  `pipe_max_chunks × pipe_coalesce_size` ≈ 8 MiB extra per worker.
- **Per-file throughput + bottleneck telemetry.** Each pushed file emits
  one INFO line at completion: `path: 1.23 GB in 12.3s (100 MB/s); HF
  wait 0.8s (6%), OCI wait 8.5s (69%)`. The two wait values come from
  the `_PipeBuffer` (time the producer spent blocked on `put` because
  the consumer was slow ⇒ OCI bottleneck; symmetric for `get` ⇒ HF
  bottleneck). Cached files emit no telemetry (no real transfer
  happened). Surfaced as a new `FileTelemetry` field returned from
  `process_one_file` alongside `(descriptor, diff_id)`.

### Changed
- **Default `--chunk-mib` raised from 8 to 32.** Empirical validation on
  real registries showed that the per-PATCH overhead (TCP RTT + HTTP
  headers + TLS) dominates on fast LAN links, and amortizing it over
  larger chunks gives a substantial speedup. Memory baseline: ~64 MiB
  pic per worker (vs. 16 MiB before); with the default 1 worker that's
  immaterial, with `--workers 8` it's ~512 MiB peak. Override with
  `--chunk-mib N` to go back down (1–1024 MiB allowed).

## [0.4.1] - 2026-05-08

### Fixed
- **Mid-stream SSL EOF is now correctly treated as transient.** Regression
  introduced in 0.4.0: the early-raise fix for SSL handshake failures was
  too broad and also caught `ssl.SSLEOFError` ("EOF occurred in violation
  of protocol") raised when an idle proxy or firewall cuts a long-running
  TCP connection mid-transfer. Symptom seen on a 1.34 GB shard. The
  exception's `__cause__` and `__context__` chains are now walked for
  `ssl.SSLEOFError` (with a string-marker fallback for wrappers that
  drop the chain), and when found the error is treated as transient:
  `HfStream` resumes via Range and `ChunkedBlobUpload` resyncs + retries.
  Pure handshake / cert-validation SSL errors remain fatal.

## [0.4.0] - 2026-05-08

### Added
- **Auth-source visibility.** `oci_auth_header` now logs an `INFO` line on
  successful resolution (`OCI auth resolved from <env|path>`) and a
  `WARNING` when no source matches and the push falls back to anonymous,
  so the user no longer needs to guess why their credentials weren't
  picked up.
- `$XDG_CONFIG_HOME/containers/auth.json` (default
  `$HOME/.config/containers/auth.json`) is now part of the auth.json
  search path, in addition to `~/.docker/config.json` and
  `$XDG_RUNTIME_DIR/containers/auth.json`. This is the default location
  for rootful `podman login` on most distros.
- **Live per-file upload progress** during `push`. Each tracked file emits
  a `<path>: NN% (<transferred> / <total>)` line at most once every 5 s
  while its layer streams, with GB/MB/KB scaling. The cumulative byte
  count flows from `HfStream` through `process_one_file` via a new
  `progress_cb: Callable[[int], None] | None` callback, throttled by the
  new `ProgressEmitter` helper in `oci_modelcar.logging`.
- Per-file headers and result lines in **multi-worker mode**. Previously
  `--workers >1` was completely silent between the `Pushing N layers`
  banner and the manifest section. Now all `[N/total] <path> (<size>)`
  headers are emitted in alphabetical order before any worker starts,
  and each worker prints `<path>: -> sha256:digest…` (or
  `<path>: failed: …` on error) on completion.

### Changed
- The mono-worker logging path is unified with multi-worker — the
  per-file `file_scope` buffer (which suppressed all output until the
  layer finished) is replaced by direct, path-prefixed emits, so live
  progress is visible in both modes.

### Fixed
- **No more silent hang on TLS / proxy misconfig.** Three retry layers were
  quietly retrying SSL handshake and proxy errors (urllib3 session-level
  `Retry`, `HfStream._next_chunk`, `ChunkedBlobUpload._patch_with_retry`).
  With each layer compounding, a misconfigured CA on a self-signed registry
  could keep the CLI silent for several minutes after `Pushing N layers`
  before the real error surfaced. A new `_SmartRetry` subclass re-raises
  `ssl.SSLError` / `urllib3.exceptions.SSLError` / `urllib3.exceptions.ProxyError`
  out of `Retry.increment()`, and the chunk-level retry loops re-raise
  `requests.exceptions.SSLError` / `ProxyError` before falling into the
  transient catch. Other transport errors still retry as before.
- **`auth.json` keys with a repo path now match.** When the user has
  `auths["artifactory.example/myproject"]` (set by `podman login
  artifactory.example/myproject`), the previous strict `auths.get(host)`
  lookup never matched, leading to silent anonymous push and a confusing
  401 later. `docker_config_auth` now does longest-prefix match across
  normalized auths keys (strips `https://`/`http://`, `/v2/` suffix,
  trailing slashes), and `oci_auth_header` is plumbed `target_repo` from
  `OciClient` so the lookup can use the full `host/repo` reference.
- **Ctrl+C is now responsive in multi-worker mode.** Previously the
  `with ThreadPoolExecutor(...)` context manager called
  `pool.shutdown(wait=True)` on the way out of a `KeyboardInterrupt`,
  so the program appeared frozen until every in-flight worker finished
  its current multi-GB file (potentially many minutes). The runner now
  uses an explicit `try/finally` with `pool.shutdown(wait=False,
  cancel_futures=True)`, plus a shared `threading.Event` `stop_event`
  threaded through `HfStream._next_chunk` and
  `ChunkedBlobUpload._patch_with_retry`. Workers in flight short-circuit
  at the next chunk boundary (sub-second at typical bandwidth) by
  raising `InterruptedError`, while the main thread re-raises so the
  CLI exits 130 promptly. State integrity is preserved: `mark_pushed`
  only runs after a layer fully closes, so partial files are never
  recorded.

## [0.3.0] - 2026-05-08

### Changed
- Lowered minimum Python version from 3.14 to **3.11**. The codebase only
  relied on `typing.Self` and `datetime.UTC` (both 3.11+); no 3.12+ syntax
  was in use. Verified with the full unit + integration suite (97 tests)
  and `mypy --strict` on Python 3.11, 3.12, 3.13, and 3.14. `requires-python`,
  `tool.ruff.target-version`, `tool.mypy.python_version`, and the trove
  classifiers were all updated together.
- CI now runs the test job as a matrix over Python 3.11–3.14
  (`fail-fast: false`). The lint job stays pinned to 3.14 since mypy is
  already targeting 3.11 via `pyproject.toml`.
- Bumped all JavaScript GitHub Actions to Node.js 24 runtimes ahead of the
  Node 20 deprecation (June 2026): `actions/checkout` v4→v6,
  `actions/setup-python` v5→v6, `actions/upload-artifact` v4→v7,
  `actions/download-artifact` v4→v8, `softprops/action-gh-release` v2→v3.

## [0.2.1] - 2026-05-08

### Fixed
- HuggingFace download retries now absorb the full set of transport-layer
  failure modes (`urllib3.exceptions.ProtocolError`, `http.client.IncompleteRead`,
  raw `OSError`, plus the existing `requests` exception family). Premature
  end-of-stream (the response generator finishing before `Content-Length` is
  reached) is also handled uniformly: the stream is reopened with
  `Range: bytes=N-` and the read continues, mirroring `wget --continue` /
  `curl --continue-at` semantics. This prevents repeated upstream cuts on
  large files (multi-GB) from aborting a push when the underlying connection
  is severed at a fixed offset.
- `HfStream.read(-1)` is no longer single-shot on resume: it shares the same
  unbounded resume loop as `read(n)`, so cascaded cuts during a full-file
  read are recovered instead of raising a truncation error.

### Added
- INFO-level log line on every Range-based resume, showing
  `path`, current offset, expected total, and percentage. Makes multi-resume
  pushes observable without enabling verbose mode.

## [0.2.0] - 2026-05-07

### Added
- `IMAGEREFDIGEST=<registry>/<repo>@sha256:<digest>` variable in `push`
  output, suitable for direct piping into `cosign sign`. Re-emitted on
  idempotent re-runs.
- `image_ref_digest` field persisted in the state file and surfaced as a
  second indented line in `oci-modelcar status` output. Legacy state files
  without this field continue to render the single existing line.
- PEP 740 digital attestations published with PyPI artifacts via Sigstore
  keyless OIDC (one-line release-workflow change). Verifiable with
  `pypi-attestations` (see README "Signing & verification").
- README "Signing & verification" section with cosign sign/verify recipes
  (keyless + static-key) for modelcar OCI images.

### Changed
- `JsonStateStore.mark_completed()` now requires `image_ref_digest=` keyword
  argument (keyword-only). Breaking change for direct API consumers; CLI
  users unaffected.

### Fixed
- `IMAGEREF` and `IMAGEREFDIGEST` no longer leak the `http://` / `https://`
  scheme prefix when `--registry` is passed with one. Both outputs now use
  the bare `host[:port]` form so they can feed directly into
  `cosign sign $IMAGEREFDIGEST`.

## [0.1.0] - 2026-05-07

Initial release.

### Added
- Stream HuggingFace models directly into OCI registries as multi-layer images
  (one tar layer per file, `digest == diff_id` for uncompressed tar layers).
- Three-level resume capability: HF Range request (intra-file network),
  OCI session resync via GET (intra-file PATCH), JSON state file (cross-process).
- Configurable parallel workers (`--workers`, cap 8) with deterministic manifest
  layer ordering regardless of completion order.
- HF revision auto-resolution: `--hf-revision main` resolves to a 40-char SHA
  via the HF API; tag defaults to `<sha[:12]>`.
- Auto-detection of loopback registries (`localhost`, `127.x.x.x`, `::1`) which
  use HTTP; remote registries default to HTTPS. Explicit scheme prefix on
  `--registry` overrides.
- `oci-modelcar` CLI with `push`, `status`, and `validate` sub-commands.
- OCI Distribution v1.1 compliance: PATCH `Content-Range: N-M` (no `bytes` prefix),
  `416` and `5xx`/`408`/`429` treated as transient with backoff + resync,
  HEAD validation cross-checks `Docker-Content-Digest`.
- HuggingFace and OCI registry auth: `HF_TOKEN` env or `~/.cache/huggingface/token`;
  `OCI_USERNAME`/`OCI_PASSWORD` env or `~/.docker/config.json` or
  `$XDG_RUNTIME_DIR/containers/auth.json`.
- Logging: text format (default) and Azure Pipelines format (auto-detected via
  `TF_BUILD`).
- `--dry-run`, `--force`, `--fail-fast` / `--continue-on-error` flags.

### Known limitations (v0.2 follow-up)
- Broader runner concurrency tests (state writes during contention, fail-fast
  cancellation timing).
- HEAD pre-check on cached layers to detect registry GC since last run.
- `_patch_with_retry` partial-acceptance edge case (most registries are atomic;
  not observed in practice).
- Cross-repo blob mount optimization for re-tagging the same model.
