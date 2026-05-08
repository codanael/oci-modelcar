# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
