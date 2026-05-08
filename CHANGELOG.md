# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
