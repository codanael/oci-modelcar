# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `IMAGEREFDIGEST=<registry>/<repo>@sha256:<digest>` variable in `push`
  output, suitable for direct piping into `cosign sign`. Re-emitted on
  idempotent re-runs.
- `image_ref_digest` field persisted in the state file and surfaced as a
  second indented line in `oci-modelcar status` output. Legacy state files
  without this field continue to render the single existing line.
- PEP 740 digital attestations published with PyPI artifacts via Sigstore
  keyless OIDC (one-line release-workflow change). Verifiable with
  `pypi-attestations` or `cosign verify-blob`.
- README "Signing & verification" section with cosign sign/verify recipes
  (keyless + static-key) for modelcar OCI images.

### Changed
- `JsonStateStore.mark_completed()` now requires `image_ref_digest=` keyword
  argument. Breaking change for direct API consumers; CLI users unaffected.

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
