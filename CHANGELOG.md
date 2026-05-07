# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Initial implementation

### Fixed
- Resume path now persists `layer_size` so the manifest's `layers[].size` is correct.
- `failed` list in parallel mode now contains file paths (was exception strings).
- `pushed` / `skipped` counters now accurate in both serial and parallel modes.

### Known limitations (v0.2 follow-up)
- Broader runner concurrency tests (state writes during contention, fail-fast cancellation timing).
- HEAD pre-check on cached layers to detect registry GC since last run.
- `_patch_with_retry` partial-acceptance edge case (most registries are atomic; not observed in practice).
