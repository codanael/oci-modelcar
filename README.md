# oci-modelcar

[![CI](https://github.com/codanael/oci-modelcar/actions/workflows/ci.yml/badge.svg)](https://github.com/codanael/oci-modelcar/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/oci-modelcar.svg)](https://pypi.org/project/oci-modelcar/)

Stream HuggingFace models directly into OCI registries as multi-layer images,
suitable for KServe with native OCI image volumes (KEP-4639).

## Why

Pushing a HuggingFace model to an OCI registry typically means:
1. Triple-trip: HF -> local cache -> registry
2. One huge layer: no cross-repo blob mount possible
3. No resume: a 5 GB shard failing at 4.5 GB starts over

`oci-modelcar` streams in pure Python:
- HF -> registry directly, no disk persistence
- One uncompressed tar layer per file (`digest == diff_id`)
- Three-level resume: HF Range request, OCI session resync, file-level
  state file
- Memory bounded to ~16 MiB per worker

## Install

```bash
pip install oci-modelcar
```

## Quick start

```bash
export HF_TOKEN=hf_...
export OCI_USERNAME=...
export OCI_PASSWORD=...

oci-modelcar push \
  --hf-repo Qwen/Qwen3-30B-A3B \
  --registry registry.example.com \
  --target-repo models/qwen3-30b
```

The image tag defaults to the first 12 characters of the resolved HF commit
SHA (e.g. `a3f47b09c8d2`).

## Authentication

**HuggingFace** (aligned with `huggingface-cli`):
- `HF_TOKEN` env var (recommended)
- `~/.cache/huggingface/token` (created by `huggingface-cli login`)

**OCI registry**:
- `OCI_USERNAME` + `OCI_PASSWORD` env vars (recommended for CI)
- `~/.docker/config.json` (`docker login` writes here)
- `$XDG_RUNTIME_DIR/containers/auth.json` (`podman login`)

Local registries (hostnames `localhost`, `127.x.x.x`, `::1`) automatically
use plain HTTP. Pass an explicit `http://` or `https://` prefix on
`--registry` to override.

## Common options

| Option | Default | Description |
|---|---|---|
| `--hf-revision` | `main` | Branch, tag, or 40-char SHA |
| `--target-tag` | `<sha[:12]>` | Image tag |
| `--also-tag` | — | CSV of alias tags |
| `--workers` | `1` | Parallel layers (cap 8) |
| `--chunk-mib` | `8` | PATCH chunk size |
| `--state-file` | `~/.local/state/oci-modelcar/state.json` | JSON resume state |
| `--fail-fast` / `--continue-on-error` | fail-fast | Failure policy |
| `--log-style` | auto | `text` or `azure` |
| `--dry-run` | — | List files, don't push |

Full list: `oci-modelcar push --help`.

## Resume after failure

State is automatically saved per file. If a push is killed (kill, OOM, network
loss), re-running the same command resumes:

```bash
# First run, killed mid-way
oci-modelcar push --hf-repo X --registry Y --target-repo Z
# ^C

# Re-run: skips files already pushed
oci-modelcar push --hf-repo X --registry Y --target-repo Z
```

Force a full re-push with `--force`.

## OCI compliance

Compliant with OCI Distribution v1.1 and OCI Image Spec v1.1:
- Chunked PATCH uploads with `Content-Range: N-M` (inclusive, no `bytes`
  prefix per OCI spec)
- Resume via `GET /v2/<repo>/blobs/uploads/<id>` and the `Range: 0-N` header
- `416 Range Not Satisfiable` is treated as "ask the server, sync, retry"
- HEAD validation cross-checks `Docker-Content-Digest`
- Layers use `application/vnd.oci.image.layer.v1.tar` (uncompressed) so
  `layer.digest == diff_id` by construction

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml` and update `CHANGELOG.md`.
2. Tag: `git tag v0.1.0 && git push --tags`.
3. The `release.yml` workflow builds, publishes to PyPI via Trusted Publishing,
   and creates a GitHub Release.

PyPI trusted publisher must be configured once: on pypi.org -> Project
Settings -> Publishing -> Add publisher with:
- Owner: `codanael`
- Repo: `oci-modelcar`
- Workflow: `release.yml`
- Environment: `pypi`

## License

MIT
