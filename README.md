# oci-modelcar v1.0

[![CI](https://github.com/codanael/oci-modelcar/actions/workflows/ci.yml/badge.svg)](https://github.com/codanael/oci-modelcar/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/oci-modelcar.svg)](https://pypi.org/project/oci-modelcar/)

Push HuggingFace models to OCI registries as multi-layer images,
suitable for KServe with native OCI image volumes (KEP-4639).

v1.0 uses a per-file pipeline (download → tar → push) with a single PATCH
per blob (Jib-style), eliminating per-PATCH routing decisions on Artifactory
HA clusters and Harbor reverse-proxy setups.

## Why

Pushing a HuggingFace model to an OCI registry typically means:
1. Triple-trip: HF -> local cache -> registry
2. One huge layer: no cross-repo blob mount possible
3. No resume: a 5 GB shard failing at 4.5 GB starts over

`oci-modelcar` handles this in pure Python:
- Per-file pipeline: download, tar, and push run concurrently per file
- One uncompressed tar layer per file (`digest == diff_id`)
- Single PATCH per blob — same wire shape as Jib and `containers/image`
- Parallel workers (`--workers`, cap 8)

## Documentation

- [User guide](./docs/user-guide.md) — complete CLI reference, CI/CD examples, troubleshooting, exit codes
- [CHANGELOG](./CHANGELOG.md) — release history, breaking changes
- Design spec: [docs/superpowers/specs/2026-05-08-oci-modelcar-v1-design.md](./docs/superpowers/specs/2026-05-08-oci-modelcar-v1-design.md)

## Install

```bash
pip install oci-modelcar
# or via uv (recommended for CI)
uv tool install oci-modelcar
```

Requires Python 3.11+.

## Quick start

```bash
export HF_TOKEN=hf_...
export OCI_USERNAME=...
export OCI_PASSWORD=...

oci-modelcar push \
  --hf-repo Qwen/Qwen2.5-7B-Instruct \
  --registry registry.acme.com \
  --target-repo models/qwen-7b
```

The image tag defaults to the first 12 characters of the resolved HF commit
SHA (e.g. `a3f47b09c8d2`).

## Disk space

v1 spools downloaded HF files and built tar layers under `--spool-dir`
(default `$TMPDIR/oci-modelcar`). Roughly 2× the largest layer per worker
in flight, plus the cumulative size of all source files unless
`--clean-hf-after-push` is set. The push aborts up-front with a clear
error if free space is insufficient.

## Migration from v0.5

v1.0 is a clean rewrite. Breaking changes:

- `--state-file` removed (registry HEAD is the source of truth)
- `--chunk-mib` removed (single PATCH per blob)
- `--upload-mode` removed (one mode)
- Default `--oci-max-retries` lowered from 10 to 5 (each retry is a full PATCH replay)
- Two new flags: `--spool-dir`, `--clean-hf-after-push`

See [CHANGELOG.md](./CHANGELOG.md) for full details.

## Authentication

**HuggingFace** (aligned with `huggingface-cli`):
- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` env var (recommended)
- `~/.cache/huggingface/token` (created by `huggingface-cli login`)
- Opt-out: set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` to skip implicit token sources

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
| `--allow-patterns` | `.safetensors .json .txt .md .model` | Files to include. Bare tokens are suffix filters; tokens with `*`/`?`/`[…]` are full `fnmatch` globs on the repo path. |
| `--ignore-patterns` | empty | Files to exclude after `--allow-patterns` admits. Same syntax. Wins on conflict. Needed for repos shipping multiple weight formats (e.g. `--ignore-patterns "consolidated*"` to skip the Mistral native layout in `mistralai/Mistral-Medium-3.5-128B`). |
| `--no-reuse-records` | off | Disable OCI referrer artifacts that let a re-run after a crashed `--clean-hf-after-push` push skip HF re-downloads. Default is on (records ARE written); the flag is an opt-out. |
| `--workers` | `1` | Parallel layers (cap 8) |
| `--spool-dir` | `$TMPDIR/oci-modelcar` | Directory for downloaded files and built tar layers |
| `--clean-hf-after-push` | off | Delete each HF file after its layer is pushed |
| `--oci-max-retries` | `5` | Max PATCH retries per blob (each is a full replay) |
| `--fail-fast` / `--continue-on-error` | fail-fast | Failure policy |
| `--log-style` | auto | `text` or `azure` |
| `--dry-run` | — | List files, don't push |

Full list: `oci-modelcar push --help`. For complete usage, scenarios, and
troubleshooting see [docs/user-guide.md](./docs/user-guide.md).

## Resume after failure / re-push

v1 uses the registry as the source of truth. No local state file is needed.

Two layers of skipping kick in on a re-run:

1. **Cross-run reuse (no HF traffic).** Every layer is annotated with the
   HuggingFace path and (for LFS files) the upstream sha256. On every push,
   the pipeline GETs the existing manifest at the target tag, indexes its
   layers by `(hf-path, hf-sha256)`, and for each unchanged file skips the
   HF download + tar build + PATCH entirely — only a HEAD-blob is issued
   to confirm the layer is still in the registry. A re-push of an unchanged
   HF revision touches HF for zero bytes.
2. **Same-run blob skip (resume after crash).** Even when the reuse map
   misses, the worker still HEADs each freshly-built layer digest and skips
   the PUT if the registry already has the blob. Combined with the cached
   spool sources (`<spool>/sources/<path>` is reused if it exists at the
   expected size), this turns a killed-mid-way push into a near-instant
   completion on the next run.

```bash
# First run, killed mid-way
oci-modelcar push --hf-repo X --registry Y --target-repo Z
# ^C

# Re-run: cached files + cached blobs picked up, only the missing pieces transfer
oci-modelcar push --hf-repo X --registry Y --target-repo Z
```

`--force` bypasses both layers (reuse map is not built, HEAD checks are
ignored on blobs and on the manifest tag).

## OCI compliance

Compliant with OCI Distribution v1.1 and OCI Image Spec v1.1:
- Single PATCH per blob with upfront `Content-Length` (Jib-style); on retry
  the full PATCH is replayed from the local spool file
- `Content-Range: N-M` (inclusive, no `bytes` prefix per OCI spec)
- HEAD validation cross-checks `Docker-Content-Digest`
- Layers use `application/vnd.oci.image.layer.v1.tar` (uncompressed) so
  `layer.digest == diff_id` by construction

## Signing & verification

`oci-modelcar` itself does not sign artifacts — signature is delegated to
[cosign](https://github.com/sigstore/cosign), the canonical OCI signing tool.
Each `push` exposes the canonical digest reference for direct piping into
cosign:

```bash
oci-modelcar push --hf-repo ... --registry ... --target-repo ...
# IMAGEREFDIGEST=registry.example.com/models/qwen3-30b@sha256:...

# Sign keyless (CI with OIDC, e.g. GitHub Actions with id-token: write)
cosign sign $IMAGEREFDIGEST

# Sign with a static key (offline / regulated environments)
cosign generate-key-pair                  # one-time, produces cosign.key + cosign.pub
cosign sign --key cosign.key $IMAGEREFDIGEST

# Verify (consumer side, e.g. KServe operator)
cosign verify $IMAGEREFDIGEST \
    --certificate-identity-regexp '^https://github\.com/your-org/' \
    --certificate-oidc-issuer 'https://token.actions.githubusercontent.com'

# Or with the static public key
cosign verify --key cosign.pub $IMAGEREFDIGEST
```

The signature is stored as an additional artifact in the same OCI registry,
attached to the manifest by digest (referrers API for OCI Distribution v1.1+,
or `:sha256-<digest>.sig` tag for legacy registries — cosign auto-detects).

### PyPI artifact

The `oci-modelcar` PyPI distribution is signed with PEP 740 digital
attestations generated by GitHub Actions in keyless OIDC mode. Verify with:

```bash
pip install pypi-attestations

# Replace with the version you want to verify
VERSION=1.0.0

python -m pypi_attestations verify pypi \
    --repository https://github.com/codanael/oci-modelcar \
    pypi:oci_modelcar-${VERSION}-py3-none-any.whl
python -m pypi_attestations verify pypi \
    --repository https://github.com/codanael/oci-modelcar \
    pypi:oci_modelcar-${VERSION}.tar.gz
```

`pypi-attestations` fetches the artifact and its provenance from PyPI on its
own when you pass `pypi:<filename>`, so no separate `pip download` step is
needed. `--repository` expects the full GitHub/GitLab URL of the publishing
repo; an `OK: <filename>` line per artifact (exit 0) means the provenance
chains correctly through Sigstore (Fulcio cert + Rekor transparency log).

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml` and update `CHANGELOG.md`.
2. Tag: `git tag -a v1.0.0 -m 'release notes' && git push origin v1.0.0`.
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
