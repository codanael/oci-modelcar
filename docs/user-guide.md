# oci-modelcar — User Guide

Complete reference for using `oci-modelcar` v1.0+ in production. For a
high-level overview see [README.md](../README.md).

## Table of contents

1. [Concepts](#concepts)
2. [Installation](#installation)
3. [Your first push](#your-first-push)
4. [CLI reference](#cli-reference)
5. [Authentication](#authentication)
6. [Multi-tag publishing](#multi-tag-publishing)
7. [Resilience: retries, resume, cancellation](#resilience-retries-resume-cancellation)
8. [Disk space planning](#disk-space-planning)
9. [CI/CD integration](#cicd-integration)
10. [Using the image with KServe](#using-the-image-with-kserve)
11. [Cosign signing & verification](#cosign-signing--verification)
12. [Troubleshooting](#troubleshooting)
13. [Exit codes](#exit-codes)

---

## Concepts

`oci-modelcar push` does three things, in order:

1. **Pre-flight** — resolve the HuggingFace revision, list files, derive the
   target tag, check the registry for an existing manifest at that tag,
   verify free disk space.
2. **Per-file pipeline** — for each matched HF file, run a worker that:
   downloads the file to a local spool directory, builds an uncompressed
   tar layer with deterministic headers, HEADs the registry to skip if
   the blob is already present, otherwise pushes the layer in a single
   streaming PATCH from the local file (Jib-style, with full-PATCH replay
   on a TCP cut), then HEAD-confirms and cleans up the tar. Workers run
   in parallel via `--workers`.
3. **Manifest** — once all blobs are present, build the OCI image config
   from the collected `diff_id`s, push the config blob, build and push
   the manifest under the target tag plus any `--also-tag`s, validate
   each tag's `Docker-Content-Digest` matches.

### Image layout

One file from HF maps to one OCI image layer. Layers are
`application/vnd.oci.image.layer.v1.tar` (uncompressed) so the layer
digest equals the diff_id by construction. The OCI image config has
no `created` field, so the config digest — and therefore the manifest
digest — is reproducible across runs of the same revision.

### Source of truth

There is no local state file. The registry's HEAD responses
(`/v2/<repo>/blobs/<digest>`, `/v2/<repo>/manifests/<tag>`) are the
authoritative source. Re-running a push of the same revision skips
already-pushed blobs automatically and returns exit 0 if the tag is
already at the expected digest.

---

## Installation

```bash
# pip (stable)
pip install oci-modelcar

# uv (recommended for CI — single binary, fast install)
uv tool install oci-modelcar

# from source (latest unreleased)
pip install git+https://github.com/codanael/oci-modelcar@main
```

Requires Python 3.14+. The runtime dependencies are `requests`,
`urllib3`, and `huggingface_hub` (used for metadata only).

Verify:

```bash
oci-modelcar --help        # shows usage
oci-modelcar push --help   # full flag list
python -c "import oci_modelcar; print(oci_modelcar.__version__)"
```

---

## Your first push

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
export OCI_USERNAME=alice
export OCI_PASSWORD=secret

oci-modelcar push \
  --hf-repo Qwen/Qwen2.5-7B-Instruct \
  --registry registry.acme.com \
  --target-repo models/qwen-7b
```

Output:

```
== Resolving HuggingFace revision ==
HF repo  : Qwen/Qwen2.5-7B-Instruct
Revision : a3f47b09c8d2e9f5b7a8c3d4e5f6789012345678
8 files matched
manifest: sha256:cafef00d...
image:    registry.acme.com/models/qwen-7b:a3f47b09c8d2
manifestDigest=sha256:cafef00d...
imageRef=registry.acme.com/models/qwen-7b:a3f47b09c8d2
imageRefDigest=registry.acme.com/models/qwen-7b@sha256:cafef00d...
```

The image tag defaults to the first 12 characters of the resolved HF
commit SHA. Override with `--target-tag`.

The last three lines are emitted as Azure DevOps task variables when
running with `--log-style azure` (see [CI/CD integration](#cicd-integration)).

---

## CLI reference

### Source (HuggingFace)

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--hf-repo <org/name>` | `HF_REPO` | required | HuggingFace repo (e.g. `Qwen/Qwen2.5-7B-Instruct`) |
| `--hf-revision <ref>` | `HF_REVISION` | `main` | Branch, tag, or 40-char SHA |
| `--hf-endpoint <url>` | `HF_ENDPOINT` | `https://huggingface.co` | Override for HF mirrors / proxies |
| `--allow-patterns <pat...>` | `ALLOW_PATTERNS` | `.safetensors .json .txt .md .model` | Space-separated extensions; only matching files are pulled |

### Target (OCI registry)

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--registry <host>` | `REGISTRY` | required | `host` or `host:port`; scheme inferred (loopback → http, else https) |
| `--target-repo <path>` | `TARGET_REPO` | required | Repository path on the registry (e.g. `models/qwen-7b`) |
| `--target-tag <tag>` | `TARGET_TAG` | sha[:12] | Primary tag |
| `--also-tag <csv>` | `ALSO_TAGS` | — | Comma-separated additional tags (e.g. `latest,prod`) |
| `--layer-prefix <path>` | `LAYER_PATH_PREFIX` | `models/` | Prefix prepended to the file path inside the tar layer |

### Pipeline / disk

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--workers <N>` | `WORKERS` | `1` | Parallel workers, capped at 8 |
| `--spool-dir <path>` | `SPOOL_DIR` | `$TMPDIR/oci-modelcar` | Where downloaded sources and built tar layers are stored |
| `--clean-hf-after-push` | `CLEAN_HF_AFTER_PUSH` | off | Delete each source file after its layer is push-confirmed (minimize disk on ephemeral CI) |

### Retries

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--hf-max-retries <N>` | `HF_MAX_RETRIES` | `10` | Per-file Range-resume retries on transient HF errors |
| `--oci-max-retries <N>` | `OCI_MAX_RETRIES` | `5` | Per-blob full-PATCH replay retries; each retry re-POSTs a fresh upload session |

### Behavior

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--fail-fast` / `--continue-on-error` | `FAIL_FAST` | fail-fast | On first file failure, fail-fast cancels remaining workers; continue-on-error collects failures and exits 7 with no manifest pushed |
| `--force` | `FORCE` | off | Overwrite primary tag if it exists at a different digest (without `--force`, the push refuses with exit 6) |
| `--dry-run` | — | off | Run pre-flight (resolve, list, disk check) and exit 0; no downloads, no uploads |

### Logging

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--log-style text\|azure` | `LOG_STYLE` | auto-detect | `text` for human, `azure` for Azure DevOps logging commands and `task.setvariable` outputs |
| `--verbose` | `LOG_VERBOSE` | off | DEBUG-level diagnostic logs |
| `--quiet` | `LOG_QUIET` | off | Suppress INFO; warnings and errors only |

### Removed in v1.0 (vs v0.5)

- `--state-file` (no local state)
- `--chunk-mib` (single PATCH per blob — chunking removed)
- `--upload-mode` (one mode)

---

## Authentication

### HuggingFace

Resolution priority:

1. `HF_TOKEN` env var
2. `HUGGING_FACE_HUB_TOKEN` env var
3. `~/.cache/huggingface/token` (created by `huggingface-cli login`)

Set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` to skip *all* implicit token sources
(useful when you want to ensure no token is sent for a public-only push).

For gated repos, the user/org must have accepted terms on
`https://huggingface.co/<repo>` before the token will work. The CLI
returns exit code 3 (`GatedRepoError`) with a clear message pointing to
the URL when it sees a `403` with `X-Error-Code: GatedRepo`.

**Security**: HF tokens are NEVER forwarded to redirect targets like S3
or CloudFront. The `_SafeSession.rebuild_auth` strips `Authorization`
on any cross-origin redirect (different netloc).

### OCI registry

Resolution priority:

1. `OCI_USERNAME` + `OCI_PASSWORD` env vars
2. `~/.docker/config.json` (`docker login` writes here)
3. `$XDG_RUNTIME_DIR/containers/auth.json` (`podman login --runtime`)
4. `$XDG_CONFIG_HOME/containers/auth.json` (`podman login` default)

Docker config keys are matched by **longest-prefix** on the normalized
key (strips `https://`, `/v2/`, trailing slashes). So a key like
`artifactory.example/myproject` matches when pushing to
`artifactory.example/myproject/models/qwen` even though the host is
just `artifactory.example`.

If no source matches, the push proceeds **anonymously** with a warning.
This succeeds against open registries (`registry:2` with no auth) and
fails with 401 on protected ones.

### Loopback registries

Hostnames `localhost`, `127.x.x.x`, `::1` automatically use plain HTTP
(no TLS). For other hosts, HTTPS is assumed. Override either way with
an explicit scheme:

```bash
--registry http://insecure-registry.example.com:5000
--registry https://localhost:5000
```

---

## Multi-tag publishing

Push under several tags atomically:

```bash
oci-modelcar push \
  --hf-repo Qwen/Qwen2.5-7B-Instruct \
  --registry registry.acme.com \
  --target-repo models/qwen-7b \
  --target-tag prod-2026.05 \
  --also-tag latest,stable
```

After this, all three tags (`prod-2026.05`, `latest`, `stable`) point to
the same manifest digest. Verify:

```bash
oci-modelcar status --registry registry.acme.com --target-repo models/qwen-7b
# Tags in models/qwen-7b @ registry.acme.com:
#   prod-2026.05  sha256:cafef00d...
#   latest        sha256:cafef00d...
#   stable        sha256:cafef00d...
```

### Tag conflict policy

For the **primary** `--target-tag`:

| Existing tag | `--force` | Action |
|---|---|---|
| Absent | any | Push |
| Present, digest matches | any | Skip job, exit 0 ("already pushed") |
| Present, digest differs | absent | Refuse, exit 6 (`PushError`) |
| Present, digest differs | present | Overwrite |

For `--also-tag` aliases: **silently overwritten**, no conflict check.
This is by design — `--also-tag latest` means "make `latest` point at
this build", same as `docker tag`. If you want safer semantics for
specific aliases, push them as primary tags in separate runs.

---

## Resilience: retries, resume, cancellation

### HF download (Range resume)

When an HF download is interrupted (TCP cut, SSL EOF mid-stream,
proxy timeout, etc.), the worker retries by re-issuing the GET with a
`Range: bytes=N-` header continuing from the last byte written to disk.
Up to `--hf-max-retries` retries (default 10), with full-jitter
exponential backoff capped at 60s.

Retry budget refreshes on progress: if the download walks forward
between two errors, the budget resets, so a long file can survive
arbitrarily many cuts as long as each one yields some bytes.

### OCI push (full-PATCH replay from spool file)

The OCI side uses a single PATCH per blob, with the body sourced from
the local spool file (`<spool>/layers/<file>.tar`). On a transient
failure (TCP cut, SSL EOF, 408/429/5xx, 404 BLOB_UPLOAD_INVALID), the
worker:

1. Sleeps with full-jitter backoff
2. POSTs `/v2/<repo>/blobs/uploads/` to start a **fresh** upload session
3. PATCHes the new Location with the full body, re-read from the spool file

Up to `--oci-max-retries` retries (default 5). The fresh-POST-per-retry
is the key invariant that handles Artifactory HA cluster + load balancer
without sticky session affinity: each PATCH = new TCP request = new LB
routing decision = entire blob lands on one node.

### Resume after partial failure

If the push is killed mid-way (`Ctrl+C`, OOM, network outage), re-run
the *exact same command*. The pre-flight will:

- Skip the job entirely if the manifest tag already points to the
  expected digest
- Otherwise, for each file, the worker's HEAD-blob check skips files
  whose digest is already present in the registry

Force a full re-push (ignoring HEAD-blob skips) with `--force`. This
also overwrites the existing manifest tag.

### Mid-stream cancellation

`Ctrl+C` (SIGINT) and `SIGTERM` are wired to a `stop_event` polled by
all workers at chunk granularity (~1 MiB). A 50 GB download in flight
aborts within ~ms of the signal — not minutes. This is critical for
CI builds with timeouts.

---

## Disk space planning

The push writes to `--spool-dir` only. Two subdirectories are created:

```
<spool_dir>/
  sources/
    <hf_path>           # downloaded file, atomic .partial → rename
  layers/
    <hf_path>.tar       # built tar layer, source for the PATCH body
```

`layers/*.tar` files are ALWAYS deleted after the push HEAD-confirms the
blob in the registry (success or skip path). `sources/*` files are
retained by default so re-runs skip the download — set
`--clean-hf-after-push` to delete them once their corresponding push
HEAD-confirms.

### Estimating required disk

Let `M` = sum of selected file sizes, `L` = size of the largest file's
tar layer (≈ file size + 10240 bytes), `W` = `--workers`.

Without `--clean-hf-after-push`:

```
needed ≈ M (sources, persistent until job end) + W × L (tars in flight)
```

With `--clean-hf-after-push`:

```
needed ≈ W × (max_source + L) × 1.2 (only files in flight)
```

The pre-flight check uses these formulas with safety margin and aborts
the job upfront with `DiskSpaceError` (exit 4) if free space falls short.
The error message includes the chosen formula and the current
`--spool-dir`.

### Examples

A 30 GB model with one 28 GB safetensors and a few small JSONs:

| Mode | `--workers` | Required disk |
|---|---|---|
| Default | 1 | ~30 GB (sources) + 28 GB (tar) = **~58 GB** |
| Default | 4 | ~30 GB + 4 × 28 GB = **~140 GB** (rarely worth it for 1 big file) |
| `--clean-hf-after-push` | 1 | ~28 GB × 2 × 1.2 = **~67 GB** transient, **~0** persistent |

For models with many small-to-medium files (e.g. 8 × 4 GB shards):

| Mode | `--workers` | Required disk |
|---|---|---|
| Default | 4 | ~32 GB sources + 4 × 4 GB tar = **~48 GB** |
| `--clean-hf-after-push` | 4 | ~4 × (4 GB + 4 GB) × 1.2 = **~38 GB** transient |

For very large models on resource-constrained CI runners,
`--clean-hf-after-push --workers 1` minimizes peak disk at the cost of
re-downloading on a re-run.

---

## CI/CD integration

### GitHub Actions

```yaml
name: Push HF model
on:
  workflow_dispatch:
    inputs:
      hf_repo:
        description: 'HuggingFace repo (org/name)'
        required: true
      target_tag:
        description: 'Optional override tag'
        default: ''

jobs:
  push:
    runs-on: ubuntu-latest
    permissions:
      id-token: write   # for cosign keyless signing
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: '3.14'
      - run: pip install oci-modelcar
      - name: Push
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          OCI_USERNAME: ${{ secrets.REGISTRY_USERNAME }}
          OCI_PASSWORD: ${{ secrets.REGISTRY_PASSWORD }}
        run: |
          oci-modelcar push \
            --hf-repo "${{ inputs.hf_repo }}" \
            --registry registry.acme.com \
            --target-repo "models/$(echo "${{ inputs.hf_repo }}" | tr '[:upper:]' '[:lower:]' | tr '/' '-')" \
            ${{ inputs.target_tag && format('--target-tag {0}', inputs.target_tag) || '' }} \
            --workers 4 \
            --clean-hf-after-push
```

### Azure Pipelines

`oci-modelcar` automatically emits Azure DevOps logging commands when
`--log-style azure` is set or detected. Three task variables are set on
success: `manifestDigest`, `imageRef`, `imageRefDigest`.

```yaml
- task: PythonScript@0
  displayName: Push HF model
  inputs:
    scriptSource: inline
    script: |
      import subprocess, sys
      sys.exit(subprocess.call([
        "oci-modelcar", "push",
        "--hf-repo", "$(hfRepo)",
        "--registry", "$(registry)",
        "--target-repo", "models/$(modelName)",
        "--workers", "4",
        "--clean-hf-after-push",
        "--log-style", "azure",
      ]))
  env:
    HF_TOKEN: $(HF_TOKEN)
    OCI_USERNAME: $(REGISTRY_USERNAME)
    OCI_PASSWORD: $(REGISTRY_PASSWORD)

- script: cosign sign $(imageRefDigest)
  displayName: Sign image
  env:
    COSIGN_EXPERIMENTAL: "true"
```

### Disk planning for shared runners

GitHub-hosted runners have ~14 GB free. For models > 5 GB, either:

- Use a self-hosted runner with more disk
- Use `--clean-hf-after-push` and `--workers 1` to minimize peak disk
- Mount a larger volume and set `--spool-dir` to it

---

## Using the image with KServe

KServe supports OCI image volumes natively
([KEP-4639](https://github.com/kubernetes/enhancements/issues/4639))
since 0.13. After pushing your model, reference it as a volume in your
`InferenceService`:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: qwen-7b
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: oci://registry.acme.com/models/qwen-7b@sha256:cafef00d...
```

Use the digest reference (`@sha256:...`) rather than a mutable tag for
reproducible deployments.

The model files appear under `/mnt/models/<layer-prefix>/<filename>`
inside the predictor pod. The default `--layer-prefix models/` puts
them at `/mnt/models/models/...`; override to `--layer-prefix ""` if
your serving framework expects them at the volume root.

---

## Cosign signing & verification

`oci-modelcar` itself does not sign artifacts. Pipe its
`imageRefDigest` output into `cosign`:

```bash
oci-modelcar push \
  --hf-repo Qwen/Qwen2.5-7B-Instruct \
  --registry registry.acme.com \
  --target-repo models/qwen-7b > push.log

# Extract the digest reference
DIGEST_REF=$(grep '^imageRefDigest=' push.log | cut -d= -f2-)

# Sign keyless (CI with OIDC)
cosign sign "$DIGEST_REF"

# Or with a static key
cosign sign --key cosign.key "$DIGEST_REF"
```

Verification (consumer side):

```bash
# Keyless
cosign verify "$DIGEST_REF" \
    --certificate-identity-regexp '^https://github\.com/your-org/' \
    --certificate-oidc-issuer 'https://token.actions.githubusercontent.com'

# Static key
cosign verify --key cosign.pub "$DIGEST_REF"
```

The signature is stored as a separate OCI artifact attached to the
manifest by digest. cosign auto-detects whether to use the OCI v1.1
referrers API or the legacy `:sha256-<digest>.sig` tag scheme.

---

## Troubleshooting

### Push exits with 2 "hf_repo is required"

You forgot `--hf-repo` (or `HF_REPO=`). The same applies to `--registry`
and `--target-repo`. Required flags can be set via env or CLI.

### Push exits with 3 "Repo X is gated"

The HuggingFace repo requires acceptance of terms. Open
`https://huggingface.co/<repo>` in a browser, log in, accept, then
re-run with the same `HF_TOKEN`.

### Push exits with 4 "Need X GB free"

The pre-flight estimated more disk than is available. Options:

- Use `--clean-hf-after-push` (drops persistent budget)
- Use `--spool-dir /path/to/larger/volume`
- Lower `--workers`

### Push exits with 6 "tag exists with different digest"

The target tag already points to a different manifest. Either:

- Re-run with `--force` to overwrite (acknowledges this is intentional)
- Pick a different `--target-tag`

This guard prevents accidentally overwriting a known-good production
tag with a different revision.

### Push exits with 6 "PATCH retries exhausted"

The upload to the registry kept failing across `--oci-max-retries`
attempts. The hint suggests bumping the retry count; in practice this
usually points to a registry-side issue (cluster misconfiguration,
quota, downtime). With `--verbose`, look at which attempt errored and
why.

If you hit this against an Artifactory HA cluster, verify load balancer
sticky session affinity is configured for `/v2/*/blobs/uploads/*`. v1.0
retries with a fresh POST per attempt to mitigate this, but if the LB
is *systematically* cutting connections, even fresh sessions won't
complete.

### Push exits with 7 "N/M files failed"

You used `--continue-on-error`. The summary lists which files failed
and why. The successful files have their blobs in the registry already
(re-run will skip them via HEAD), so the retry cost is only the failed
files. The manifest is NOT pushed when there are failures, even with
`--continue-on-error` — partial models are not published.

### Worker "anonymously" warning

No OCI credentials matched. If your registry requires auth, this will
fail later with 401. Check:

- `OCI_USERNAME` / `OCI_PASSWORD` env vars are set
- `docker login <registry>` was run (writes to `~/.docker/config.json`)
- `podman login <registry>` was run (writes to
  `$XDG_RUNTIME_DIR/containers/auth.json` or
  `~/.config/containers/auth.json`)

The auth lookup uses **longest-prefix matching** on normalized keys.
A key like `artifactory.example/myproject` matches a push to
`artifactory.example/myproject/models/qwen`.

### `ConfigError: no files matched allow_patterns`

Either:

- The HF revision has no files matching the default
  `--allow-patterns .safetensors .json .txt .md .model`
- `--allow-patterns` was set too narrowly

Inspect the repo at `https://huggingface.co/<repo>/tree/<revision>`
and adjust. To pull everything, set
`--allow-patterns ".bin .safetensors .json .txt .md .model .tokenizer"`
or similar (space-separated extensions; matched as suffix).

### "HF SSL EOF mid-stream" warnings

These are expected on flaky networks. The worker resumes via Range and
retries up to `--hf-max-retries` times. As long as the push completes,
these are informational. If they cause exhaustion, bump
`--hf-max-retries` or investigate the network path.

### Stale tar files in `--spool-dir`

After a SIGKILL, the worker's `finally` block doesn't run, leaving
`<spool>/layers/*.tar` files behind. These are safe to delete manually:
they're rebuilt deterministically on the next run. Use a
`tempfile`-backed spool dir on systems that auto-clean.

### Verifying a push manually

```bash
# List tags
oci-modelcar status --registry registry.acme.com --target-repo models/qwen-7b

# Validate manifest + all referenced blobs are present
oci-modelcar validate \
  --registry registry.acme.com \
  --target-repo models/qwen-7b \
  --target-tag a3f47b09c8d2
# Output: "manifest at a3f47b09c8d2 is coherent (8 layers)"
```

---

## Exit codes

For CI/CD branching:

| Code | Class | Meaning |
|---|---|---|
| 0 | — | Success (or no-op skip) |
| 1 | — | Generic error (unhandled exception, signal, etc.) |
| 2 | `ConfigError` | Invalid CLI/env (missing required, bad value) |
| 3 | `GatedRepoError` | HF repo requires terms acceptance |
| 4 | `DiskSpaceError` | Insufficient free disk in `--spool-dir` |
| 5 | `DownloadError` (incl. `RevisionNotFoundError`, `EntryNotFoundError`) | HF download failed (file or revision not found, retries exhausted) |
| 6 | `PushError` | OCI push failed (PATCH retries exhausted, tag conflict refused without `--force`) |
| 7 | `PartialFailure` | `--continue-on-error` mode: some files failed, manifest NOT pushed |

Example shell branching:

```bash
oci-modelcar push --hf-repo X --registry Y --target-repo Z
case $? in
  0)  echo "pushed"        ;;
  3)  echo "accept terms at huggingface.co/X" ;;
  4)  echo "free disk first" ;;
  6)  echo "tag conflict: investigate before --force" ;;
  *)  echo "other failure"; exit 1 ;;
esac
```

---

## Further reading

- [README.md](../README.md) — overview and install
- [CHANGELOG.md](../CHANGELOG.md) — release history
- [docs/superpowers/specs/2026-05-08-oci-modelcar-v1-design.md](./superpowers/specs/2026-05-08-oci-modelcar-v1-design.md) — full v1.0 design rationale
- [OCI Distribution Spec v1.1](https://github.com/opencontainers/distribution-spec/blob/main/spec.md)
- [OCI Image Spec v1.1](https://github.com/opencontainers/image-spec/blob/main/spec.md)
- [KEP-4639: OCI image volumes](https://github.com/kubernetes/enhancements/issues/4639)
