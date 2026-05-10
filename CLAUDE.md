# CLAUDE.md

Guidance for AI assistants and future Claude sessions working on this repo.

## What this is

`oci-modelcar` — a Python CLI that pushes HuggingFace models to OCI registries
as multi-layer images, suitable for KServe with native OCI image volumes
(KEP-4639). Public package on PyPI, MIT licensed. Repo:
`github.com/codanael/oci-modelcar`. Latest released: v1.0.0.

## Read in this order

1. `README.md` — user-facing overview, install, quick start, OCI compliance notes.
2. `docs/superpowers/specs/2026-05-08-oci-modelcar-v1-design.md` — the full
   v1.0 design spec with rationale for every choice. The single best file to
   read before making any non-trivial change.
3. `docs/superpowers/plans/2026-05-08-oci-modelcar-v1.md` — the 11-phase TDD
   plan that built v1.0. Use this as a template for future feature plans.
4. `CHANGELOG.md` — what's released, what changed from v0.x to v1.0.

## Architecture

v1.0 uses a per-file pipeline: each HF file is downloaded to `--spool-dir`,
wrapped in a tar layer, then pushed to the registry in a single PATCH per blob
(Jib-style). On PATCH failure the full PATCH is replayed from the local spool
file. The registry HEAD is the source of truth for resumability; no local state
file is needed. `huggingface_hub.HfApi` handles metadata (revision resolve, file
listing, LFS sha256 detection); bytes are streamed by our own code so mid-stream
cancellation works on multi-GB downloads.

```
src/oci_modelcar/
├── __init__.py     version metadata
├── __main__.py     python -m oci_modelcar entry
├── cli.py          argparse dispatch; push/status/validate sub-commands; exit codes
├── config.py       Config dataclass + ConfigError; env+CLI parsing, validation
├── errors.py       GatedRepoError, RevisionNotFoundError, EntryNotFoundError,
│                   DiskSpaceError, PushError, PartialFailureError; exit code map
├── http.py         _SafeSession (cross-origin Authorization stripping),
│                   hf_session(), oci_session(), oci_auth_header(), hf_token()
├── logging.py      PipelineLogger, TextFormatter, AzureFormatter
├── download.py     HfDownloader (HfApi metadata + streamed bytes), atomic write,
│                   Range-200 fallback
├── layer.py        make_tar_info, build_layer_tar, tar_layer_size; TarLayerInfo
├── manifest.py     build_config_bytes, build_manifest_bytes, derive_tag
├── registry.py     RegistryClient, BlobDescriptor, SinglePatchUpload,
│                   push_small_blob, head_blob, push_manifest, tag_conflict_policy,
│                   _is_loopback
└── pipeline.py     FileWorker (download→tar→push→cleanup), Pipeline (pre-flight,
                    disk check, ThreadPoolExecutor, fail-fast, manifest assembly),
                    RunResult
```

Module dependency graph is acyclic. Wiring lives only in `pipeline.py` and
`cli.py`. Each module is independently testable.

## Dev environment (NixOS)

A `shell.nix` exists at the repo root but is **gitignored**. It provisions
Python 3.14, ruff, mypy, pre-commit, skopeo, gh, git. Always run commands via:

```bash
nix-shell ./shell.nix --command "<command>"
```

The `shellHook` auto-creates `.venv/`, runs `pip install -e '.[dev,e2e]'`, and
runs `pre-commit install`. It also forces nix-installed `ruff`/`mypy` ahead of
venv copies in `PATH` because pip-installed ruff is a glibc-linked binary that
won't run on NixOS.

If `shell.nix` is missing on a fresh checkout, recreate it from the design spec
§13 or copy from another dev's machine — it's not in git on purpose.

## Quality gates (must all pass before commit)

Pre-commit hooks (`.pre-commit-config.yaml`) enforce:
- `ruff check --fix`
- `ruff format`
- `mypy --strict src/`
- `pytest -m "not e2e" -q`

All hooks use `language: system` so they pick up nix-installed binaries
naturally. The pytest and mypy hooks invoke `python3.14 -m ...` rather than
bare `pytest`/`mypy` because nix's PATH may resolve to Python 3.13 binaries
that can't see the venv install.

Manual full check:
```bash
nix-shell ./shell.nix --command "ruff check . && ruff format --check . && python3.14 -m mypy --strict src/"
nix-shell ./shell.nix --command "python3.14 -m pytest tests/unit tests/integration -m 'not e2e' --cov=oci_modelcar --cov-report=term"
nix-shell ./shell.nix --command "python3.14 -m pytest tests/e2e -m e2e -v"   # needs Docker + network
```

## Conventions

- **TDD strict**: write the failing test first, run to confirm failure, write
  minimal impl, run to confirm pass, commit. The plan in `docs/superpowers/plans/`
  encodes this rhythm — replicate it for new features.
- **Conventional commits**: `feat(scope): ...`, `fix(scope): ...`, `docs:`,
  `ci:`, `chore:`, `test:`, `style:`. See `git log --oneline` for examples.
- **AI-assisted commits**: include the `Co-Authored-By: Claude ...` trailer.
- **One feature, one branch**: develop on `impl/<feature>` or `fix/<topic>`,
  fast-forward merge to `main`, push tags for releases.
- **Default-no comments**. Comments only for non-obvious WHY (hidden constraint,
  workaround, surprising invariant). The codebase mostly avoids comments by design.

## Locked design decisions — do NOT change without re-reading the spec

These choices are load-bearing. Changing them requires updating the spec and
explaining why in the PR.

- Layer mediaType is **uncompressed** `application/vnd.oci.image.layer.v1.tar`.
  This makes `layer.digest == diff_id` by construction (OCI image-spec v1.1).
  Don't add gzip — safetensors compress ~2%, gzip burns CPU and breaks the
  digest equality.
- **No `created` field** in OCI image config (`manifest.py:build_config_bytes`).
  Preserves byte-identical config across runs → identical config digest →
  identical manifest digest. Adding `created` breaks idempotence.
- **mtime=0, uid=gid=0, uname=gname=""** in all tar headers
  (`layer.py:make_tar_info`). Reproducibility.
- **Single PATCH per blob from local spool file (Jib-style replay-on-cut).**
  `registry.py:SinglePatchUpload` issues one PATCH with upfront `Content-Length`.
  On failure the full PATCH is replayed from the spool file. This eliminates
  per-PATCH LB routing decisions on Artifactory HA clusters. Don't reintroduce
  chunked mode without updating the spec.
- **`Content-Range: N-M`** on PATCH (no `bytes ` prefix, both bounds inclusive).
  This is the OCI Distribution v1.1 format — distinct from RFC 7233.
  `registry.py:SinglePatchUpload` line emitting this is wire-spec critical.
- **Manifest layers ordered by alphabetical `hf_path`** regardless of worker
  completion order (`pipeline.py:Pipeline.run` reconstructs from `layers_by_idx`).
  This makes the manifest digest deterministic across `--workers` settings.
- **Workers default 1, hard cap 8**. The proxy/HF bottleneck is reached around
  N=4 in practice; larger N adds disk pressure and memory without throughput.
- **Default HF endpoint is `https://huggingface.co`**. Use `--hf-endpoint` to
  override (e.g. for an Artifactory HF proxy).
- **Loopback registries (`localhost`, `127.x.x.x`, `::1`) auto-use HTTP**
  (`registry.py:_is_loopback`). Remote = HTTPS. Override with explicit
  `http://`/`https://` prefix on `--registry`.
- **Cross-origin Authorization stripping in `_SafeSession`** (`http.py`).
  HF Bearer tokens must NOT be forwarded on HF→S3/CloudFront redirects.
  `_SafeSession.rebuild_auth` strips the Authorization header when origin
  changes. Do not remove this guard.
- **`huggingface_hub.HfApi` for metadata only; bytes via our streamer.**
  `HfApi` is used for revision resolution and file listing (including LFS
  sha256 detection). Download bytes go through our own `HfDownloader` so
  mid-stream cancellation (via `threading.Event`) works on multi-GB files.
  Do not replace the byte streaming with `hf_hub_download` — it can't be
  cancelled mid-stream.
- **Atomic write semantics for downloaded files** (`.partial` → rename).
  `download.py:HfDownloader` writes to `<path>.partial` and renames on
  completion. A partial file left on disk is automatically overwritten on
  retry; a complete file is never re-downloaded if `--clean-hf-after-push`
  is not set.
- **`--clean-hf-after-push` is opt-in; default keeps source files.**
  This allows re-running with `--force` without re-downloading. Set it in
  space-constrained environments (containers with small ephemeral storage).
- **`errors.py` exit codes 0..7** mapped to specific exception classes.
  Exit code contract: 0=success, 1=generic/unhandled, 2=`ConfigError`,
  3=`GatedRepoError`, 4=`DiskSpaceError`, 5=`DownloadError` (incl.
  `RevisionNotFoundError`, `EntryNotFoundError`), 6=`PushError`,
  7=`PartialFailureError`. Don't renumber without updating `cli.py`,
  `docs/user-guide.md` (the §Exit codes table), and the spec.
- **Registry HEAD is the source of truth for resumability.** No local state
  file (`state.json` was removed in v1.0). Blobs already present in the
  registry are detected via HEAD check and skipped. `--force` bypasses HEAD.
- **CLI uses argparse sub-commands** (`push`, `status`, `validate`) dispatched
  by `cli.py:main`. `Config` has its own parser scoped to `push` arguments.
  Don't collapse sub-commands into `argv[0]` dispatch (reverted from v0.x).
- **`derive_tag` lives in `manifest.py`** (migrated from the removed `tags.py`).
  40-char SHA → first 12 chars; other revision strings are sanitized.
  Don't split it out again.

## NixOS-specific gotchas

- `pip install ruff` ships a glibc binary; `shell.nix` works around with
  nix-provided `pkgs.ruff`/`pkgs.mypy` taking PATH precedence.
- `ruff format` on this Python-3.14-targeted codebase will normalize
  `except (A, B):` to the unparenthesized `except A, B:` form. Any
  occurrence carrying `# fmt: skip` keeps parens for readability.
  **Don't remove `# fmt: skip`** without first running `ruff format` to
  confirm behavior — both forms are semantically equivalent in Python
  3.14+ but the parens form reads cleanly to humans.
- Tests' pytest fixtures sometimes need `werkzeug.wrappers.Response` directly
  to set `Content-Length` headers that werkzeug would otherwise recompute.
  See `tests/unit/test_oci_misc.py` for the pattern.

## How to add a new feature

1. **Update or extend the design spec** if scope is non-trivial. Save under
   `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`.
2. **Write an implementation plan** with TDD-shaped bite-sized tasks. Save
   under `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`. The v0.1.0 plan is
   a good template — verbose, with full code per step, no placeholders.
3. **Branch**: `git checkout -b feat/<topic>`
4. **Implement task by task** following TDD. Commit per task with
   conventional commit messages.
5. **Run all gates** locally (see "Quality gates" above).
6. **Push, open PR, merge** when CI is green.
7. **Update `CHANGELOG.md`** under `[Unreleased]`.
8. **For a release**: bump `pyproject.toml` version, add a CHANGELOG section,
   `git tag -a vX.Y.Z && git push origin vX.Y.Z`. The `release.yml` workflow
   handles PyPI publish (Trusted Publishing OIDC, no token needed) and GH
   Release.

## Test layout

- `tests/unit/` — fast (sub-second) per-module tests with `pytest-httpserver`
  mocks. The registry compliance tests are the most spec-critical; don't let
  them lose coverage. ~149 tests across all v1 modules.
- `tests/integration/` — multi-module pipeline tests with mocked HTTP.
  Covers fail-fast cancellation timing and partial failure (continue-on-error).
- `tests/e2e/` — real HuggingFace `hf-internal-testing/tiny-random-LlamaForCausalLM`
  pinned at SHA `9fb191250dd56d0ba7ec9785a025ed29c03d5998`, against a Docker
  `registry:2`. Marked `@pytest.mark.e2e`, gated. Update the pinned SHA only
  if the upstream repo retired the commit (rare for `hf-internal-testing`).

## Things NOT to do

- Don't add `created` to OCI image config (breaks idempotence).
- Don't compress layers (gzip/zstd) without recomputing both digest AND diff_id.
- Don't introduce mutable default factories on dataclasses without
  `field(default_factory=...)`.
- Don't widen runtime dependencies casually. v1.0 ships with `requests`,
  `urllib3`, and `huggingface_hub`. Any new dep must be justified in the spec.
- Don't `pip install` Rust-based or mypyc-compiled tools (ruff, mypy) on
  NixOS dev — use `pkgs.<tool>` in `shell.nix`.
- Don't raise `requires-python` above 3.11 without a concrete reason. The
  v1.0 codebase only relies on `typing.Self` (3.11) and union/generic
  builtins; no 3.12+ syntax is used. Verify any change with `mypy --strict`
  on the lowest supported version. The CI matrix tests 3.11/3.12/3.13/3.14.
- Don't bypass `RegistryClient` when constructing registry URLs. Always go
  through `RegistryClient.url(...)`.
- Don't hardcode `HF_TOKEN` or `OCI_PASSWORD` anywhere. Auth resolution lives
  in `http.py`; extend it there if a new auth source is needed.
- Don't remove the `_SafeSession.rebuild_auth` cross-origin Authorization
  stripping — it prevents leaking Bearer tokens to S3/CloudFront.
- Don't replace `HfDownloader` byte streaming with `hf_hub_download` — it
  can't be cancelled mid-stream via `threading.Event`.
- Don't reintroduce `state.json` or `JsonStateStore`. The registry HEAD is the
  source of truth. If you need cross-run state, justify it in the spec first.
- Don't commit `.venv/`, `shell.nix`, or `stream_modelcar.py` (the original
  prototype, kept locally for reference). All are in `.gitignore`.

## Common debugging tips

- **Wire-format issues**: enable `--verbose` to log all OCI PATCH transitions
  (upload location URL, retry counts, replay from spool).
- **Reproducibility check**: run a push twice with `--force`. Manifest digest
  must be identical. If not, something has crept in (likely `created` field,
  or non-deterministic tar header).
- **Disk space**: the spool directory grows by ~2× the largest layer per worker.
  Use `--clean-hf-after-push` to reclaim space after each layer is pushed.
  The pre-flight check will refuse to start if space is insufficient.
- **Token issues**: set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` to skip implicit
  token sources and verify you're using the right `HF_TOKEN`.
- **Network blips**: OCI PATCH is replayed from spool file on cut. If retries
  exhaust (`--oci-max-retries`, default 5), the file fails.
  Check warnings with `--verbose`.

## Useful commands

```bash
# Local registry for E2E
docker run -d --rm --name oci-modelcar-reg -p 5000:5000 registry:2
docker stop oci-modelcar-reg

# Manual push to local registry
nix-shell ./shell.nix --command "oci-modelcar push \
  --hf-repo Qwen/Qwen2.5-0.5B-Instruct \
  --registry localhost:5000 \
  --target-repo demo/qwen-05b"

# Push with disk cleanup after each layer
nix-shell ./shell.nix --command "oci-modelcar push \
  --hf-repo Qwen/Qwen2.5-0.5B-Instruct \
  --registry localhost:5000 \
  --target-repo demo/qwen-05b \
  --clean-hf-after-push"

# Inspect what was pushed
nix-shell ./shell.nix --command "skopeo --insecure-policy inspect --tls-verify=false docker://localhost:5000/demo/qwen-05b:<sha12>"

# Find all TODOs and known limitations
grep -rn -E "(TODO|FIXME)" src/ docs/

# Re-run the whole test suite
nix-shell ./shell.nix --command "python3.14 -m pytest tests/ -v"
```

## When stuck

- The design spec (`docs/superpowers/specs/2026-05-08-oci-modelcar-v1-design.md`)
  has the rationale for every choice. Read it before second-guessing a pattern.
- The OCI Distribution v1.1 and Image Spec v1.1 are the source of truth for
  wire format. URLs in `docs/superpowers/specs/...` reference the relevant
  sections.
- The HF surface in use:
  - `huggingface_hub.HfApi.model_info()` → resolve revision SHA
  - `huggingface_hub.HfApi.list_repo_tree()` → file listing with LFS sha256
  - `GET /{repo}/resolve/{rev}/{path}` → file bytes (Range-supportable),
    called directly by `download.py:HfDownloader` (not via HfApi, for
    cancellation support)
- For "is this Python 3.14 valid?" questions, parse with `ast.parse(...)` to
  see what the AST does. Some surprising syntactic forms are accepted.
- For OCI behavior verification, `registry:2` is your friend — it implements
  the full OCI Distribution spec including the PATCH-from-zero semantics.
