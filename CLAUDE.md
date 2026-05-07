# CLAUDE.md

Guidance for AI assistants and future Claude sessions working on this repo.

## What this is

`oci-modelcar` — a Python CLI that streams HuggingFace models directly into OCI
registries as multi-layer images, suitable for KServe with native OCI image
volumes (KEP-4639). Public package on PyPI, MIT licensed. Repo:
`github.com/codanael/oci-modelcar`. Latest released: v0.1.0.

## Read in this order

1. `README.md` — user-facing overview, install, quick start, OCI compliance notes.
2. `docs/superpowers/specs/2026-05-07-oci-modelcar-design.md` — the full design
   spec with rationale for every choice. The single best file to read before
   making any non-trivial change.
3. `docs/superpowers/plans/2026-05-07-oci-modelcar-implementation.md` — the
   26-task TDD plan that built v0.1.0. Use this as a template for future
   feature plans.
4. `CHANGELOG.md` — what's released, what's planned (search "v0.2 follow-up").

## Architecture

```
src/oci_modelcar/
├── __init__.py         version metadata
├── __main__.py         python -m oci_modelcar entry
├── cli.py              argparse dispatch on argv[0] for push/status/validate
├── config.py           Config dataclass + ConfigError; env+CLI parsing, validation
├── http.py             build_session(), oci_auth_header(), huggingface_token()
├── logging.py          PipelineLogger, TextFormatter, AzureFormatter, FileScopedLogger
├── hf.py               HfClient (resolve_revision, list_files), HfFile, HfStream (Range resume)
├── oci.py              OciClient, BlobDescriptor, ChunkedBlobUpload, push_small_blob,
│                       head_blob, push_manifest, validate_manifest_tag, _is_loopback
├── tar_layer.py        make_tar_info, build_layer_tar_bytes, stream_layer_to
├── manifest.py         build_config_bytes, build_manifest_bytes
├── state.py            JsonStateStore (atomic JSON, threading.Lock), JobState, FileState
├── runner.py           process_one_file, run_push, RunResult
└── tags.py             derive_tag (40-char SHA -> [:12] or sanitized name)
```

Module dependency graph is acyclic. Wiring lives only in `runner.py` and
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
  (`tar_layer.py:make_tar_info`). Reproducibility.
- **File-level resume across processes**, not chunk-level. `hashlib.sha256` is
  not serializable cross-process and OCI upload sessions expire. Intra-process
  retries handle network blips via Range (HF) and PATCH resync via GET (OCI).
- **`Content-Range: N-M`** on PATCH (no `bytes ` prefix, both bounds inclusive).
  This is the OCI Distribution v1.1 format — distinct from RFC 7233.
  `oci.py:_patch_with_retry` line emitting this is wire-spec critical.
- **GET on upload session returns 204** with optional `Range: 0-N` header.
  Header absent → 0 bytes received. Header `0-0` → 1 byte received.
  `oci.py:_resync` handles all three cases.
- **Manifest layers ordered by alphabetical `hf_path`** regardless of worker
  completion order (`runner.py:run_push` reconstructs from `layers_by_idx`).
  This makes the manifest digest deterministic across `--workers` settings.
- **Workers default 1, hard cap 8**. The proxy/HF bottleneck is reached around
  N=4 in practice; larger N adds memory without throughput.
- **Default HF endpoint is `https://huggingface.co`**. The original use case
  was an Artifactory HF proxy, but the v0.1.0 release deliberately makes no
  reference to any specific provider. Use `--hf-endpoint` to override.
- **Loopback registries (`localhost`, `127.x.x.x`, `::1`) auto-use HTTP**
  (`oci.py:_is_loopback`). Remote = HTTPS. Override with explicit
  `http://`/`https://` prefix on `--registry`.
- **CLI dispatches manually on `argv[0]`** (`cli.py:main`), then calls
  `Config.from_env_and_args(argv[1:])` — `Config`'s parser is at top-level
  (no `push` subparser). Don't reintroduce subparsers in `Config`; the split
  is intentional.

## NixOS-specific gotchas

- `pip install ruff` ships a glibc binary; `shell.nix` works around with
  nix-provided `pkgs.ruff`/`pkgs.mypy` taking PATH precedence.
- `ruff format` on this Python-3.14-targeted codebase will normalize
  `except (A, B):` to the unparenthesized `except A, B:` form. The two
  occurrences (in `state.py` and `logging.py`) carry `# fmt: skip` to keep
  parens for readability. **Don't remove the `# fmt: skip`** without first
  running `ruff format` to confirm behavior — both forms are semantically
  equivalent in Python 3.14+ but the parens form reads cleanly to humans.
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
  mocks. The OCI compliance tests are the most spec-critical; don't let
  them lose coverage.
- `tests/integration/` — multi-module pipeline tests with mocked HTTP. Note
  `tests/integration/test_runner_multi.py` currently has minimal coverage of
  the parallel path — broader concurrency testing is a v0.2 follow-up
  (cross-thread state writes contention, fail-fast cancellation timing).
- `tests/e2e/` — real HuggingFace `hf-internal-testing/tiny-random-LlamaForCausalLM`
  pinned at SHA `9fb191250dd56d0ba7ec9785a025ed29c03d5998`, against a Docker
  `registry:2`. Marked `@pytest.mark.e2e`, gated. Update the pinned SHA only
  if the upstream repo retired the commit (rare for `hf-internal-testing`).

## Things NOT to do

- Don't add `created` to OCI image config (breaks idempotence).
- Don't compress layers (gzip/zstd) without recomputing both digest AND diff_id.
- Don't introduce mutable default factories on dataclasses without
  `field(default_factory=...)`.
- Don't widen runtime dependencies casually. v0.1.0 ships with **only**
  `requests` and `urllib3`. Any new dep must be justified in the design spec.
- Don't `pip install` Rust-based or mypyc-compiled tools (ruff, mypy) on
  NixOS dev — use `pkgs.<tool>` in `shell.nix`.
- Don't lower `requires-python` below 3.14 — the floor was a deliberate user
  choice. If you need to lower it for compatibility, you'll also need to
  audit Python 3.14-only syntax (the unparenthesized `except A, B:` clauses,
  any future `Self` from `typing`, free-threaded primitives, etc.).
- Don't bypass `OciClient` when constructing registry URLs (the `validate`
  sub-command was previously broken because of this — fixed in commit
  `688659b`). Always go through `OciClient.url(...)`.
- Don't hardcode `HF_TOKEN` or `OCI_PASSWORD` anywhere. Auth resolution lives
  in `http.py`; extend it there if a new auth source is needed.
- Don't commit `.venv/`, `shell.nix`, or `stream_modelcar.py` (the original
  prototype, kept locally for reference). All are in `.gitignore`.

## Common debugging tips

- **Wire-format issues**: enable `--verbose` to log all OCI session
  transitions (offset progression, location URL changes).
- **Reproducibility check**: run a push twice with `--force`. Manifest digest
  must be identical. If not, something has crept in (likely `created` field,
  or non-deterministic tar header).
- **Resume issues**: inspect `~/.local/state/oci-modelcar/state.json`
  (configurable via `--state-file`). Each job carries `files{}` with
  `digest`, `diff_id`, and `layer_size`. The `layer_size` field was added in
  v0.1.0 release fix — older state files may lack it; the runner re-pushes
  affected files automatically.
- **Network blips**: HF Range and OCI resync should mask single failures.
  If retries exhaust, the file fails. Check warnings with `--verbose`.

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

# Inspect what was pushed
nix-shell ./shell.nix --command "skopeo --insecure-policy inspect --tls-verify=false docker://localhost:5000/demo/qwen-05b:<sha12>"

# View state
cat ~/.local/state/oci-modelcar/state.json | python3 -m json.tool

# Find all TODOs and known limitations
grep -rn -E "(TODO|FIXME|v0\.2)" src/ docs/

# Re-run the whole test suite
nix-shell ./shell.nix --command "python3.14 -m pytest tests/ -v"
```

## When stuck

- The design spec has the rationale for every choice. Read it before second-
  guessing a pattern in the code.
- The OCI Distribution v1.1 and Image Spec v1.1 are the source of truth for
  wire format. URLs in `docs/superpowers/specs/...` reference the relevant
  sections.
- The HF API surface in use:
  - `GET /api/models/{repo}` → `{sha: ...}` for the default branch
  - `GET /api/models/{repo}/revision/{rev}` → `{sha: ...}` for a specific revision
  - `GET /api/models/{repo}/tree/{rev}?recursive=true` → list of files
  - `GET /{repo}/resolve/{rev}/{path}` → file content (Range-supportable)
- For "is this Python 3.14 valid?" questions, parse with `ast.parse(...)` to
  see what the AST does. Some surprising syntactic forms are accepted.
- For OCI behavior verification, `registry:2` is your friend — it implements
  the full OCI Distribution spec including the resume semantics.
