# oci-modelcar v1.2 — Glob-aware file filtering with `--ignore-patterns`

**Status**: design approved, implementation in progress.
**Author**: Anael Latassa with Claude.
**Date**: 2026-05-11.
**Extends**: `2026-05-08-oci-modelcar-v1-design.md` (v1.0), `2026-05-11-cross-run-layer-reuse-design.md` (v1.1).

## 1. Why

Some HuggingFace repos ship more than one complete weight set in the same
root directory, sharing the `.safetensors` extension. The motivating case
is `mistralai/Mistral-Medium-3.5-128B`, which packs both:

- HF transformers layout — `model-*.safetensors` (~134 GB) +
  `model.safetensors.index.json` + `config.json`. Consumed by vLLM,
  SGLang, transformers — the runtimes a KServe ModelCar typically
  dispatches to.
- Mistral native layout — `consolidated-*.safetensors` (~134 GB) +
  `consolidated.safetensors.index.json` + `params.json`. Consumed by
  `mistral-inference` and Mistral's own tooling.

The two sets are distinguished by filename **prefix** only. Today's
`--allow-patterns` is suffix-only (`download.py:list_files` calls
`entry.path.endswith(ext)`), so a single push cannot exclude one set
without excluding the other. For Mistral-Medium-3.5-128B that means
either pushing ~267 GB (both layouts) or hand-curating an unreadable
suffix list. The cost surfaces directly on every CI re-push and in
every consumer's registry pull.

Other repos with the same shape: any Llama-derived release that ships
the original Meta format under `original/` alongside the HF format at
the root, and quantized/precision variants packaged in subdirectories.

## 2. Goals and non-goals

**Goals**

- Let users include or exclude specific files in an HF repo with
  expressive patterns, not just by extension.
- Match `huggingface_hub.snapshot_download`'s `allow_patterns` /
  `ignore_patterns` semantics, so user mental models transfer.
- Preserve exact backward compatibility for the existing default
  (`--allow-patterns ".safetensors .json .txt .md .model"`) and for any
  caller already passing bare extensions.
- Keep the surface area tiny: one new flag, one new env var, one new
  Config field, one helper function. No data-flow or wire-format
  changes.

**Non-goals**

- A `--variant hf|native` preset. Useful ergonomically but couples the
  tool to one repo's conventions. Defer.
- A `**` recursive glob operator. Standard `fnmatch.fnmatchcase` `*`
  already matches across `/`, so plain `*` suffices.
- Case-insensitive matching. HF paths are case-sensitive (Git-backed).
- Per-file allowlists driven by a YAML/JSON config. CLI + env keeps
  parity with the rest of the surface.

## 3. The matching semantics

Inclusion rule:

```
file_included(path) ==
    any(fnmatchcase(path, p) for p in allow_compiled) and
    not any(fnmatchcase(path, p) for p in ignore_compiled)
```

That is: a path passes when at least one `allow` pattern matches **and**
zero `ignore` patterns match. Ignore wins over allow when both match —
the same precedence `huggingface_hub.snapshot_download` uses.

`fnmatch.fnmatchcase` is the right primitive:

- It is the case-sensitive variant of `fnmatch`. We need
  case-sensitivity because HF paths are stored verbatim, and Python's
  `fnmatch.fnmatch` is case-insensitive on Windows (does
  `os.path.normcase` on both sides). We want identical behavior on every
  OS.
- Its `*` matches any character **including `/`**. So `consolidated-*`
  matches `consolidated-00001-of-00003.safetensors`, `images/*` matches
  anything under `images/`, and `*.json` matches both
  `model.safetensors.index.json` and `subdir/foo.json`. This is the
  same `*` semantics `huggingface_hub` documents.
- It is in the standard library; no new dependency.

## 4. The backwards-compat heuristic

The existing default `_DEFAULT_ALLOW = ".safetensors .json .txt .md .model"`
must continue to behave identically. Today these are treated as
suffixes (`endswith`). Under glob semantics, a bare `.safetensors`
matches nothing — it's a literal filename starting with a dot.

We solve this with a one-line compile step applied to both `allow` and
`ignore` lists at the call site:

```python
def _compile_filter(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """A token containing none of *?[ is rewritten as *<token>.
    A token containing any of *?[ is kept verbatim."""
    out: list[str] = []
    for p in patterns:
        if any(c in p for c in "*?["):
            out.append(p)
        else:
            out.append(f"*{p}")
    return tuple(out)
```

Examples after compilation:

| User input | Compiled | Matches |
|---|---|---|
| `.safetensors` | `*.safetensors` | every `*.safetensors` at any depth |
| `consolidated-*` | `consolidated-*` | files whose path begins with `consolidated-` |
| `original/*.safetensors` | `original/*.safetensors` | safetensors under `original/` |
| `images/*` | `images/*` | anything under `images/` |
| `model-*.safetensors` | `model-*.safetensors` | sharded HF weights |
| `*.bin` | `*.bin` | every `*.bin` at any depth |

Existing users — and the default — notice **zero** behavior change. New
users get full glob expressiveness by adding `*`, `?`, or `[…]` to any
token.

## 5. Surface

**CLI flag** (mirrors `--allow-patterns`):

```
--ignore-patterns <pat...>      space-separated; default empty
```

**Env var** (mirrors `ALLOW_PATTERNS`):

```
IGNORE_PATTERNS                 space-separated; default empty
```

**Config field** (`config.py:Config`):

```python
ignore_patterns: tuple[str, ...] = field(default_factory=tuple)
```

Parsed by `Config.from_env_and_args` the same way as `allow_patterns`:

```python
ignore_patterns=tuple(
    (ns.ignore_patterns or _envstr("IGNORE_PATTERNS", "")).split()
),
```

The empty default is intentional: a missing `IGNORE_PATTERNS` env var
must compile to an empty tuple so the `not any(...)` branch
short-circuits to `True` for every file.

**Plumbing**: `Pipeline._preflight` passes `cfg.ignore_patterns` into
`HfDownloader.list_files`, whose signature gains a third parameter:

```python
def list_files(
    self,
    repo: str,
    revision: str,
    allow: tuple[str, ...],
    ignore: tuple[str, ...] = (),
) -> list[HfFile]:
```

Defaulting `ignore=()` keeps existing test call sites working.

## 6. The user-facing example (Mistral-Medium-3.5-128B)

To push only the HF layout — what KServe + vLLM consumes — and skip
the parallel Mistral native layout and the README image assets:

```bash
oci-modelcar push \
  --hf-repo mistralai/Mistral-Medium-3.5-128B \
  --registry registry.example.com \
  --target-repo models/mistral-medium-3.5 \
  --ignore-patterns "consolidated-*" "params.json" "images/*"
```

Resulting file set (∼134 GB instead of ∼267 GB):

```
model-00001-of-00003.safetensors      49.8 GB
model-00002-of-00003.safetensors      49.8 GB
model-00003-of-00003.safetensors      34.0 GB
model.safetensors.index.json          256 KB
config.json                           1.92 KB
generation_config.json                131 B
tokenizer.json                        17.1 MB
tokenizer_config.json                 21.2 KB
processor_config.json                 660 B
chat_template.jinja                   13.5 KB
tekken.json                           16.3 MB
README.md, SYSTEM_PROMPT.txt          (docs)
```

To push only the Mistral native layout instead:

```bash
oci-modelcar push ... \
  --ignore-patterns "model-*" "config.json" "generation_config.json" "images/*"
```

## 7. What is NOT affected

All v1.0 / v1.1 invariants survive because filtering is upstream of layer
construction:

- **Layer mediaType, no `created` field, mtime=0, deterministic
  manifest digest** — untouched. Push the same file set twice with the
  same flags → identical manifest digest.
- **Layer ordering by `hf_path`** (`pipeline.py:_assemble_manifest`)
  — unchanged. The filter shrinks the input set; the sort happens
  after.
- **Reuse map and HEAD-based resumability** (v1.1) — unchanged. The
  `(hf_path, hf_sha256)` keys in the reuse map only cover files that
  survived the filter, so a re-push of an unchanged filtered subset
  still hits every layer.
- **OCI wire format, exit codes, sub-commands, registry pre-flight,
  cross-origin auth strip** — none touched.

Changing the file set **does** change the manifest digest of course;
that's the intended behavior of "push a different set of files."

## 8. Error message

The existing "no files matched" error in `pipeline.py:_preflight` gets
the ignore tuple appended for diagnosability:

```python
raise ConfigError(
    f"no files matched allow_patterns {self.cfg.allow_patterns} "
    f"with ignore_patterns {self.cfg.ignore_patterns} "
    f"in {self.cfg.hf_repo}@{revision}"
)
```

When `ignore_patterns` is the empty tuple `()`, the second clause still
prints — explicit and unambiguous in CI logs.

## 9. Implementation outline

Files touched:

- `src/oci_modelcar/download.py`
  - new private `_compile_filter(patterns) -> tuple[str, ...]`
  - new private `_path_matches(path, compiled) -> bool` (single
    `any(fnmatchcase(...))`) or inlined at the call site
  - `list_files` gains `ignore: tuple[str, ...] = ()`; the `endswith`
    line is replaced with the allow ∧ ¬ignore check
- `src/oci_modelcar/config.py`
  - new field `ignore_patterns: tuple[str, ...] = field(default_factory=tuple)`
  - new arg `--ignore-patterns` in `_build_parser`
  - new env-var read in `from_env_and_args`
- `src/oci_modelcar/pipeline.py`
  - `_preflight` passes `self.cfg.ignore_patterns` into
    `list_files`
  - error message extended
- `tests/unit/test_download.py`
  - new table-driven test for `_compile_filter`
  - extend existing `test_list_files_filters_by_allow_patterns` with
    glob + ignore cases
- `tests/unit/test_config.py`
  - assert `--ignore-patterns` parses, env var resolves, default is
    `()`
- `tests/integration/test_pipeline_filtering.py` (new)
  - mocked HF tree with both `model-*.safetensors` and
    `consolidated-*.safetensors`; assert
    `--ignore-patterns "consolidated-*"` produces a manifest with
    exactly the HF-layout layers
- `docs/user-guide.md`
  - new row for `--ignore-patterns` in the source-side flag table
  - new short section "Filtering when a repo ships multiple weight
    formats" with the Mistral example
- `README.md`
  - one-line mention under features
- `CHANGELOG.md`
  - `[Unreleased]` entry
- `pyproject.toml`
  - bump to `1.2.0` at release time (not in this PR; release lifecycle
    is decoupled from feature merge per CLAUDE.md)

Estimated diff size: ~60–80 lines of source + ~120 lines of tests +
docs.

## 10. Risks and rejected alternatives

- **Add `--ignore-patterns` with `endswith` semantics only.** Rejected.
  Cannot express "files starting with `consolidated-`" — the very case
  motivating the feature. Would require shipping a v1.3 that revisits
  the matcher.
- **Treat all patterns as full globs without the backwards-compat
  heuristic.** Rejected. Breaks every existing caller that passes
  `.safetensors` — including `_DEFAULT_ALLOW`. Migration would require
  a major version bump and CHANGELOG warning.
- **Use `pathlib.PurePath.match()` instead of `fnmatch.fnmatchcase`.**
  Rejected. `PurePath.match` anchors at the **end** of the path and
  treats `/` specially — `original/*` would not match
  `original/foo.bin` because `*` doesn't span beyond the segment. The
  HF Hub convention is `fnmatch`-style flat matching, and we want
  parity.
- **Add a `--variant {hf|native}` preset.** Deferred. Can be added on
  top of glob filtering once we observe whether real-world repos
  cluster on shared conventions worth canonicalizing.

## 11. Test strategy

Unit (`tests/unit/test_download.py`):

| Case | Inputs | Expected |
|---|---|---|
| bare ext compiles to suffix | `_compile_filter((".safetensors",))` | `("*.safetensors",)` |
| glob stays verbatim | `_compile_filter(("consolidated-*",))` | `("consolidated-*",)` |
| mixed | `_compile_filter((".json", "images/*"))` | `("*.json", "images/*")` |
| `?` triggers passthrough | `_compile_filter(("?ata.bin",))` | `("?ata.bin",)` |
| `[` triggers passthrough | `_compile_filter(("file[0-9].bin",))` | `("file[0-9].bin",)` |
| `list_files` honors default suffix behavior | `allow=(".safetensors",)` only | matches every `*.safetensors` |
| `list_files` honors ignore glob | `allow=(".safetensors",), ignore=("consolidated-*",)` | drops `consolidated-*` |
| ignore wins | file matches both `*.safetensors` and `consolidated-*` | excluded |
| empty ignore | `ignore=()` | identical to no ignore arg |

Unit (`tests/unit/test_config.py`):

- `--ignore-patterns "a b"` → `cfg.ignore_patterns == ("a", "b")`
- env var `IGNORE_PATTERNS="a b"` (no CLI flag) → same
- neither set → `cfg.ignore_patterns == ()`

Integration (`tests/integration/test_pipeline_filtering.py`):

- HF tree mock with `model-*.safetensors` (3 shards) and
  `consolidated-*.safetensors` (3 shards), plus shared `config.json`
- Run pipeline with `--ignore-patterns "consolidated-*"`
- Assert exactly the three `model-*` shards plus `config.json` end up
  in the manifest layers; assert manifest digest is deterministic
  across two consecutive runs with `--force`

## 12. Out of scope for this design

- Annotating excluded files in the manifest (e.g. for audit). The
  manifest already records exactly what's there; what's missing is by
  definition uninteresting to the consumer.
- `huggingface_hub.HfApi` upstream filtering. Today we list the whole
  tree and filter in Python. For repos with thousands of files this
  could matter; for the Mistral case (10 weight files) it does not.
  Defer until profiling shows the listing call as the bottleneck.
- File-set persistence across runs. Filters are stateless CLI args.
  Users who want reproducibility encode them in their CI script (the
  current state of the world for `--allow-patterns`).
