# oci-modelcar v1.1 — Cross-run layer reuse (HEAD before download)

**Status**: implemented.
**Author**: Anael Latassa with Claude.
**Date**: 2026-05-11.
**Extends**: `2026-05-08-oci-modelcar-v1-design.md` (v1.0 design).

## 1. Why

The v1.0 pipeline runs four phases per HuggingFace file (download → tar →
HEAD → push). On a re-push of an unchanged model — common in CI when the
image rebuilds without the HF revision changing — every byte is
re-downloaded from HF before the HEAD check at phase c can detect that
the resulting layer blob is already in the registry. For a 30 GB model
this is roughly 30 GB of waste on every CI run.

Two regressions also surfaced after v1.0:

1. The per-file progress logs that v0.x emitted via `ProgressEmitter`
   were dropped in the rewrite. After "X files matched" the tool went
   silent for the entire push.

2. `HfDownloader.download()` did not skip cached files in
   `<spool>/sources/<path>`, contradicting the CLAUDE.md statement
   that a complete prior file is never re-downloaded.

## 2. The constraint that rules out the naive "HEAD before download"

The user-facing intuition is: HF exposes the file's sha256 (for LFS
files), so we should be able to HEAD the corresponding OCI blob and skip
the download. This does not work directly.

The OCI layer blob is a **tar archive** wrapping the file:

```
sha256(layer) = sha256( tar_header(512 B) || file_content || padding || EOF(1024 B) )
```

`sha256` is not composable — knowing `sha256(file_content)` gives us no
way to compute `sha256(layer)` without actually hashing the wrapped
content. Pre-computing or caching `file_sha256 → layer_digest` locally
would solve it but violates the v1.0 invariant "registry HEAD is the
source of truth, no local state file".

Changing the wire format so `layer.digest == file.sha256` (raw blob
layer) is also off the table: the consumer kubelet / containerd image
volume path requires tar-formatted layers to unpack the file onto the
volume filesystem (KEP-4639). Without tar, the runtime has no filesystem
contract.

## 3. Design — annotations on the layer descriptors

The reusable mapping `(hf_path, hf_sha256) → layer_digest` already
exists implicitly in the previous push's manifest. We make it explicit
by emitting two OCI annotations on every layer descriptor:

```
io.github.codanael.modelcar.hf-path:    <hf_path>           # always
io.github.codanael.modelcar.hf-sha256:  <64-hex>            # LFS files only
```

Constants live in `manifest.py:ANN_HF_PATH`, `ANN_HF_SHA256`. Sorted-keys
JSON keeps the manifest digest deterministic.

On every push, after `get_manifest_digest_at_tag` confirms a manifest
exists at the target tag (and `--force` is off), the pipeline GETs the
manifest body and `pipeline.py:build_reuse_map` indexes its layers by
the tuple `(hf-path, hf-sha256)`. The reuse-map is handed to every
`FileWorker`.

`FileWorker.process` gains a phase 0 that runs before download:

```
0. REUSE PRE-CHECK
   reuse_hit = reuse_map.get((hf_file.path, hf_file.lfs_sha256))
   if reuse_hit and head_blob(reuse_hit.digest) is not None:
       log "<path>: reusing cached layer <digest> (NN MB) — HF skipped"
       return reuse_hit
```

The `head_blob` confirmation is critical — the registry may have
garbage-collected the blob since the last manifest was pushed. On a
404, the worker falls through to the normal phase a (download).

The first push of any model produces an empty reuse-map (no existing
manifest). From the second push onward, every unchanged file is a hit:
zero HF bytes, one HEAD-blob per file, manifest comes out byte-identical.

## 4. Why not a local cache file

A `<spool>/blob-index.json` mapping `(repo@rev, hf_path, hf_sha256) →
layer_digest` would also work and would cover cross-tag scenarios (push
the same revision under tag `prod-A` then under `prod-B`, second one
reuses without going through HF). But:

- It violates the v1.0 "registry HEAD is source of truth" invariant.
- It introduces a stale-cache failure mode that the registry-anchored
  approach doesn't have.
- The cross-tag case is rare in practice — most re-pushes are CI
  rebuilds of the same `(repo, tag)`.

The annotations approach is bounded by what the registry already stores
and survives any cache wipe.

## 5. Why not switch the layer mediaType

We considered making `layer.digest == file.sha256` by switching the
layer mediaType to a raw-blob format. This trivializes "HEAD before
download" because we know the file sha256 from HF directly. Rejected:

- KServe's OCI Image Volume contract (KEP-4639) requires the runtime
  (containerd / CRI-O) to unpack the layer onto a filesystem. Both
  runtimes hardcode tar-family formats (`tar`, `tar+gzip`, `tar+zstd`);
  a raw-blob layer would be rejected at pod start.
- Breaks the v1.0 locked decision "layer.digest == diff_id by
  construction".

Out of scope.

## 6. Companion fixes

### 6.1 Progress logs (regression from v1.0 rewrite)

`logging.py` regains `fmt_bytes` and `ProgressEmitter` (port from v0.x).
`PipelineLogger` emits are now mutex-guarded so parallel workers cannot
interleave bytes mid-line.

`FileWorker.process` announces each phase via `PipelineLogger.info`:

```
<path>: downloading (NN MB)
<path>: 30% (NN MB / NN MB)         # ProgressEmitter, throttle 5s
<path>: pushing layer sha256:abc12 (NN MB)
<path>: pushed sha256:abc12
or  <path>: reusing cached layer sha256:abc12 (NN MB) — HF skipped
or  <path>: reusing existing blob sha256:abc12 (skip push)
```

The pre-v1.0 standard library `log.info(...)` calls in `pipeline.py` are
silenced (no `logging.basicConfig`); they were ornamental in v1.0 since
no handler was attached.

### 6.2 Skip cached source files (consistency fix)

`HfDownloader.download()` short-circuits when
`<spool>/sources/<path>` exists at the expected size and returns the
cached path without any HTTP. The atomic-rename invariant guarantees a
present `final` file is a complete + verified prior download — re-hashing
on every re-run would defeat the optimization.

## 7. Test surface

- `tests/unit/test_logging.py` — `fmt_bytes`, `ProgressEmitter`
  throttling, `PipelineLogger` line atomicity under thread contention.
- `tests/unit/test_manifest.py` — annotations emitted on
  `BlobDescriptor.to_dict()`; manifest digest unchanged across runs.
- `tests/unit/test_pipeline.py` — `fetch_manifest_at_tag`,
  `build_reuse_map`, `FileWorker` reuse-hit / miss / hit-but-blob-gone.
- `tests/unit/test_download.py` — skip-if-final-exists guard.
- `tests/integration/test_pipeline_reuse.py` — end-to-end: first run
  downloads + pushes; second run reuses every layer (`downloader.download`
  called zero times, no PUTs, byte-identical manifest digest).

## 8. Compatibility

- **Older manifests without annotations**: `build_reuse_map` returns an
  empty map (no path annotation → entry is skipped). The push proceeds
  through the normal download+push path. After this push, the new
  manifest carries annotations and subsequent re-pushes benefit.
- **Non-LFS files** (small JSON / TXT): no `hf-sha256` annotation
  available; `(path, None)` is the lookup key. Equivalent on both sides
  in the reuse-map; matches as expected.
- **`--force`**: short-circuits the reuse-map build entirely. Existing
  semantics preserved.
- **Manifest digest stability**: annotations are sorted-keys JSON inside
  each layer descriptor; the manifest digest is reproducible across runs
  of the same revision (verified in
  `test_manifest_digest_stable_across_runs_with_annotations`).
