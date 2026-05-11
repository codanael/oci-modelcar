# oci-modelcar v1.3 — Crash-resilient reuse via OCI referrers

**Status**: design approved, implementation in progress.
**Author**: Anael Latassa with Claude.
**Date**: 2026-05-11.
**Extends**: v1.0 (`2026-05-08-oci-modelcar-v1-design.md`), v1.1
(`2026-05-11-cross-run-layer-reuse-design.md`), v1.2
(`2026-05-11-file-filtering-design.md`).

## 1. Why

v1.1 introduced cross-run layer reuse by reading per-layer annotations
(`io.github.codanael.modelcar.hf-path`,
`io.github.codanael.modelcar.hf-sha256`) from the manifest at the target
tag. A re-push of an unchanged HuggingFace revision then touches HF for
zero bytes.

That mechanism breaks in one specific failure mode: **a run that pushes
layer blobs but crashes before the final manifest is committed, with
`--clean-hf-after-push` on**. In that state:

- Every layer blob is in the registry (the v1.0 design correctly
  uploads each layer's blob as soon as its file is downloaded).
- The HuggingFace source files have been deleted by
  `--clean-hf-after-push` after each successful push.
- The target tag has no manifest yet, so v1.1's reuse-map is empty.

The next run sees the empty reuse-map, finds the source files gone, and
re-downloads the entire model from HuggingFace. For
`mistralai/Mistral-Medium-3.5-128B` that is ~134 GB of wasted HF
bandwidth, even though all the corresponding layer blobs are sitting in
the registry. The phase-c HEAD skip prevents re-uploading the blobs,
but the HF round-trip is unavoidable: there is no surviving mapping
from `(hf_path, hf_sha256)` to `layer_digest`.

The brute-force fix — never delete sources until manifest commit —
defeats the purpose of `--clean-hf-after-push` (peak-disk minimization
on ephemeral CI / small containers). We need a mapping that is durable
across crashes **and** stored in the registry, not on local disk.

## 2. Constraints and rejected alternatives

The mapping `(hf_path, hf_sha256) → layer_digest` must:

1. Survive a crash between layer-push and manifest-commit.
2. Be reachable on a fresh CI runner (no local disk persistence).
3. Be addressable by run inputs (`hf_repo`, `hf_revision`,
   `allow_patterns`, `ignore_patterns`, `layer_prefix`) so two
   different filter configurations don't pollute each other.
4. Cost no more than ~1 KB per layer.
5. Not pollute the image-tag namespace with transient state.

Rejected:

- **Local state file** (re-introduce v0.x `state.json`). Violates the
  v1.0 invariant "registry HEAD is the source of truth" and fails
  constraint 2 (ephemeral CI scratch).
- **Rolling `<tag>.partial` manifest**, updated after each worker.
  Works but pollutes the tag namespace, races on shared writes when
  workers complete concurrently, and a partial manifest can be
  confused for a real image (anyone pulling `<tag>.partial` gets a
  broken model).
- **Per-layer "marker" tags** (`<tag>-l<N>`). Worse on constraint 5
  (N transient tags per run, `O(repo_tags)` scan on resume).
- **Defer cleanup until manifest commit**. Defeats
  `--clean-hf-after-push`'s low-disk purpose.

## 3. Design — OCI 1.1 referrer artifacts anchored on a stub manifest

The OCI Distribution Spec v1.1 introduced the **Referrers API**: a
manifest can carry a `subject` field pointing at another manifest by
digest. Clients query `GET /v2/<repo>/referrers/<digest>` to retrieve
an image index listing every manifest that names `<digest>` as its
subject. When the registry lacks native support, the spec mandates a
fallback: an image index hosted at the tag `<algo>-<hex>` (truncated
per spec), maintained by clients.

Both Artifactory (≥ 7.90.1, 24 Sep 2024) and Docker `registry:2`
(≥ 2.8) support the native API. The fallback tag schema works on every
OCI 1.0 registry.

We use this primitive to persist the reuse mapping as **referrer
artifacts** anchored on a **deterministic stub manifest**.

### 3.1 The stub manifest

A tiny content-addressed manifest whose digest is a pure function of
the run inputs:

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "artifactType": "application/vnd.codanael.modelcar.reuse-anchor.v1",
  "config": {
    "mediaType": "application/vnd.oci.empty.v1+json",
    "digest": "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "size": 2
  },
  "layers": [],
  "annotations": {
    "io.github.codanael.modelcar.hf-repo":         "mistralai/Mistral-Medium-3.5-128B",
    "io.github.codanael.modelcar.hf-revision":     "abc123...",
    "io.github.codanael.modelcar.allow-patterns":  "*.safetensors *.json *.txt *.md *.model",
    "io.github.codanael.modelcar.ignore-patterns": "consolidated* params.json",
    "io.github.codanael.modelcar.layer-prefix":    "models/"
  }
}
```

Properties:

- `config` uses the OCI-spec-defined empty descriptor
  (`application/vnd.oci.empty.v1+json`, content `{}`, fixed digest
  `sha256:4413...8a`, size 2). The 2-byte `{}` blob is pushable as a
  small blob exactly once per repo; subsequent attempts no-op via
  HEAD.
- `layers: []` is permitted on OCI 1.1 artifact manifests.
- `annotations` carry the run inputs verbatim for audit and so
  inspectors (`skopeo inspect --raw`, `oras manifest fetch`) can show
  what configuration produced these blobs.
- The whole manifest is serialized with `json.dumps(...,
  separators=(",", ":"), sort_keys=True)` and SHA-256'd — same
  deterministic pattern as v1.0's `build_manifest_bytes`.

The stub's digest is the run's **anchor**. Two runs with identical
inputs compute the same digest. PUT-by-digest is idempotent: HEAD
first, push only if absent.

### 3.2 The reuse-record artifact

After every worker completes phase d/e (push + verify) successfully,
the pipeline pushes one tiny artifact manifest per layer:

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "artifactType": "application/vnd.codanael.modelcar.reuse-record.v1",
  "config": {
    "mediaType": "application/vnd.oci.empty.v1+json",
    "digest": "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "size": 2
  },
  "subject": {
    "mediaType": "application/vnd.oci.image.manifest.v1+json",
    "digest": "<stub_digest>",
    "size":   <stub_size>
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar",
      "digest":    "<actual_layer_digest>",
      "size":       <actual_layer_size>,
      "annotations": {
        "io.github.codanael.modelcar.hf-path":   "model-00001-of-00003.safetensors",
        "io.github.codanael.modelcar.hf-sha256": "abc123..."
      }
    }
  ]
}
```

`layers[0].digest` **references** the blob already pushed by the
pipeline's normal flow — this artifact does not re-upload the blob. The
artifact is pure metadata, ~1 KB.

PUT-by-digest. Idempotent if same content (same layer → same artifact
digest).

### 3.3 Resume protocol

At pipeline preflight (after revision resolve, before disk-space
check):

1. Compute `stub_digest = sha256(build_anchor_manifest_bytes(cfg))`.
2. HEAD `/v2/<repo>/manifests/sha256:<stub_digest>`.
   - 200: stub already present from a prior partial run.
   - 404: push the stub now (PUT-by-digest), idempotent.
3. GET `/v2/<repo>/referrers/sha256:<stub_digest>?artifactType=application/vnd.codanael.modelcar.reuse-record.v1`
   - **200**: parse the image index, accumulate descriptors.
   - **404**: the registry does not implement OCI 1.1 referrers
     natively. Fall back: GET `/v2/<repo>/manifests/sha256-<stub_hex>`
     (the OCI fallback tag schema). Same content shape (image index),
     same parsing.
4. For each descriptor in the index, GET the manifest. Extract
   `layers[0].digest`, `layers[0].size`, and the `hf-path` /
   `hf-sha256` annotations. Build a `referrer_reuse_map: dict[(path,
   sha), BlobDescriptor]`.
5. Merge with v1.1's manifest-based reuse-map: union; on key
   collision, v1.1's entry wins (it came from a successful final
   manifest, more authoritative than a referrer from a possibly
   partial run).

The merged reuse-map is handed to workers as today. Worker phase 0
behavior (HEAD the blob to confirm it's still in the registry, return
the existing descriptor on hit) is unchanged.

### 3.4 Recording protocol

In `FileWorker.process`, after phase e (verify) and before phase f
(cleanup), if the worker actually pushed a new blob (i.e. didn't skip
via reuse-map or phase-c HEAD), push the corresponding reuse-record
artifact. On reuse-map / phase-c hit, **don't** re-push the record —
it's already there (it was the source of the reuse-map entry, or
it'll be present from the previous successful run that put the blob
in the registry).

The worker calls a new injected helper, e.g. `record_reuse(layer_desc:
BlobDescriptor) -> None`, that internally PUTs the record manifest and
maintains the fallback tag when needed (see 3.5).

### 3.5 Native API detection + fallback maintenance

Per OCI Distribution Spec 1.1: when a registry accepts a manifest with
a `subject` field, a referrers-capable registry **MUST** echo back the
header `OCI-Subject: <subject_digest>` on the 201/202 response. A
registry that omits this header signals it does not maintain the
fallback tag, and the client is responsible.

Implementation:

- On the first reuse-record PUT of the run, inspect the response.
- If `OCI-Subject` is present, set `state.native_referrers = True` and
  do nothing else.
- If absent, set `state.native_referrers = False` and from this PUT
  onward, also maintain the fallback tag:
  - GET `/v2/<repo>/manifests/sha256-<stub_hex>` (or treat as empty
    index on 404).
  - Append the new descriptor to the index's `manifests` array.
  - PUT the updated index at `sha256-<stub_hex>`.
- Serialize the fallback-tag GET-modify-PUT through a per-run mutex so
  parallel workers don't race. Native-mode workers do not block on
  each other.

The first PUT can't know in advance whether the registry is native —
that's fine. The record() flow becomes: (1) PUT the record manifest by
digest, (2) inspect response headers for `OCI-Subject`, (3) if absent,
set `native_referrers=False` and synchronously update the fallback
index for **this** record before returning (GET fallback tag, append
descriptor, PUT under the run-wide lock). From record() call #2
onward, the cached `native_referrers` flag short-circuits the
detection: native-mode workers do nothing extra; fallback-mode workers
always update the fallback index under the lock.

Simpler alternative considered and rejected: always maintain the
fallback tag, native or not. Doubles the work on Artifactory; rejected
because Artifactory is the primary production target.

### 3.6 What does NOT change

- Layer mediaType: still `application/vnd.oci.image.layer.v1.tar`
  uncompressed.
- Image manifest format for the final image: unchanged.
- Image config (no `created` field, deterministic): unchanged.
- v1.1 reuse-map (from manifest annotations at target tag):
  unchanged, consulted first.
- v1.2 filter syntax (`fnmatch` globs, `--ignore-patterns`):
  unchanged.
- Worker phases a→f and the FileWorker contract: only addition is the
  `record_reuse` callback after phase e.
- Exit codes, sub-commands, retry budgets, disk planning:
  unchanged.

## 4. New module: `src/oci_modelcar/reuse.py`

A dedicated module to keep `manifest.py` focused on the image manifest
itself.

```python
# Public surface (final names may shift slightly during impl):

EMPTY_CONFIG_DIGEST = "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
EMPTY_CONFIG_SIZE = 2
EMPTY_CONFIG_BYTES = b"{}"

ARTIFACT_TYPE_ANCHOR = "application/vnd.codanael.modelcar.reuse-anchor.v1"
ARTIFACT_TYPE_RECORD = "application/vnd.codanael.modelcar.reuse-record.v1"

ANN_HF_REPO       = "io.github.codanael.modelcar.hf-repo"
ANN_HF_REVISION   = "io.github.codanael.modelcar.hf-revision"
ANN_ALLOW         = "io.github.codanael.modelcar.allow-patterns"
ANN_IGNORE        = "io.github.codanael.modelcar.ignore-patterns"
ANN_LAYER_PREFIX  = "io.github.codanael.modelcar.layer-prefix"

def build_anchor_manifest_bytes(
    hf_repo: str,
    hf_revision: str,
    allow_patterns: tuple[str, ...],
    ignore_patterns: tuple[str, ...],
    layer_prefix: str,
) -> bytes: ...

def build_record_manifest_bytes(
    layer: BlobDescriptor,                 # already has hf-path / hf-sha256
    stub_digest: str,
    stub_size: int,
) -> bytes: ...

def fallback_referrers_tag(stub_digest: str) -> str:
    """sha256:<hex> → sha256-<hex>, per OCI spec fallback schema."""

@dataclass
class RegistryReuseStore:
    client: OciClient
    repo: str
    plog: PipelineLogger
    _native_referrers: bool | None = None  # set lazily on first record()
    _fallback_lock: threading.Lock = field(default_factory=threading.Lock)

    def ensure_anchor(self, anchor_bytes: bytes, anchor_digest: str) -> None:
        """HEAD the stub; PUT-by-digest if absent. Idempotent."""

    def load_reuse_map(
        self,
        anchor_digest: str,
    ) -> dict[tuple[str, str | None], BlobDescriptor]:
        """GET referrers (native or fallback), fetch each manifest,
        return reuse-map keyed by (hf_path, hf_sha256)."""

    def record(
        self,
        layer: BlobDescriptor,
        anchor_digest: str,
        anchor_size: int,
    ) -> None:
        """PUT the record manifest by digest. Detect native via
        OCI-Subject header. Update fallback tag if needed (under lock)."""
```

`RegistryReuseStore` is owned by `Pipeline`, passed to each
`FileWorker` (which calls `record()` after phase e).

## 5. Pipeline integration

`pipeline.py:Pipeline.run`, after preflight:

```python
# v1.1 reuse-map (existing)
existing_tag_digest = get_manifest_digest_at_tag(...)
manifest_reuse_map = {}
if existing_tag_digest and not self.cfg.force:
    existing_manifest = fetch_manifest_at_tag(...)
    if existing_manifest:
        manifest_reuse_map = build_reuse_map(existing_manifest)

# v1.3 reuse-store + anchor
anchor_bytes = build_anchor_manifest_bytes(cfg.hf_repo, revision, ...)
anchor_digest = "sha256:" + sha256(anchor_bytes).hexdigest()
reuse_store = RegistryReuseStore(client=..., repo=..., plog=...)
if not self.cfg.force:
    reuse_store.ensure_anchor(anchor_bytes, anchor_digest)
    referrer_reuse_map = reuse_store.load_reuse_map(anchor_digest)
else:
    referrer_reuse_map = {}

# Merge: v1.1 wins on conflict
reuse_map = {**referrer_reuse_map, **manifest_reuse_map}
```

`FileWorker` gains a `reuse_store` argument and an `anchor_digest`
argument. After phase e, if the file's path was actually pushed (not
reused), the worker calls `reuse_store.record(layer_desc, anchor_digest,
anchor_size)`. On phase-0 or phase-c reuse, no record is written.

`--force` short-circuits the referrer **load** (same as it does for
the v1.1 reuse-map): the user is saying "ignore prior state, rebuild
the manifest". `--force` does **not** stop the run from **writing**
records — record PUTs are idempotent by digest, so a forced re-push
just no-ops on records that are identical and adds records for any
genuinely new layers. Stale records from a prior partial run with the
same inputs remain in the registry as audit history; their layer
blobs are unchanged.

## 6. Backwards compatibility

- Older oci-modelcar versions don't push records. A repo with v1.1
  layers but no v1.3 records still benefits from the v1.1 manifest
  reuse-map. No regression.
- v1.3 pushes records every run. If a repo is pushed alternately by a
  v1.3 client and a v1.1 client, the v1.1 client ignores referrers and
  re-downloads from HF on crash; the v1.3 client picks them up.
  Mixing is permitted, behavior degrades gracefully.
- The stub anchor manifest carries the full input fingerprint in its
  annotations. A v1.3 client reading a stub written by a future v1.4
  client (with extra annotations) sees the annotations it understands;
  unknown annotations are ignored.

## 7. Wire-format constants

To avoid magic strings:

```python
EMPTY_CONFIG_DIGEST = "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
```

This is the digest of the literal byte-string `b"{}"` (2 bytes) under
SHA-256. The OCI spec defines this as the canonical empty descriptor
content. We push the 2-byte blob exactly once per repo (HEAD-skip),
and reference it as the `config` of every stub and every record.

Fallback referrers tag format (OCI 1.1):

```
sha256:abc123...   →   sha256-abc123...
```

Replace `:` with `-`. Total length: `sha256-` (7) + 64 hex = 71 chars.
Well within the 128-char tag limit (`_TAG_RE` allows up to 128).

## 8. Failure modes

| Scenario | v1.0 | v1.1 | v1.2 | v1.3 |
|---|---|---|---|---|
| Re-push unchanged revision, sources cached | re-DL + re-push | reuse via manifest, no DL/push | same | same |
| Re-push unchanged revision, sources cleaned | re-DL only | reuse via manifest, no DL/push | same | same |
| Crash after some pushes, before manifest, sources cached | re-DL the rest, HEAD-skips pushed ones | same | same | **plus**: skip re-DL of crashed-mid-stream files (referrer reuse-map) |
| Crash after some pushes, before manifest, sources cleaned | full re-DL, HEAD-skips uploads | full re-DL, HEAD-skips uploads | same | **no re-DL** for any file with a record; one-by-one DL for any that didn't make it to PUT before the crash |

## 9. Tests

### 9.1 Unit (new `tests/unit/test_reuse.py`)

- `build_anchor_manifest_bytes`:
  - same inputs → byte-identical
  - different `ignore_patterns` → different digest
  - annotations sorted, empty config descriptor exact-equal
  - layers field is `[]`
  - `schemaVersion == 2`, `artifactType == ARTIFACT_TYPE_ANCHOR`
- `build_record_manifest_bytes`:
  - `subject` populated with stub digest/size
  - `layers[0]` has expected digest/size/annotations
  - `config` references the empty descriptor
  - same input layer → same record digest
- `fallback_referrers_tag`:
  - `sha256:abc...` → `sha256-abc...` (8 alg variants if you want, only `sha256` matters here)
- `RegistryReuseStore.ensure_anchor`:
  - HEAD 200 → no PUT
  - HEAD 404 → PUT-by-digest happens, exactly once
- `RegistryReuseStore.load_reuse_map`:
  - native referrers 200 → parses correctly
  - native 404 → falls back to tag GET
  - tag GET 404 → returns empty map (no crash)
  - records with missing annotations → silently skipped
- `RegistryReuseStore.record`:
  - first PUT, response has `OCI-Subject` → no fallback PUT
  - first PUT, response lacks `OCI-Subject` → fallback PUT happens
  - subsequent PUTs in non-native mode → fallback PUT under lock

### 9.2 Integration (new `tests/integration/test_pipeline_referrer_resume.py`)

- 3-file scenario, simulated crash after file 2 completes:
  - First Pipeline run: stop after file 2's reuse-record is written
    (use a stop-event in the worker after phase e for one file only).
  - Second Pipeline run: assert files 1 and 2 are reused via the
    referrer map (downloader.download NOT called for them), file 3 is
    downloaded as normal. Final manifest digest matches the
    no-interruption baseline.
- Native registry path: ensure record PUT response carries
  `OCI-Subject`, fallback tag NOT created.
- Fallback path: same scenario, registry mock returns 404 on
  `/referrers/`, omits `OCI-Subject` on PUTs. Assert client writes
  to `sha256-<hex>` tag, the index there has 2 entries after the
  first run, all 3 after the second.

### 9.3 E2E (gated, optional this release)

Real `registry:2` (≥ 2.8 supports referrers natively). Skip in CI.

## 10. CLI / Config surface

**None.** The whole feature is automatic. Power users can opt out via:

```
--no-reuse-records       (CONFIG: no_reuse_records=False)
```

Defaults to **off** (records ARE written). Useful for users who want
to keep the repo audit-clean of metadata artifacts, or who push
through a registry that mishandles unknown artifact types (none
expected today). Documented as a corner-case escape hatch.

## 11. Documentation

- `docs/user-guide.md`: new section "Crash-resilient reuse via OCI
  referrers" right after the existing "Resume after partial failure"
  section. Explains the mechanism in user terms, the
  `--no-reuse-records` opt-out, and how to inspect records via
  `oras discover` / `skopeo inspect`.
- `README.md`: one-line mention under features.
- `CHANGELOG.md`: `[Unreleased]` entry.
- `CLAUDE.md`: add the new annotations + artifact types to the locked
  design decisions section, alongside the existing
  `ANN_HF_PATH` / `ANN_HF_SHA256` entry.

## 12. Open questions

- **Garbage collection of records on long-running repos**. After
  months of pushes, a repo accumulates one record per layer per
  unique-input run. At ~1 KB each, even 10,000 records = 10 MB —
  manageable. Defer a `prune-reuse-records` sub-command to a future
  release if user demand emerges.
- **Records for layers reused from a prior run**. Today's design says
  "don't re-record on reuse hit". An alternative is to record always,
  producing duplicate records with identical content (and identical
  digest, so they collapse on idempotent PUT). Slightly cleaner —
  every successful pipeline run leaves N records, regardless of
  origin. Worth picking during implementation.
- **Stub annotation size limit**. OCI doesn't impose a hard limit but
  some registries cap manifest size at 4 MB. Our annotations are
  bounded by the length of `allow_patterns` + `ignore_patterns`,
  which in practice is < 1 KB. No risk in v1.3.
