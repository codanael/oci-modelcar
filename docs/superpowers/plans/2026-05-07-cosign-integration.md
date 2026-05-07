# Cosign Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Sigstore-based signing at two levels: PEP 740 attestations on PyPI artifacts (via release.yml flag) and a digest-based image reference in `push` output that the user can pipe directly into `cosign sign`.

**Architecture:** Minimal-by-design. Adds zero runtime/dev dependencies, zero new modules, zero CLI subcommands. PyPI side is a one-line workflow change. Modelcar side adds a single output variable `IMAGEREFDIGEST=<host>/<repo>@sha256:<digest>` plus state persistence, leaving signing to the user's `cosign` invocation.

**Tech Stack:** Python 3.14, `pypa/gh-action-pypi-publish@release/v1` (already used), `pytest`, `pre-commit` with ruff/mypy hooks. NixOS dev shell (`nix-shell ./shell.nix --command "..."`).

**Spec:** `docs/superpowers/specs/2026-05-07-cosign-integration-design.md`

---

## Pre-flight

- [ ] **Step 0.1: Verify clean working tree**

```bash
git status
git branch --show-current
```

Expected: working tree clean, on `main`.

- [ ] **Step 0.2: Create feature branch**

```bash
git checkout -b feat/cosign-integration
```

Expected: switched to new branch.

- [ ] **Step 0.3: Sanity check — run existing test suite**

```bash
nix-shell ./shell.nix --command "python3.14 -m pytest tests/unit tests/integration -m 'not e2e' -q"
```

Expected: all pass. If anything fails, stop and investigate before changing anything.

---

## Task 1: Persist `image_ref_digest` in state

**Files:**
- Modify: `src/oci_modelcar/state.py:138-143` (signature of `mark_completed`)
- Modify: `tests/unit/test_state.py:126-139` (existing test) and add new test

- [ ] **Step 1.1: Write the failing test for new round-trip**

Append to `tests/unit/test_state.py`:

```python
def test_mark_completed_persists_image_ref_digest(tmp_path: Path):
    store = JsonStateStore(tmp_path / "state.json")
    job = JobState(
        hf_repo="foo/bar",
        hf_revision_input="main",
        hf_revision_resolved="a" * 40,
        registry="r.example",
        target_repo="m/x",
        target_tag="v1",
    )
    store.upsert_job("k1", job)
    store.mark_completed(
        "k1",
        manifest_digest="sha256:abc",
        image_ref_digest="r.example/m/x@sha256:abc",
    )
    store.save()

    fresh = JsonStateStore(tmp_path / "state.json")
    raw_job = fresh.get_job("k1")
    assert raw_job is not None
    assert raw_job["manifest_digest"] == "sha256:abc"
    assert raw_job["image_ref_digest"] == "r.example/m/x@sha256:abc"
```

Also update the existing `test_mark_completed` (line 138) to pass the new kwarg:

```python
def test_mark_completed(tmp_path: Path):
    store = JsonStateStore(tmp_path / "state.json")
    job = JobState(
        hf_repo="foo/bar",
        hf_revision_input="main",
        hf_revision_resolved="a" * 40,
        registry="r",
        target_repo="m",
        target_tag="v1",
    )
    store.upsert_job("k1", job)
    assert not store.is_completed("k1")
    store.mark_completed(
        "k1",
        manifest_digest="sha256:abc",
        image_ref_digest="r/m@sha256:abc",
    )
    assert store.is_completed("k1")
```

- [ ] **Step 1.2: Run tests to verify failure**

```bash
nix-shell ./shell.nix --command "python3.14 -m pytest tests/unit/test_state.py -v"
```

Expected:
- `test_mark_completed_persists_image_ref_digest` FAILS with `TypeError: mark_completed() got an unexpected keyword argument 'image_ref_digest'`.
- `test_mark_completed` FAILS with same error.

- [ ] **Step 1.3: Update `mark_completed` signature in `src/oci_modelcar/state.py`**

Replace the existing `mark_completed` method (around line 138):

```python
    def mark_completed(
        self,
        job_key: str,
        manifest_digest: str,
        image_ref_digest: str,
    ) -> None:
        with self._lock:
            job = self._data["jobs"][job_key]
            job["manifest_digest"] = manifest_digest
            job["image_ref_digest"] = image_ref_digest
            job["completed_at"] = _now_iso()
            job["updated_at"] = _now_iso()
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
nix-shell ./shell.nix --command "python3.14 -m pytest tests/unit/test_state.py -v"
```

Expected: all 9 tests pass (8 existing + 1 new).

- [ ] **Step 1.5: Commit**

```bash
git add src/oci_modelcar/state.py tests/unit/test_state.py
git commit -m "$(cat <<'EOF'
feat(state): persist image_ref_digest on completion

Add a required image_ref_digest kwarg to mark_completed so the canonical
digest reference is stored alongside manifest_digest. Surfaced by
oci-modelcar status and re-emitted on idempotent re-runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: pre-commit hooks pass, commit lands. If hooks fail, fix and create a NEW commit (don't `--amend`).

---

## Task 2: Wire `image_ref_digest` through happy-path of `run_push`

**Files:**
- Modify: `src/oci_modelcar/runner.py:271-275` (call site of `mark_completed` + output_variable)

This task is non-TDD (the runner full pipeline isn't unit-tested locally; emission validation lives in the E2E test, updated in Task 5). The state round-trip in Task 1 covers the persistence side; here we wire it in.

- [ ] **Step 2.1: Update the manifest-push block in `runner.py`**

Replace the block from line 271 to line 275 (the lines between `for t in [target_tag, *cfg.also_tags]:` validation and the `return RunResult(...)`):

Before:
```python
    state.mark_completed(job_key, manifest_digest=manifest_digest)
    state.save()

    plog.output_variable("manifestDigest", manifest_digest)
    plog.output_variable("imageRef", image_ref)
```

After:
```python
    image_ref_digest = f"{cfg.registry}/{cfg.target_repo}@{manifest_digest}"

    state.mark_completed(
        job_key,
        manifest_digest=manifest_digest,
        image_ref_digest=image_ref_digest,
    )
    state.save()

    plog.output_variable("manifestDigest", manifest_digest)
    plog.output_variable("imageRef", image_ref)
    plog.output_variable("imageRefDigest", image_ref_digest)
```

- [ ] **Step 2.2: Run unit + integration tests**

```bash
nix-shell ./shell.nix --command "python3.14 -m pytest tests/unit tests/integration -m 'not e2e' -q"
```

Expected: all pass. If `mypy` complains about the kwarg, the call site is now consistent with the new signature — should be green.

- [ ] **Step 2.3: Run mypy explicitly**

```bash
nix-shell ./shell.nix --command "python3.14 -m mypy --strict src/"
```

Expected: `Success: no issues found`.

- [ ] **Step 2.4: Commit**

```bash
git add src/oci_modelcar/runner.py
git commit -m "$(cat <<'EOF'
feat(runner): emit IMAGEREFDIGEST and persist image_ref_digest after push

Construct the canonical digest reference <registry>/<repo>@sha256:<digest>
once the manifest digest is known, persist it via mark_completed, and emit
it through PipelineLogger so consumers can pipe it directly into cosign sign.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire `image_ref_digest` through idempotent re-run path

**Files:**
- Modify: `src/oci_modelcar/runner.py:111-120` (the `is_completed` early return block)

When `state.is_completed(job_key)` is true and `--force` is not set, we currently log "already completed" and return without re-emitting any output variable. Consumers of the second run get no `IMAGEREFDIGEST`. We need to re-emit it, with a fallback that reconstructs the reference from `manifest_digest` for state files written by v0.1.0 (which don't have `image_ref_digest`).

- [ ] **Step 3.1: Update the early return block**

Replace lines 111-120 (the `if state.is_completed(...) and not cfg.force:` block):

Before:
```python
    if state.is_completed(job_key) and not cfg.force:
        existing = state.get_job(job_key)
        assert existing is not None
        plog.info(f"Job already completed: {existing['manifest_digest']}")
        return RunResult(
            job_key=job_key,
            manifest_digest=str(existing["manifest_digest"]),
            image_ref=image_ref,
            layers=[],
        )
```

After:
```python
    if state.is_completed(job_key) and not cfg.force:
        existing = state.get_job(job_key)
        assert existing is not None
        manifest_digest = str(existing["manifest_digest"])
        image_ref_digest = (
            existing.get("image_ref_digest")
            or f"{cfg.registry}/{cfg.target_repo}@{manifest_digest}"
        )
        plog.info(f"Job already completed: {manifest_digest}")
        plog.output_variable("manifestDigest", manifest_digest)
        plog.output_variable("imageRef", image_ref)
        plog.output_variable("imageRefDigest", image_ref_digest)
        return RunResult(
            job_key=job_key,
            manifest_digest=manifest_digest,
            image_ref=image_ref,
            layers=[],
        )
```

- [ ] **Step 3.2: Run tests**

```bash
nix-shell ./shell.nix --command "python3.14 -m pytest tests/unit tests/integration -m 'not e2e' -q"
```

Expected: all pass.

- [ ] **Step 3.3: Run mypy**

```bash
nix-shell ./shell.nix --command "python3.14 -m mypy --strict src/"
```

Expected: `Success: no issues found`.

- [ ] **Step 3.4: Commit**

```bash
git add src/oci_modelcar/runner.py
git commit -m "$(cat <<'EOF'
feat(runner): re-emit IMAGEREFDIGEST on idempotent re-run

When a job is already completed, emit MANIFESTDIGEST/IMAGEREF/IMAGEREFDIGEST
as if the push happened just now, so downstream cosign-sign chains see the
reference regardless of whether the layers were re-pushed. Fall back to
reconstructing image_ref_digest from manifest_digest for legacy state files
written by v0.1.0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Surface `image_ref_digest` in `oci-modelcar status`

**Files:**
- Modify: `src/oci_modelcar/cli.py:91-104` (the `_run_status` rendering loop)
- Add: `tests/integration/test_cli.py` (new tests for status formatting)

- [ ] **Step 4.1: Write the failing tests**

Append to `tests/integration/test_cli.py`:

```python
import json
from pathlib import Path


def _write_state(tmp_path: Path, with_image_ref_digest: bool) -> Path:
    state_path = tmp_path / "state.json"
    job: dict = {
        "source": {
            "hf_repo": "foo/bar",
            "hf_revision_input": "main",
            "hf_revision_resolved": "a" * 40,
        },
        "target": {
            "registry": "r.example",
            "repo": "m/x",
            "tag": "abcd1234",
            "also_tags": [],
        },
        "started_at": "2026-05-07T12:00:00Z",
        "updated_at": "2026-05-07T12:00:00Z",
        "completed_at": "2026-05-07T12:01:00Z",
        "manifest_digest": "sha256:" + "f" * 64,
        "files": {},
    }
    if with_image_ref_digest:
        job["image_ref_digest"] = "r.example/m/x@sha256:" + "f" * 64
    state_path.write_text(json.dumps({"version": 1, "jobs": {"abc123def456": job}}))
    return state_path


def test_status_shows_image_ref_digest_when_present(tmp_path):
    state_path = _write_state(tmp_path, with_image_ref_digest=True)
    proc = subprocess.run(
        [sys.executable, "-m", "oci_modelcar", "status", "--state-file", str(state_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ref=r.example/m/x@sha256:" in proc.stdout


def test_status_graceful_when_image_ref_digest_missing(tmp_path):
    state_path = _write_state(tmp_path, with_image_ref_digest=False)
    proc = subprocess.run(
        [sys.executable, "-m", "oci_modelcar", "status", "--state-file", str(state_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "job=abc123def456" in proc.stdout
    assert "ref=" not in proc.stdout
```

- [ ] **Step 4.2: Run tests to verify failure**

```bash
nix-shell ./shell.nix --command "python3.14 -m pytest tests/integration/test_cli.py -v"
```

Expected: `test_status_shows_image_ref_digest_when_present` FAILS (assertion `'ref=r.example/...' in proc.stdout` is false because `_run_status` doesn't emit that line yet). `test_status_graceful_when_image_ref_digest_missing` PASSES (the `ref=` line isn't emitted by definition, and the rest of `status` works).

- [ ] **Step 4.3: Update `_run_status` rendering loop**

In `src/oci_modelcar/cli.py`, find the loop that begins at line 91 (`for key in keys:`) and modify the body. The current body is one `sys.stdout.write` call producing one line; we add a conditional second line if `image_ref_digest` is present.

Replace:

```python
    for key in keys:
        job = store.get_job(key)
        if job is None:
            continue
        src = job.get("source", {})
        tgt = job.get("target", {})
        digest = job.get("manifest_digest") or "(pending)"
        completed = job.get("completed_at") or "(in-progress)"
        sys.stdout.write(
            f"job={key[:12]}  "
            f"{src.get('hf_repo', '?')}@{src.get('hf_revision_resolved', '?')}  "
            f"-> {tgt.get('registry', '?')}/{tgt.get('repo', '?')}:{tgt.get('tag', '?')}  "
            f"digest={digest[:23]}  completed={completed}\n"
        )
    return _EX_OK
```

With:

```python
    for key in keys:
        job = store.get_job(key)
        if job is None:
            continue
        src = job.get("source", {})
        tgt = job.get("target", {})
        digest = job.get("manifest_digest") or "(pending)"
        completed = job.get("completed_at") or "(in-progress)"
        sys.stdout.write(
            f"job={key[:12]}  "
            f"{src.get('hf_repo', '?')}@{src.get('hf_revision_resolved', '?')}  "
            f"-> {tgt.get('registry', '?')}/{tgt.get('repo', '?')}:{tgt.get('tag', '?')}  "
            f"digest={digest[:23]}  completed={completed}\n"
        )
        image_ref_digest = job.get("image_ref_digest")
        if image_ref_digest:
            sys.stdout.write(f"    ref={image_ref_digest}\n")
    return _EX_OK
```

- [ ] **Step 4.4: Run tests to verify pass**

```bash
nix-shell ./shell.nix --command "python3.14 -m pytest tests/integration/test_cli.py -v"
```

Expected: both new tests PASS, plus the 3 existing tests in this file.

- [ ] **Step 4.5: Run mypy**

```bash
nix-shell ./shell.nix --command "python3.14 -m mypy --strict src/"
```

Expected: `Success: no issues found`.

- [ ] **Step 4.6: Commit**

```bash
git add src/oci_modelcar/cli.py tests/integration/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): show image_ref_digest in oci-modelcar status output

Append a second indented line per job exposing the canonical digest
reference, when the state stores it. Legacy state files (without the
field) render only the existing single line — graceful absence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: E2E test asserts `IMAGEREFDIGEST` in push output

**Files:**
- Modify: `tests/e2e/test_real_huggingface.py:55-65` (existing assertions on `IMAGEREF` / `MANIFESTDIGEST`)

This task adds a single assertion to the E2E so we have end-to-end coverage of the runner emitting `IMAGEREFDIGEST` over a real `registry:2`. The test still requires Docker + skopeo + network, gated by `@pytest.mark.e2e`.

- [ ] **Step 5.1: Add the assertion**

In `tests/e2e/test_real_huggingface.py`, locate the block in `test_push_tiny_llama` that currently looks like:

```python
    assert proc.returncode == 0, proc.stderr
    expected_tag = HF_TEST_REVISION[:12]
    assert f"IMAGEREF={local_registry.host}/test/tiny-llama:{expected_tag}" in proc.stdout
    m = re.search(r"^MANIFESTDIGEST=(sha256:[0-9a-f]{64})$", proc.stdout, re.MULTILINE)
    assert m, f"no MANIFEST= in stdout:\n{proc.stdout}"
```

Add a new assertion **after** `m = re.search(...)`:

```python
    assert proc.returncode == 0, proc.stderr
    expected_tag = HF_TEST_REVISION[:12]
    assert f"IMAGEREF={local_registry.host}/test/tiny-llama:{expected_tag}" in proc.stdout
    m = re.search(r"^MANIFESTDIGEST=(sha256:[0-9a-f]{64})$", proc.stdout, re.MULTILINE)
    assert m, f"no MANIFEST= in stdout:\n{proc.stdout}"
    expected_digest_ref = f"IMAGEREFDIGEST={local_registry.host}/test/tiny-llama@{m.group(1)}"
    assert expected_digest_ref in proc.stdout, (
        f"expected '{expected_digest_ref}' in stdout:\n{proc.stdout}"
    )
```

- [ ] **Step 5.2: Run E2E (skip if no Docker on this machine)**

If Docker is available:

```bash
nix-shell ./shell.nix --command "python3.14 -m pytest tests/e2e -m e2e -v -k test_push_tiny_llama"
```

Expected: PASS.

If Docker isn't available locally, skip running it but verify the assertion is syntactically correct by running:

```bash
nix-shell ./shell.nix --command "python3.14 -m pytest tests/e2e --collect-only -q"
```

Expected: tests are collected without import errors.

- [ ] **Step 5.3: Commit**

```bash
git add tests/e2e/test_real_huggingface.py
git commit -m "$(cat <<'EOF'
test(e2e): assert IMAGEREFDIGEST is emitted by push

Adds a final assertion that push outputs the canonical digest reference
in the form <host>/<repo>@sha256:<digest>, matching the manifest digest
captured from the same run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Enable PEP 740 attestations in release workflow

**Files:**
- Modify: `.github/workflows/release.yml:42-44` (the `pypa/gh-action-pypi-publish` step)

- [ ] **Step 6.1: Add `attestations: true` to the publish step**

Replace:

```yaml
      - uses: pypa/gh-action-pypi-publish@release/v1
        # Trusted Publisher: configure on pypi.org under
        # Project Settings -> Publishing, link this repo + workflow.
```

With:

```yaml
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          attestations: true
        # Trusted Publisher: configure on pypi.org under
        # Project Settings -> Publishing, link this repo + workflow.
        # attestations:true enables PEP 740 digital attestations via
        # Sigstore keyless OIDC. id-token: write is already set at the
        # workflow level (line 10).
```

- [ ] **Step 6.2: Validate YAML syntax**

```bash
nix-shell ./shell.nix --command "python3.14 -c 'import yaml; yaml.safe_load(open(\".github/workflows/release.yml\"))'"
```

Expected: no output, exit 0 (parse succeeds).

- [ ] **Step 6.3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "$(cat <<'EOF'
ci(release): enable PEP 740 attestations on PyPI publish

Add attestations: true to pypa/gh-action-pypi-publish so each release
publishes Sigstore-signed digital attestations alongside the dist files.
Identity derives from the OIDC token: keyless, no key management. The
existing id-token: write permission is already set at workflow level.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Document signing in README

**Files:**
- Modify: `README.md:92-103` (insert new section after "OCI compliance" block, before "Releasing (maintainers)")

- [ ] **Step 7.1: Insert the new section**

In `README.md`, find the line `## Releasing (maintainers)`. Immediately before it, insert the following section (preserve a blank line on both sides):

```markdown
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
python -m pypi_attestations verify pypi \
    --repository codanael/oci-modelcar \
    "$(pip download --no-deps --no-build-isolation -d . oci-modelcar | tail -1 | awk '{print $NF}')"
```
```

- [ ] **Step 7.2: Verify the README still renders cleanly**

```bash
grep -n "^## " README.md
```

Expected: section order is preserved — "Why", "Install", "Quick start", "Authentication", "Common options", "Resume after failure", "OCI compliance", "Signing & verification" (new), "Releasing (maintainers)", "License".

- [ ] **Step 7.3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(readme): add Signing & verification section

Show cosign sign recipes for both keyless OIDC (CI) and static-key
(offline) modes, plus consumer-side cosign verify. Also document
PEP 740 verification for the PyPI artifact via pypi-attestations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md:5` (under `[Unreleased]`)

- [ ] **Step 8.1: Add entry under [Unreleased]**

Replace:

```markdown
## [Unreleased]

## [0.1.0] - 2026-05-07
```

With:

```markdown
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
```

- [ ] **Step 8.2: Commit**

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(changelog): record cosign integration changes under [Unreleased]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Final quality gates

- [ ] **Step 9.1: Run ruff lint and format check**

```bash
nix-shell ./shell.nix --command "ruff check . && ruff format --check ."
```

Expected: both clean. If `ruff format --check` reports diffs, run `ruff format .`, re-stage, and create a *new* commit. Don't `--amend`.

- [ ] **Step 9.2: Run mypy strict**

```bash
nix-shell ./shell.nix --command "python3.14 -m mypy --strict src/"
```

Expected: `Success: no issues found`.

- [ ] **Step 9.3: Run unit + integration tests with coverage**

```bash
nix-shell ./shell.nix --command "python3.14 -m pytest tests/unit tests/integration -m 'not e2e' --cov=oci_modelcar --cov-report=term"
```

Expected: all pass. Coverage shouldn't drop materially (the changes are tiny).

- [ ] **Step 9.4: Verify branch state**

```bash
git log --oneline main..HEAD
git status
```

Expected:
- 8 commits ahead of `main` (one per Task 1–8).
- Working tree clean.

- [ ] **Step 9.5: Summary check against the spec**

Re-read `docs/superpowers/specs/2026-05-07-cosign-integration-design.md` §2.1 ("Surface modifiée"). Each row in that table should correspond to a closed checkbox above. If anything's missing, open a new task to address it before declaring done.

---

## Notes

- **Pre-commit hooks**: every `git commit` in this plan triggers ruff (lint+format), mypy --strict, and pytest non-e2e via the `.pre-commit-config.yaml` hooks. If a hook fails, the commit is aborted; fix the issue and retry. Never use `--no-verify`.
- **NixOS shell**: each `nix-shell ./shell.nix --command "..."` reuses the cached `.venv/`. First invocation in a fresh checkout will be slower (initial pip install).
- **Conventional commits**: scopes used here — `state`, `runner`, `cli`, `e2e`, `release` (maps to ci scope), `readme` (maps to docs scope), `changelog` (maps to docs).
- **No --amend**: per CLAUDE.md, always create new commits. If a hook fails on commit, the commit didn't happen — fix and re-stage and re-run `git commit`.
- **Branch merge** (post-implementation, out of scope of this plan): fast-forward `main` → `feat/cosign-integration` via PR or `git merge --ff-only`. Tag a v0.2.0 release at maintainer's discretion (separate flow, not part of this plan).
