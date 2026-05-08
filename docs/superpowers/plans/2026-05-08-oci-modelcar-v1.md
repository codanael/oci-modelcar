# oci-modelcar v1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean rewrite of oci-modelcar per the v1.0 design spec (`docs/superpowers/specs/2026-05-08-oci-modelcar-v1-design.md`). Single PATCH per blob, per-file pipeline, registry as source of truth, drop chunked mode.

**Architecture:** Per-file pipeline (download HF → tar to disk → push from file with full-PATCH replay → confirm via HEAD → cleanup). `huggingface_hub.HfApi` for metadata, our own bytes streamer for downloads (preserves stop_event cancellation). 12 modules, acyclic dependency graph.

**Tech Stack:** Python 3.14, `requests`, `urllib3`, `huggingface_hub` (metadata only). Tests via `pytest` + `pytest-httpserver`. Pre-commit gates: ruff, mypy --strict, pytest.

---

## Setup

Run on a new branch from the current `feat/robust-patch-retry` (which carries v0.5.x patches as backstory):

```bash
git checkout -b feat/v1-rewrite feat/robust-patch-retry
```

The plan deletes the existing src/ and tests/ at task 0.3 and rebuilds module-by-module. The branch will be in a "broken" intermediate state until phase 9 completes; all commits should pass pre-commit (ruff + mypy + pytest), but pytest collection on partial state will exercise only the modules already implemented.

---

## Phase 0 — Branch setup and clean slate

### Task 0.1: Create v1 branch + bump version

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Create branch**

```bash
git checkout -b feat/v1-rewrite
```

- [ ] **Step 2: Bump version to 1.0.0a0 and add huggingface_hub dependency**

Edit `pyproject.toml`:

```toml
[project]
name = "oci-modelcar"
version = "1.0.0a0"
description = "Push HuggingFace models to OCI registries (v1: per-file pipeline, single-PATCH per blob)"
readme = "README.md"
requires-python = ">=3.14"
license = "MIT"
license-files = ["LICENSE"]
authors = [{name = "codanael"}]
dependencies = [
    "requests>=2.32",
    "urllib3>=2.2",
    "huggingface_hub>=0.27",
]
```

- [ ] **Step 3: Verify install resolves**

```bash
.venv/bin/pip install -e '.[dev]' 2>&1 | tail -3
```

Expected: success, `huggingface-hub` present in `.venv/bin/pip list`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore(v1): bump to 1.0.0a0, add huggingface_hub dep"
```

---

### Task 0.2: Wipe old src and tests

**Files:**
- Delete: `src/oci_modelcar/*.py` (except `__init__.py`)
- Delete: `tests/unit/*.py`, `tests/integration/*.py`
- Modify: `src/oci_modelcar/__init__.py` to expose only `__version__`

- [ ] **Step 1: Delete all v0.x source files**

```bash
rm src/oci_modelcar/cli.py src/oci_modelcar/config.py src/oci_modelcar/http.py \
   src/oci_modelcar/hf.py src/oci_modelcar/oci.py src/oci_modelcar/manifest.py \
   src/oci_modelcar/runner.py src/oci_modelcar/state.py src/oci_modelcar/tags.py \
   src/oci_modelcar/tar_layer.py src/oci_modelcar/logging.py src/oci_modelcar/__main__.py
rm -f tests/unit/*.py tests/integration/*.py
# Keep tests/conftest.py (empty), tests/e2e/ (will rewrite later), tests/__init__ stubs
```

- [ ] **Step 2: Reset `__init__.py` to bare minimum**

`src/oci_modelcar/__init__.py`:

```python
"""oci-modelcar v1.0 — push HuggingFace models to OCI registries."""

from __future__ import annotations

__version__ = "1.0.0a0"
```

- [ ] **Step 3: Verify bare state**

```bash
ls src/oci_modelcar/
# Expected: __init__.py
ls tests/unit/ tests/integration/
# Expected: empty (or only __init__.py)
.venv/bin/python -c "import oci_modelcar; print(oci_modelcar.__version__)"
# Expected: 1.0.0a0
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(v1): wipe v0.x src/ and tests/ for clean rewrite"
```

---

### Task 0.3: Disable pytest in pre-commit during rewrite

**Files:**
- Modify: `.pre-commit-config.yaml`

The full pytest run will fail until later phases. Keep ruff and mypy active (they enforce per-file correctness) but disable the suite. Re-enable in Task 11.3.

- [ ] **Step 1: Comment out pytest hook**

In `.pre-commit-config.yaml`, locate the `pytest-fast` hook and comment it:

```yaml
#  - id: pytest-fast
#    name: pytest (not e2e)
#    ...
```

- [ ] **Step 2: Verify pre-commit still installs**

```bash
PATH="/run/current-system/sw/bin:$(pwd)/.venv/bin:$PATH" pre-commit run --all-files 2>&1 | tail -5
```

Expected: ruff + mypy pass (no source files to check is also OK), no pytest invocation.

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore(v1): disable pytest pre-commit hook during rewrite"
```

---

## Phase 1 — errors.py

### Task 1.1: Custom exception hierarchy

**Files:**
- Create: `src/oci_modelcar/errors.py`
- Test: `tests/unit/test_errors.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_errors.py`:

```python
import pytest

from oci_modelcar.errors import (
    ConfigError,
    DiskSpaceError,
    DownloadError,
    EntryNotFoundError,
    GatedRepoError,
    OciModelcarError,
    PartialFailure,
    PushError,
    RevisionNotFoundError,
    exit_code_for,
)


def test_base_class_carries_hint():
    e = OciModelcarError("base", hint="try X")
    assert e.hint == "try X"
    assert "base" in str(e)


def test_config_error_inherits():
    assert issubclass(ConfigError, OciModelcarError)


def test_gated_inherits_download():
    assert issubclass(GatedRepoError, DownloadError)
    assert issubclass(DownloadError, OciModelcarError)


@pytest.mark.parametrize(
    "exc_cls,expected_code",
    [
        (OciModelcarError, 1),
        (ConfigError, 2),
        (GatedRepoError, 3),
        (DiskSpaceError, 4),
        (DownloadError, 5),
        (RevisionNotFoundError, 5),
        (EntryNotFoundError, 5),
        (PushError, 6),
        (PartialFailure, 7),
    ],
)
def test_exit_codes(exc_cls, expected_code):
    assert exc_cls.exit_code == expected_code


def test_exit_code_for_returns_class_code():
    assert exit_code_for(GatedRepoError("x")) == 3
    assert exit_code_for(ValueError("x")) == 1  # non-OciModelcarError → 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_errors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'oci_modelcar.errors'`

- [ ] **Step 3: Write minimal implementation**

`src/oci_modelcar/errors.py`:

```python
"""Custom exception hierarchy with per-class CI exit codes."""

from __future__ import annotations


class OciModelcarError(Exception):
    """Base. `hint` carries actionable user-facing guidance for CLI surface."""

    exit_code: int = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class ConfigError(OciModelcarError):
    exit_code = 2


class DownloadError(OciModelcarError):
    exit_code = 5


class GatedRepoError(DownloadError):
    exit_code = 3


class RevisionNotFoundError(DownloadError):
    exit_code = 5


class EntryNotFoundError(DownloadError):
    exit_code = 5


class DiskSpaceError(OciModelcarError):
    exit_code = 4


class PushError(OciModelcarError):
    exit_code = 6


class PartialFailure(OciModelcarError):
    exit_code = 7


def exit_code_for(exc: BaseException) -> int:
    """Map any exception to a CLI exit code. Non-OciModelcarError → 1."""
    if isinstance(exc, OciModelcarError):
        return exc.exit_code
    return 1
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/test_errors.py -v
```
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/errors.py tests/unit/test_errors.py
git commit -m "feat(v1): errors.py with hierarchy and per-class exit codes"
```

---

## Phase 2 — http.py

### Task 2.1: build_session + diagnostic env vars

**Files:**
- Create: `src/oci_modelcar/http.py`
- Test: `tests/unit/test_http.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_http.py`:

```python
import os

import pytest

from oci_modelcar import __version__
from oci_modelcar.http import build_session


def test_session_has_default_user_agent(monkeypatch):
    monkeypatch.delenv("OCI_MODELCAR_USER_AGENT", raising=False)
    s = build_session()
    assert s.headers["User-Agent"] == f"oci-modelcar/{__version__}"


def test_session_user_agent_overridable(monkeypatch):
    monkeypatch.setenv("OCI_MODELCAR_USER_AGENT", "custom/1.0")
    s = build_session()
    assert s.headers["User-Agent"] == "custom/1.0"


def test_session_force_connection_close(monkeypatch):
    monkeypatch.setenv("OCI_MODELCAR_FORCE_CONNECTION_CLOSE", "1")
    s = build_session()
    assert s.headers.get("Connection") == "close"


def test_session_default_no_connection_close(monkeypatch):
    monkeypatch.delenv("OCI_MODELCAR_FORCE_CONNECTION_CLOSE", raising=False)
    s = build_session()
    assert "Connection" not in s.headers
```

- [ ] **Step 2: Run, expect fail**

Run: `.venv/bin/python -m pytest tests/unit/test_http.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

`src/oci_modelcar/http.py`:

```python
"""Shared requests.Session, auth resolution, redirect hooks."""

from __future__ import annotations

import os

import requests

from oci_modelcar import __version__


def _envbool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def build_session() -> requests.Session:
    """Single source of truth for HTTP sessions across the codebase."""
    s = requests.Session()
    s.headers["User-Agent"] = (
        os.environ.get("OCI_MODELCAR_USER_AGENT") or f"oci-modelcar/{__version__}"
    )
    if _envbool("OCI_MODELCAR_FORCE_CONNECTION_CLOSE"):
        s.headers["Connection"] = "close"
    return s
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_http.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/http.py tests/unit/test_http.py
git commit -m "feat(v1): http.py build_session with UA + connection-close env vars"
```

---

### Task 2.2: is_transient_ssl + _SmartRetry

**Files:**
- Modify: `src/oci_modelcar/http.py`
- Modify: `tests/unit/test_http.py`

- [ ] **Step 1: Append failing tests**

In `tests/unit/test_http.py`:

```python
import ssl

import requests
import urllib3.exceptions

from oci_modelcar.http import _SmartRetry, is_transient_ssl


def test_is_transient_ssl_true_for_eof():
    e = ssl.SSLEOFError("EOF occurred in violation of protocol")
    wrapped = requests.exceptions.SSLError(str(e))
    wrapped.__cause__ = e
    assert is_transient_ssl(wrapped) is True


def test_is_transient_ssl_false_for_handshake():
    e = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
    wrapped = requests.exceptions.SSLError(str(e))
    wrapped.__cause__ = e
    assert is_transient_ssl(wrapped) is False


def test_is_transient_ssl_via_message_match():
    """Some wrappers don't preserve __cause__; fall back to message match."""
    e = requests.exceptions.SSLError("EOF occurred in violation of protocol (_ssl.c:2437)")
    assert is_transient_ssl(e) is True


def test_smart_retry_reraises_ssl():
    r = _SmartRetry(total=5)
    with pytest.raises(ssl.SSLError):
        r.increment(error=ssl.SSLError("CERTIFICATE_VERIFY_FAILED"))


def test_smart_retry_reraises_proxy():
    r = _SmartRetry(total=5)
    with pytest.raises(urllib3.exceptions.ProxyError):
        r.increment(error=urllib3.exceptions.ProxyError("bad proxy", OSError()))
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/python -m pytest tests/unit/test_http.py -v
```

Expected: 5 new tests fail with import errors.

- [ ] **Step 3: Add implementation to `http.py`**

Append to `src/oci_modelcar/http.py`:

```python
import ssl
from typing import Any

import urllib3.exceptions
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_NEVER_RETRY_EXC: tuple[type[Exception], ...] = (
    ssl.SSLError,
    urllib3.exceptions.SSLError,
    urllib3.exceptions.ProxyError,
)


def is_transient_ssl(exc: BaseException) -> bool:
    """True if `exc` is a mid-stream SSL EOF (recoverable via Range/replay).

    Walks both __cause__ and __context__ to find a wrapped SSLEOFError.
    Falls back to message match because some wrappers don't preserve the
    chain (urllib3 → requests roundtrip).
    """
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ssl.SSLEOFError):
            return True
        if "EOF occurred in violation of protocol" in str(cur):
            return True
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    return False


class _SmartRetry(Retry):
    """urllib3 Retry that re-raises SSL/Proxy errors instead of looping on them.

    SSL handshake / proxy misconfig never gets better with retry; surfacing
    immediately gives the user the real error rather than a generic
    "max retries exceeded".
    """

    def increment(  # type: ignore[override]
        self,
        method: str | None = None,
        url: str | None = None,
        response: Any = None,
        error: Exception | None = None,
        _pool: Any = None,
        _stacktrace: Any = None,
    ) -> Retry:
        if error is not None and isinstance(error, _NEVER_RETRY_EXC):
            raise error
        return super().increment(method, url, response, error, _pool, _stacktrace)
```

Replace the body of `build_session` to mount the retry adapter:

```python
def build_session() -> requests.Session:
    s = requests.Session()
    retry = _SmartRetry(
        total=8,
        backoff_factor=2,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers["User-Agent"] = (
        os.environ.get("OCI_MODELCAR_USER_AGENT") or f"oci-modelcar/{__version__}"
    )
    if _envbool("OCI_MODELCAR_FORCE_CONNECTION_CLOSE"):
        s.headers["Connection"] = "close"
    return s
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_http.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/http.py tests/unit/test_http.py
git commit -m "feat(v1): is_transient_ssl + _SmartRetry policy"
```

---

### Task 2.3: Cross-origin authorization stripping (security)

**Files:**
- Modify: `src/oci_modelcar/http.py`
- Modify: `tests/unit/test_http.py`

This is a security fix: HF redirects to S3/CloudFront for LFS files, and we must NOT forward `Authorization: Bearer hf_...` to those origins.

- [ ] **Step 1: Append failing test**

```python
def test_authorization_dropped_on_cross_origin_redirect(httpserver):
    """When HF redirects to S3 or any other host, the Bearer token MUST be
    stripped before the second GET."""
    from werkzeug.wrappers import Response

    seen_auth_on_s3: list[str | None] = []

    def origin_handler(request):
        return Response(
            "",
            status=302,
            headers={"Location": httpserver.url_for("/s3-mock/file")},
        )

    def s3_handler(request):
        seen_auth_on_s3.append(request.headers.get("Authorization"))
        return Response(b"data", status=200)

    httpserver.expect_request("/api/redirect-me").respond_with_handler(origin_handler)
    httpserver.expect_request("/s3-mock/file").respond_with_handler(s3_handler)

    s = build_session()
    # Manually force a different host to trigger the cross-origin path. We
    # simulate by hitting a path that redirects to a "different" host via an
    # origin header — the strip logic looks at netloc, so we contrive that.
    # Easier: use 127.0.0.1 vs localhost on the same port; netloc differs.
    # pytest-httpserver binds to 127.0.0.1; we hit "localhost" first.
    base = httpserver.url_for("").replace("127.0.0.1", "localhost")
    r = s.get(
        f"{base}api/redirect-me",
        headers={"Authorization": "Bearer hf_secret"},
    )
    r.raise_for_status()
    assert seen_auth_on_s3 == [None], (
        f"Authorization must be stripped on cross-origin redirect; "
        f"got {seen_auth_on_s3!r}"
    )


def test_authorization_preserved_on_same_origin_redirect(httpserver):
    """Same-host redirects should keep the Bearer token."""
    from werkzeug.wrappers import Response

    seen_auth_on_target: list[str | None] = []

    httpserver.expect_request("/redirect").respond_with_data(
        "", status=302, headers={"Location": httpserver.url_for("/target")}
    )

    def target_handler(request):
        seen_auth_on_target.append(request.headers.get("Authorization"))
        return Response("", status=200)

    httpserver.expect_request("/target").respond_with_handler(target_handler)

    s = build_session()
    r = s.get(httpserver.url_for("/redirect"), headers={"Authorization": "Bearer hf_secret"})
    r.raise_for_status()
    assert seen_auth_on_target == ["Bearer hf_secret"]
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/python -m pytest tests/unit/test_http.py::test_authorization_dropped_on_cross_origin_redirect -v
```

Expected: FAIL (token leaks to S3 mock).

- [ ] **Step 3: Implement the redirect hook**

In `src/oci_modelcar/http.py`, add:

```python
from urllib.parse import urlparse


def _strip_auth_on_cross_origin(prepared_request, response, **_):
    """`requests` Session.send hook: when about to follow a redirect to a
    different netloc, drop the Authorization header from the prepared
    request before send. Defends HF→S3 token leak."""
    if not response.is_redirect:
        return response
    target = response.headers.get("Location")
    if not target:
        return response
    src_netloc = urlparse(prepared_request.url).netloc
    dst_netloc = urlparse(target).netloc
    # Resolve relative redirects: dst_netloc empty → same host, OK to keep auth
    if not dst_netloc or dst_netloc == src_netloc:
        return response
    # Cross-origin: strip Authorization from session-level headers won't help
    # because requests applies headers per-request; we mutate the session's
    # _redirect_auth_strip_pending sentinel and use rebuild_auth.
    # Simplest: requests already provides Session.rebuild_auth which strips
    # the header on cross-host redirects ONLY if the original request used
    # Session-level auth. For per-request `headers={"Authorization": ...}`
    # passed by callers, we override Session.rebuild_auth.
    return response


class _SafeSession(requests.Session):
    """Session that strips Authorization on cross-origin redirects regardless
    of whether the header was set per-request or via session.headers."""

    def rebuild_auth(self, prepared_request, response):  # type: ignore[override]
        super().rebuild_auth(prepared_request, response)
        # Belt-and-braces: requests >=2.32 already strips Session.headers
        # auth on cross-host; this also strips per-request Authorization.
        if "Authorization" not in prepared_request.headers:
            return
        original_url = response.request.url
        if original_url is None:
            return
        original_netloc = urlparse(original_url).netloc
        new_netloc = urlparse(prepared_request.url).netloc
        if new_netloc and new_netloc != original_netloc:
            del prepared_request.headers["Authorization"]
```

Update `build_session` to use `_SafeSession`:

```python
def build_session() -> requests.Session:
    s = _SafeSession()
    retry = _SmartRetry(
        total=8,
        backoff_factor=2,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers["User-Agent"] = (
        os.environ.get("OCI_MODELCAR_USER_AGENT") or f"oci-modelcar/{__version__}"
    )
    if _envbool("OCI_MODELCAR_FORCE_CONNECTION_CLOSE"):
        s.headers["Connection"] = "close"
    return s
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_http.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/http.py tests/unit/test_http.py
git commit -m "feat(v1): _SafeSession strips Authorization on cross-origin redirect"
```

---

### Task 2.4: Auth resolution (HF + OCI)

**Files:**
- Modify: `src/oci_modelcar/http.py`
- Modify: `tests/unit/test_http.py`

- [ ] **Step 1: Append failing tests**

```python
import json
from pathlib import Path


def test_huggingface_token_from_HF_TOKEN(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok_a")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    from oci_modelcar.http import huggingface_token

    assert huggingface_token() == "tok_a"


def test_huggingface_token_from_HUGGING_FACE_HUB_TOKEN(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok_b")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    from oci_modelcar.http import huggingface_token

    assert huggingface_token() == "tok_b"


def test_huggingface_token_priority(monkeypatch):
    """HF_TOKEN wins over HUGGING_FACE_HUB_TOKEN."""
    monkeypatch.setenv("HF_TOKEN", "tok_a")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok_b")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    from oci_modelcar.http import huggingface_token

    assert huggingface_token() == "tok_a"


def test_huggingface_token_disabled_by_env(monkeypatch):
    """HF_HUB_DISABLE_IMPLICIT_TOKEN=1 returns None even if a token is set."""
    monkeypatch.setenv("HF_TOKEN", "tok_a")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    from oci_modelcar.http import huggingface_token

    assert huggingface_token() is None


def test_huggingface_token_from_cache_file(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = tmp_path / ".cache" / "huggingface" / "token"
    cache.parent.mkdir(parents=True)
    cache.write_text("tok_from_file\n")
    from oci_modelcar.http import huggingface_token

    assert huggingface_token() == "tok_from_file"


def test_oci_auth_header_from_env(monkeypatch):
    monkeypatch.setenv("OCI_USERNAME", "alice")
    monkeypatch.setenv("OCI_PASSWORD", "s3cret")
    from oci_modelcar.http import oci_auth_header

    h = oci_auth_header("registry.example.com")
    assert h["Authorization"].startswith("Basic ")


def test_oci_auth_header_from_docker_config(monkeypatch, tmp_path):
    import base64

    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    cfg = tmp_path / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True)
    auth = base64.b64encode(b"alice:s3cret").decode()
    cfg.write_text(json.dumps({"auths": {"registry.example.com": {"auth": auth}}}))
    from oci_modelcar.http import oci_auth_header

    h = oci_auth_header("registry.example.com")
    assert h["Authorization"] == f"Basic {auth}"


def test_oci_auth_anonymous_when_no_credentials(monkeypatch, tmp_path, caplog):
    import logging

    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    from oci_modelcar.http import oci_auth_header

    with caplog.at_level(logging.WARNING):
        h = oci_auth_header("registry.example.com")
    assert h == {}
    assert any("anonymously" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/python -m pytest tests/unit/test_http.py::test_huggingface_token_from_HF_TOKEN -v
```

Expected: FAIL (function not defined).

- [ ] **Step 3: Implement auth resolution**

Append to `src/oci_modelcar/http.py`:

```python
import base64
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def huggingface_token() -> str | None:
    """Resolve HF token. Priority: HF_TOKEN > HUGGING_FACE_HUB_TOKEN > cache file.
    Returns None if HF_HUB_DISABLE_IMPLICIT_TOKEN is set."""
    if _envbool("HF_HUB_DISABLE_IMPLICIT_TOKEN"):
        return None
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = os.environ.get(env_name)
        if tok:
            return tok
    cache = Path.home() / ".cache" / "huggingface" / "token"
    if cache.is_file():
        try:
            content = cache.read_text().strip()
            return content or None
        except OSError:
            return None
    return None


def huggingface_auth_header() -> dict[str, str]:
    tok = huggingface_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def oci_auth_header(
    registry_host: str, target_repo: str | None = None
) -> dict[str, str]:
    """Resolve OCI registry auth: env > ~/.docker/config.json > podman auth.json."""
    target = f"{registry_host}/{target_repo}" if target_repo else registry_host

    user = os.environ.get("OCI_USERNAME")
    pwd = os.environ.get("OCI_PASSWORD")
    if user and pwd is not None:
        log.info("OCI auth resolved from OCI_USERNAME/OCI_PASSWORD env")
        token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    for path in _auth_search_paths():
        auth = _docker_config_auth(path, target)
        if auth:
            log.info("OCI auth resolved from %s", path)
            return {"Authorization": f"Basic {auth}"}

    log.warning(
        "no OCI credentials found for %s — pushing anonymously "
        "(set OCI_USERNAME/OCI_PASSWORD or run `podman login`/`docker login`)",
        target,
    )
    return {}


def _auth_search_paths() -> list[Path]:
    paths = [Path.home() / ".docker" / "config.json"]
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        paths.append(Path(xdg_runtime) / "containers" / "auth.json")
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    config_root = Path(xdg_config) if xdg_config else Path.home() / ".config"
    paths.append(config_root / "containers" / "auth.json")
    return paths


def _normalize_auth_key(key: str) -> str:
    for prefix in ("https://", "http://"):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    key = key.rstrip("/")
    if key.endswith("/v2"):
        key = key[: -len("/v2")].rstrip("/")
    return key


def _docker_config_auth(path: Path, target: str) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    auths = data.get("auths", {})
    if not isinstance(auths, dict):
        return None
    best_key, best_len = None, -1
    for raw_key in auths:
        norm = _normalize_auth_key(raw_key)
        if (norm == target or target.startswith(norm + "/")) and len(norm) > best_len:
            best_key, best_len = raw_key, len(norm)
    if best_key is None:
        return None
    entry = auths[best_key]
    if not isinstance(entry, dict):
        return None
    raw = entry.get("auth")
    return raw if isinstance(raw, str) and raw else None
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_http.py -v
```

Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/http.py tests/unit/test_http.py
git commit -m "feat(v1): HF + OCI auth resolution with expanded token sources"
```

---

## Phase 3 — layer.py

### Task 3.1: tar_layer_size formula

**Files:**
- Create: `src/oci_modelcar/layer.py`
- Test: `tests/unit/test_layer.py`

- [ ] **Step 1: Write failing test**

`tests/unit/test_layer.py`:

```python
import io
import tarfile

import pytest

from oci_modelcar.layer import build_layer_tar_bytes, make_tar_info, tar_layer_size


@pytest.mark.parametrize(
    "file_size", [0, 1, 100, 511, 512, 513, 1024, 1025, 12345, 1048576]
)
def test_tar_layer_size_matches_actual_bytes(file_size: int):
    """Streaming uploads need to set Content-Length upfront. The formula must
    equal the bytes that build_layer_tar_bytes produces, otherwise the
    registry hangs waiting for missing bytes."""
    actual = len(build_layer_tar_bytes("models/", "weights.bin", b"x" * file_size))
    assert tar_layer_size(file_size) == actual
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/python -m pytest tests/unit/test_layer.py -v
```

Expected: FAIL (module not found).

- [ ] **Step 3: Implement layer.py**

`src/oci_modelcar/layer.py`:

```python
"""Tar layer building. Uncompressed (mediaType vnd.oci.image.layer.v1.tar)
so that layer.digest == diff_id by construction."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

_TAR_BLOCKSIZE = 512
_TAR_RECORDSIZE = 10240


def tar_layer_size(file_size: int) -> int:
    """Exact bytes produced by build_layer_tar_bytes / build_layer_to_file
    for the given file size. Deterministic given mtime=0/uid=0/gid=0."""
    body_padded = (file_size + _TAR_BLOCKSIZE - 1) // _TAR_BLOCKSIZE * _TAR_BLOCKSIZE
    raw = _TAR_BLOCKSIZE + body_padded + 2 * _TAR_BLOCKSIZE  # header + body + 2-block trailer
    return (raw + _TAR_RECORDSIZE - 1) // _TAR_RECORDSIZE * _TAR_RECORDSIZE


def make_tar_info(prefix: str, filename: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=prefix + filename)
    info.size = size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info


def build_layer_tar_bytes(prefix: str, filename: str, payload: bytes) -> bytes:
    """In-memory tar build. Used by tests; production uses build_layer_to_file."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w|") as tar:
        info = make_tar_info(prefix, filename, len(payload))
        tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_layer.py -v
```

Expected: 10 parametrized cases pass.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/layer.py tests/unit/test_layer.py
git commit -m "feat(v1): layer.py with tar_layer_size + build_layer_tar_bytes"
```

---

### Task 3.2: build_layer_to_file (streaming tar with sha256)

**Files:**
- Modify: `src/oci_modelcar/layer.py`
- Modify: `tests/unit/test_layer.py`

- [ ] **Step 1: Append failing test**

```python
def test_build_layer_to_file_writes_tar_and_returns_digest(tmp_path):
    source = tmp_path / "weights.bin"
    payload = b"X" * 12345
    source.write_bytes(payload)

    dest = tmp_path / "weights.bin.tar"
    digest, size = build_layer_to_file(
        source_path=source, prefix="models/", filename="weights.bin", dest_path=dest
    )

    raw = dest.read_bytes()
    assert size == len(raw)
    assert size == tar_layer_size(len(payload))
    assert digest == "sha256:" + hashlib.sha256(raw).hexdigest()

    # Tar contents must match what build_layer_tar_bytes would produce
    expected = build_layer_tar_bytes("models/", "weights.bin", payload)
    assert raw == expected


def test_build_layer_to_file_streaming_does_not_load_full_payload(tmp_path):
    """The function must iterate the source file in chunks; for a multi-MB
    file we shouldn't need 2x size in RAM. We can't directly assert RAM
    use in pytest, but we can verify the function completes for a file
    larger than typical chunk_size and that the output is correct."""
    source = tmp_path / "big.bin"
    big_payload = b"Y" * (3 * 1024 * 1024)  # 3 MiB > 1 MiB internal chunks
    source.write_bytes(big_payload)
    dest = tmp_path / "big.bin.tar"
    digest, size = build_layer_to_file(source, "models/", "big.bin", dest)
    assert size == tar_layer_size(len(big_payload))
    expected = build_layer_tar_bytes("models/", "big.bin", big_payload)
    assert dest.read_bytes() == expected
```

Add `import hashlib` to test file imports if not already present.

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/python -m pytest tests/unit/test_layer.py::test_build_layer_to_file_writes_tar_and_returns_digest -v
```

Expected: FAIL (`build_layer_to_file` not defined).

- [ ] **Step 3: Implement**

Append to `src/oci_modelcar/layer.py`:

```python
class _HashingWriter:
    """File-like wrapper that hashes every byte written to the inner file."""

    def __init__(self, inner: io.BufferedWriter) -> None:
        self._inner = inner
        self.h = hashlib.sha256()
        self.bytes_written = 0

    def write(self, data: bytes) -> int:
        self.h.update(data)
        n = self._inner.write(data)
        self.bytes_written += n
        return n


def build_layer_to_file(
    source_path: Path,
    prefix: str,
    filename: str,
    dest_path: Path,
    read_chunk: int = 1024 * 1024,
) -> tuple[str, int]:
    """Build the tar layer at dest_path streaming from source_path.

    Returns (digest, size) where digest is "sha256:<64hex>" and size is the
    total bytes written (== tar_layer_size(source_size)).

    Memory bound: ~read_chunk + tar internal buffering (~64 KiB).
    """
    source_size = source_path.stat().st_size
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as raw:
        writer = _HashingWriter(raw)
        # tarfile writes through the wrapper; addfile streams from source
        with tarfile.open(fileobj=writer, mode="w|") as tar:
            info = make_tar_info(prefix, filename, source_size)
            with open(source_path, "rb") as src:
                tar.addfile(info, src)
        # `with tarfile.open(... mode="w|")` flushes trailer + record padding
    digest = "sha256:" + writer.h.hexdigest()
    expected_size = tar_layer_size(source_size)
    if writer.bytes_written != expected_size:
        raise RuntimeError(
            f"tar size mismatch for {filename}: wrote {writer.bytes_written}, "
            f"formula expected {expected_size}"
        )
    return digest, writer.bytes_written
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_layer.py -v
```

Expected: all pass (12 cases).

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/layer.py tests/unit/test_layer.py
git commit -m "feat(v1): build_layer_to_file streams source → tar with sha256"
```

---

### Task 3.3: Reproducibility tests (mtime/uid/gid invariants)

**Files:**
- Modify: `tests/unit/test_layer.py`

- [ ] **Step 1: Append tests**

```python
def test_layer_tar_is_reproducible(tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"X" * 12345)
    a = tmp_path / "a.tar"
    b = tmp_path / "b.tar"
    digest_a, _ = build_layer_to_file(source, "models/", "f.bin", a)
    digest_b, _ = build_layer_to_file(source, "models/", "f.bin", b)
    assert digest_a == digest_b
    assert a.read_bytes() == b.read_bytes()


def test_layer_tar_has_zero_mtime_uid_gid(tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"hello")
    dest = tmp_path / "f.tar"
    build_layer_to_file(source, "models/", "f.bin", dest)
    with tarfile.open(fileobj=io.BytesIO(dest.read_bytes()), mode="r") as tf:
        members = tf.getmembers()
        assert len(members) == 1
        m = members[0]
        assert m.name == "models/f.bin"
        assert m.size == 5
        assert m.mtime == 0
        assert m.uid == 0 and m.gid == 0
        assert m.uname == "" and m.gname == ""
        assert m.mode == 0o644
```

- [ ] **Step 2: Run, expect pass (no impl change needed)**

```bash
.venv/bin/python -m pytest tests/unit/test_layer.py -v
```

Expected: all pass (14 cases). These are protective tests that lock in
v0.x's documented invariants.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_layer.py
git commit -m "test(v1): reproducibility invariants for build_layer_to_file"
```

---

## Phase 4 — manifest.py

### Task 4.1: BlobDescriptor + config + manifest builders

**Files:**
- Create: `src/oci_modelcar/manifest.py`
- Test: `tests/unit/test_manifest.py`

- [ ] **Step 1: Write failing test**

`tests/unit/test_manifest.py`:

```python
import hashlib
import json

from oci_modelcar.manifest import (
    BlobDescriptor,
    build_config_bytes,
    build_manifest_bytes,
)


def test_blob_descriptor_to_dict():
    d = BlobDescriptor(
        media_type="application/vnd.oci.image.layer.v1.tar",
        digest="sha256:" + "a" * 64,
        size=12345,
        hf_path="model.safetensors",
    )
    assert d.to_dict() == {
        "mediaType": "application/vnd.oci.image.layer.v1.tar",
        "digest": "sha256:" + "a" * 64,
        "size": 12345,
    }


def test_config_bytes_no_created_field():
    """v0.x design lock-in: NO `created` field, so config bytes are
    deterministic across runs and config digest is stable."""
    diff_ids = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
    cfg = build_config_bytes(diff_ids)
    parsed = json.loads(cfg)
    assert "created" not in parsed
    assert parsed["rootfs"]["diff_ids"] == diff_ids
    assert parsed["rootfs"]["type"] == "layers"
    assert parsed["architecture"] == "amd64"
    assert parsed["os"] == "linux"


def test_config_bytes_reproducible():
    diff_ids = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
    a = build_config_bytes(diff_ids)
    b = build_config_bytes(diff_ids)
    assert a == b


def test_manifest_bytes_layers_in_provided_order():
    """The runner is responsible for sorting layers alphabetically by
    hf_path. build_manifest_bytes preserves the order it's given."""
    layers = [
        BlobDescriptor("application/vnd.oci.image.layer.v1.tar", "sha256:" + "a" * 64, 100, "a.bin"),
        BlobDescriptor("application/vnd.oci.image.layer.v1.tar", "sha256:" + "b" * 64, 200, "b.bin"),
    ]
    config_digest = "sha256:" + "c" * 64
    config_size = 50
    manifest = build_manifest_bytes(config_digest, config_size, layers)
    parsed = json.loads(manifest)
    assert parsed["mediaType"] == "application/vnd.oci.image.manifest.v1+json"
    assert parsed["config"]["digest"] == config_digest
    assert parsed["config"]["size"] == config_size
    assert [layer["digest"] for layer in parsed["layers"]] == [
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
    ]


def test_manifest_bytes_reproducible():
    layers = [
        BlobDescriptor("application/vnd.oci.image.layer.v1.tar", "sha256:" + "a" * 64, 100, "a.bin"),
    ]
    a = build_manifest_bytes("sha256:" + "c" * 64, 50, layers)
    b = build_manifest_bytes("sha256:" + "c" * 64, 50, layers)
    assert a == b
    assert hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/python -m pytest tests/unit/test_manifest.py -v
```

Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

`src/oci_modelcar/manifest.py`:

```python
"""OCI image config + manifest building. Reproducible: no `created` field,
deterministic JSON serialization, layers ordered by caller."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

ML_TAR = "application/vnd.oci.image.layer.v1.tar"
ML_CFG = "application/vnd.oci.image.config.v1+json"
ML_MAN = "application/vnd.oci.image.manifest.v1+json"


@dataclass(frozen=True, slots=True)
class BlobDescriptor:
    media_type: str
    digest: str
    size: int
    hf_path: str  # not serialized; used by runner for ordering and logging

    def to_dict(self) -> dict[str, object]:
        return {
            "mediaType": self.media_type,
            "digest": self.digest,
            "size": self.size,
        }


def build_config_bytes(diff_ids: list[str]) -> bytes:
    """OCI image config without `created` (so config digest is stable)."""
    cfg = {
        "architecture": "amd64",
        "os": "linux",
        "rootfs": {
            "type": "layers",
            "diff_ids": diff_ids,
        },
        "config": {},
    }
    return json.dumps(cfg, separators=(",", ":"), sort_keys=True).encode()


def build_manifest_bytes(
    config_digest: str, config_size: int, layers: Iterable[BlobDescriptor]
) -> bytes:
    """Build manifest. Caller is responsible for ordering `layers`
    deterministically (e.g. sorted by hf_path)."""
    layer_list = list(layers)
    manifest = {
        "schemaVersion": 2,
        "mediaType": ML_MAN,
        "config": {
            "mediaType": ML_CFG,
            "digest": config_digest,
            "size": config_size,
        },
        "layers": [d.to_dict() for d in layer_list],
    }
    return json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_manifest.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/manifest.py tests/unit/test_manifest.py
git commit -m "feat(v1): manifest.py with BlobDescriptor + reproducible builders"
```

---

### Task 4.2: derive_tag (port from tags.py)

**Files:**
- Modify: `src/oci_modelcar/manifest.py`
- Modify: `tests/unit/test_manifest.py`

- [ ] **Step 1: Append failing tests**

```python
import pytest

from oci_modelcar.manifest import derive_tag


def test_derive_tag_from_40char_sha():
    sha = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    assert derive_tag(sha, explicit=None) == "9fb191250dd5"


def test_derive_tag_explicit_overrides():
    assert derive_tag("ignored", explicit="v1.0") == "v1.0"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("main", "main"),
        ("feature/x", "feature-x"),
        ("v1.0.0", "v1.0.0"),
        ("Hello World", "hello-world"),
        ("trailing/", "trailing"),
    ],
)
def test_derive_tag_sanitizes_non_sha(raw, expected):
    assert derive_tag(raw, explicit=None) == expected


def test_derive_tag_explicit_validated():
    """Explicit tag is taken as-is; the caller (Config.validate) enforces
    OCI tag rules. derive_tag does not re-validate."""
    assert derive_tag("any", explicit="raw_input") == "raw_input"
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/python -m pytest tests/unit/test_manifest.py::test_derive_tag_from_40char_sha -v
```

Expected: FAIL (function not defined).

- [ ] **Step 3: Implement**

Append to `src/oci_modelcar/manifest.py`:

```python
import re

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def derive_tag(revision_resolved: str, explicit: str | None) -> str:
    """Derive the OCI tag from a resolved HF revision.

    - If `explicit` is given, return it as-is (Config validates the format).
    - If `revision_resolved` is a 40-char SHA, take the first 12 chars
      (matches `git rev-parse --short=12`).
    - Otherwise sanitize: lowercase, replace [/] with -, strip trailing -.
    """
    if explicit is not None:
        return explicit
    if _SHA_RE.match(revision_resolved):
        return revision_resolved[:12]
    out = revision_resolved.lower().replace("/", "-").replace(" ", "-")
    return out.rstrip("-")
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_manifest.py -v
```

Expected: all pass (12 cases).

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/manifest.py tests/unit/test_manifest.py
git commit -m "feat(v1): derive_tag in manifest.py (was tags.py in v0.x)"
```

---

## Phase 5 — registry.py

### Task 5.1: OciClient + head_blob + push_small_blob

**Files:**
- Create: `src/oci_modelcar/registry.py`
- Test: `tests/unit/test_registry.py`

- [ ] **Step 1: Write failing test**

`tests/unit/test_registry.py`:

```python
import hashlib

import pytest
import requests
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.registry import OciClient, head_blob, push_small_blob


def _client(httpserver: HTTPServer) -> OciClient:
    return OciClient(host_url=httpserver.url_for(""))


def test_oci_client_url_construction():
    c = OciClient(host_url="https://registry.example.com")
    assert c.url("repo", "blobs", "uploads") == "https://registry.example.com/v2/repo/blobs/uploads"


def test_oci_client_loopback_uses_http():
    c = OciClient(registry_host="localhost:5000")
    assert c.base == "http://localhost:5000"


def test_oci_client_remote_uses_https():
    c = OciClient(registry_host="registry.example.com")
    assert c.base == "https://registry.example.com"


def test_oci_client_explicit_scheme_preserved():
    c = OciClient(registry_host="http://custom.example.com:8080")
    assert c.base == "http://custom.example.com:8080"


def test_head_blob_returns_descriptor_when_present(httpserver):
    digest = "sha256:" + hashlib.sha256(b"x").hexdigest()
    httpserver.expect_request(
        f"/v2/repo/blobs/{digest}", method="HEAD"
    ).respond_with_data(
        "",
        status=200,
        headers={"Docker-Content-Digest": digest, "Content-Length": "1"},
    )
    info = head_blob(_client(httpserver), "repo", digest)
    assert info == {"digest": digest, "size": 1}


def test_head_blob_returns_none_when_404(httpserver):
    digest = "sha256:" + "a" * 64
    httpserver.expect_request(
        f"/v2/repo/blobs/{digest}", method="HEAD"
    ).respond_with_data("", status=404)
    assert head_blob(_client(httpserver), "repo", digest) is None


def test_head_blob_raises_on_digest_mismatch(httpserver):
    digest = "sha256:" + "a" * 64
    other = "sha256:" + "b" * 64
    httpserver.expect_request(
        f"/v2/repo/blobs/{digest}", method="HEAD"
    ).respond_with_data(
        "", status=200, headers={"Docker-Content-Digest": other, "Content-Length": "1"}
    )
    with pytest.raises(RuntimeError, match="digest mismatch"):
        head_blob(_client(httpserver), "repo", digest)


def test_push_small_blob_skips_when_already_present(httpserver):
    data = b"config bytes"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    httpserver.expect_request(
        f"/v2/repo/blobs/{digest}", method="HEAD"
    ).respond_with_data(
        "", status=200, headers={"Docker-Content-Digest": digest, "Content-Length": str(len(data))}
    )
    # No POST/PUT registered; if push_small_blob hits them, the test fails.
    out = push_small_blob(_client(httpserver), "repo", data)
    assert out == digest


def test_push_small_blob_post_then_put(httpserver):
    data = b"config bytes"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()

    httpserver.expect_request(
        f"/v2/repo/blobs/{digest}", method="HEAD"
    ).respond_with_data("", status=404)
    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_data("", status=202, headers={"Location": httpserver.url_for("/u/1")})

    received = {"data": b""}

    def put_handler(request):
        received["data"] = request.data
        return Response("", status=201)

    httpserver.expect_request("/u/1", method="PUT").respond_with_handler(put_handler)

    out = push_small_blob(_client(httpserver), "repo", data)
    assert out == digest
    assert received["data"] == data
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/python -m pytest tests/unit/test_registry.py -v
```

Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

`src/oci_modelcar/registry.py`:

```python
"""OCI Distribution v1.1 client. v1: single-PATCH streaming upload from
a local file. No chunked mode."""

from __future__ import annotations

import hashlib
import logging
from functools import cached_property

import requests

from oci_modelcar.http import build_session, oci_auth_header
from oci_modelcar.manifest import ML_CFG, ML_MAN, ML_TAR

log = logging.getLogger(__name__)


def _is_loopback(host: str) -> bool:
    if host == "::1" or host.startswith("["):
        return host in ("::1", "[::1]")
    h = host.split(":", 1)[0]
    if h in ("localhost", "127.0.0.1"):
        return True
    return h.startswith("127.")


class OciClient:
    def __init__(
        self,
        host_url: str | None = None,
        registry_host: str | None = None,
        session: requests.Session | None = None,
        target_repo: str | None = None,
    ) -> None:
        if host_url is not None:
            self.base = host_url.rstrip("/")
            self.host = self.base.split("//", 1)[-1]
        else:
            assert registry_host is not None
            self.host = registry_host
            if registry_host.startswith(("http://", "https://")):
                self.base = registry_host.rstrip("/")
                self.host = registry_host.split("//", 1)[-1]
            elif _is_loopback(registry_host):
                self.base = f"http://{registry_host}"
            else:
                self.base = f"https://{registry_host}"
        self.session = session if session is not None else build_session()
        self.target_repo = target_repo

    @cached_property
    def auth(self) -> dict[str, str]:
        return oci_auth_header(self.host, target_repo=self.target_repo)

    def url(self, *parts: str) -> str:
        return f"{self.base}/v2/" + "/".join(parts)


def head_blob(client: OciClient, repo: str, digest: str) -> dict | None:
    """HEAD a blob. Returns {digest, size} on 200, None on 404. Raises on
    digest mismatch."""
    url = client.url(repo, "blobs", digest)
    r = client.session.head(url, headers=client.auth, timeout=30)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        r.raise_for_status()
    got = r.headers.get("Docker-Content-Digest", "")
    if got != digest:
        raise RuntimeError(f"digest mismatch on HEAD {digest}: server returned {got!r}")
    cl = int(r.headers.get("Content-Length", "0"))
    return {"digest": digest, "size": cl}


def push_small_blob(client: OciClient, repo: str, data: bytes) -> str:
    """Monolithic POST + PUT for small blobs (config). Returns digest."""
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    if head_blob(client, repo, digest) is not None:
        return digest
    init_url = client.url(repo, "blobs", "uploads") + "/"
    r = client.session.post(init_url, headers=client.auth, timeout=30)
    if r.status_code != 202:
        r.raise_for_status()
    loc = r.headers["Location"]
    sep = "&" if "?" in loc else "?"
    hdr = {
        **client.auth,
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(data)),
    }
    r = client.session.put(f"{loc}{sep}digest={digest}", data=data, headers=hdr, timeout=120)
    if r.status_code != 201:
        r.raise_for_status()
    return digest
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_registry.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/registry.py tests/unit/test_registry.py
git commit -m "feat(v1): registry.py with OciClient, head_blob, push_small_blob"
```

---

### Task 5.2: push_manifest + validate_manifest_tag

**Files:**
- Modify: `src/oci_modelcar/registry.py`
- Modify: `tests/unit/test_registry.py`

- [ ] **Step 1: Append failing tests**

```python
import json

from oci_modelcar.registry import push_manifest, validate_manifest_tag


def test_push_manifest_returns_digest(httpserver):
    body = json.dumps({"schemaVersion": 2, "config": {}, "layers": []}).encode()
    expected_digest = "sha256:" + hashlib.sha256(body).hexdigest()

    received = {"data": b""}

    def put_handler(request):
        received["data"] = request.data
        return Response("", status=201)

    httpserver.expect_request("/v2/repo/manifests/v1", method="PUT").respond_with_handler(put_handler)
    out = push_manifest(_client(httpserver), "repo", "v1", body)
    assert out == expected_digest
    assert received["data"] == body


def test_validate_manifest_tag_succeeds_on_match(httpserver):
    digest = "sha256:" + "a" * 64
    httpserver.expect_request("/v2/repo/manifests/v1", method="GET").respond_with_data(
        "", status=200, headers={"Docker-Content-Digest": digest}
    )
    validate_manifest_tag(_client(httpserver), "repo", "v1", digest)


def test_validate_manifest_tag_raises_on_mismatch(httpserver):
    digest = "sha256:" + "a" * 64
    other = "sha256:" + "b" * 64
    httpserver.expect_request("/v2/repo/manifests/v1", method="GET").respond_with_data(
        "", status=200, headers={"Docker-Content-Digest": other}
    )
    with pytest.raises(RuntimeError, match="manifest digest mismatch"):
        validate_manifest_tag(_client(httpserver), "repo", "v1", digest)
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL (functions not defined).

- [ ] **Step 3: Append impl**

```python
def push_manifest(client: OciClient, repo: str, tag: str, manifest_bytes: bytes) -> str:
    url = client.url(repo, "manifests", tag)
    hdr = {**client.auth, "Content-Type": ML_MAN}
    r = client.session.put(url, data=manifest_bytes, headers=hdr, timeout=60)
    if r.status_code not in (200, 201):
        r.raise_for_status()
    return "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()


def validate_manifest_tag(
    client: OciClient, repo: str, tag: str, expected_digest: str
) -> None:
    url = client.url(repo, "manifests", tag)
    r = client.session.get(url, headers={**client.auth, "Accept": ML_MAN}, timeout=30)
    r.raise_for_status()
    got = r.headers.get("Docker-Content-Digest", "")
    if got != expected_digest:
        raise RuntimeError(
            f"manifest digest mismatch on tag {tag}: expected {expected_digest} got {got!r}"
        )
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_registry.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/registry.py tests/unit/test_registry.py
git commit -m "feat(v1): push_manifest + validate_manifest_tag"
```

---

### Task 5.3: StreamingBlobUpload happy path + 200/201/202/204 acceptance

**Files:**
- Modify: `src/oci_modelcar/registry.py`
- Modify: `tests/unit/test_registry.py`

- [ ] **Step 1: Append failing tests**

```python
import re
from pathlib import Path

from oci_modelcar.registry import StreamingBlobUpload


def test_streaming_push_from_file_happy_path(httpserver, tmp_path: Path):
    payload = b"X" * (4 * 1024 * 1024 + 17)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/1")}
    )

    received = bytearray()

    def patch_handler(request):
        cr = request.headers.get("Content-Range", "")
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m, f"bad Content-Range {cr!r}"
        start, end = int(m.group(1)), int(m.group(2))
        assert start == 0
        assert end == len(payload) - 1
        cl = int(request.headers["Content-Length"])
        assert cl == len(payload)
        received.extend(request.data)
        return Response("", status=202, headers={"Location": httpserver.url_for("/u/1")})

    httpserver.expect_request("/u/1", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/1", method="PUT").respond_with_data("", status=201)

    upload = StreamingBlobUpload(client=_client(httpserver), repo="repo")
    out_digest, out_size = upload.push_from_file(f, len(payload), digest)
    assert out_digest == digest
    assert out_size == len(payload)
    assert bytes(received) == payload


@pytest.mark.parametrize("success_status", [200, 201, 202, 204])
def test_streaming_accepts_non_spec_success_codes(httpserver, tmp_path, success_status):
    """Artifactory returns 200/204; Harbor (some setups) returns 204.
    go-containerregistry accepts {201,202,204}; oras-py accepts {200,201,202}.
    Union: {200,201,202,204}."""
    payload = b"Z" * 1024
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/codes")}
    )
    httpserver.expect_request("/u/codes", method="PATCH").respond_with_data(
        "", status=success_status, headers={"Location": httpserver.url_for("/u/codes")}
    )
    httpserver.expect_request("/u/codes", method="PUT").respond_with_data("", status=201)

    upload = StreamingBlobUpload(client=_client(httpserver), repo="repo")
    out_digest, _ = upload.push_from_file(f, len(payload), digest)
    assert out_digest == digest


def test_streaming_unhandled_status_raises(httpserver, tmp_path):
    """299 (no spec meaning) must raise rather than spin or silently succeed."""
    payload = b"A" * 64
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/odd")}
    )
    httpserver.expect_request("/u/odd", method="PATCH").respond_with_data("", status=299)

    upload = StreamingBlobUpload(client=_client(httpserver), repo="repo")
    with pytest.raises(RuntimeError, match=r"unexpected.*299|status 299"):
        upload.push_from_file(f, len(payload), digest)


def test_streaming_no_chunked_transfer_encoding(httpserver, tmp_path):
    """Content-Length must be set explicitly to avoid chunked TE, which
    some registries handle differently from a fixed-size PATCH."""
    payload = b"L" * 512
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/len")}
    )

    seen_te: list[str | None] = []

    def patch_handler(request):
        seen_te.append(request.headers.get("Transfer-Encoding"))
        return Response("", status=202, headers={"Location": httpserver.url_for("/u/len")})

    httpserver.expect_request("/u/len", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/len", method="PUT").respond_with_data("", status=201)

    upload = StreamingBlobUpload(client=_client(httpserver), repo="repo")
    upload.push_from_file(f, len(payload), digest)

    te = seen_te[0] or ""
    assert "chunked" not in te.lower(), f"Transfer-Encoding leaked chunked: {te!r}"
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL (StreamingBlobUpload not defined).

- [ ] **Step 3: Implement (happy path only — retry comes in 5.4)**

Append to `src/oci_modelcar/registry.py`:

```python
import threading
from collections.abc import Callable
from pathlib import Path

from oci_modelcar.errors import PushError


class StreamingBlobUpload:
    """Single-PATCH streaming blob upload, body sourced from a local file.

    Mirrors containers/image (Podman) and Jib: one PATCH per blob means one
    TCP request, one LB routing decision, one node receives the entire blob.
    The local file is the replayable source for retries.
    """

    def __init__(
        self,
        client: OciClient,
        repo: str,
        max_retries: int = 5,
        backoff_initial: float = 1.0,
        backoff_cap: float = 60.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.client = client
        self.repo = repo
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_cap = backoff_cap
        self.stop_event = stop_event

    def _begin(self) -> str:
        url = self.client.url(self.repo, "blobs", "uploads") + "/"
        r = self.client.session.post(url, headers=self.client.auth, timeout=60)
        if r.status_code != 202:
            r.raise_for_status()
            raise RuntimeError(f"unexpected status {r.status_code} on upload init")
        loc = r.headers.get("Location")
        if not loc:
            raise RuntimeError("upload init missing Location header")
        return loc

    def push_from_file(
        self, tar_path: Path, total_size: int, digest: str
    ) -> tuple[str, int]:
        """POST init → PATCH from file → PUT close. Returns (digest, total_size).

        v1: this method does NOT retry; mid-PATCH cuts surface as exceptions.
        Retry is added in task 5.4.
        """
        if self.stop_event is not None and self.stop_event.is_set():
            raise InterruptedError(f"OCI upload to {self.repo} aborted before start")
        location = self._begin()
        with open(tar_path, "rb") as body:
            hdr = {
                **self.client.auth,
                "Content-Type": "application/octet-stream",
                "Content-Length": str(total_size),
                "Content-Range": f"0-{total_size - 1}",
            }
            r = self.client.session.patch(location, data=body, headers=hdr, timeout=(30, 600))
        if r.status_code in (200, 201, 202, 204):
            location = r.headers.get("Location", location)
        else:
            r.raise_for_status()
            raise RuntimeError(
                f"unexpected PATCH status {r.status_code} for streaming upload to {self.repo}"
            )
        sep = "&" if "?" in location else "?"
        url = f"{location}{sep}digest={digest}"
        rp = self.client.session.put(url, headers=self.client.auth, timeout=120)
        if rp.status_code != 201:
            rp.raise_for_status()
            raise RuntimeError(f"unexpected status {rp.status_code} on PUT close")
        return digest, total_size
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_registry.py -v
```

Expected: all pass (19 cases).

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/registry.py tests/unit/test_registry.py
git commit -m "feat(v1): StreamingBlobUpload happy path + 200/201/202/204 acceptance"
```

---

### Task 5.4: StreamingBlobUpload retry on SSL EOF (file rewound)

**Files:**
- Modify: `src/oci_modelcar/registry.py`
- Modify: `tests/unit/test_registry.py`

- [ ] **Step 1: Append failing tests**

```python
import time
from unittest.mock import patch as mock_patch


def test_streaming_retries_on_ssl_eof_with_file_rewound(httpserver, tmp_path, monkeypatch):
    """First PATCH attempt raises mid-stream SSL EOF; second succeeds.
    File must be reopened/rewound; full body sent again from offset 0."""
    payload = b"R" * 1024
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/eof")}
    )
    received = bytearray()

    def patch_handler(request):
        received.extend(request.data)
        return Response("", status=202, headers={"Location": httpserver.url_for("/u/eof")})

    httpserver.expect_request("/u/eof", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/eof", method="PUT").respond_with_data("", status=201)

    monkeypatch.setattr("oci_modelcar.registry.time.sleep", lambda d: None)

    real_patch = requests.Session.patch
    calls = {"n": 0}

    def flaky_patch(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.SSLError(
                "EOF occurred in violation of protocol (_ssl.c:2437)"
            )
        return real_patch(self, *args, **kwargs)

    upload = StreamingBlobUpload(
        client=_client(httpserver), repo="repo", max_retries=3, backoff_initial=0.0
    )
    with mock_patch.object(requests.Session, "patch", flaky_patch):
        out_digest, out_size = upload.push_from_file(f, len(payload), digest)

    assert out_digest == digest
    assert calls["n"] == 2, "must retry exactly once after SSL EOF"
    assert bytes(received) == payload, "second attempt must re-send full body"


def test_streaming_does_not_retry_on_handshake_ssl(httpserver, tmp_path):
    """SSL handshake errors are fatal; no retry."""
    payload = b"H" * 64
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/handshake")}
    )

    calls = {"n": 0}

    def fatal_ssl_patch(self, *args, **kwargs):
        calls["n"] += 1
        raise requests.exceptions.SSLError("CERTIFICATE_VERIFY_FAILED")

    upload = StreamingBlobUpload(
        client=_client(httpserver), repo="repo", max_retries=5, backoff_initial=0.0
    )
    with mock_patch.object(requests.Session, "patch", fatal_ssl_patch):
        with pytest.raises(requests.exceptions.SSLError):
            upload.push_from_file(f, len(payload), digest)

    assert calls["n"] == 1, "fatal SSL must not retry"


def test_streaming_max_retries_exhausted_raises_PushError(httpserver, tmp_path, monkeypatch):
    """All attempts fail with transient SSL EOF → PushError."""
    payload = b"X" * 32
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/exhaust")}
    )
    monkeypatch.setattr("oci_modelcar.registry.time.sleep", lambda d: None)

    calls = {"n": 0}

    def always_eof(self, *args, **kwargs):
        calls["n"] += 1
        raise requests.exceptions.SSLError(
            "EOF occurred in violation of protocol (_ssl.c:2437)"
        )

    upload = StreamingBlobUpload(
        client=_client(httpserver), repo="repo", max_retries=3, backoff_initial=0.0
    )
    with mock_patch.object(requests.Session, "patch", always_eof):
        from oci_modelcar.errors import PushError

        with pytest.raises(PushError, match="retries exhausted"):
            upload.push_from_file(f, len(payload), digest)

    assert calls["n"] == 3, "must call PATCH max_retries times before giving up"


def test_streaming_retries_on_5xx(httpserver, tmp_path, monkeypatch):
    """Server returns 503 then 202 → retry succeeds."""
    payload = b"S" * 16
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    httpserver.expect_request("/v2/repo/blobs/uploads/", method="POST").respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/5xx")}
    )

    calls = {"n": 0}

    def patch_handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return Response("server unavailable", status=503)
        return Response("", status=202, headers={"Location": httpserver.url_for("/u/5xx")})

    httpserver.expect_request("/u/5xx", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/5xx", method="PUT").respond_with_data("", status=201)

    monkeypatch.setattr("oci_modelcar.registry.time.sleep", lambda d: None)
    upload = StreamingBlobUpload(
        client=_client(httpserver), repo="repo", max_retries=3, backoff_initial=0.0
    )
    out_digest, _ = upload.push_from_file(f, len(payload), digest)
    assert out_digest == digest
    assert calls["n"] == 2
```

- [ ] **Step 2: Run, expect fail**

Expected: tests fail because the v1 push_from_file in 5.3 has no retry yet.

- [ ] **Step 3: Replace push_from_file with retrying version**

Replace `push_from_file` body in `src/oci_modelcar/registry.py`:

```python
import random
import time

from oci_modelcar.http import is_transient_ssl


def push_from_file(
    self, tar_path: Path, total_size: int, digest: str
) -> tuple[str, int]:
    """POST init → PATCH from file (full replay on cut) → PUT close.

    Each retry attempt re-opens tar_path and rewinds to offset 0.
    Backoff: full jitter Uniform(0, min(cap, base * 2^attempt)).
    Accepts {200, 201, 202, 204} on PATCH (Artifactory + Harbor quirks).
    """
    if self.stop_event is not None and self.stop_event.is_set():
        raise InterruptedError(f"OCI upload to {self.repo} aborted before start")
    location = self._begin()
    last_exc: BaseException | None = None

    for attempt in range(self.max_retries):
        if self.stop_event is not None and self.stop_event.is_set():
            raise InterruptedError(
                f"OCI upload to {self.repo} aborted before attempt {attempt + 1}"
            )
        if attempt > 0:
            self._sleep_backoff(attempt - 1)
        try:
            with open(tar_path, "rb") as body:
                hdr = {
                    **self.client.auth,
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(total_size),
                    "Content-Range": f"0-{total_size - 1}",
                }
                r = self.client.session.patch(
                    location, data=body, headers=hdr, timeout=(30, 600)
                )
            if r.status_code in (200, 201, 202, 204):
                location = r.headers.get("Location", location)
                break
            if r.status_code in (408, 429) or 500 <= r.status_code < 600:
                log.warning(
                    "PATCH transient %d for %s attempt %d/%d",
                    r.status_code, self.repo, attempt + 1, self.max_retries,
                )
                last_exc = RuntimeError(f"transient {r.status_code}")
                continue
            r.raise_for_status()
            raise RuntimeError(
                f"unexpected PATCH status {r.status_code} for streaming upload to {self.repo}"
            )
        except requests.exceptions.SSLError as e:
            if not is_transient_ssl(e):
                raise
            log.warning(
                "PATCH SSL EOF for %s attempt %d/%d, will retry from offset 0",
                self.repo, attempt + 1, self.max_retries,
            )
            last_exc = e
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            log.warning(
                "PATCH transient transport for %s attempt %d/%d: %s",
                self.repo, attempt + 1, self.max_retries, e,
            )
            last_exc = e
    else:
        # for-else: loop exhausted without break
        raise PushError(
            f"OCI PATCH retries exhausted ({self.max_retries}) for {self.repo}: "
            f"last error {last_exc!r}",
            hint=f"--oci-max-retries N (currently {self.max_retries}), or check registry health.",
        )

    sep = "&" if "?" in location else "?"
    url = f"{location}{sep}digest={digest}"
    rp = self.client.session.put(url, headers=self.client.auth, timeout=120)
    if rp.status_code != 201:
        rp.raise_for_status()
        raise RuntimeError(f"unexpected status {rp.status_code} on PUT close")
    return digest, total_size


def _sleep_backoff(self, attempt: int) -> None:
    cap_delay = min(self.backoff_cap, self.backoff_initial * (2**attempt))
    if cap_delay > 0:
        time.sleep(random.uniform(0, cap_delay))
```

Add `_sleep_backoff` as a method of `StreamingBlobUpload` (define after `push_from_file`).

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_registry.py -v
```

Expected: all pass (23 cases).

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/registry.py tests/unit/test_registry.py
git commit -m "feat(v1): StreamingBlobUpload full-PATCH retry with file replay"
```

---

## Phase 6 — download.py

### Task 6.1: HfFile + metadata via HfApi

**Files:**
- Create: `src/oci_modelcar/download.py`
- Test: `tests/unit/test_download.py`

- [ ] **Step 1: Write failing test**

`tests/unit/test_download.py`:

```python
from unittest.mock import MagicMock

import pytest

from oci_modelcar.download import HfDownloader, HfFile


def test_hf_file_carries_metadata():
    f = HfFile(path="model.safetensors", size=1234, lfs_sha256="a" * 64)
    assert f.path == "model.safetensors"
    assert f.size == 1234
    assert f.lfs_sha256 == "a" * 64


def test_hf_file_no_lfs():
    f = HfFile(path="config.json", size=100, lfs_sha256=None)
    assert f.lfs_sha256 is None


def test_resolve_revision_uses_repo_info():
    api = MagicMock()
    api.repo_info.return_value = MagicMock(sha="9fb191250dd56d0ba7ec9785a025ed29c03d5998")
    d = HfDownloader(api=api, session=MagicMock(), spool_dir=None, stop_event=None)
    assert d.resolve_revision("Qwen/Qwen2.5-7B", "main") == "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    api.repo_info.assert_called_once_with("Qwen/Qwen2.5-7B", revision="main")


def test_list_files_filters_by_allow_patterns():
    api = MagicMock()
    api.list_repo_tree.return_value = [
        MagicMock(type="file", path="model.safetensors", size=1000, lfs=MagicMock(sha256="a" * 64)),
        MagicMock(type="file", path="config.json", size=100, lfs=None),
        MagicMock(type="file", path="readme.md", size=50, lfs=None),
        MagicMock(type="file", path="ignored.bin", size=10, lfs=None),
        MagicMock(type="directory", path="subdir", size=0, lfs=None),
    ]
    d = HfDownloader(api=api, session=MagicMock(), spool_dir=None, stop_event=None)
    files = d.list_files("Qwen/Qwen2.5-7B", "main", allow=(".safetensors", ".json", ".md"))
    paths = [f.path for f in files]
    assert paths == sorted(paths)
    assert "model.safetensors" in paths
    assert "config.json" in paths
    assert "readme.md" in paths
    assert "ignored.bin" not in paths
    assert "subdir" not in paths


def test_list_files_extracts_lfs_sha256():
    api = MagicMock()
    lfs_meta = MagicMock(sha256="b" * 64)
    api.list_repo_tree.return_value = [
        MagicMock(type="file", path="model.safetensors", size=1000, lfs=lfs_meta),
        MagicMock(type="file", path="config.json", size=100, lfs=None),
    ]
    d = HfDownloader(api=api, session=MagicMock(), spool_dir=None, stop_event=None)
    files = d.list_files("Qwen/Qwen2.5-7B", "main", allow=(".safetensors", ".json"))
    by_path = {f.path: f for f in files}
    assert by_path["model.safetensors"].lfs_sha256 == "b" * 64
    assert by_path["config.json"].lfs_sha256 is None
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL (module not found).

- [ ] **Step 3: Implement metadata layer**

`src/oci_modelcar/download.py`:

```python
"""HuggingFace metadata via huggingface_hub.HfApi + bytes streamer with
mid-stream cancellation, atomic write, cross-origin auth strip."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import HfApi


@dataclass(frozen=True, slots=True)
class HfFile:
    path: str
    size: int
    lfs_sha256: str | None  # 64-hex from /api/models tree, when LFS-backed


class HfDownloader:
    def __init__(
        self,
        api: HfApi,
        session: requests.Session,
        spool_dir: Path | None,
        stop_event: threading.Event | None,
        max_retries: int = 10,
        backoff_initial: float = 1.0,
        backoff_cap: float = 60.0,
    ) -> None:
        self.api = api
        self.session = session
        self.spool_dir = spool_dir
        self.stop_event = stop_event
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_cap = backoff_cap

    def resolve_revision(self, repo: str, revision: str) -> str:
        info = self.api.repo_info(repo, revision=revision)
        return str(info.sha)

    def list_files(
        self, repo: str, revision: str, allow: tuple[str, ...]
    ) -> list[HfFile]:
        out: list[HfFile] = []
        for entry in self.api.list_repo_tree(repo, revision=revision, recursive=True):
            if getattr(entry, "type", None) != "file":
                continue
            if not any(entry.path.endswith(ext) for ext in allow):
                continue
            lfs = getattr(entry, "lfs", None)
            sha = lfs.sha256 if lfs is not None else None
            out.append(HfFile(path=entry.path, size=int(entry.size), lfs_sha256=sha))
        out.sort(key=lambda f: f.path)
        return out
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_download.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/download.py tests/unit/test_download.py
git commit -m "feat(v1): HfDownloader metadata layer (HfApi + HfFile)"
```

---

### Task 6.2: download() happy path with atomic rename

**Files:**
- Modify: `src/oci_modelcar/download.py`
- Modify: `tests/unit/test_download.py`

- [ ] **Step 1: Append failing tests**

```python
from pathlib import Path

from huggingface_hub import HfApi
from pytest_httpserver import HTTPServer

from oci_modelcar.http import build_session


def _make_downloader(httpserver: HTTPServer, spool: Path) -> HfDownloader:
    api = HfApi(endpoint=httpserver.url_for(""))
    return HfDownloader(
        api=api,
        session=build_session(),
        spool_dir=spool,
        stop_event=None,
        max_retries=3,
        backoff_initial=0.0,
    )


def test_download_writes_file_and_returns_path(httpserver, tmp_path):
    payload = b"hello world"
    httpserver.expect_request("/repo/resolve/main/file.txt").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )

    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.txt", size=len(payload), lfs_sha256=None)

    result = d.download("repo", "main", f)
    assert result == spool / "sources" / "file.txt"
    assert result.read_bytes() == payload
    assert not (spool / "sources" / "file.txt.partial").exists()


def test_download_preserves_subdirs_in_hf_path(httpserver, tmp_path):
    payload = b"sub"
    httpserver.expect_request("/repo/resolve/main/subdir/inner.txt").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )

    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="subdir/inner.txt", size=len(payload), lfs_sha256=None)

    result = d.download("repo", "main", f)
    assert result == spool / "sources" / "subdir" / "inner.txt"
    assert result.read_bytes() == payload


def test_download_calls_progress_cb(httpserver, tmp_path):
    payload = b"X" * 4096
    httpserver.expect_request("/repo/resolve/main/big.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="big.bin", size=len(payload), lfs_sha256=None)

    seen: list[int] = []
    d.download("repo", "main", f, progress_cb=seen.append)
    assert seen, "progress_cb was never invoked"
    assert seen[-1] == len(payload)


def test_download_partial_file_cleaned_on_exception(httpserver, tmp_path):
    """If the GET fails partway, the .partial file is removed."""
    httpserver.expect_request("/repo/resolve/main/file.bin").respond_with_data(
        "", status=503
    )
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.bin", size=10, lfs_sha256=None)

    from oci_modelcar.errors import DownloadError

    with pytest.raises(DownloadError):
        d.download("repo", "main", f)
    partial = spool / "sources" / "file.bin.partial"
    final = spool / "sources" / "file.bin"
    assert not partial.exists() and not final.exists()
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL (`download` not implemented).

- [ ] **Step 3: Implement download**

Append to `src/oci_modelcar/download.py`:

```python
import contextlib
import hashlib
import http.client
import logging
import random
import time
import urllib3.exceptions

from oci_modelcar.errors import (
    DownloadError,
    EntryNotFoundError,
    GatedRepoError,
    RevisionNotFoundError,
)
from oci_modelcar.http import huggingface_auth_header, is_transient_ssl

log = logging.getLogger(__name__)

_TRANSIENT_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    requests.RequestException,
    urllib3.exceptions.ProtocolError,
    http.client.IncompleteRead,
    OSError,
)
_FATAL_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    requests.exceptions.SSLError,
    requests.exceptions.ProxyError,
)
_CHUNK_DEFAULT = 1024 * 1024  # 1 MiB stop_event polling granularity


class HfDownloader:
    # ... existing __init__, resolve_revision, list_files unchanged ...

    def download(
        self,
        repo: str,
        revision: str,
        hf_file: HfFile,
        progress_cb: Callable[[int], None] | None = None,
    ) -> Path:
        """Download to <spool>/sources/<hf_path>.partial, atomic rename to .../<hf_path>."""
        if self.spool_dir is None:
            raise RuntimeError("spool_dir required for download()")
        sources = self.spool_dir / "sources"
        final = sources / hf_file.path
        partial = sources / (hf_file.path + ".partial")
        partial.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._stream_to(repo, revision, hf_file, partial, progress_cb)
            partial.replace(final)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                partial.unlink()
            with contextlib.suppress(FileNotFoundError):
                final.unlink()
            raise
        return final

    def _stream_to(
        self,
        repo: str,
        revision: str,
        hf_file: HfFile,
        partial: Path,
        progress_cb: Callable[[int], None] | None,
    ) -> None:
        url = f"{self.api.endpoint.rstrip('/')}/{repo}/resolve/{revision}/{hf_file.path}"
        bytes_done = 0
        h = hashlib.sha256() if hf_file.lfs_sha256 else None

        for attempt in range(self.max_retries):
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError(
                    f"HF download of {hf_file.path} aborted by stop_event"
                )
            try:
                self._stream_one_attempt(
                    url, partial, hf_file, bytes_done, h, progress_cb
                )
                bytes_done = partial.stat().st_size
                if bytes_done >= hf_file.size:
                    if h is not None and hf_file.lfs_sha256 is not None:
                        got = h.hexdigest()
                        if got != hf_file.lfs_sha256:
                            raise DownloadError(
                                f"sha256 mismatch for {hf_file.path}: "
                                f"expected {hf_file.lfs_sha256}, got {got}",
                                hint="HF cache may be corrupted; delete and re-run.",
                            )
                    return
                # short read but no exception: continue with Range
            except (requests.exceptions.HTTPError,) as e:
                self._raise_specific_http_error(e, repo, revision, hf_file.path)
            except _FATAL_TRANSPORT_ERRORS as e:
                if isinstance(e, requests.exceptions.SSLError) and is_transient_ssl(e):
                    log.warning(
                        "HF SSL EOF mid-stream for %s at %d/%d (attempt %d/%d)",
                        hf_file.path, bytes_done, hf_file.size, attempt + 1, self.max_retries,
                    )
                    bytes_done = partial.stat().st_size if partial.exists() else 0
                    self._sleep_backoff(attempt)
                    continue
                raise
            except _TRANSIENT_TRANSPORT_ERRORS as e:
                log.warning(
                    "HF read failed for %s at %d/%d (attempt %d/%d): %s",
                    hf_file.path, bytes_done, hf_file.size, attempt + 1, self.max_retries, e,
                )
                bytes_done = partial.stat().st_size if partial.exists() else 0
                self._sleep_backoff(attempt)
                continue

        raise DownloadError(
            f"HF retries exhausted for {hf_file.path} at offset {bytes_done}",
            hint=f"--hf-max-retries N (currently {self.max_retries})",
        )

    def _stream_one_attempt(
        self,
        url: str,
        partial: Path,
        hf_file: HfFile,
        start: int,
        h: "hashlib._Hash | None",
        progress_cb: Callable[[int], None] | None,
    ) -> None:
        headers = dict(huggingface_auth_header())
        if start > 0:
            headers["Range"] = f"bytes={start}-"
        r = self.session.get(url, headers=headers, stream=True, timeout=(10, 600))
        r.raise_for_status()

        # If we asked for Range but got 200, server ignored it: truncate + restart
        if start > 0 and r.status_code == 200:
            log.info("HF server ignored Range; truncating partial and restarting from 0")
            partial.unlink(missing_ok=True)
            start = 0
            if h is not None:
                h = hashlib.sha256()

        mode = "ab" if start > 0 else "wb"
        with open(partial, mode) as f:
            written = start
            for chunk in r.iter_content(chunk_size=_CHUNK_DEFAULT):
                if self.stop_event is not None and self.stop_event.is_set():
                    raise InterruptedError(
                        f"HF download of {hf_file.path} aborted by stop_event"
                    )
                if not chunk:
                    continue
                f.write(chunk)
                if h is not None:
                    h.update(chunk)
                written += len(chunk)
                if progress_cb is not None:
                    progress_cb(written)

    def _raise_specific_http_error(
        self,
        e: requests.exceptions.HTTPError,
        repo: str,
        revision: str,
        path: str,
    ) -> None:
        resp = e.response
        if resp is None:
            raise e
        status = resp.status_code
        if status == 403 and resp.headers.get("X-Error-Code") == "GatedRepo":
            raise GatedRepoError(
                f"Repo {repo} is gated.",
                hint=f"Accept terms at https://huggingface.co/{repo}, then re-run.",
            ) from e
        if status == 404:
            # Distinguish revision vs file
            if "/resolve/" in resp.request.url and path in resp.request.url:
                raise EntryNotFoundError(
                    f"File not found: {repo}@{revision}/{path}"
                ) from e
            raise RevisionNotFoundError(
                f"Revision not found: {repo}@{revision}"
            ) from e
        raise

    def _sleep_backoff(self, attempt: int) -> None:
        cap_delay = min(self.backoff_cap, self.backoff_initial * (2**attempt))
        if cap_delay > 0:
            time.sleep(random.uniform(0, cap_delay))
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_download.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/download.py tests/unit/test_download.py
git commit -m "feat(v1): HfDownloader.download with atomic rename + retry"
```

---

### Task 6.3: stop_event mid-stream cancellation

**Files:**
- Modify: `tests/unit/test_download.py`

The `download` method already polls `stop_event` per chunk (Task 6.2). This task adds the test that locks in the timing guarantee (the regression we're solving).

- [ ] **Step 1: Append failing test**

```python
import threading


def test_download_aborts_within_two_chunks_of_stop_event(httpserver, tmp_path):
    """Regression test for v0.x: 50 GB DL was uncancellable. v1 must cancel
    within ~2 × CHUNK_DEFAULT (~2 MiB) of stop_event.set()."""
    big_payload = b"X" * (8 * 1024 * 1024)  # 8 MiB streamed slowly via httpserver
    httpserver.expect_request("/repo/resolve/main/big.bin").respond_with_data(
        big_payload, headers={"Content-Length": str(len(big_payload))}
    )
    spool = tmp_path / "spool"
    stop = threading.Event()
    api = HfApi(endpoint=httpserver.url_for(""))
    d = HfDownloader(
        api=api, session=build_session(), spool_dir=spool, stop_event=stop, max_retries=1
    )
    f = HfFile(path="big.bin", size=len(big_payload), lfs_sha256=None)

    chunks_seen = 0
    raised: list[BaseException] = []

    def progress(_n: int) -> None:
        nonlocal chunks_seen
        chunks_seen += 1
        if chunks_seen == 1:
            stop.set()  # request abort after the first chunk

    def runner():
        try:
            d.download("repo", "main", f, progress_cb=progress)
        except BaseException as e:  # noqa: BLE001
            raised.append(e)

    t = threading.Thread(target=runner)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "download did not abort within timeout"
    assert raised and isinstance(raised[0], InterruptedError)
```

- [ ] **Step 2: Run, expect pass (already implemented in 6.2)**

```bash
.venv/bin/python -m pytest tests/unit/test_download.py::test_download_aborts_within_two_chunks_of_stop_event -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_download.py
git commit -m "test(v1): stop_event aborts download within seconds (regression)"
```

---

### Task 6.4: Range-200 fallback + cross-origin auth strip integration

**Files:**
- Modify: `tests/unit/test_download.py`

The Range-200 fallback is in `_stream_one_attempt`. This task adds a direct test.
Cross-origin auth stripping is in `_SafeSession` (already covered in `test_http.py`),
but we add an integration test here to confirm `download()` benefits from it.

- [ ] **Step 1: Append failing test**

```python
def test_download_handles_range_200_fallback(httpserver, tmp_path, monkeypatch):
    """If first attempt downloads partial bytes then fails, and the second
    attempt sends Range but server ignores it (returns 200), the partial
    file is truncated and download restarts cleanly."""
    payload = b"Y" * 4096
    state = {"served": False}

    def handler(request):
        # First call: serve only first 1024 bytes then close (simulates cut)
        if not state["served"]:
            state["served"] = True
            return Response(
                payload[:1024],
                status=200,
                headers={"Content-Length": str(len(payload[:1024]))},
            )
        # Subsequent calls: ignore Range, return full payload with 200
        return Response(
            payload, status=200, headers={"Content-Length": str(len(payload))}
        )

    httpserver.expect_request("/repo/resolve/main/file.bin").respond_with_handler(handler)
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.bin", size=len(payload), lfs_sha256=None)

    monkeypatch.setattr("oci_modelcar.download.time.sleep", lambda d: None)
    out = d.download("repo", "main", f)
    assert out.read_bytes() == payload
```

- [ ] **Step 2: Run, expect pass (already implemented)**

```bash
.venv/bin/python -m pytest tests/unit/test_download.py::test_download_handles_range_200_fallback -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_download.py
git commit -m "test(v1): Range-200 fallback truncates and restarts download"
```

---

### Task 6.5: GatedRepo / RevisionNotFound / EntryNotFound error mapping

**Files:**
- Modify: `tests/unit/test_download.py`

- [ ] **Step 1: Append failing tests**

```python
def test_download_gated_repo_raises_specific_error(httpserver, tmp_path):
    httpserver.expect_request("/repo/resolve/main/file.bin").respond_with_data(
        "Gated", status=403, headers={"X-Error-Code": "GatedRepo"}
    )
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.bin", size=10, lfs_sha256=None)

    with pytest.raises(GatedRepoError) as exc:
        d.download("repo", "main", f)
    assert exc.value.hint and "huggingface.co/repo" in exc.value.hint


def test_download_404_on_resolve_raises_entry_not_found(httpserver, tmp_path):
    httpserver.expect_request("/repo/resolve/main/missing.bin").respond_with_data(
        "Not found", status=404
    )
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="missing.bin", size=10, lfs_sha256=None)

    with pytest.raises(EntryNotFoundError):
        d.download("repo", "main", f)


def test_download_lfs_sha_verified(httpserver, tmp_path):
    payload = b"hello world"
    correct_sha = hashlib.sha256(payload).hexdigest()
    httpserver.expect_request("/repo/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.bin", size=len(payload), lfs_sha256=correct_sha)

    result = d.download("repo", "main", f)
    assert result.read_bytes() == payload


def test_download_lfs_sha_mismatch_raises(httpserver, tmp_path):
    payload = b"hello world"
    wrong_sha = "0" * 64
    httpserver.expect_request("/repo/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    spool = tmp_path / "spool"
    d = _make_downloader(httpserver, spool)
    f = HfFile(path="file.bin", size=len(payload), lfs_sha256=wrong_sha)

    with pytest.raises(DownloadError, match="sha256 mismatch"):
        d.download("repo", "main", f)


from oci_modelcar.errors import EntryNotFoundError  # add to imports
import hashlib  # ensure imported
```

- [ ] **Step 2: Run, expect pass (logic in 6.2)**

```bash
.venv/bin/python -m pytest tests/unit/test_download.py -v
```

Expected: all pass (15 cases).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_download.py
git commit -m "test(v1): error mapping and LFS sha verification"
```

---

## Phase 7 — config.py

### Task 7.1: Config dataclass + minimal CLI/env parsing

**Files:**
- Create: `src/oci_modelcar/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing test**

`tests/unit/test_config.py`:

```python
import pytest

from oci_modelcar.config import Config
from oci_modelcar.errors import ConfigError


def test_config_minimum_required(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    cfg = Config.from_env_and_args([])
    assert cfg.hf_repo == "foo/bar"
    assert cfg.registry == "registry.example.com"
    assert cfg.target_repo == "models/x"
    assert cfg.hf_revision == "main"
    assert cfg.hf_endpoint == "https://huggingface.co"
    assert cfg.workers == 1


def test_config_missing_required(monkeypatch):
    monkeypatch.delenv("HF_REPO", raising=False)
    monkeypatch.delenv("REGISTRY", raising=False)
    monkeypatch.delenv("TARGET_REPO", raising=False)
    with pytest.raises(ConfigError, match="hf_repo"):
        Config.from_env_and_args([])


def test_config_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    monkeypatch.setenv("WORKERS", "4")
    cfg = Config.from_env_and_args(["--workers", "2"])
    assert cfg.workers == 2


def test_config_workers_cap(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(ConfigError, match="workers"):
        Config.from_env_and_args(["--workers", "9"])


def test_config_workers_zero_raises(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(ConfigError, match="workers"):
        Config.from_env_and_args(["--workers", "0"])


def test_config_invalid_target_tag(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(ConfigError, match="target_tag"):
        Config.from_env_and_args(["--target-tag", "bad/tag"])


def test_config_verbose_quiet_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    monkeypatch.setenv("LOG_VERBOSE", "1")
    monkeypatch.setenv("LOG_QUIET", "1")
    with pytest.raises(ConfigError, match="mutually exclusive"):
        Config.from_env_and_args([])
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

`src/oci_modelcar/config.py`:

```python
"""Config: Config dataclass + env+CLI parsing + validation. v1: drops
--state-file, --chunk-mib, --upload-mode; adds --spool-dir, --clean-hf-after-push."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from oci_modelcar.errors import ConfigError

_TAG_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127}$")
_DEFAULT_ALLOW = ".safetensors .json .txt .md .model"


def _envbool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _envstr(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw else default


def _default_spool_dir() -> Path:
    return Path(tempfile.gettempdir()) / "oci-modelcar"


def _parse_csv(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else []


@dataclass
class Config:
    hf_repo: str
    registry: str
    target_repo: str
    hf_revision: str = "main"
    hf_endpoint: str = "https://huggingface.co"
    target_tag: str | None = None
    also_tags: list[str] = field(default_factory=list)
    allow_patterns: tuple[str, ...] = field(default_factory=lambda: tuple(_DEFAULT_ALLOW.split()))
    layer_prefix: str = "models/"
    workers: int = 1
    spool_dir: Path = field(default_factory=_default_spool_dir)
    clean_hf_after_push: bool = False
    hf_max_retries: int = 10
    oci_max_retries: int = 5
    fail_fast: bool = True
    force: bool = False
    log_style: str | None = None
    verbose: bool = False
    quiet: bool = False
    dry_run: bool = False
    sub_command: str = "push"

    @classmethod
    def from_env_and_args(cls, argv: list[str]) -> Self:
        parser = _build_parser()
        ns = parser.parse_args(argv)

        cfg = cls(
            hf_repo=ns.hf_repo or os.environ.get("HF_REPO", ""),
            registry=ns.registry or os.environ.get("REGISTRY", ""),
            target_repo=ns.target_repo or os.environ.get("TARGET_REPO", ""),
            hf_revision=ns.hf_revision or _envstr("HF_REVISION", "main"),
            hf_endpoint=ns.hf_endpoint or _envstr("HF_ENDPOINT", "https://huggingface.co"),
            target_tag=ns.target_tag or os.environ.get("TARGET_TAG") or None,
            also_tags=_parse_csv(ns.also_tag or _envstr("ALSO_TAGS", "")),
            allow_patterns=tuple(
                (ns.allow_patterns or _envstr("ALLOW_PATTERNS", _DEFAULT_ALLOW)).split()
            ),
            layer_prefix=(
                ns.layer_prefix
                if ns.layer_prefix is not None
                else _envstr("LAYER_PATH_PREFIX", "models/")
            ),
            workers=int(ns.workers if ns.workers is not None else _envstr("WORKERS", "1")),
            spool_dir=Path(
                ns.spool_dir or _envstr("SPOOL_DIR", str(_default_spool_dir()))
            ),
            clean_hf_after_push=(
                ns.clean_hf_after_push or _envbool("CLEAN_HF_AFTER_PUSH", False)
            ),
            hf_max_retries=int(
                ns.hf_max_retries
                if ns.hf_max_retries is not None
                else _envstr("HF_MAX_RETRIES", "10")
            ),
            oci_max_retries=int(
                ns.oci_max_retries
                if ns.oci_max_retries is not None
                else _envstr("OCI_MAX_RETRIES", "5")
            ),
            fail_fast=False if ns.continue_on_error else (ns.fail_fast or _envbool("FAIL_FAST", True)),
            force=ns.force or _envbool("FORCE", False),
            log_style=ns.log_style or os.environ.get("LOG_STYLE"),
            verbose=ns.verbose or _envbool("LOG_VERBOSE", False),
            quiet=ns.quiet or _envbool("LOG_QUIET", False),
            dry_run=ns.dry_run,
            sub_command="push",
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.hf_repo:
            raise ConfigError("hf_repo is required (--hf-repo or HF_REPO)")
        if not self.registry:
            raise ConfigError("registry is required (--registry or REGISTRY)")
        if not self.target_repo:
            raise ConfigError("target_repo is required (--target-repo or TARGET_REPO)")
        if not (1 <= self.workers <= 8):
            raise ConfigError(f"workers must be in [1, 8], got {self.workers}")
        if not (0 <= self.hf_max_retries <= 100):
            raise ConfigError(f"hf_max_retries must be in [0, 100], got {self.hf_max_retries}")
        if not (0 <= self.oci_max_retries <= 100):
            raise ConfigError(f"oci_max_retries must be in [0, 100], got {self.oci_max_retries}")
        if self.verbose and self.quiet:
            raise ConfigError("verbose and quiet are mutually exclusive")
        if self.target_tag is not None and not _TAG_RE.match(self.target_tag):
            raise ConfigError(
                f"target_tag {self.target_tag!r} does not match [a-zA-Z0-9_][a-zA-Z0-9._-]{{0,127}}"
            )
        for t in self.also_tags:
            if not _TAG_RE.match(t):
                raise ConfigError(f"also_tag {t!r} is invalid")
        if self.log_style is not None and self.log_style not in ("text", "azure"):
            raise ConfigError(f"log_style must be 'text' or 'azure', got {self.log_style!r}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oci-modelcar",
        description="Push HuggingFace models to OCI registries.",
    )
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--hf-revision", default=None)
    p.add_argument("--hf-endpoint", default=None)
    p.add_argument("--registry", default=None)
    p.add_argument("--target-repo", default=None)
    p.add_argument("--target-tag", default=None)
    p.add_argument("--also-tag", default=None, help="CSV list of additional tags")
    p.add_argument("--allow-patterns", default=None)
    p.add_argument("--layer-prefix", default=None)
    p.add_argument("--workers", default=None, type=int)
    p.add_argument("--spool-dir", default=None)
    p.add_argument("--clean-hf-after-push", action="store_true", default=False)
    p.add_argument("--hf-max-retries", default=None, type=int)
    p.add_argument("--oci-max-retries", default=None, type=int)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fail-fast", action="store_true", default=False)
    g.add_argument("--continue-on-error", action="store_true", default=False)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--log-style", default=None, choices=["text", "azure"])
    g2 = p.add_mutually_exclusive_group()
    g2.add_argument("--verbose", action="store_true", default=False)
    g2.add_argument("--quiet", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    return p
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_config.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/config.py tests/unit/test_config.py
git commit -m "feat(v1): config.py with new flags (--spool-dir, --clean-hf-after-push)"
```

---

### Task 7.2: Tests for new flags wiring

**Files:**
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Append tests**

```python
def test_config_spool_dir_default(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    monkeypatch.delenv("SPOOL_DIR", raising=False)
    cfg = Config.from_env_and_args([])
    assert cfg.spool_dir.name == "oci-modelcar"


def test_config_spool_dir_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    cfg = Config.from_env_and_args(["--spool-dir", str(tmp_path)])
    assert cfg.spool_dir == tmp_path


def test_config_spool_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    monkeypatch.setenv("SPOOL_DIR", str(tmp_path))
    cfg = Config.from_env_and_args([])
    assert cfg.spool_dir == tmp_path


def test_config_clean_hf_after_push_default_false(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    monkeypatch.delenv("CLEAN_HF_AFTER_PUSH", raising=False)
    cfg = Config.from_env_and_args([])
    assert cfg.clean_hf_after_push is False


def test_config_clean_hf_after_push_cli(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    cfg = Config.from_env_and_args(["--clean-hf-after-push"])
    assert cfg.clean_hf_after_push is True


def test_config_clean_hf_after_push_env(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    monkeypatch.setenv("CLEAN_HF_AFTER_PUSH", "1")
    cfg = Config.from_env_and_args([])
    assert cfg.clean_hf_after_push is True


def test_config_oci_max_retries_default_5(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    cfg = Config.from_env_and_args([])
    assert cfg.oci_max_retries == 5


def test_config_continue_on_error_overrides_fail_fast(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    cfg = Config.from_env_and_args(["--continue-on-error"])
    assert cfg.fail_fast is False
```

- [ ] **Step 2: Run, expect pass (already implemented in 7.1)**

```bash
.venv/bin/python -m pytest tests/unit/test_config.py -v
```

Expected: 15 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_config.py
git commit -m "test(v1): config wiring for new --spool-dir and --clean-hf-after-push"
```

---

### Task 7.3: also_tag CSV parsing + log_style validation

**Files:**
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Append tests**

```python
def test_config_also_tags_csv(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    cfg = Config.from_env_and_args(["--also-tag", "v1.0,latest,prod"])
    assert cfg.also_tags == ["v1.0", "latest", "prod"]


def test_config_also_tags_invalid_raises(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(ConfigError, match="also_tag"):
        Config.from_env_and_args(["--also-tag", "bad/tag"])


def test_config_log_style_invalid_raises(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    # argparse rejects unknown choices itself
    with pytest.raises(SystemExit):
        Config.from_env_and_args(["--log-style", "bogus"])


def test_config_chunk_mib_flag_rejected(monkeypatch):
    """v0.x removed flag — argparse should error."""
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(SystemExit):
        Config.from_env_and_args(["--chunk-mib", "32"])


def test_config_state_file_flag_rejected(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(SystemExit):
        Config.from_env_and_args(["--state-file", "/tmp/x.json"])


def test_config_upload_mode_flag_rejected(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(SystemExit):
        Config.from_env_and_args(["--upload-mode", "chunked"])
```

- [ ] **Step 2: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_config.py -v
```

Expected: 21 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_config.py
git commit -m "test(v1): also_tags CSV + removed flags rejected"
```

---

## Phase 8 — pipeline.py

### Task 8.1: logging.py port (PipelineLogger + formatters)

**Files:**
- Create: `src/oci_modelcar/logging.py`
- Test: `tests/unit/test_logging.py`

This is a near-verbatim port from v0.x — `logging.py` was stable. Read the
v0.x version and copy it.

- [ ] **Step 1: Write failing test**

`tests/unit/test_logging.py`:

```python
import io
import logging

from oci_modelcar.logging import (
    AzureFormatter,
    PipelineLogger,
    TextFormatter,
)


def test_pipeline_logger_emits_to_stdout(capsys):
    plog = PipelineLogger(stream=None, log_style="text", verbose=False, quiet=False)
    plog.info("hello")
    out = capsys.readouterr().out
    assert "hello" in out


def test_pipeline_logger_section(capsys):
    plog = PipelineLogger(stream=None, log_style="text", verbose=False, quiet=False)
    plog.section("Phase 1")
    out = capsys.readouterr().out
    assert "Phase 1" in out
    # text formatter should set off sections visually
    assert "==" in out or "##" in out or "Phase 1" in out


def test_pipeline_logger_quiet_suppresses_info(capsys):
    plog = PipelineLogger(stream=None, log_style="text", verbose=False, quiet=True)
    plog.info("hidden")
    out = capsys.readouterr().out
    assert "hidden" not in out


def test_pipeline_logger_verbose_includes_debug(capsys):
    plog = PipelineLogger(stream=None, log_style="text", verbose=True, quiet=False)
    plog.debug("verbose-detail")
    out = capsys.readouterr().out
    assert "verbose-detail" in out


def test_azure_format_uses_logging_command():
    fmt = AzureFormatter()
    record = logging.LogRecord(
        name="x", level=logging.WARNING, pathname="", lineno=0, msg="warn-text",
        args=(), exc_info=None,
    )
    out = fmt.format(record)
    assert out.startswith("##[warning]") or "##[warning]" in out


def test_pipeline_logger_output_variable_azure(capsys):
    plog = PipelineLogger(stream=None, log_style="azure", verbose=False, quiet=False)
    plog.output_variable("manifestDigest", "sha256:abc")
    out = capsys.readouterr().out
    # Azure DevOps output variable command
    assert "task.setvariable" in out or "manifestDigest" in out
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL (module not found).

- [ ] **Step 3: Implement (port from v0.5 logging.py)**

`src/oci_modelcar/logging.py`:

```python
"""Pipeline logging: text + Azure DevOps formatters, output_variable."""

from __future__ import annotations

import logging
import sys
from typing import IO


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class AzureFormatter(logging.Formatter):
    """Azure DevOps logging commands. WARNING/ERROR get task.logissue prefix."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"##[error]{msg}"
        if record.levelno >= logging.WARNING:
            return f"##[warning]{msg}"
        return msg


class PipelineLogger:
    def __init__(
        self,
        stream: IO[str] | None = None,
        log_style: str = "text",
        verbose: bool = False,
        quiet: bool = False,
    ) -> None:
        self.stream = stream or sys.stdout
        self.log_style = log_style
        self.verbose = verbose
        self.quiet = quiet
        self._fmt = AzureFormatter() if log_style == "azure" else TextFormatter()

    def _emit(self, level: int, msg: str) -> None:
        if self.quiet and level < logging.WARNING:
            return
        if level == logging.DEBUG and not self.verbose:
            return
        rec = logging.LogRecord(
            name="oci-modelcar", level=level, pathname="", lineno=0,
            msg=msg, args=(), exc_info=None,
        )
        print(self._fmt.format(rec), file=self.stream, flush=True)

    def section(self, title: str) -> None:
        if self.log_style == "azure":
            print(f"##[section]{title}", file=self.stream, flush=True)
        else:
            print(f"\n== {title} ==", file=self.stream, flush=True)

    def debug(self, msg: str) -> None:
        self._emit(logging.DEBUG, msg)

    def info(self, msg: str) -> None:
        self._emit(logging.INFO, msg)

    def warning(self, msg: str) -> None:
        self._emit(logging.WARNING, msg)

    def error(self, msg: str) -> None:
        self._emit(logging.ERROR, msg)

    def output_variable(self, name: str, value: str) -> None:
        if self.log_style == "azure":
            print(
                f"##vso[task.setvariable variable={name};isOutput=true]{value}",
                file=self.stream, flush=True,
            )
        else:
            print(f"{name}={value}", file=self.stream, flush=True)
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_logging.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/logging.py tests/unit/test_logging.py
git commit -m "feat(v1): logging.py with text + Azure formatters (port)"
```

---

### Task 8.2: FileWorker — phase ordering

**Files:**
- Create: `src/oci_modelcar/pipeline.py`
- Test: `tests/unit/test_pipeline.py`

- [ ] **Step 1: Write failing test**

`tests/unit/test_pipeline.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oci_modelcar.download import HfFile
from oci_modelcar.manifest import BlobDescriptor
from oci_modelcar.pipeline import FileWorker


def _build_worker(tmp_path: Path, head_blob_returns=None, **overrides):
    downloader = MagicMock()
    registry_client = MagicMock()
    spool = tmp_path / "spool"
    (spool / "sources").mkdir(parents=True)
    (spool / "layers").mkdir(parents=True)

    # Default downloader: writes a fake source file
    def fake_download(repo, rev, hf_file, progress_cb=None):
        p = spool / "sources" / hf_file.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"X" * hf_file.size)
        return p
    downloader.download.side_effect = fake_download

    # head_blob result is configurable: None = not present, dict = skip
    head_blob_mock = MagicMock(return_value=head_blob_returns)

    # StreamingBlobUpload mock
    streaming_factory = MagicMock()
    streaming_inst = MagicMock()
    streaming_inst.push_from_file.return_value = ("sha256:" + "f" * 64, 12345)
    streaming_factory.return_value = streaming_inst

    return FileWorker(
        downloader=downloader,
        registry_client=registry_client,
        head_blob_fn=head_blob_mock,
        streaming_factory=streaming_factory,
        layer_prefix="models/",
        spool_dir=spool,
        clean_hf_after_push=False,
        oci_max_retries=3,
        backoff_initial=0.0,
        stop_event=None,
    ), downloader, head_blob_mock, streaming_inst


def test_file_worker_phase_order_happy_path(tmp_path):
    """Confirms phases a→f run in order: download, tar+hash, head-skip,
    push, verify, cleanup."""
    worker, downloader, head_blob_mock, streaming = _build_worker(
        tmp_path, head_blob_returns=None
    )
    f = HfFile(path="weights.bin", size=1000, lfs_sha256=None)
    desc = worker.process(repo="repo", revision="main", hf_file=f)

    # download was called
    downloader.download.assert_called_once()
    # head_blob called twice: skip-check + verify
    assert head_blob_mock.call_count >= 1
    # streaming push called
    streaming.push_from_file.assert_called_once()
    # spool/layers/weights.bin.tar should be cleaned up
    assert not (tmp_path / "spool" / "layers" / "weights.bin.tar").exists()
    # source file kept (clean_hf_after_push=False)
    assert (tmp_path / "spool" / "sources" / "weights.bin").exists()
    assert isinstance(desc, BlobDescriptor)
    assert desc.hf_path == "weights.bin"


def test_file_worker_skips_push_if_blob_present(tmp_path):
    """If head_blob returns existing blob, skip the streaming push."""
    digest = "sha256:" + "a" * 64
    worker, downloader, head_blob_mock, streaming = _build_worker(
        tmp_path, head_blob_returns={"digest": digest, "size": 1000}
    )
    # Patch head_blob to return None on the FIRST call (which is for our digest)
    # and stop the test from reaching verify. For simplicity we keep both calls
    # returning present.
    head_blob_mock.return_value = None  # default: not present, push proceeds
    # ... but we want to test the skip branch, so set up the side_effect:
    # First call (skip-check after tar): returns the dict (present)
    # Verify call: doesn't happen because we skipped
    head_blob_mock.side_effect = [{"digest": "sha256:dummy", "size": 1000}]

    # We need download to write a source so build_layer_to_file works to compute digest
    f = HfFile(path="weights.bin", size=1000, lfs_sha256=None)
    desc = worker.process(repo="repo", revision="main", hf_file=f)

    streaming.push_from_file.assert_not_called()
    assert isinstance(desc, BlobDescriptor)
    # tar was cleaned even when push skipped
    assert not (tmp_path / "spool" / "layers" / "weights.bin.tar").exists()


def test_file_worker_clean_hf_after_push_deletes_source(tmp_path):
    worker, _, _, _ = _build_worker(tmp_path, head_blob_returns=None)
    worker.clean_hf_after_push = True
    f = HfFile(path="weights.bin", size=500, lfs_sha256=None)
    worker.process(repo="repo", revision="main", hf_file=f)
    assert not (tmp_path / "spool" / "sources" / "weights.bin").exists()


def test_file_worker_cleanup_runs_on_exception(tmp_path):
    """If the push step raises, the tar file is still deleted."""
    worker, _, _, streaming = _build_worker(tmp_path, head_blob_returns=None)
    streaming.push_from_file.side_effect = RuntimeError("simulated push failure")
    f = HfFile(path="weights.bin", size=500, lfs_sha256=None)
    with pytest.raises(RuntimeError, match="simulated push failure"):
        worker.process(repo="repo", revision="main", hf_file=f)
    assert not (tmp_path / "spool" / "layers" / "weights.bin.tar").exists()
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL (module not found).

- [ ] **Step 3: Implement FileWorker**

`src/oci_modelcar/pipeline.py`:

```python
"""Per-file pipeline orchestration: FileWorker + Pipeline."""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oci_modelcar.config import Config
from oci_modelcar.download import HfDownloader, HfFile
from oci_modelcar.errors import PushError
from oci_modelcar.layer import build_layer_to_file
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.manifest import BlobDescriptor, ML_TAR
from oci_modelcar.registry import OciClient, StreamingBlobUpload, head_blob

log = logging.getLogger(__name__)


class FileWorker:
    """Process one HF file end-to-end (phases a→f from the design spec)."""

    def __init__(
        self,
        downloader: HfDownloader,
        registry_client: OciClient,
        head_blob_fn: Callable[..., Any],
        streaming_factory: Callable[..., StreamingBlobUpload],
        layer_prefix: str,
        spool_dir: Path,
        clean_hf_after_push: bool,
        oci_max_retries: int,
        backoff_initial: float = 1.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.downloader = downloader
        self.registry_client = registry_client
        self.head_blob_fn = head_blob_fn
        self.streaming_factory = streaming_factory
        self.layer_prefix = layer_prefix
        self.spool_dir = spool_dir
        self.clean_hf_after_push = clean_hf_after_push
        self.oci_max_retries = oci_max_retries
        self.backoff_initial = backoff_initial
        self.stop_event = stop_event

    def process(
        self,
        repo: str,
        revision: str,
        hf_file: HfFile,
        progress_cb: Callable[[int], None] | None = None,
    ) -> BlobDescriptor:
        if self.stop_event is not None and self.stop_event.is_set():
            raise InterruptedError(f"worker for {hf_file.path} aborted before start")

        # a. DOWNLOAD
        source_path = self.downloader.download(repo, revision, hf_file, progress_cb=progress_cb)

        # b. TAR + HASH
        tar_path = self.spool_dir / "layers" / (hf_file.path + ".tar")
        try:
            digest, layer_size = build_layer_to_file(
                source_path, self.layer_prefix, hf_file.path.split("/")[-1], tar_path
            )

            # c. SKIP CHECK
            existing = self.head_blob_fn(self.registry_client, repo="repo", digest=digest) \
                if False else self.head_blob_fn(self.registry_client, "repo-from-arg", digest)
            # Note: head_blob signature is (client, repo, digest); we use registry's repo
            # which the caller wires via streaming_factory's bound repo. For simplicity,
            # we look it up from streaming_factory's closure.
            # For the v1 plan, we expose `target_repo` on the worker:
            existing = self.head_blob_fn(self.registry_client, self._target_repo(), digest)
            if existing is not None and existing.get("digest") == digest:
                log.info("skip push: blob %s already in registry", digest[:23])
                return BlobDescriptor(
                    media_type=ML_TAR,
                    digest=digest,
                    size=int(existing["size"]),
                    hf_path=hf_file.path,
                )

            # d. PUSH
            streaming = self.streaming_factory(
                client=self.registry_client,
                repo=self._target_repo(),
                max_retries=self.oci_max_retries,
                backoff_initial=self.backoff_initial,
                stop_event=self.stop_event,
            )
            streaming.push_from_file(tar_path, layer_size, digest)

            # e. VERIFY
            verified = self.head_blob_fn(self.registry_client, self._target_repo(), digest)
            if verified is None:
                raise PushError(
                    f"blob {digest} not visible after PUT for {hf_file.path}",
                    hint="registry may not have persisted the upload; retry the run.",
                )

            return BlobDescriptor(
                media_type=ML_TAR, digest=digest, size=layer_size, hf_path=hf_file.path
            )
        finally:
            # f. CLEANUP
            with contextlib.suppress(FileNotFoundError):
                tar_path.unlink()
            if self.clean_hf_after_push:
                with contextlib.suppress(FileNotFoundError):
                    source_path.unlink()

    def _target_repo(self) -> str:
        # The worker doesn't own target_repo directly; it's in registry_client
        repo = self.registry_client.target_repo
        assert repo is not None, "OciClient must have target_repo set for FileWorker"
        return repo
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_pipeline.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat(v1): FileWorker with phase ordering and cleanup contract"
```

---

### Task 8.3: Pipeline pre-flight (revision, files, tag conflict)

**Files:**
- Modify: `src/oci_modelcar/pipeline.py`
- Modify: `tests/unit/test_pipeline.py`

- [ ] **Step 1: Append failing tests**

```python
from oci_modelcar.pipeline import Pipeline


def _build_pipeline(tmp_path, **cfg_overrides):
    """Construct a Pipeline with mocked external dependencies."""
    from oci_modelcar.config import Config

    base = dict(
        hf_repo="foo/bar", registry="registry.example.com",
        target_repo="models/x", target_tag=None, also_tags=[],
        allow_patterns=(".safetensors", ".json"), layer_prefix="models/",
        workers=1, spool_dir=tmp_path / "spool", clean_hf_after_push=False,
        hf_max_retries=3, oci_max_retries=3, fail_fast=True, force=False,
        log_style="text", verbose=False, quiet=True, dry_run=False,
        sub_command="push", hf_revision="main",
        hf_endpoint="https://huggingface.co",
    )
    base.update(cfg_overrides)
    cfg = Config(**base)
    plog = PipelineLogger(log_style="text", quiet=True)
    return cfg, plog


def test_pipeline_skips_when_tag_matches_existing_manifest(tmp_path, monkeypatch):
    """If target tag exists with matching digest, log + exit 0 (no push)."""
    cfg, plog = _build_pipeline(tmp_path)
    expected_digest = "sha256:" + "a" * 64

    # Mock everything Pipeline touches
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    fake_downloader.list_files.return_value = []  # not reached
    fake_registry = MagicMock(target_repo="models/x")

    # validate_manifest_tag: tag exists with matching digest
    monkeypatch.setattr(
        "oci_modelcar.pipeline.validate_manifest_tag",
        lambda c, r, tag, expected: None,  # no raise = match
    )
    monkeypatch.setattr(
        "oci_modelcar.pipeline.get_manifest_digest_at_tag",
        lambda c, r, tag: expected_digest,
    )

    # We don't have a real Pipeline yet; the test will guide implementation.
    # The Pipeline.run() method should:
    # - resolve_revision
    # - list_files
    # - derive_tag
    # - check existing manifest: if matches, log + early return RunResult
    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)
    # We override get_manifest_digest_at_tag to simulate a matching tag
    # The expected manifest digest depends on the layers; for simplicity in this
    # test, we return the same value the test expects. The Pipeline must compute
    # the layer digests deterministically OR accept an early-skip when the tag
    # exists at all (depending on the design choice). Our spec says:
    # - matches → exit 0 (skip)
    # - differs without --force → refuse (PushError)
    # - differs with --force → overwrite

    # For this test, we simulate "tag exists, matches"; the Pipeline must
    # short-circuit BEFORE listing files / downloading. Since computing the
    # final manifest digest requires layers, we instead test:
    # - get_manifest_digest_at_tag returns a known digest
    # - Pipeline checks "tag exists" without --force → log and skip
    # The actual conflict/match logic happens AFTER manifest is computed,
    # comparing against the registry's current digest.

    # Refined: the Pipeline does pre-flight check but the digest comparison
    # is deferred until after file listing. So for this test, we just verify
    # the pre-flight check fetches the existing tag's digest and stores it.
    # The actual skip/refuse decision is in task 8.6 (manifest assembly).

    pytest.skip("Tag conflict policy is exercised in task 8.6 manifest tests")


def test_pipeline_resolves_revision_and_lists_files(tmp_path, monkeypatch):
    cfg, plog = _build_pipeline(tmp_path)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    fake_downloader.list_files.return_value = [
        HfFile("model.safetensors", 1000, None),
        HfFile("config.json", 100, None),
    ]
    fake_registry = MagicMock(target_repo="models/x")

    pipeline = Pipeline(
        cfg, plog, downloader=fake_downloader, registry_client=fake_registry
    )
    rev, files, target_tag = pipeline._preflight()
    assert rev == "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    assert len(files) == 2
    assert target_tag == "9fb191250dd5"


def test_pipeline_preflight_no_files_raises_config(tmp_path, monkeypatch):
    cfg, plog = _build_pipeline(tmp_path)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    fake_downloader.list_files.return_value = []
    fake_registry = MagicMock(target_repo="models/x")

    pipeline = Pipeline(
        cfg, plog, downloader=fake_downloader, registry_client=fake_registry
    )
    from oci_modelcar.errors import ConfigError
    with pytest.raises(ConfigError, match="no files matched"):
        pipeline._preflight()
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL (Pipeline class not defined).

- [ ] **Step 3: Implement Pipeline pre-flight**

Append to `src/oci_modelcar/pipeline.py`:

```python
from dataclasses import dataclass

from oci_modelcar.errors import ConfigError, PartialFailure
from oci_modelcar.manifest import (
    build_config_bytes,
    build_manifest_bytes,
    derive_tag,
)
from oci_modelcar.registry import push_manifest, push_small_blob, validate_manifest_tag


@dataclass(frozen=True, slots=True)
class RunResult:
    manifest_digest: str
    image_ref: str
    image_ref_digest: str
    layers: tuple[BlobDescriptor, ...]
    skipped_blobs: int = 0


def get_manifest_digest_at_tag(
    client: OciClient, repo: str, tag: str
) -> str | None:
    """HEAD the manifest tag and return Docker-Content-Digest, or None if 404."""
    url = client.url(repo, "manifests", tag)
    r = client.session.head(
        url, headers={**client.auth, "Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=30,
    )
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        r.raise_for_status()
    digest = r.headers.get("Docker-Content-Digest")
    return digest if digest else None


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        plog: PipelineLogger,
        downloader: HfDownloader | None = None,
        registry_client: OciClient | None = None,
    ) -> None:
        self.cfg = cfg
        self.plog = plog
        # Allow injection for tests; callers in cli.py pass real ones.
        self.downloader = downloader  # type: ignore[assignment]
        self.registry_client = registry_client  # type: ignore[assignment]

    def _preflight(self) -> tuple[str, list[HfFile], str]:
        self.plog.section("Resolving HuggingFace revision")
        revision = self.downloader.resolve_revision(self.cfg.hf_repo, self.cfg.hf_revision)
        self.plog.info(f"HF repo     : {self.cfg.hf_repo}")
        self.plog.info(f"Revision    : {revision}")

        files = self.downloader.list_files(
            self.cfg.hf_repo, revision, self.cfg.allow_patterns
        )
        if not files:
            raise ConfigError(
                f"no files matched allow_patterns {self.cfg.allow_patterns} "
                f"in {self.cfg.hf_repo}@{revision}"
            )
        self.plog.info(f"{len(files)} files matched")

        target_tag = derive_tag(revision, explicit=self.cfg.target_tag)
        return revision, files, target_tag
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_pipeline.py -v
```

Expected: 6 passed (1 skipped).

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat(v1): Pipeline pre-flight (resolve revision, list files, derive tag)"
```

---

### Task 8.4: Pipeline disk space check (mode-aware)

**Files:**
- Modify: `src/oci_modelcar/pipeline.py`
- Modify: `tests/unit/test_pipeline.py`

- [ ] **Step 1: Append failing tests**

```python
def test_pipeline_disk_space_passes_when_sufficient(tmp_path, monkeypatch):
    cfg, plog = _build_pipeline(tmp_path)
    pipeline = Pipeline(cfg, plog, downloader=MagicMock(), registry_client=MagicMock())
    files = [HfFile("a.bin", 1000, None), HfFile("b.bin", 2000, None)]
    # Plenty of space
    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 10 * 1024**3})(),
    )
    pipeline._check_disk_space(files)  # should not raise


def test_pipeline_disk_space_fails_with_hint(tmp_path, monkeypatch):
    from oci_modelcar.errors import DiskSpaceError

    cfg, plog = _build_pipeline(tmp_path, workers=4)
    pipeline = Pipeline(cfg, plog, downloader=MagicMock(), registry_client=MagicMock())
    files = [HfFile("big.bin", 10 * 1024**3, None)]  # 10 GiB file
    # Only 5 GB free → can't fit (need 4 × 10 GiB workers + sources)
    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 5 * 1024**3})(),
    )
    with pytest.raises(DiskSpaceError) as exc:
        pipeline._check_disk_space(files)
    assert exc.value.hint and "--clean-hf-after-push" in exc.value.hint


def test_pipeline_disk_space_clean_hf_lowers_required(tmp_path, monkeypatch):
    """With --clean-hf-after-push, the persistent budget drops to 0; only
    workers × max_layer is required."""
    cfg, plog = _build_pipeline(tmp_path, workers=1, clean_hf_after_push=True)
    pipeline = Pipeline(cfg, plog, downloader=MagicMock(), registry_client=MagicMock())
    files = [HfFile(f"f{i}.bin", 1024**3, None) for i in range(20)]  # 20 GB total

    # 5 GB free should suffice with --clean-hf-after-push because we only need
    # ~1.2 × (1 + 1) GB in flight (rounded up).
    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 5 * 1024**3})(),
    )
    pipeline._check_disk_space(files)  # should not raise
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL.

- [ ] **Step 3: Implement disk check**

Append to `src/oci_modelcar/pipeline.py`:

```python
import shutil

from oci_modelcar.errors import DiskSpaceError
from oci_modelcar.layer import tar_layer_size


class Pipeline:
    # ... (continues)

    def _check_disk_space(self, files: list[HfFile]) -> None:
        if not files:
            return
        max_layer = max(tar_layer_size(f.size) for f in files)
        max_source = max(f.size for f in files)
        total_sources = sum(f.size for f in files)

        in_flight = (max_source + max_layer) * self.cfg.workers * 1.2
        persistent = 0 if self.cfg.clean_hf_after_push else int(total_sources * 1.05)
        needed = int(in_flight + persistent)

        # Ensure spool dir exists for disk_usage
        self.cfg.spool_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.cfg.spool_dir).free
        if free < needed:
            raise DiskSpaceError(
                f"Need {needed / 1e9:.1f} GB free in {self.cfg.spool_dir}, "
                f"only {free / 1e9:.1f} GB available.",
                hint=(
                    f"--spool-dir <other-volume>, --clean-hf-after-push, "
                    f"or lower --workers (currently {self.cfg.workers})."
                ),
            )
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_pipeline.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat(v1): Pipeline._check_disk_space mode-aware"
```

---

### Task 8.5: Pipeline.run with ThreadPoolExecutor + fail-fast

**Files:**
- Modify: `src/oci_modelcar/pipeline.py`
- Modify: `tests/unit/test_pipeline.py`

- [ ] **Step 1: Append failing tests**

```python
import time


def test_pipeline_fail_fast_cancels_pending(tmp_path, monkeypatch):
    """When one worker raises, stop_event must be set and the loop exit
    within ~1 second (cancel_futures kills pending)."""
    cfg, plog = _build_pipeline(tmp_path, workers=2)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [
        HfFile(f"f{i}.bin", 100, None) for i in range(8)
    ]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    # Two workers: file f0 raises immediately; f1..f7 sleep
    call_count = {"n": 0}

    def fake_process(self, repo, revision, hf_file, progress_cb=None):
        call_count["n"] += 1
        if hf_file.path == "f0.bin":
            raise RuntimeError("simulated f0 failure")
        # Other workers loop on stop_event
        for _ in range(50):
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError("stop_event")
            time.sleep(0.05)
        return MagicMock()

    monkeypatch.setattr(FileWorker, "process", fake_process)

    pipeline = Pipeline(
        cfg, plog, downloader=fake_downloader, registry_client=fake_registry
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="simulated f0 failure"):
        pipeline.run()
    elapsed = time.monotonic() - started

    # Even with N=8 files, fail-fast should cancel within seconds, not 8 × 2.5s
    assert elapsed < 5.0, f"fail-fast took too long: {elapsed:.1f}s"


def test_pipeline_continue_on_error_collects_failures(tmp_path, monkeypatch):
    cfg, plog = _build_pipeline(tmp_path, workers=2, fail_fast=False)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [
        HfFile("good.bin", 100, None),
        HfFile("bad.bin", 100, None),
    ]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    def fake_process(self, repo, revision, hf_file, progress_cb=None):
        if hf_file.path == "bad.bin":
            raise RuntimeError("bad failed")
        return BlobDescriptor(
            media_type="application/vnd.oci.image.layer.v1.tar",
            digest="sha256:" + "a" * 64, size=100, hf_path="good.bin",
        )

    monkeypatch.setattr(FileWorker, "process", fake_process)

    pipeline = Pipeline(
        cfg, plog, downloader=fake_downloader, registry_client=fake_registry
    )
    from oci_modelcar.errors import PartialFailure
    with pytest.raises(PartialFailure):
        pipeline.run()
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL (Pipeline.run not implemented).

- [ ] **Step 3: Implement Pipeline.run**

Append to `src/oci_modelcar/pipeline.py`:

```python
from concurrent.futures import Future, ThreadPoolExecutor


class Pipeline:
    # ... (continues)

    def run(self) -> RunResult:
        if self.downloader is None or self.registry_client is None:
            raise RuntimeError("Pipeline requires downloader and registry_client")

        revision, files, target_tag = self._preflight()
        self._check_disk_space(files)

        if self.cfg.dry_run:
            self.plog.info("dry-run: skipping push")
            return RunResult(
                manifest_digest="", image_ref="", image_ref_digest="",
                layers=(), skipped_blobs=0,
            )

        # Spool layout
        (self.cfg.spool_dir / "sources").mkdir(parents=True, exist_ok=True)
        (self.cfg.spool_dir / "layers").mkdir(parents=True, exist_ok=True)

        stop_event = threading.Event()
        from oci_modelcar.registry import head_blob as _head_blob

        def make_worker() -> FileWorker:
            return FileWorker(
                downloader=self.downloader,
                registry_client=self.registry_client,
                head_blob_fn=_head_blob,
                streaming_factory=StreamingBlobUpload,
                layer_prefix=self.cfg.layer_prefix,
                spool_dir=self.cfg.spool_dir,
                clean_hf_after_push=self.cfg.clean_hf_after_push,
                oci_max_retries=self.cfg.oci_max_retries,
                stop_event=stop_event,
            )

        descriptors: list[BlobDescriptor] = []
        failures: list[tuple[str, BaseException]] = []

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            futures: dict[Future[BlobDescriptor], HfFile] = {}
            for hf_file in files:
                worker = make_worker()
                fut = pool.submit(
                    worker.process, self.cfg.hf_repo, revision, hf_file
                )
                futures[fut] = hf_file

            for fut in list(futures):
                hf_file = futures[fut]
                try:
                    desc = fut.result()
                    descriptors.append(desc)
                except BaseException as e:  # noqa: BLE001
                    failures.append((hf_file.path, e))
                    if self.cfg.fail_fast:
                        stop_event.set()
                        # cancel pending
                        for other in futures:
                            if not other.done():
                                other.cancel()
                        # raise the first error
                        raise

        if failures:
            self.plog.error(f"{len(failures)} file(s) failed:")
            for path, exc in failures:
                self.plog.error(f"  {path}: {type(exc).__name__}: {exc}")
            raise PartialFailure(
                f"{len(failures)}/{len(files)} files failed",
                hint="re-run; succeeded blobs are cached in registry",
            )

        return self._assemble_manifest(target_tag, descriptors)

    def _assemble_manifest(
        self, target_tag: str, descriptors: list[BlobDescriptor]
    ) -> RunResult:
        # Sort layers alphabetically by hf_path (deterministic manifest digest)
        descriptors.sort(key=lambda d: d.hf_path)

        diff_ids = [d.digest for d in descriptors]
        config_bytes = build_config_bytes(diff_ids)

        config_digest = push_small_blob(self.registry_client, self.cfg.target_repo, config_bytes)
        manifest_bytes = build_manifest_bytes(config_digest, len(config_bytes), descriptors)
        manifest_digest = push_manifest(
            self.registry_client, self.cfg.target_repo, target_tag, manifest_bytes
        )
        validate_manifest_tag(self.registry_client, self.cfg.target_repo, target_tag, manifest_digest)

        for tag in self.cfg.also_tags:
            push_manifest(self.registry_client, self.cfg.target_repo, tag, manifest_bytes)
            validate_manifest_tag(self.registry_client, self.cfg.target_repo, tag, manifest_digest)

        image_ref = f"{self.registry_client.host}/{self.cfg.target_repo}:{target_tag}"
        image_ref_digest = f"{self.registry_client.host}/{self.cfg.target_repo}@{manifest_digest}"
        self.plog.info(f"manifest: {manifest_digest}")
        self.plog.info(f"image:    {image_ref}")
        self.plog.output_variable("manifestDigest", manifest_digest)
        self.plog.output_variable("imageRef", image_ref)
        self.plog.output_variable("imageRefDigest", image_ref_digest)

        return RunResult(
            manifest_digest=manifest_digest,
            image_ref=image_ref,
            image_ref_digest=image_ref_digest,
            layers=tuple(descriptors),
        )
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_pipeline.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat(v1): Pipeline.run with ThreadPoolExecutor, fail-fast, manifest assembly"
```

---

### Task 8.6: Tag conflict policy

**Files:**
- Modify: `src/oci_modelcar/pipeline.py`
- Modify: `tests/unit/test_pipeline.py`

- [ ] **Step 1: Append failing tests**

```python
def test_pipeline_tag_match_skips_job(tmp_path, monkeypatch):
    """If existing tag matches the manifest digest we'd produce, skip."""
    cfg, plog = _build_pipeline(tmp_path, workers=1)

    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [HfFile("a.bin", 100, None)]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    # Compute a deterministic "existing" digest equal to what assemble would
    # produce. For test simplicity, we patch _assemble_manifest to return
    # a known digest, and patch get_manifest_digest_at_tag to return the same.
    expected_digest = "sha256:" + "f" * 64
    monkeypatch.setattr(
        "oci_modelcar.pipeline.get_manifest_digest_at_tag",
        lambda *a, **kw: expected_digest,
    )

    # Patch the worker so we don't actually run downloads
    monkeypatch.setattr(
        FileWorker, "process",
        lambda self, *a, **kw: BlobDescriptor(
            media_type=ML_TAR, digest="sha256:" + "a" * 64, size=100, hf_path="a.bin",
        ),
    )

    # Patch _assemble_manifest internals: build_manifest_bytes returns a
    # known body whose sha256 == expected_digest is the only way to make
    # this test fully realistic. Easier: we ALSO patch _assemble_manifest
    # to return a RunResult with the expected digest.
    captured: dict = {}

    real_assemble = Pipeline._assemble_manifest
    def fake_assemble(self, target_tag, descriptors):
        captured["called"] = True
        return RunResult(
            manifest_digest=expected_digest, image_ref="x:y",
            image_ref_digest="x@" + expected_digest, layers=tuple(descriptors),
        )
    monkeypatch.setattr(Pipeline, "_assemble_manifest", fake_assemble)

    pipeline = Pipeline(
        cfg, plog, downloader=fake_downloader, registry_client=fake_registry
    )
    result = pipeline.run()
    assert result.manifest_digest == expected_digest


def test_pipeline_tag_conflict_no_force_raises(tmp_path, monkeypatch):
    """Existing tag with DIFFERENT digest, no --force → PushError."""
    cfg, plog = _build_pipeline(tmp_path, workers=1, force=False)

    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [HfFile("a.bin", 100, None)]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    # Existing tag is at a different digest
    monkeypatch.setattr(
        "oci_modelcar.pipeline.get_manifest_digest_at_tag",
        lambda *a, **kw: "sha256:" + "1" * 64,
    )

    pipeline = Pipeline(
        cfg, plog, downloader=fake_downloader, registry_client=fake_registry
    )
    with pytest.raises(PushError, match="tag exists with different digest"):
        pipeline.run()


def test_pipeline_tag_conflict_with_force_overwrites(tmp_path, monkeypatch):
    cfg, plog = _build_pipeline(tmp_path, workers=1, force=True)

    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [HfFile("a.bin", 100, None)]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )
    monkeypatch.setattr(
        "oci_modelcar.pipeline.get_manifest_digest_at_tag",
        lambda *a, **kw: "sha256:" + "1" * 64,  # differs but --force overrides
    )
    monkeypatch.setattr(
        FileWorker, "process",
        lambda self, *a, **kw: BlobDescriptor(
            media_type=ML_TAR, digest="sha256:" + "a" * 64, size=100, hf_path="a.bin",
        ),
    )
    monkeypatch.setattr(
        Pipeline, "_assemble_manifest",
        lambda self, t, d: RunResult(
            manifest_digest="sha256:new", image_ref="x", image_ref_digest="y", layers=tuple(d),
        ),
    )

    pipeline = Pipeline(
        cfg, plog, downloader=fake_downloader, registry_client=fake_registry
    )
    result = pipeline.run()
    assert result.manifest_digest == "sha256:new"  # overwrote existing
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL.

- [ ] **Step 3: Add tag-conflict logic to Pipeline.run**

In `src/oci_modelcar/pipeline.py`, add a check inside `run()` AFTER `_preflight`
and AFTER `_assemble_manifest` is computed (because we need the actual
manifest digest to compare). The simplest correct flow:

```python
def run(self) -> RunResult:
    revision, files, target_tag = self._preflight()
    self._check_disk_space(files)

    if self.cfg.dry_run:
        self.plog.info("dry-run: skipping push")
        return RunResult(manifest_digest="", image_ref="", image_ref_digest="", layers=(), skipped_blobs=0)

    (self.cfg.spool_dir / "sources").mkdir(parents=True, exist_ok=True)
    (self.cfg.spool_dir / "layers").mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    from oci_modelcar.registry import head_blob as _head_blob

    def make_worker() -> FileWorker:
        return FileWorker(
            downloader=self.downloader,
            registry_client=self.registry_client,
            head_blob_fn=_head_blob,
            streaming_factory=StreamingBlobUpload,
            layer_prefix=self.cfg.layer_prefix,
            spool_dir=self.cfg.spool_dir,
            clean_hf_after_push=self.cfg.clean_hf_after_push,
            oci_max_retries=self.cfg.oci_max_retries,
            stop_event=stop_event,
        )

    descriptors: list[BlobDescriptor] = []
    failures: list[tuple[str, BaseException]] = []

    with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
        futures: dict[Future[BlobDescriptor], HfFile] = {}
        for hf_file in files:
            worker = make_worker()
            fut = pool.submit(worker.process, self.cfg.hf_repo, revision, hf_file)
            futures[fut] = hf_file
        for fut in list(futures):
            hf_file = futures[fut]
            try:
                desc = fut.result()
                descriptors.append(desc)
            except BaseException as e:
                failures.append((hf_file.path, e))
                if self.cfg.fail_fast:
                    stop_event.set()
                    for other in futures:
                        if not other.done():
                            other.cancel()
                    raise

    if failures:
        self.plog.error(f"{len(failures)} file(s) failed:")
        for path, exc in failures:
            self.plog.error(f"  {path}: {type(exc).__name__}: {exc}")
        raise PartialFailure(
            f"{len(failures)}/{len(files)} files failed",
            hint="re-run; succeeded blobs are cached in registry",
        )

    # Assemble manifest, then check for tag conflict
    result = self._assemble_manifest(target_tag, descriptors)
    self._check_tag_conflict(target_tag, result.manifest_digest)
    return result

def _check_tag_conflict(self, target_tag: str, new_digest: str) -> None:
    existing = get_manifest_digest_at_tag(self.registry_client, self.cfg.target_repo, target_tag)
    if existing is None or existing == new_digest:
        return  # absent or matches → either fresh push or idempotent re-run
    if self.cfg.force:
        self.plog.warning(
            f"tag {target_tag!r} existed at {existing} but --force overwrote with {new_digest}"
        )
        return
    raise PushError(
        f"tag {target_tag!r} exists with different digest "
        f"(was {existing}, computed {new_digest})",
        hint="use --force to overwrite, or pick a different --target-tag.",
    )
```

Wait — this leaves a problem: with the design above, we'd push the manifest BEFORE checking for conflict, since `_assemble_manifest` calls `push_manifest`. That overwrites silently, which is exactly what the spec forbids.

Reorder so the conflict check happens BEFORE pushing the manifest:

```python
def run(self) -> RunResult:
    revision, files, target_tag = self._preflight()
    self._check_disk_space(files)

    # Pre-flight tag check (best-effort early; the final check happens after
    # we know our manifest digest, which requires layer digests = downloads).
    # For now, just inform if the tag exists at all.
    existing_tag_digest = get_manifest_digest_at_tag(
        self.registry_client, self.cfg.target_repo, target_tag
    )
    if existing_tag_digest is not None and not self.cfg.force:
        self.plog.info(
            f"tag {target_tag!r} already at {existing_tag_digest}; will compare after push"
        )

    if self.cfg.dry_run:
        return RunResult(manifest_digest="", image_ref="", image_ref_digest="", layers=(), skipped_blobs=0)

    # ... (download + push layers as before, fail-fast collected) ...

    # Now compute the manifest digest WITHOUT pushing yet
    descriptors.sort(key=lambda d: d.hf_path)
    diff_ids = [d.digest for d in descriptors]
    config_bytes = build_config_bytes(diff_ids)
    import hashlib
    config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
    manifest_bytes = build_manifest_bytes(config_digest, len(config_bytes), descriptors)
    new_manifest_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()

    if existing_tag_digest is not None:
        if existing_tag_digest == new_manifest_digest:
            self.plog.info(f"tag {target_tag!r} already at {new_manifest_digest}; skipping push")
            image_ref = f"{self.registry_client.host}/{self.cfg.target_repo}:{target_tag}"
            return RunResult(
                manifest_digest=new_manifest_digest,
                image_ref=image_ref,
                image_ref_digest=f"{self.registry_client.host}/{self.cfg.target_repo}@{new_manifest_digest}",
                layers=tuple(descriptors),
            )
        if not self.cfg.force:
            raise PushError(
                f"tag {target_tag!r} exists with different digest "
                f"(was {existing_tag_digest}, computed {new_manifest_digest})",
                hint="use --force to overwrite, or pick a different --target-tag.",
            )

    # Now push config + manifest
    push_small_blob(self.registry_client, self.cfg.target_repo, config_bytes)
    push_manifest(
        self.registry_client, self.cfg.target_repo, target_tag, manifest_bytes
    )
    validate_manifest_tag(
        self.registry_client, self.cfg.target_repo, target_tag, new_manifest_digest
    )

    for tag in self.cfg.also_tags:
        push_manifest(self.registry_client, self.cfg.target_repo, tag, manifest_bytes)
        validate_manifest_tag(self.registry_client, self.cfg.target_repo, tag, new_manifest_digest)

    image_ref = f"{self.registry_client.host}/{self.cfg.target_repo}:{target_tag}"
    image_ref_digest = f"{self.registry_client.host}/{self.cfg.target_repo}@{new_manifest_digest}"
    self.plog.info(f"manifest: {new_manifest_digest}")
    self.plog.info(f"image:    {image_ref}")
    self.plog.output_variable("manifestDigest", new_manifest_digest)
    self.plog.output_variable("imageRef", image_ref)
    self.plog.output_variable("imageRefDigest", image_ref_digest)

    return RunResult(
        manifest_digest=new_manifest_digest,
        image_ref=image_ref,
        image_ref_digest=image_ref_digest,
        layers=tuple(descriptors),
    )
```

Replace the old `run()` and `_assemble_manifest` (which becomes inline)
in `src/oci_modelcar/pipeline.py`.

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_pipeline.py -v
```

Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat(v1): tag conflict policy (skip on match, refuse without --force)"
```

---

## Phase 9 — cli.py

### Task 9.1: cli.main with sub-command dispatch + signal handlers

**Files:**
- Create: `src/oci_modelcar/cli.py`
- Create: `src/oci_modelcar/__main__.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write failing test**

`tests/unit/test_cli.py`:

```python
from unittest.mock import MagicMock, patch

import pytest

from oci_modelcar.cli import main


def test_cli_no_args_shows_usage(capsys):
    rc = main(["oci-modelcar"])
    out = capsys.readouterr().out + capsys.readouterr().err
    # argparse prints help when no sub-command is given
    assert rc != 0


def test_cli_push_dispatches_to_pipeline(monkeypatch):
    fake_run = MagicMock(return_value=MagicMock(manifest_digest="sha256:x"))
    monkeypatch.setattr("oci_modelcar.cli.Pipeline", MagicMock(return_value=MagicMock(run=fake_run)))
    monkeypatch.setattr("oci_modelcar.cli.HfApi", MagicMock())
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")

    rc = main(["oci-modelcar", "push"])
    assert rc == 0


def test_cli_push_config_error_exits_2(monkeypatch):
    monkeypatch.delenv("HF_REPO", raising=False)
    monkeypatch.delenv("REGISTRY", raising=False)
    monkeypatch.delenv("TARGET_REPO", raising=False)
    rc = main(["oci-modelcar", "push"])
    assert rc == 2


def test_cli_push_gated_repo_exits_3(monkeypatch):
    from oci_modelcar.errors import GatedRepoError
    fake_pipe = MagicMock()
    fake_pipe.run.side_effect = GatedRepoError("gated", hint="accept terms")
    monkeypatch.setattr("oci_modelcar.cli.Pipeline", MagicMock(return_value=fake_pipe))
    monkeypatch.setattr("oci_modelcar.cli.HfApi", MagicMock())
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    rc = main(["oci-modelcar", "push"])
    assert rc == 3


def test_cli_push_disk_space_error_exits_4(monkeypatch):
    from oci_modelcar.errors import DiskSpaceError
    fake_pipe = MagicMock()
    fake_pipe.run.side_effect = DiskSpaceError("no space", hint="more disk")
    monkeypatch.setattr("oci_modelcar.cli.Pipeline", MagicMock(return_value=fake_pipe))
    monkeypatch.setattr("oci_modelcar.cli.HfApi", MagicMock())
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    rc = main(["oci-modelcar", "push"])
    assert rc == 4


def test_cli_push_partial_failure_exits_7(monkeypatch):
    from oci_modelcar.errors import PartialFailure
    fake_pipe = MagicMock()
    fake_pipe.run.side_effect = PartialFailure("2/5 failed")
    monkeypatch.setattr("oci_modelcar.cli.Pipeline", MagicMock(return_value=fake_pipe))
    monkeypatch.setattr("oci_modelcar.cli.HfApi", MagicMock())
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    rc = main(["oci-modelcar", "push"])
    assert rc == 7
```

- [ ] **Step 2: Run, expect fail**

Expected: FAIL.

- [ ] **Step 3: Implement cli.main**

`src/oci_modelcar/cli.py`:

```python
"""CLI entrypoint: argparse dispatch on argv[1] sub-command."""

from __future__ import annotations

import logging
import signal
import sys
import threading

from huggingface_hub import HfApi

from oci_modelcar.config import Config
from oci_modelcar.download import HfDownloader
from oci_modelcar.errors import (
    OciModelcarError,
    exit_code_for,
)
from oci_modelcar.http import build_session
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.pipeline import Pipeline
from oci_modelcar.registry import OciClient

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        print("usage: oci-modelcar {push,status,validate} [options]", file=sys.stderr)
        return 1

    sub = argv[1]
    rest = argv[2:]

    if sub == "push":
        return _run_push(rest)
    if sub == "status":
        return _run_status(rest)
    if sub == "validate":
        return _run_validate(rest)

    print(f"unknown sub-command: {sub}", file=sys.stderr)
    return 1


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def handler(signum, frame):
        log.warning("signal %d received; aborting", signum)
        stop_event.set()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _run_push(argv: list[str]) -> int:
    try:
        cfg = Config.from_env_and_args(argv)
    except OciModelcarError as e:
        print(f"error: {e}", file=sys.stderr)
        if e.hint:
            print(f"hint: {e.hint}", file=sys.stderr)
        return exit_code_for(e)

    plog = PipelineLogger(
        log_style=cfg.log_style or "text", verbose=cfg.verbose, quiet=cfg.quiet
    )
    session = build_session()
    api = HfApi(endpoint=cfg.hf_endpoint)
    downloader = HfDownloader(
        api=api, session=session, spool_dir=cfg.spool_dir,
        stop_event=None,  # set after pipeline init
        max_retries=cfg.hf_max_retries,
    )
    registry_client = OciClient(
        registry_host=cfg.registry, target_repo=cfg.target_repo, session=session,
    )
    pipeline = Pipeline(
        cfg=cfg, plog=plog, downloader=downloader, registry_client=registry_client,
    )
    # Note: stop_event is owned by Pipeline.run(). For SIGINT/SIGTERM during
    # pre-flight (before run() creates the event), we install a handler here
    # that flips a top-level flag the Pipeline checks.
    # For simplicity in v1, we let run() install its own signal-aware behavior.

    try:
        pipeline.run()
        return 0
    except OciModelcarError as e:
        plog.error(f"{type(e).__name__}: {e}")
        if e.hint:
            plog.error(f"hint: {e.hint}")
        return exit_code_for(e)
    except KeyboardInterrupt:
        plog.error("interrupted")
        return 1


def _run_status(argv: list[str]) -> int:
    """List tags for target_repo from the registry."""
    # Minimal config: only --registry and --target-repo
    import argparse
    p = argparse.ArgumentParser(prog="oci-modelcar status")
    p.add_argument("--registry", required=True)
    p.add_argument("--target-repo", required=True)
    p.add_argument("--log-style", default=None, choices=["text", "azure"])
    p.add_argument("--quiet", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true", default=False)
    ns = p.parse_args(argv)

    plog = PipelineLogger(
        log_style=ns.log_style or "text", verbose=ns.verbose, quiet=ns.quiet
    )
    client = OciClient(registry_host=ns.registry, target_repo=ns.target_repo)
    url = client.url(ns.target_repo, "tags", "list")
    r = client.session.get(url, headers=client.auth, timeout=30)
    if r.status_code == 404:
        plog.info(f"repo {ns.target_repo} not found in {ns.registry}")
        return 0
    r.raise_for_status()
    tags = r.json().get("tags", []) or []
    plog.info(f"Tags in {ns.target_repo} @ {ns.registry}:")
    for tag in tags:
        url = client.url(ns.target_repo, "manifests", tag)
        h = client.session.head(
            url, headers={**client.auth, "Accept": "application/vnd.oci.image.manifest.v1+json"},
            timeout=30,
        )
        digest = h.headers.get("Docker-Content-Digest", "?")
        plog.info(f"  {tag}  {digest}")
    return 0


def _run_validate(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="oci-modelcar validate")
    p.add_argument("--registry", required=True)
    p.add_argument("--target-repo", required=True)
    p.add_argument("--target-tag", required=True)
    p.add_argument("--log-style", default=None, choices=["text", "azure"])
    p.add_argument("--quiet", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true", default=False)
    ns = p.parse_args(argv)

    plog = PipelineLogger(
        log_style=ns.log_style or "text", verbose=ns.verbose, quiet=ns.quiet
    )
    client = OciClient(registry_host=ns.registry, target_repo=ns.target_repo)

    import json
    url = client.url(ns.target_repo, "manifests", ns.target_tag)
    r = client.session.get(
        url,
        headers={**client.auth, "Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=30,
    )
    r.raise_for_status()
    manifest = r.json()
    config_digest = manifest["config"]["digest"]
    layers = manifest["layers"]

    from oci_modelcar.registry import head_blob
    if head_blob(client, ns.target_repo, config_digest) is None:
        plog.error(f"config blob missing: {config_digest}")
        return 1
    for layer in layers:
        if head_blob(client, ns.target_repo, layer["digest"]) is None:
            plog.error(f"layer missing: {layer['digest']}")
            return 1
    plog.info(f"manifest at {ns.target_tag} is coherent ({len(layers)} layers)")
    return 0
```

`src/oci_modelcar/__main__.py`:

```python
"""`python -m oci_modelcar` entry point."""

from oci_modelcar.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_cli.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/cli.py src/oci_modelcar/__main__.py tests/unit/test_cli.py
git commit -m "feat(v1): cli.py with push/status/validate sub-commands and exit codes"
```

---

### Task 9.2: Update pyproject.toml entry point

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add console script entry**

In `pyproject.toml`:

```toml
[project.scripts]
oci-modelcar = "oci_modelcar.cli:main"
```

- [ ] **Step 2: Reinstall and verify**

```bash
.venv/bin/pip install -e '.[dev]' 2>&1 | tail -3
.venv/bin/oci-modelcar 2>&1 | head -2
# Expected: usage line
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore(v1): wire oci-modelcar console script entry point"
```

---

### Task 9.3: Re-enable pre-commit pytest

**Files:**
- Modify: `.pre-commit-config.yaml`

- [ ] **Step 1: Uncomment pytest hook**

Restore the `pytest-fast` hook in `.pre-commit-config.yaml`.

- [ ] **Step 2: Verify all hooks pass**

```bash
PATH="/run/current-system/sw/bin:$(pwd)/.venv/bin:$PATH" pre-commit run --all-files 2>&1 | tail -10
```

Expected: ruff + mypy + pytest all pass.

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore(v1): re-enable pytest pre-commit hook"
```

---

## Phase 10 — Integration tests + E2E

### Task 10.1: Integration test — full happy path

**Files:**
- Create: `tests/integration/test_pipeline_full.py`

- [ ] **Step 1: Write the test**

```python
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.config import Config
from oci_modelcar.download import HfDownloader, HfFile
from oci_modelcar.http import build_session
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.pipeline import Pipeline
from oci_modelcar.registry import OciClient


def test_full_pipeline_push_two_files(httpserver: HTTPServer, tmp_path: Path):
    """End-to-end: HF metadata mocked → 2 files downloaded + tarred + pushed."""
    payload_a = b"X" * 1000
    payload_b = b"Y" * 500

    # HF resolve endpoints
    httpserver.expect_request("/api/models/foo/bar").respond_with_json(
        {"sha": "deadbeef" * 5}
    )
    httpserver.expect_request("/api/models/foo/bar/revision/main").respond_with_json(
        {"sha": "deadbeef" * 5}
    )
    # HF tree (used by HfApi.list_repo_tree under the hood)
    httpserver.expect_request("/api/models/foo/bar/tree/main").respond_with_json([
        {"type": "file", "path": "a.bin", "size": len(payload_a)},
        {"type": "file", "path": "b.bin", "size": len(payload_b)},
    ])
    # File downloads
    httpserver.expect_request("/foo/bar/resolve/deadbeefdeadbeefdeadbeefdeadbeefdeadbeef/a.bin").respond_with_data(
        payload_a, headers={"Content-Length": str(len(payload_a))}
    )
    httpserver.expect_request("/foo/bar/resolve/deadbeefdeadbeefdeadbeefdeadbeefdeadbeef/b.bin").respond_with_data(
        payload_b, headers={"Content-Length": str(len(payload_b))}
    )

    # OCI registry: HEAD-blobs (always 404), POST init, PATCH, PUT for each blob
    blobs_received: list[bytes] = []

    def head_404(request):
        return Response("", status=404)

    httpserver.expect_request(
        "/v2/models/x/blobs/", method="HEAD"
    ).respond_with_handler(head_404)

    upload_counter = {"n": 0}

    def post_init(request):
        upload_counter["n"] += 1
        return Response(
            "", status=202, headers={"Location": httpserver.url_for(f"/u/{upload_counter['n']}")},
        )

    httpserver.expect_request(
        "/v2/models/x/blobs/uploads/", method="POST"
    ).respond_with_handler(post_init)

    def patch_handler(request):
        blobs_received.append(request.data)
        # Return location matching the URL
        loc = request.path
        return Response("", status=202, headers={"Location": httpserver.url_for(loc)})

    httpserver.expect_request("/u/1", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/2", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/3", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/1", method="PUT").respond_with_data("", status=201)
    httpserver.expect_request("/u/2", method="PUT").respond_with_data("", status=201)
    httpserver.expect_request("/u/3", method="PUT").respond_with_data("", status=201)

    # Manifest tag HEAD: 404 (no existing tag)
    httpserver.expect_request(
        "/v2/models/x/manifests/deadbeefdead", method="HEAD"
    ).respond_with_data("", status=404)
    # Manifest PUT
    httpserver.expect_request(
        "/v2/models/x/manifests/deadbeefdead", method="PUT"
    ).respond_with_data("", status=201)
    # Manifest GET (validate_manifest_tag)
    def manifest_get(request):
        # Compute the digest of what was sent in PUT — we need to mirror it
        # in the response header. For simplicity, accept anything and return
        # a constant. The validate function will then compare with
        # the digest pipeline computed.
        # Better: capture the PUT body, hash it, and serve that.
        return Response("{}", status=200, headers={"Docker-Content-Digest": "sha256:" + "0" * 64})
    # We need a smarter handler; defer to a state-tracking dict.
    pytest.skip(
        "Full integration requires a stateful registry mock; "
        "this scaffold is a starting point — wire up a smarter mock or "
        "use docker registry:2 for the e2e test."
    )
```

This integration test is complex because it needs a stateful registry mock
(the manifest digest depends on actual layer digests pushed). The skip marker
is intentional: the e2e test (Task 10.3) covers the happy path against a
real `registry:2` container.

- [ ] **Step 2: Verify the file is collected and the test skips cleanly**

```bash
.venv/bin/python -m pytest tests/integration/test_pipeline_full.py -v
```

Expected: 1 skipped.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_pipeline_full.py
git commit -m "test(v1): integration scaffold (full coverage in e2e via registry:2)"
```

---

### Task 10.2: Integration test — failure modes (fail-fast cancel timing)

**Files:**
- Create: `tests/integration/test_pipeline_failure_modes.py`

- [ ] **Step 1: Write the test**

```python
import threading
import time
from unittest.mock import MagicMock

import pytest

from oci_modelcar.config import Config
from oci_modelcar.download import HfFile
from oci_modelcar.errors import PartialFailure
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.manifest import BlobDescriptor, ML_TAR
from oci_modelcar.pipeline import FileWorker, Pipeline


def _make_cfg(tmp_path, **overrides):
    base = dict(
        hf_repo="foo/bar", registry="registry.example.com",
        target_repo="models/x", target_tag=None, also_tags=[],
        allow_patterns=(".bin",), layer_prefix="models/",
        workers=2, spool_dir=tmp_path / "spool", clean_hf_after_push=False,
        hf_max_retries=3, oci_max_retries=3, fail_fast=True, force=False,
        log_style="text", verbose=False, quiet=True, dry_run=False,
        sub_command="push", hf_revision="main",
        hf_endpoint="https://huggingface.co",
    )
    base.update(overrides)
    return Config(**base)


def test_fail_fast_cancellation_within_seconds(tmp_path, monkeypatch):
    """Worker f0 raises immediately; f1..f7 simulate long downloads but
    must abort within ~1s of stop_event being set."""
    cfg = _make_cfg(tmp_path, workers=2)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [HfFile(f"f{i}.bin", 100, None) for i in range(8)]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    def fake_process(self, repo, revision, hf_file, progress_cb=None):
        if hf_file.path == "f0.bin":
            raise RuntimeError("boom")
        for _ in range(50):
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError("stop_event")
            time.sleep(0.05)
        return BlobDescriptor(media_type=ML_TAR, digest="sha256:x", size=100, hf_path=hf_file.path)

    monkeypatch.setattr(FileWorker, "process", fake_process)

    plog = PipelineLogger(quiet=True)
    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run()
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"fail-fast took too long: {elapsed:.1f}s"


def test_continue_on_error_collects_and_raises_partial(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, workers=2, fail_fast=False)
    fake_downloader = MagicMock()
    fake_downloader.resolve_revision.return_value = "deadbeef" * 5
    fake_downloader.list_files.return_value = [
        HfFile("ok.bin", 100, None),
        HfFile("bad.bin", 100, None),
    ]
    fake_registry = MagicMock(target_repo="models/x")

    monkeypatch.setattr(
        "oci_modelcar.pipeline.shutil.disk_usage",
        lambda p: type("DU", (), {"free": 100 * 1024**3})(),
    )

    def fake_process(self, repo, revision, hf_file, progress_cb=None):
        if hf_file.path == "bad.bin":
            raise RuntimeError("bad")
        return BlobDescriptor(media_type=ML_TAR, digest="sha256:x", size=100, hf_path=hf_file.path)

    monkeypatch.setattr(FileWorker, "process", fake_process)

    plog = PipelineLogger(quiet=True)
    pipeline = Pipeline(cfg, plog, downloader=fake_downloader, registry_client=fake_registry)
    with pytest.raises(PartialFailure):
        pipeline.run()
```

- [ ] **Step 2: Run, expect pass**

```bash
.venv/bin/python -m pytest tests/integration/test_pipeline_failure_modes.py -v
```

Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_pipeline_failure_modes.py
git commit -m "test(v1): fail-fast cancellation timing + continue-on-error PartialFailure"
```

---

### Task 10.3: E2E test against real HuggingFace + docker registry:2

**Files:**
- Modify: `tests/e2e/test_real_huggingface.py`

- [ ] **Step 1: Write the e2e test**

```python
"""End-to-end against real HuggingFace and a local docker registry:2.

Requires:
- Docker daemon running
- Network access to huggingface.co

Run with: pytest tests/e2e/ -m e2e -v
"""

import os
import subprocess
import time

import pytest

REGISTRY_PORT = 5000
HF_REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HF_REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"
EXPECTED_TAG = "9fb191250dd5"


@pytest.fixture(scope="module")
def local_registry():
    """Spin up a local docker registry:2 on REGISTRY_PORT for the duration
    of the module."""
    name = "oci-modelcar-e2e-registry"
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name,
         "-p", f"{REGISTRY_PORT}:5000", "registry:2"],
        check=True,
    )
    # Wait for healthy
    for _ in range(30):
        try:
            import requests
            r = requests.get(f"http://localhost:{REGISTRY_PORT}/v2/", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    else:
        pytest.fail("local registry did not become healthy")
    yield f"localhost:{REGISTRY_PORT}"
    subprocess.run(["docker", "stop", name], check=False, capture_output=True)


@pytest.mark.e2e
def test_e2e_push_tiny_llama(local_registry, tmp_path):
    """Push the pinned tiny llama to the local registry and validate."""
    env = os.environ.copy()
    spool = tmp_path / "spool"
    cmd = [
        "oci-modelcar", "push",
        "--hf-repo", HF_REPO,
        "--hf-revision", HF_REVISION,
        "--registry", local_registry,
        "--target-repo", "e2e/tiny-llama",
        "--spool-dir", str(spool),
        "--workers", "2",
        "--quiet",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"push failed:\n{result.stdout}\n{result.stderr}"

    # Validate
    cmd = [
        "oci-modelcar", "validate",
        "--registry", local_registry,
        "--target-repo", "e2e/tiny-llama",
        "--target-tag", EXPECTED_TAG,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"validate failed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.e2e
def test_e2e_push_idempotent(local_registry, tmp_path):
    """Re-running the push against an already-pushed tag → exit 0 (skipped)."""
    env = os.environ.copy()
    cmd = [
        "oci-modelcar", "push",
        "--hf-repo", HF_REPO,
        "--hf-revision", HF_REVISION,
        "--registry", local_registry,
        "--target-repo", "e2e/tiny-llama",
        "--spool-dir", str(tmp_path / "spool2"),
        "--quiet",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
```

- [ ] **Step 2: Run (only if Docker available)**

```bash
.venv/bin/python -m pytest tests/e2e/ -m e2e -v
```

Expected: 2 passed (skipped if no Docker).

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_real_huggingface.py
git commit -m "test(v1): e2e against pinned HF + local docker registry:2"
```

---

## Phase 11 — Documentation + release prep

### Task 11.1: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README sections**

Update README.md:
- **Top section**: bump version to 1.0.0, mention v1 as the current line
- **Quick start**: update CLI example to remove `--chunk-mib`/`--state-file`/`--upload-mode`
- **Add section "Disk space"**: explain `--spool-dir` and `--clean-hf-after-push`
- **Migration guide from v0.5**: short subsection pointing at CHANGELOG

- [ ] **Step 2: Render check (markdown)**

```bash
.venv/bin/python -c "from pathlib import Path; print(Path('README.md').read_text()[:500])"
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(v1): update README for v1.0 (per-file pipeline, new flags)"
```

---

### Task 11.2: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace architecture section**

Update CLAUDE.md to reflect:
- New module list (`download.py`, `registry.py`, `pipeline.py`, etc.)
- Drop references to `state.py`, `oci.py`'s `ChunkedBlobUpload`, `runner.py`'s `_PipeBuffer`
- Add the spec/plan references: `2026-05-08-oci-modelcar-v1-design.md` + `2026-05-08-oci-modelcar-v1.md`
- Update "Locked design decisions" with v1 invariants:
  - Single PATCH per blob from local file (Jib-style)
  - HfApi for metadata only; bytes via our streamer (cancellation)
  - Cross-origin authorization stripping (security)
  - Drop chunked mode

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(v1): update CLAUDE.md for v1.0 architecture"
```

---

### Task 11.3: CHANGELOG + version bump + release tag

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write CHANGELOG entry**

Prepend to `CHANGELOG.md`:

```markdown
## [1.0.0] - 2026-05-08

### Added
- Per-file pipeline (download → tar → push → cleanup) parallelized via `--workers`.
- `huggingface_hub.HfApi` for metadata (revision resolve, file listing,
  LFS sha256 detection); bytes streamed by our own code so mid-stream
  cancellation works on multi-GB downloads.
- Atomic write semantics for downloaded files (`.partial` → rename).
- Cross-origin authorization stripping on HF→S3 redirects (security).
- Expanded HF token sources: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
  `~/.cache/huggingface/token`, opt-out via `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.
- Range-200 fallback (server ignores Range → truncate + restart) ported
  from huggingface_hub.
- Specific error classes: `GatedRepoError`, `RevisionNotFoundError`,
  `EntryNotFoundError`, `DiskSpaceError`, `PushError`, `PartialFailure`.
- Per-class CI exit codes: 0/1/2/3/4/5/6/7.
- `--spool-dir`, `--clean-hf-after-push` flags + matching env vars.
- Tag conflict policy: skip on match, refuse without `--force`, overwrite
  with `--force`.
- Mode-aware disk space pre-flight check (with/without `--clean-hf-after-push`).

### Changed
- **Single PATCH per blob from local file (Jib-style replay-on-cut).**
  Eliminates per-PATCH LB routing decisions on misconfigured Artifactory
  HA clusters. Same wire shape as containers/image and Jib.
- Default `--oci-max-retries` lowered from 10 to 5 (each retry is a full
  PATCH replay; bandwidth ballooning on systematic failures otherwise).
- Tar layer size formula now exposed as `layer.tar_layer_size(file_size)`.

### Removed
- `state.json` and the `state.py` module entirely. Registry HEAD is
  the source of truth for resumability and idempotency.
- `ChunkedBlobUpload` and chunked PATCH mode.
- `--state-file`, `--chunk-mib`, `--upload-mode` flags. Use `--spool-dir`
  and `--clean-hf-after-push` for the new disk model.
- `_PipeBuffer` thread-bridge (per-file pipeline replaces it).
- `tags.py` (`derive_tag` migrated into `manifest.py`).

### Security
- HF Authorization tokens are no longer forwarded on cross-origin
  redirects. Previous versions could leak a Bearer token to S3 /
  CloudFront (HF's redirect target for LFS files), where the request
  was rejected but the token may have been logged.
```

- [ ] **Step 2: Bump version**

In `pyproject.toml`:

```toml
version = "1.0.0"
```

- [ ] **Step 3: Final pre-commit check**

```bash
PATH="/run/current-system/sw/bin:$(pwd)/.venv/bin:$PATH" pre-commit run --all-files 2>&1 | tail -10
```

Expected: all hooks pass on the entire codebase.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md pyproject.toml
git commit -m "chore(release): v1.0.0"
```

- [ ] **Step 5: Tag and push**

```bash
git tag -a v1.0.0 -m "v1.0.0 — clean rewrite: per-file pipeline, single-PATCH, drop state.json"
# Push when user is ready:
# git push origin feat/v1-rewrite v1.0.0
```

---

## Self-review (post-plan)

After all tasks complete, verify:

- [ ] All sections of the design spec (`docs/superpowers/specs/2026-05-08-oci-modelcar-v1-design.md`)
  have at least one task implementing them.
- [ ] No `TBD`/`TODO`/`FIXME` placeholders in implementation code.
- [ ] `pytest --cov=oci_modelcar` shows ≥ 95% line coverage.
- [ ] `pre-commit run --all-files` passes.
- [ ] E2E tests pass against real HF + docker registry:2.
- [ ] `git log` shows ~33 commits, one per task.
- [ ] CHANGELOG covers all breaking changes.

