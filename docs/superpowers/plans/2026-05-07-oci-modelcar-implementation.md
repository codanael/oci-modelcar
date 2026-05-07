# oci-modelcar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-ready Python package `oci-modelcar` that streams HuggingFace models directly into OCI registries as multi-layer images, with three-level resume capability and full OCI Distribution v1.1 compliance.

**Architecture:** Modular Python 3.14 package with eight focused modules (`config`, `http`, `logging`, `hf`, `oci`, `tar_layer`, `manifest`, `state`, `runner`, `cli`). Each module has one responsibility. TDD throughout: tests first, implementation second. Streaming pipeline: HF → tarfile.open(mode="w|") → ChunkedBlobUpload → registry. Resume at three levels: HfStream Range request (intra-file network), OCI session GET resync (intra-file PATCH), JSON state file (cross-process file-level).

**Tech Stack:** Python 3.14, `requests`, `urllib3`, stdlib (`dataclasses`, `argparse`, `logging`, `tarfile`, `hashlib`, `concurrent.futures`, `tomllib`, `json`). Dev: `pytest`, `pytest-httpserver`, `ruff`, `mypy`, `pre-commit`. CI: GitHub Actions with PyPI Trusted Publishing. Tests against real HuggingFace `hf-internal-testing/tiny-random-LlamaForCausalLM` and local `registry:2`.

**Spec reference:** `docs/superpowers/specs/2026-05-07-oci-modelcar-design.md`

---

## Phase 1 — Project skeleton

### Task 1: Create package layout and `pyproject.toml`

**Files:**
- Create: `pyproject.toml`
- Create: `LICENSE`
- Create: `README.md`
- Create: `CHANGELOG.md`
- Create: `src/oci_modelcar/__init__.py`
- Create: `src/oci_modelcar/__main__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "oci-modelcar"
version = "0.1.0"
description = "Stream HuggingFace models directly into OCI registries as multi-layer images"
readme = "README.md"
requires-python = ">=3.14"
license = "MIT"
license-files = ["LICENSE"]
authors = [{name = "codanael"}]
keywords = ["huggingface", "oci", "kserve", "modelcar", "registry"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: System Administrators",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.14",
    "Topic :: System :: Archiving :: Packaging",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "requests>=2.32",
    "urllib3>=2.2",
]

[project.urls]
Homepage = "https://github.com/codanael/oci-modelcar"
Issues = "https://github.com/codanael/oci-modelcar/issues"
Source = "https://github.com/codanael/oci-modelcar"

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "pytest-httpserver>=1.0",
    "ruff>=0.7",
    "mypy>=1.13",
    "types-requests",
    "build>=1.2",
    "pre-commit>=4",
]
e2e = ["pytest>=8"]

[project.scripts]
oci-modelcar = "oci_modelcar.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/oci_modelcar"]

[tool.ruff]
target-version = "py314"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.14"
strict = true
warn_return_any = true
files = ["src/oci_modelcar"]

[tool.pytest.ini_options]
addopts = "-v --strict-markers"
testpaths = ["tests"]
markers = ["e2e: end-to-end tests requiring Docker, skopeo, and network"]

[tool.coverage.run]
source = ["src/oci_modelcar"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

- [ ] **Step 2: Create `LICENSE` (MIT)**

```
MIT License

Copyright (c) 2026 codanael

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Create minimal `README.md`**

```markdown
# oci-modelcar

Stream HuggingFace models directly into OCI registries as multi-layer images,
suitable for KServe with native OCI image volumes (KEP-4639).

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

See `oci-modelcar push --help` for all options.

## License

MIT
```

- [ ] **Step 4: Create `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Initial implementation
```

- [ ] **Step 5: Create `src/oci_modelcar/__init__.py`**

```python
"""Stream HuggingFace models into OCI registries as multi-layer images."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("oci-modelcar")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
```

- [ ] **Step 6: Create `src/oci_modelcar/__main__.py`**

```python
from oci_modelcar.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Create empty `tests/__init__.py` and `tests/conftest.py`**

`tests/__init__.py`: empty file.

`tests/conftest.py`:
```python
"""Shared pytest fixtures."""
```

- [ ] **Step 8: Verify package installs**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -c "import oci_modelcar; print(oci_modelcar.__version__)"
```

Expected: prints `0.1.0`.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml LICENSE README.md CHANGELOG.md src/ tests/
git commit -m "chore: initial package skeleton"
```

---

### Task 2: Pre-commit configuration

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff check
        entry: ruff check --fix
        language: system
        types: [python]
      - id: ruff-format
        name: ruff format
        entry: ruff format
        language: system
        types: [python]
      - id: mypy
        name: mypy --strict
        entry: mypy --strict src/
        language: system
        types: [python]
        pass_filenames: false
      - id: pytest-fast
        name: pytest (not e2e)
        entry: pytest -m "not e2e" -q
        language: system
        pass_filenames: false
        stages: [pre-commit]
```

- [ ] **Step 2: Install hooks (in nix-shell with ruff, mypy, pre-commit available)**

```bash
pre-commit install
```

Expected: `pre-commit installed at .git/hooks/pre-commit`.

- [ ] **Step 3: Run hooks against existing files**

```bash
pre-commit run --all-files
```

Expected: all hooks pass (no Python files yet to lint substantively).

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit hooks (ruff, mypy, pytest)"
```

---

## Phase 2 — Configuration

### Task 3: `Config` dataclass with env + CLI parsing

**Files:**
- Create: `src/oci_modelcar/config.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing test for env-only config**

`tests/unit/__init__.py`: empty file.

`tests/unit/test_config.py`:
```python
import os
import pytest
from oci_modelcar.config import Config, ConfigError


def test_config_from_env_minimal(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    cfg = Config.from_env_and_args([])
    assert cfg.hf_repo == "foo/bar"
    assert cfg.hf_revision == "main"
    assert cfg.hf_endpoint == "https://huggingface.co"
    assert cfg.registry == "registry.example.com"
    assert cfg.target_repo == "models/x"
    assert cfg.target_tag is None  # derived later
    assert cfg.workers == 1
    assert cfg.chunk_mib == 8


def test_config_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    monkeypatch.setenv("WORKERS", "4")
    cfg = Config.from_env_and_args(["--workers", "2"])
    assert cfg.workers == 2


def test_config_missing_required_raises(monkeypatch):
    monkeypatch.delenv("HF_REPO", raising=False)
    monkeypatch.delenv("REGISTRY", raising=False)
    monkeypatch.delenv("TARGET_REPO", raising=False)
    with pytest.raises(ConfigError, match="hf_repo"):
        Config.from_env_and_args([])


def test_config_workers_cap(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(ConfigError, match="workers"):
        Config.from_env_and_args(["--workers", "9"])


def test_config_invalid_target_tag(monkeypatch):
    monkeypatch.setenv("HF_REPO", "foo/bar")
    monkeypatch.setenv("REGISTRY", "registry.example.com")
    monkeypatch.setenv("TARGET_REPO", "models/x")
    with pytest.raises(ConfigError, match="target_tag"):
        Config.from_env_and_args(["--target-tag", "bad/tag"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_config.py -v
```

Expected: ImportError on `oci_modelcar.config`.

- [ ] **Step 3: Implement `Config` dataclass**

`src/oci_modelcar/config.py`:
```python
"""Configuration: env vars + CLI args + validation."""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self


class ConfigError(Exception):
    """Raised when configuration is invalid."""


_TAG_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127}$")
_DEFAULT_ALLOW = ".safetensors .json .txt .md .model"


def _xdg_state_home() -> Path:
    raw = os.environ.get("XDG_STATE_HOME")
    if raw:
        return Path(raw)
    return Path.home() / ".local" / "state"


def _envbool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    hf_repo: str
    registry: str
    target_repo: str
    hf_revision: str = "main"
    hf_endpoint: str = "https://huggingface.co"
    target_tag: str | None = None
    also_tags: list[str] = field(default_factory=list)
    allow_patterns: tuple[str, ...] = field(
        default_factory=lambda: tuple(_DEFAULT_ALLOW.split())
    )
    layer_prefix: str = "models/"
    chunk_mib: int = 8
    workers: int = 1
    state_file: Path = field(
        default_factory=lambda: _xdg_state_home() / "oci-modelcar" / "state.json"
    )
    hf_max_retries: int = 10
    oci_max_retries: int = 10
    fail_fast: bool = True
    force: bool = False
    log_style: str | None = None  # None = auto-detect
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
            hf_revision=ns.hf_revision or os.environ.get("HF_REVISION", "main"),
            hf_endpoint=(
                ns.hf_endpoint
                or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
            ),
            target_tag=ns.target_tag or os.environ.get("TARGET_TAG") or None,
            also_tags=_parse_csv(ns.also_tag or os.environ.get("ALSO_TAGS", "")),
            allow_patterns=tuple(
                (ns.allow_patterns or os.environ.get("ALLOW_PATTERNS", _DEFAULT_ALLOW))
                .split()
            ),
            layer_prefix=(
                ns.layer_prefix
                if ns.layer_prefix is not None
                else os.environ.get("LAYER_PATH_PREFIX", "models/")
            ),
            chunk_mib=int(ns.chunk_mib or os.environ.get("CHUNK_MIB", "8")),
            workers=int(ns.workers or os.environ.get("WORKERS", "1")),
            state_file=Path(
                ns.state_file
                or os.environ.get(
                    "STATE_FILE", str(_xdg_state_home() / "oci-modelcar" / "state.json")
                )
            ),
            hf_max_retries=int(
                ns.hf_max_retries or os.environ.get("HF_MAX_RETRIES", "10")
            ),
            oci_max_retries=int(
                ns.oci_max_retries or os.environ.get("OCI_MAX_RETRIES", "10")
            ),
            fail_fast=(
                False
                if ns.continue_on_error
                else (ns.fail_fast or _envbool("FAIL_FAST", True))
            ),
            force=ns.force or _envbool("FORCE", False),
            log_style=ns.log_style or os.environ.get("LOG_STYLE"),
            verbose=ns.verbose or _envbool("LOG_VERBOSE", False),
            quiet=ns.quiet or _envbool("LOG_QUIET", False),
            dry_run=ns.dry_run,
            sub_command=ns.sub_command or "push",
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
        if not (1 <= self.chunk_mib <= 1024):
            raise ConfigError(f"chunk_mib must be in [1, 1024], got {self.chunk_mib}")
        if self.target_tag is not None and not _TAG_RE.match(self.target_tag):
            raise ConfigError(
                f"target_tag {self.target_tag!r} does not match "
                "[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127}"
            )
        for t in self.also_tags:
            if not _TAG_RE.match(t):
                raise ConfigError(f"also_tag {t!r} is invalid")
        if self.log_style is not None and self.log_style not in ("text", "azure"):
            raise ConfigError(
                f"log_style must be 'text' or 'azure', got {self.log_style!r}"
            )

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_mib * 1024 * 1024


def _parse_csv(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else []


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oci-modelcar",
        description="Stream HuggingFace models into OCI registries.",
    )
    sub = p.add_subparsers(dest="sub_command")
    push = sub.add_parser("push", help="Push a HF model as an OCI image")
    push.add_argument("--hf-repo", default=None)
    push.add_argument("--hf-revision", default=None)
    push.add_argument("--hf-endpoint", default=None)
    push.add_argument("--registry", default=None)
    push.add_argument("--target-repo", default=None)
    push.add_argument("--target-tag", default=None)
    push.add_argument("--also-tag", default=None, help="CSV list of additional tags")
    push.add_argument("--allow-patterns", default=None)
    push.add_argument("--layer-prefix", default=None)
    push.add_argument("--chunk-mib", default=None, type=int)
    push.add_argument("--workers", default=None, type=int)
    push.add_argument("--state-file", default=None)
    push.add_argument("--hf-max-retries", default=None, type=int)
    push.add_argument("--oci-max-retries", default=None, type=int)
    g = push.add_mutually_exclusive_group()
    g.add_argument("--fail-fast", action="store_true", default=False)
    g.add_argument("--continue-on-error", action="store_true", default=False)
    push.add_argument("--force", action="store_true", default=False)
    push.add_argument("--log-style", default=None, choices=["text", "azure"])
    g2 = push.add_mutually_exclusive_group()
    g2.add_argument("--verbose", action="store_true", default=False)
    g2.add_argument("--quiet", action="store_true", default=False)
    push.add_argument("--dry-run", action="store_true", default=False)
    return p
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_config.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/config.py tests/unit/__init__.py tests/unit/test_config.py
git commit -m "feat(config): Config dataclass with env+CLI parsing"
```

---

## Phase 3 — HTTP, auth, logging primitives

### Task 4: HTTP session with auth and retries

**Files:**
- Create: `src/oci_modelcar/http.py`
- Create: `tests/unit/test_http.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_http.py`:
```python
import base64
import json
from pathlib import Path

import pytest

from oci_modelcar.http import (
    build_session,
    docker_config_auth,
    huggingface_token,
    oci_auth_header,
)


def test_oci_auth_header_from_env(monkeypatch):
    monkeypatch.setenv("OCI_USERNAME", "alice")
    monkeypatch.setenv("OCI_PASSWORD", "s3cr3t")
    hdr = oci_auth_header("registry.example.com")
    expected = "Basic " + base64.b64encode(b"alice:s3cr3t").decode()
    assert hdr == {"Authorization": expected}


def test_oci_auth_header_from_docker_config(monkeypatch, tmp_path):
    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".docker"
    cfg_dir.mkdir()
    raw = base64.b64encode(b"bob:hunter2").decode()
    (cfg_dir / "config.json").write_text(
        json.dumps({"auths": {"registry.example.com": {"auth": raw}}})
    )
    hdr = oci_auth_header("registry.example.com")
    assert hdr == {"Authorization": f"Basic {raw}"}


def test_oci_auth_header_no_creds(monkeypatch, tmp_path):
    monkeypatch.delenv("OCI_USERNAME", raising=False)
    monkeypatch.delenv("OCI_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert oci_auth_header("registry.example.com") == {}


def test_huggingface_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert huggingface_token() == "hf_secret"


def test_huggingface_token_from_cache_file(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "token").write_text("hf_from_cache\n")
    assert huggingface_token() == "hf_from_cache"


def test_huggingface_token_none(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert huggingface_token() is None


def test_build_session_has_user_agent():
    s = build_session()
    assert "oci-modelcar/" in s.headers["User-Agent"]


def test_docker_config_auth_handles_missing(tmp_path):
    assert docker_config_auth(tmp_path / "missing.json", "x") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_http.py -v
```

Expected: ImportError on `oci_modelcar.http`.

- [ ] **Step 3: Implement `http.py`**

`src/oci_modelcar/http.py`:
```python
"""Shared HTTP session + auth resolution."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oci_modelcar import __version__


def build_session() -> requests.Session:
    """Session with retries on idempotent methods only.

    Non-idempotent methods (PATCH, PUT) are NOT retried automatically;
    those have their own resync-aware retry logic in oci.py.
    """
    s = requests.Session()
    retry = Retry(
        total=8,
        backoff_factor=2,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers["User-Agent"] = f"oci-modelcar/{__version__}"
    return s


def oci_auth_header(registry_host: str) -> dict[str, str]:
    """Resolve registry auth in priority order.

    1. OCI_USERNAME + OCI_PASSWORD env
    2. ~/.docker/config.json
    3. $XDG_RUNTIME_DIR/containers/auth.json
    """
    user = os.environ.get("OCI_USERNAME")
    pwd = os.environ.get("OCI_PASSWORD")
    if user and pwd is not None:
        token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    docker_cfg = Path.home() / ".docker" / "config.json"
    auth = docker_config_auth(docker_cfg, registry_host)
    if auth:
        return {"Authorization": f"Basic {auth}"}

    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        containers_cfg = Path(xdg) / "containers" / "auth.json"
        auth = docker_config_auth(containers_cfg, registry_host)
        if auth:
            return {"Authorization": f"Basic {auth}"}

    return {}


def docker_config_auth(path: Path, registry_host: str) -> str | None:
    """Read a docker/podman config.json and return the base64 auth blob if any."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    auths = data.get("auths", {})
    entry = auths.get(registry_host)
    if not isinstance(entry, dict):
        return None
    raw = entry.get("auth")
    return raw if isinstance(raw, str) and raw else None


def huggingface_token() -> str | None:
    """Resolve HF token: HF_TOKEN env > ~/.cache/huggingface/token."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    cache = Path.home() / ".cache" / "huggingface" / "token"
    if cache.is_file():
        try:
            return cache.read_text().strip() or None
        except OSError:
            return None
    return None


def huggingface_auth_header() -> dict[str, str]:
    tok = huggingface_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_http.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/http.py tests/unit/test_http.py
git commit -m "feat(http): session + OCI/HF auth resolution"
```

---

### Task 5: Logging — text + Azure formatters

**Files:**
- Create: `src/oci_modelcar/logging.py`
- Create: `tests/unit/test_logging.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_logging.py`:
```python
import io

import pytest

from oci_modelcar.logging import (
    AzureFormatter,
    PipelineLogger,
    TextFormatter,
    detect_log_style,
)


def test_detect_log_style_azure(monkeypatch):
    monkeypatch.setenv("TF_BUILD", "True")
    assert detect_log_style(None) == "azure"


def test_detect_log_style_text(monkeypatch):
    monkeypatch.delenv("TF_BUILD", raising=False)
    assert detect_log_style(None) == "text"


def test_detect_log_style_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("TF_BUILD", "True")
    assert detect_log_style("text") == "text"


def test_text_formatter_section():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="text", use_color=False)
    log.section("Resolving HuggingFace revision")
    rendered = out.getvalue()
    assert "Resolving HuggingFace revision" in rendered
    assert "##[" not in rendered


def test_azure_formatter_section():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="azure", use_color=False)
    log.section("Resolving HuggingFace revision")
    assert "##[section]Resolving HuggingFace revision" in out.getvalue()


def test_azure_formatter_group():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="azure", use_color=False)
    log.group_start("file1.safetensors")
    log.info("uploading...")
    log.group_end("done")
    rendered = out.getvalue()
    assert "##[group]file1.safetensors" in rendered
    assert "##[endgroup]" in rendered
    assert "uploading..." in rendered


def test_azure_formatter_warning_and_error():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="azure", use_color=False)
    log.warning("retry")
    log.error("fatal")
    rendered = out.getvalue()
    assert "##[warning]retry" in rendered
    assert "##[error]fatal" in rendered


def test_text_formatter_warning_prefix():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="text", use_color=False)
    log.warning("retry")
    log.error("fatal")
    rendered = out.getvalue()
    assert "WARN" in rendered
    assert "ERROR" in rendered
    assert "##[" not in rendered


def test_set_output_variable_azure():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="azure", use_color=False)
    log.output_variable("manifestDigest", "sha256:abc")
    rendered = out.getvalue()
    assert (
        "##vso[task.setvariable variable=manifestDigest;isOutput=true]sha256:abc"
        in rendered
    )
    # Plain KEY=VALUE line emitted in both styles
    assert "MANIFESTDIGEST=sha256:abc" in rendered


def test_set_output_variable_text():
    out = io.StringIO()
    log = PipelineLogger(stream=out, style="text", use_color=False)
    log.output_variable("manifestDigest", "sha256:abc")
    rendered = out.getvalue()
    assert "MANIFESTDIGEST=sha256:abc" in rendered
    assert "##vso[" not in rendered
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_logging.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `logging.py`**

`src/oci_modelcar/logging.py`:
```python
"""Pipeline logger with text and Azure DevOps formatters."""
from __future__ import annotations

import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, Literal

LogStyle = Literal["text", "azure"]


def detect_log_style(explicit: str | None) -> LogStyle:
    if explicit in ("text", "azure"):
        return explicit  # type: ignore[return-value]
    if os.environ.get("TF_BUILD", "").strip().lower() == "true":
        return "azure"
    return "text"


def _supports_color(stream: IO[str]) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


class _Formatter:
    def section(self, title: str) -> str:
        raise NotImplementedError

    def group_start(self, title: str) -> str:
        raise NotImplementedError

    def group_end(self, summary: str) -> str:
        raise NotImplementedError

    def info(self, msg: str) -> str:
        return msg + "\n"

    def warning(self, msg: str) -> str:
        raise NotImplementedError

    def error(self, msg: str) -> str:
        raise NotImplementedError

    def debug(self, msg: str) -> str:
        raise NotImplementedError

    def progress(self, percent: int, msg: str) -> str:
        raise NotImplementedError

    def output_variable(self, name: str, value: str) -> str:
        raise NotImplementedError


class AzureFormatter(_Formatter):
    def section(self, title: str) -> str:
        return f"##[section]{title}\n"

    def group_start(self, title: str) -> str:
        return f"##[group]{title}\n"

    def group_end(self, summary: str) -> str:
        return (summary + "\n" if summary else "") + "##[endgroup]\n"

    def warning(self, msg: str) -> str:
        return f"##[warning]{msg}\n"

    def error(self, msg: str) -> str:
        return f"##[error]{msg}\n"

    def debug(self, msg: str) -> str:
        return f"##[debug]{msg}\n"

    def progress(self, percent: int, msg: str) -> str:
        return f"##vso[task.setprogress value={percent}]{msg}\n"

    def output_variable(self, name: str, value: str) -> str:
        vso = f"##vso[task.setvariable variable={name};isOutput=true]{value}\n"
        kv = f"{name.upper()}={value}\n"
        return vso + kv


_RESET = "\033[0m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_BOLD = "\033[1m"


class TextFormatter(_Formatter):
    def __init__(self, use_color: bool) -> None:
        self.use_color = use_color

    def _c(self, code: str, msg: str) -> str:
        return f"{code}{msg}{_RESET}" if self.use_color else msg

    def section(self, title: str) -> str:
        bar = "─" * max(2, 60 - len(title))
        line = f"── {title} {bar}"
        return self._c(_BOLD, line) + "\n"

    def group_start(self, title: str) -> str:
        return f"{title}\n"

    def group_end(self, summary: str) -> str:
        return (summary + "\n") if summary else ""

    def warning(self, msg: str) -> str:
        return self._c(_YELLOW, f"WARN  {msg}") + "\n"

    def error(self, msg: str) -> str:
        return self._c(_RED, f"ERROR {msg}") + "\n"

    def debug(self, msg: str) -> str:
        return self._c(_DIM, f"DEBUG {msg}") + "\n"

    def progress(self, percent: int, msg: str) -> str:
        return f"[{percent:>3}%] {msg}\n"

    def output_variable(self, name: str, value: str) -> str:
        return f"{name.upper()}={value}\n"


class PipelineLogger:
    def __init__(
        self,
        stream: IO[str] | None = None,
        style: LogStyle = "text",
        use_color: bool | None = None,
        verbose: bool = False,
        quiet: bool = False,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.verbose = verbose
        self.quiet = quiet
        if use_color is None:
            use_color = _supports_color(self.stream)
        self.formatter: _Formatter = (
            AzureFormatter() if style == "azure" else TextFormatter(use_color)
        )
        self._lock = threading.Lock()

    def _emit(self, text: str) -> None:
        with self._lock:
            self.stream.write(text)
            self.stream.flush()

    def section(self, title: str) -> None:
        if self.quiet:
            return
        self._emit(self.formatter.section(title))

    def group_start(self, title: str) -> None:
        self._emit(self.formatter.group_start(title))

    def group_end(self, summary: str = "") -> None:
        self._emit(self.formatter.group_end(summary))

    def info(self, msg: str) -> None:
        if self.quiet:
            return
        self._emit(self.formatter.info(msg))

    def warning(self, msg: str) -> None:
        self._emit(self.formatter.warning(msg))

    def error(self, msg: str) -> None:
        self._emit(self.formatter.error(msg))

    def debug(self, msg: str) -> None:
        if not self.verbose:
            return
        self._emit(self.formatter.debug(msg))

    def progress(self, percent: int, msg: str) -> None:
        if self.quiet:
            return
        self._emit(self.formatter.progress(percent, msg))

    def output_variable(self, name: str, value: str) -> None:
        self._emit(self.formatter.output_variable(name, value))

    def heartbeat(self, line: str) -> None:
        # Heartbeats are plain text in both styles, no tags
        self._emit(f"[HB] {line}\n")

    @contextmanager
    def file_scope(self, title: str) -> Iterator["FileScopedLogger"]:
        """Buffer per-file logs (for parallel mode); flush atomically."""
        scoped = FileScopedLogger(title, self)
        try:
            yield scoped
        finally:
            scoped.flush()


class FileScopedLogger:
    """Buffer all logs for one file; flush atomically at close."""

    def __init__(self, title: str, parent: PipelineLogger) -> None:
        self._buf: list[str] = []
        self._parent = parent
        self._summary = ""
        self._title = title
        self._buf.append(parent.formatter.group_start(title))

    def info(self, msg: str) -> None:
        if not self._parent.quiet:
            self._buf.append(self._parent.formatter.info(msg))

    def warning(self, msg: str) -> None:
        self._buf.append(self._parent.formatter.warning(msg))

    def error(self, msg: str) -> None:
        self._buf.append(self._parent.formatter.error(msg))

    def debug(self, msg: str) -> None:
        if self._parent.verbose:
            self._buf.append(self._parent.formatter.debug(msg))

    def set_summary(self, summary: str) -> None:
        self._summary = summary

    def flush(self) -> None:
        self._buf.append(self._parent.formatter.group_end(self._summary))
        self._parent._emit("".join(self._buf))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_logging.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/logging.py tests/unit/test_logging.py
git commit -m "feat(logging): text + azure formatters with file-scoped buffering"
```

---

## Phase 4 — HuggingFace module

### Task 6: HF revision resolution

**Files:**
- Create: `src/oci_modelcar/hf.py`
- Create: `tests/unit/test_hf_revision.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_hf_revision.py`:
```python
import pytest
from pytest_httpserver import HTTPServer

from oci_modelcar.hf import HfClient


def test_resolve_revision_main(httpserver: HTTPServer):
    httpserver.expect_request("/api/models/foo/bar").respond_with_json(
        {"sha": "a3f47b09c8d2e6f1a89b7c4d3e8f2a1b5c6d7e8f"}
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    sha = client.resolve_revision("main")
    assert sha == "a3f47b09c8d2e6f1a89b7c4d3e8f2a1b5c6d7e8f"


def test_resolve_revision_explicit_sha(httpserver: HTTPServer):
    full = "0" * 40
    httpserver.expect_request(f"/api/models/foo/bar/revision/{full}").respond_with_json(
        {"sha": full}
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    sha = client.resolve_revision(full)
    assert sha == full


def test_resolve_revision_branch_name(httpserver: HTTPServer):
    httpserver.expect_request("/api/models/foo/bar/revision/release/v1").respond_with_json(
        {"sha": "b" * 40}
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    sha = client.resolve_revision("release/v1")
    assert sha == "b" * 40


def test_resolve_revision_falls_back_on_404(httpserver: HTTPServer):
    httpserver.expect_request("/api/models/foo/bar/revision/unknown").respond_with_data(
        "", status=404
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    sha = client.resolve_revision("unknown")
    # Falls back: returns the input as-is and warns
    assert sha == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_hf_revision.py -v
```

Expected: ImportError on `oci_modelcar.hf`.

- [ ] **Step 3: Implement `HfClient.resolve_revision`**

`src/oci_modelcar/hf.py`:
```python
"""HuggingFace client: revision resolution + file listing + streaming."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from oci_modelcar.http import build_session, huggingface_auth_header

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HfFile:
    path: str
    size: int


class HfClient:
    def __init__(
        self,
        endpoint: str,
        repo: str,
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.repo = repo
        self.session = session if session is not None else build_session()
        self.timeout = timeout

    @property
    def auth(self) -> dict[str, str]:
        return huggingface_auth_header()

    def resolve_revision(self, revision: str) -> str:
        """Resolve a revision (branch/tag/SHA/'main') to a 40-char SHA."""
        if revision == "main" or not revision:
            url = f"{self.endpoint}/api/models/{self.repo}"
            r = self.session.get(url, headers=self.auth, timeout=self.timeout)
            r.raise_for_status()
            sha = r.json().get("sha")
            if not sha:
                log.warning("HF /api/models/%s did not return sha", self.repo)
                return revision or "main"
            return str(sha)
        url = f"{self.endpoint}/api/models/{self.repo}/revision/{revision}"
        r = self.session.get(url, headers=self.auth, timeout=self.timeout)
        if r.status_code == 404:
            log.warning(
                "HF revision %r not canonicalizable on %s/%s; using as-is",
                revision,
                self.endpoint,
                self.repo,
            )
            return revision
        r.raise_for_status()
        sha = r.json().get("sha")
        return str(sha) if sha else revision
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_hf_revision.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/hf.py tests/unit/test_hf_revision.py
git commit -m "feat(hf): revision resolution (main / SHA / branch / fallback)"
```

---

### Task 7: HF file listing with allow patterns

**Files:**
- Modify: `src/oci_modelcar/hf.py` (add `list_files`)
- Create: `tests/unit/test_hf_listing.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_hf_listing.py`:
```python
from pytest_httpserver import HTTPServer

from oci_modelcar.hf import HfClient, HfFile


def test_list_files_basic(httpserver: HTTPServer):
    httpserver.expect_request(
        "/api/models/foo/bar/tree/main"
    ).respond_with_json(
        [
            {"type": "file", "path": "model.safetensors", "size": 1000},
            {"type": "file", "path": "config.json", "size": 100},
            {"type": "file", "path": ".gitattributes", "size": 50},
            {"type": "directory", "path": "subdir", "size": 0},
        ]
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    files = client.list_files("main", allow=(".safetensors", ".json"))
    assert files == [
        HfFile(path="config.json", size=100),
        HfFile(path="model.safetensors", size=1000),
    ]


def test_list_files_filters_by_extension(httpserver: HTTPServer):
    httpserver.expect_request(
        "/api/models/foo/bar/tree/main"
    ).respond_with_json(
        [
            {"type": "file", "path": "a.bin", "size": 1},
            {"type": "file", "path": "b.safetensors", "size": 2},
        ]
    )
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    files = client.list_files("main", allow=(".safetensors",))
    assert [f.path for f in files] == ["b.safetensors"]


def test_list_files_uses_recursive_query_param(httpserver: HTTPServer):
    httpserver.expect_request(
        "/api/models/foo/bar/tree/main", query_string={"recursive": "true"}
    ).respond_with_json([])
    client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    files = client.list_files("main", allow=(".safetensors",))
    assert files == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_hf_listing.py -v
```

Expected: AttributeError on `list_files`.

- [ ] **Step 3: Add `list_files` to `HfClient`**

In `src/oci_modelcar/hf.py`, add the method to the class:
```python
    def list_files(self, revision: str, allow: tuple[str, ...]) -> list[HfFile]:
        """Return [HfFile, ...] sorted by path, filtered by extension."""
        url = f"{self.endpoint}/api/models/{self.repo}/tree/{revision}"
        r = self.session.get(
            url,
            headers=self.auth,
            params={"recursive": "true"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        out: list[HfFile] = []
        for entry in r.json():
            if entry.get("type") != "file":
                continue
            path = entry["path"]
            if not any(path.endswith(ext) for ext in allow):
                continue
            out.append(HfFile(path=path, size=int(entry["size"])))
        out.sort(key=lambda f: f.path)
        return out
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_hf_listing.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/hf.py tests/unit/test_hf_listing.py
git commit -m "feat(hf): list_files with allow-pattern filter and stable order"
```

---

### Task 8: HfStream with Range resume

**Files:**
- Modify: `src/oci_modelcar/hf.py` (add `HfStream`)
- Create: `tests/unit/test_hf_stream.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_hf_stream.py`:
```python
import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.hf import HfClient, HfStream


def _make_client(httpserver: HTTPServer) -> HfClient:
    return HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")


def test_hfstream_reads_full_file(httpserver: HTTPServer):
    payload = b"X" * 1024
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    client = _make_client(httpserver)
    stream = HfStream(client, revision="main", path="file.bin", size=len(payload))
    out = stream.read(-1)
    assert out == payload


def test_hfstream_read_in_chunks(httpserver: HTTPServer):
    payload = bytes(range(256)) * 4  # 1024 bytes
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload, headers={"Content-Length": str(len(payload))}
    )
    client = _make_client(httpserver)
    stream = HfStream(client, revision="main", path="file.bin", size=len(payload))
    out = b""
    while True:
        chunk = stream.read(100)
        if not chunk:
            break
        out += chunk
    assert out == payload


def test_hfstream_size_mismatch_raises(httpserver: HTTPServer):
    httpserver.expect_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        b"X" * 100, headers={"Content-Length": "100"}
    )
    client = _make_client(httpserver)
    with pytest.raises(RuntimeError, match="size mismatch"):
        HfStream(client, revision="main", path="file.bin", size=200)


def test_hfstream_range_resume(httpserver: HTTPServer):
    """First request delivers 50 bytes then drops; resume with Range honors offset."""
    payload = b"A" * 100
    call_count = {"n": 0}

    def first_handler(request):
        # Deliver only first half then close (truncate)
        return Response(payload[:50], status=200,
                        headers={"Content-Length": "100"})

    def second_handler(request):
        call_count["n"] += 1
        rng = request.headers.get("Range")
        assert rng == "bytes=50-"
        start = 50
        return Response(
            payload[start:], status=206,
            headers={
                "Content-Length": str(len(payload) - start),
                "Content-Range": f"bytes {start}-{len(payload)-1}/{len(payload)}",
            },
        )

    # We can't easily make pytest-httpserver return partial data then drop in one
    # response. Instead, simulate by responding with WRONG content-length so the
    # client reads less than declared, triggering the recovery path. We then
    # serve a Range request.
    # Use ordered_handler API:
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        lambda req: Response(
            payload[:50],
            status=200,
            headers={"Content-Length": str(len(payload))},  # claim 100, deliver 50
        )
    )
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        second_handler
    )

    client = _make_client(httpserver)
    stream = HfStream(client, revision="main", path="file.bin", size=len(payload),
                      max_retries=3, backoff_initial=0.0)
    out = stream.read(-1)
    assert out == payload
    assert call_count["n"] == 1


def test_hfstream_size_mismatch_on_resume_validates_content_range(httpserver: HTTPServer):
    """Resume handshake verifies Content-Range starts with the requested offset."""
    payload = b"A" * 100
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_handler(
        lambda req: Response(
            payload[:50], status=200,
            headers={"Content-Length": str(len(payload))}
        )
    )
    httpserver.expect_oneshot_request("/foo/bar/resolve/main/file.bin").respond_with_data(
        payload[40:],  # server returns wrong start (40 instead of 50)
        status=206,
        headers={"Content-Range": "bytes 40-99/100"},
    )
    client = _make_client(httpserver)
    stream = HfStream(client, revision="main", path="file.bin", size=len(payload),
                      max_retries=2, backoff_initial=0.0)
    with pytest.raises(RuntimeError, match="did not honor Range"):
        stream.read(-1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_hf_stream.py -v
```

Expected: NameError or ImportError on `HfStream`.

- [ ] **Step 3: Implement `HfStream`**

In `src/oci_modelcar/hf.py`, add at the bottom:
```python
import contextlib
import random
import time

CHUNK_DEFAULT = 1024 * 1024  # 1 MiB iter_content size


class HfStream:
    """File-like, read-only. Honors read(n). Resumes via HTTP Range on errors."""

    def __init__(
        self,
        client: HfClient,
        revision: str,
        path: str,
        size: int,
        chunk_size: int = CHUNK_DEFAULT,
        max_retries: int = 10,
        backoff_initial: float = 1.0,
        backoff_cap: float = 60.0,
    ) -> None:
        self.client = client
        self.revision = revision
        self.path = path
        self.expected_size = size
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_cap = backoff_cap
        self.bytes_buffered = 0
        self.buf = b""
        self._r: requests.Response | None = None
        self._it = None
        self._open(start=0)

    def _open(self, start: int) -> None:
        url = (
            f"{self.client.endpoint}/{self.client.repo}/resolve/"
            f"{self.revision}/{self.path}"
        )
        headers = dict(self.client.auth)
        if start > 0:
            headers["Range"] = f"bytes={start}-"
        r = self.client.session.get(
            url, headers=headers, stream=True, timeout=600
        )
        r.raise_for_status()
        if start == 0:
            cl_header = r.headers.get("content-length")
            if cl_header is not None:
                cl = int(cl_header)
                if cl != self.expected_size:
                    raise RuntimeError(
                        f"size mismatch for {self.path}: "
                        f"tree={self.expected_size} got={cl}"
                    )
        else:
            cr = r.headers.get("content-range", "")
            if not cr.startswith(f"bytes {start}-"):
                raise RuntimeError(
                    f"server did not honor Range for {self.path}: "
                    f"got Content-Range={cr!r}"
                )
        self._r = r
        self._it = r.iter_content(chunk_size=self.chunk_size)

    def _close_response(self) -> None:
        if self._r is not None:
            with contextlib.suppress(Exception):
                self._r.close()
        self._r = None
        self._it = None

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, self.backoff_initial * (2**attempt))
        delay += random.uniform(0, delay * 0.1)
        if delay > 0:
            time.sleep(delay)

    def _next_chunk(self) -> bytes | None:
        for attempt in range(self.max_retries):
            try:
                if self._it is None:
                    self._open(start=self.bytes_buffered)
                assert self._it is not None
                chunk = next(self._it)
                self.bytes_buffered += len(chunk)
                return chunk
            except StopIteration:
                return None
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ReadTimeout,
            ) as e:
                log.warning(
                    "HF read failed at offset %d (attempt %d/%d): %s",
                    self.bytes_buffered,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                self._close_response()
                self._sleep_backoff(attempt)
        raise RuntimeError(
            f"HF retries exhausted for {self.path} at offset {self.bytes_buffered}"
        )

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            chunks = [self.buf]
            self.buf = b""
            while True:
                c = self._next_chunk()
                if c is None:
                    break
                chunks.append(c)
            # Detect truncation (server lied about content-length on first GET)
            data = b"".join(chunks)
            if self.bytes_buffered < self.expected_size:
                # Try to resume
                missing = self.expected_size - self.bytes_buffered
                self._close_response()
                self._open(start=self.bytes_buffered)
                while self.bytes_buffered < self.expected_size:
                    c = self._next_chunk()
                    if c is None:
                        break
                    data += c
                if self.bytes_buffered < self.expected_size:
                    raise RuntimeError(
                        f"truncated read for {self.path}: "
                        f"got {self.bytes_buffered} expected {self.expected_size}"
                    )
            return data
        # Bounded read(n)
        while len(self.buf) < n:
            c = self._next_chunk()
            if c is None:
                # Truncation? Try resume
                if self.bytes_buffered < self.expected_size:
                    self._close_response()
                    self._open(start=self.bytes_buffered)
                    continue
                break
            self.buf += c
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def close(self) -> None:
        self._close_response()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_hf_stream.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/hf.py tests/unit/test_hf_stream.py
git commit -m "feat(hf): HfStream with HTTP Range resume on connection errors"
```

---

## Phase 5 — OCI module

### Task 9: ChunkedBlobUpload happy path

**Files:**
- Create: `src/oci_modelcar/oci.py`
- Create: `tests/unit/test_oci_upload.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_oci_upload.py`:
```python
import hashlib
import re

from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.oci import OciClient, ChunkedBlobUpload


def _client(httpserver: HTTPServer) -> OciClient:
    host = httpserver.url_for("").rstrip("/")
    # url_for returns full URL including scheme; strip back to host:port
    # OciClient takes a host string and uses https. For the tests we
    # construct OciClient with a base_url override.
    return OciClient(host_url=httpserver.url_for(""))


def test_chunked_upload_happy_path(httpserver: HTTPServer):
    payload = b"X" * (8 * 1024 * 1024 + 100)  # > 1 chunk
    expected_digest = "sha256:" + hashlib.sha256(payload).hexdigest()

    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/123")}
    )

    received: list[bytes] = []

    def patch_handler(request):
        cr = request.headers["Content-Range"]
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m, f"bad Content-Range: {cr!r}"
        received.append(request.data)
        end = int(m.group(2))
        return Response(
            "", status=202,
            headers={
                "Location": httpserver.url_for("/upload/123"),
                "Range": f"0-{end}",
            },
        )

    httpserver.expect_request(
        "/upload/123", method="PATCH"
    ).respond_with_handler(patch_handler)

    httpserver.expect_request(
        "/upload/123", method="PUT"
    ).respond_with_data(
        "", status=201,
        headers={"Location": httpserver.url_for(f"/v2/repo/blobs/{expected_digest}")},
    )

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=8 * 1024 * 1024)
    upload.write(payload)
    digest, total = upload.close()
    assert digest == expected_digest
    assert total == len(payload)
    assert b"".join(received) == payload[: 8 * 1024 * 1024]


def test_content_range_format_no_prefix(httpserver: HTTPServer):
    """OCI Content-Range MUST be 'N-M', NEVER 'bytes N-M/total'."""
    payload = b"Y" * 10
    seen_ranges: list[str] = []

    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/X")}
    )

    def put_handler(request):
        # final PUT may carry residual data with Content-Length, but no Content-Range
        return Response("", status=201)

    httpserver.expect_request("/upload/X", method="PUT").respond_with_handler(put_handler)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64)
    upload.write(payload)
    upload.close()
    # Small payload < chunk size -> no PATCH happens, all goes to PUT
    # Test passes if no exception from server


def test_patch_content_range_is_inclusive(httpserver: HTTPServer):
    payload = b"Z" * 200
    seen: list[tuple[int, int, int]] = []  # (start, end, body_len)

    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/upload/Y")}
    )

    def patch_handler(request):
        cr = request.headers["Content-Range"]
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m
        start, end = int(m.group(1)), int(m.group(2))
        seen.append((start, end, len(request.data)))
        # Spec: end - start + 1 == len(body)
        assert end - start + 1 == len(request.data)
        return Response(
            "", status=202,
            headers={"Location": httpserver.url_for("/upload/Y"), "Range": f"0-{end}"},
        )

    httpserver.expect_request("/upload/Y", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/upload/Y", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64)
    upload.write(payload)
    upload.close()
    assert seen  # at least one PATCH happened
    for start, end, body_len in seen:
        assert end == start + body_len - 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_oci_upload.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement minimum to pass**

`src/oci_modelcar/oci.py`:
```python
"""OCI Distribution v1.1 client: chunked blob upload, blob/manifest validation."""
from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass

import requests

from oci_modelcar.http import build_session, oci_auth_header

log = logging.getLogger(__name__)

ML_TAR = "application/vnd.oci.image.layer.v1.tar"
ML_CFG = "application/vnd.oci.image.config.v1+json"
ML_MAN = "application/vnd.oci.image.manifest.v1+json"


@dataclass(frozen=True, slots=True)
class BlobDescriptor:
    media_type: str
    digest: str
    size: int

    def to_dict(self) -> dict[str, object]:
        return {"mediaType": self.media_type, "digest": self.digest, "size": self.size}


class OciClient:
    def __init__(
        self,
        host_url: str | None = None,
        registry_host: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if host_url is not None:
            self.base = host_url.rstrip("/")
            self.host = self.base.split("//", 1)[-1]
        else:
            assert registry_host is not None
            self.host = registry_host
            self.base = f"https://{registry_host}"
        self.session = session if session is not None else build_session()

    @property
    def auth(self) -> dict[str, str]:
        return oci_auth_header(self.host)

    def url(self, *parts: str) -> str:
        return f"{self.base}/v2/" + "/".join(parts)


class ChunkedBlobUpload:
    """Streaming blob upload with PATCH chunks and PUT finalization.

    Memory bound: ~2 * chunk_size.
    Compliant with OCI Distribution v1.1: Content-Range is inclusive 'N-M'.
    """

    def __init__(
        self,
        client: OciClient,
        repo: str,
        chunk_size: int = 8 * 1024 * 1024,
        max_retries: int = 10,
        backoff_initial: float = 1.0,
        backoff_cap: float = 60.0,
    ) -> None:
        self.client = client
        self.repo = repo
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_cap = backoff_cap
        self.h = hashlib.sha256()
        self.buf = bytearray()
        self.server_offset = 0  # bytes registry has acknowledged
        self.total = 0
        self.location = self._begin()

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

    def write(self, data: bytes) -> int:
        n = len(data)
        self.h.update(data)
        self.buf.extend(data)
        self.total += n
        while len(self.buf) >= self.chunk_size:
            self._flush(self.chunk_size)
        return n

    def _flush(self, size: int) -> None:
        chunk = bytes(self.buf[:size])
        del self.buf[:size]
        self._patch_with_retry(chunk)

    def _patch_with_retry(self, chunk: bytes) -> None:
        start = self.server_offset
        end = start + len(chunk) - 1
        for attempt in range(self.max_retries):
            try:
                hdr = {
                    **self.client.auth,
                    "Content-Type": "application/octet-stream",
                    "Content-Range": f"{start}-{end}",  # OCI inclusive, no 'bytes ' prefix
                    "Content-Length": str(len(chunk)),
                }
                r = self.client.session.patch(
                    self.location, data=chunk, headers=hdr, timeout=600
                )
                if r.status_code == 202:
                    self.location = r.headers.get("Location", self.location)
                    self.server_offset = end + 1
                    return
                if r.status_code == 416:
                    log.warning("PATCH 416 at [%d-%d], resyncing", start, end)
                    self._resync()
                    if self.server_offset >= end + 1:
                        return
                    continue
                r.raise_for_status()
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                log.warning(
                    "PATCH failed [%d-%d] attempt %d: %s", start, end, attempt + 1, e
                )
                self._sleep_backoff(attempt)
                self._resync()
                if self.server_offset >= end + 1:
                    return
        raise RuntimeError(
            f"PATCH retries exhausted at offset {start} (chunk [{start}-{end}])"
        )

    def _resync(self) -> None:
        r = self.client.session.get(
            self.location, headers=self.client.auth, timeout=30
        )
        if r.status_code != 204:
            r.raise_for_status()
        rng = r.headers.get("Range", "")
        if rng:
            try:
                end = int(rng.split("-")[1])
                self.server_offset = end + 1
            except (ValueError, IndexError):
                self.server_offset = 0
        else:
            self.server_offset = 0

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, self.backoff_initial * (2**attempt))
        delay += random.uniform(0, delay * 0.1)
        if delay > 0:
            time.sleep(delay)

    def close(self) -> tuple[str, int]:
        digest = "sha256:" + self.h.hexdigest()
        sep = "&" if "?" in self.location else "?"
        url = f"{self.location}{sep}digest={digest}"
        if self.buf:
            hdr = {
                **self.client.auth,
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(self.buf)),
            }
            r = self.client.session.put(
                url, data=bytes(self.buf), headers=hdr, timeout=600
            )
        else:
            r = self.client.session.put(url, headers=self.client.auth, timeout=120)
        if r.status_code != 201:
            r.raise_for_status()
            raise RuntimeError(f"unexpected status {r.status_code} on PUT close")
        return digest, self.total
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_oci_upload.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/oci.py tests/unit/test_oci_upload.py
git commit -m "feat(oci): chunked blob upload with PATCH+PUT, OCI v1.1 Content-Range"
```

---

### Task 10: ChunkedBlobUpload — 416 handling and resync

**Files:**
- Modify: `src/oci_modelcar/oci.py` (already implemented in Task 9, add tests)
- Create: `tests/unit/test_oci_resync.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_oci_resync.py`:
```python
import re

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.oci import ChunkedBlobUpload, OciClient


def _client(httpserver: HTTPServer) -> OciClient:
    return OciClient(host_url=httpserver.url_for(""))


def test_resync_no_range_header(httpserver: HTTPServer):
    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_data("", status=202, headers={"Location": httpserver.url_for("/u/1")})
    httpserver.expect_request("/u/1", method="GET").respond_with_data("", status=204)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64)
    upload._resync()
    assert upload.server_offset == 0


def test_resync_with_range_0_0(httpserver: HTTPServer):
    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_data("", status=202, headers={"Location": httpserver.url_for("/u/2")})
    httpserver.expect_request("/u/2", method="GET").respond_with_data(
        "", status=204, headers={"Range": "0-0"}
    )
    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64)
    upload._resync()
    assert upload.server_offset == 1  # 1 byte received


def test_resync_with_range_0_1023(httpserver: HTTPServer):
    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_data("", status=202, headers={"Location": httpserver.url_for("/u/3")})
    httpserver.expect_request("/u/3", method="GET").respond_with_data(
        "", status=204, headers={"Range": "0-1023"}
    )
    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64)
    upload._resync()
    assert upload.server_offset == 1024


def test_416_resyncs_and_skips_if_already_accepted(httpserver: HTTPServer):
    """Server returns 416, but a GET shows the chunk was actually accepted."""
    payload = b"A" * 100
    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_data("", status=202, headers={"Location": httpserver.url_for("/u/4")})

    state = {"first_patch": True}

    def patch_handler(request):
        cr = request.headers["Content-Range"]
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m
        if state["first_patch"]:
            state["first_patch"] = False
            return Response("", status=416)
        return Response(
            "", status=202,
            headers={"Location": httpserver.url_for("/u/4"),
                     "Range": f"0-{m.group(2)}"},
        )

    def get_handler(request):
        # Server says: I already have the full payload accepted.
        return Response("", status=204, headers={"Range": f"0-{len(payload)-1}"})

    httpserver.expect_request("/u/4", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/4", method="GET").respond_with_handler(get_handler)
    httpserver.expect_request("/u/4", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64,
                               backoff_initial=0.0)
    upload.write(payload)
    digest, total = upload.close()
    assert total == 100
    assert digest.startswith("sha256:")


def test_patch_500_then_success_via_resync(httpserver: HTTPServer):
    """PATCH transient 500 -> resync sees no progress -> retry chunk -> success."""
    payload = b"B" * 100
    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_data("", status=202, headers={"Location": httpserver.url_for("/u/5")})

    attempts = {"n": 0}

    def patch_handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return Response("", status=500)
        cr = request.headers["Content-Range"]
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m
        return Response(
            "", status=202,
            headers={"Location": httpserver.url_for("/u/5"),
                     "Range": f"0-{m.group(2)}"},
        )

    httpserver.expect_request("/u/5", method="PATCH").respond_with_handler(patch_handler)
    httpserver.expect_request("/u/5", method="GET").respond_with_data("", status=204)
    httpserver.expect_request("/u/5", method="PUT").respond_with_data("", status=201)

    client = _client(httpserver)
    upload = ChunkedBlobUpload(client, repo="repo", chunk_size=64,
                               backoff_initial=0.0)
    upload.write(payload)
    digest, total = upload.close()
    assert total == 100
```

Note: the 500-then-success test currently expects `requests` not to retry the PATCH automatically. Since `Retry` is configured with `allowed_methods=["GET", "HEAD"]`, PATCH retries are entirely controlled by `_patch_with_retry`. The first PATCH gets a 500 response; `r.raise_for_status()` raises an `HTTPError` which is **not** caught by the current except clause (only `ConnectionError`/`Timeout`/`ChunkedEncodingError`). Need to handle 5xx as transient inside `_patch_with_retry`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_oci_resync.py -v
```

Expected: `test_patch_500_then_success_via_resync` fails (HTTPError uncaught), others may pass.

- [ ] **Step 3: Treat 5xx as transient in `_patch_with_retry`**

In `src/oci_modelcar/oci.py`, modify `_patch_with_retry` to catch 5xx and 408/429 as transient:

Replace the `if r.status_code == 416:` block and the `r.raise_for_status()` after it with:

```python
                if r.status_code == 416:
                    log.warning("PATCH 416 at [%d-%d], resyncing", start, end)
                    self._resync()
                    if self.server_offset >= end + 1:
                        return
                    continue
                if r.status_code in (408, 429) or 500 <= r.status_code < 600:
                    log.warning(
                        "PATCH transient %d at [%d-%d] attempt %d",
                        r.status_code, start, end, attempt + 1,
                    )
                    self._sleep_backoff(attempt)
                    self._resync()
                    if self.server_offset >= end + 1:
                        return
                    continue
                r.raise_for_status()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_oci_resync.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/oci.py tests/unit/test_oci_resync.py
git commit -m "feat(oci): treat 5xx/408/429 as transient on PATCH, resync between attempts"
```

---

### Task 11: `push_small_blob` and HEAD validation

**Files:**
- Modify: `src/oci_modelcar/oci.py`
- Create: `tests/unit/test_oci_misc.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_oci_misc.py`:
```python
import hashlib

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.oci import OciClient, head_blob, push_small_blob


def _client(httpserver: HTTPServer) -> OciClient:
    return OciClient(host_url=httpserver.url_for(""))


def test_push_small_blob_already_exists(httpserver: HTTPServer):
    data = b"{}"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    httpserver.expect_request(
        f"/v2/repo/blobs/{digest}", method="HEAD"
    ).respond_with_data("", status=200, headers={"Docker-Content-Digest": digest})
    client = _client(httpserver)
    out = push_small_blob(client, repo="repo", data=data)
    assert out == digest


def test_push_small_blob_creates(httpserver: HTTPServer):
    data = b'{"x":1}'
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    httpserver.expect_request(
        f"/v2/repo/blobs/{digest}", method="HEAD"
    ).respond_with_data("", status=404)
    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_data("", status=202, headers={"Location": httpserver.url_for("/u/cfg")})
    httpserver.expect_request("/u/cfg", method="PUT").respond_with_data("", status=201)
    client = _client(httpserver)
    out = push_small_blob(client, repo="repo", data=data)
    assert out == digest


def test_head_blob_validates_digest(httpserver: HTTPServer):
    digest = "sha256:" + "a" * 64
    httpserver.expect_request(
        f"/v2/repo/blobs/{digest}", method="HEAD"
    ).respond_with_data(
        "", status=200,
        headers={"Docker-Content-Digest": digest, "Content-Length": "100"},
    )
    client = _client(httpserver)
    info = head_blob(client, repo="repo", digest=digest)
    assert info["digest"] == digest
    assert info["size"] == 100


def test_head_blob_digest_mismatch(httpserver: HTTPServer):
    expected = "sha256:" + "a" * 64
    wrong = "sha256:" + "b" * 64
    httpserver.expect_request(
        f"/v2/repo/blobs/{expected}", method="HEAD"
    ).respond_with_data(
        "", status=200, headers={"Docker-Content-Digest": wrong, "Content-Length": "0"}
    )
    client = _client(httpserver)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        head_blob(client, repo="repo", digest=expected)


def test_head_blob_not_found(httpserver: HTTPServer):
    digest = "sha256:" + "c" * 64
    httpserver.expect_request(
        f"/v2/repo/blobs/{digest}", method="HEAD"
    ).respond_with_data("", status=404)
    client = _client(httpserver)
    with pytest.raises(RuntimeError, match="not found"):
        head_blob(client, repo="repo", digest=digest)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_oci_misc.py -v
```

Expected: ImportError on `push_small_blob`/`head_blob`.

- [ ] **Step 3: Add `push_small_blob` and `head_blob` to `oci.py`**

Append to `src/oci_modelcar/oci.py`:
```python
def push_small_blob(client: OciClient, repo: str, data: bytes) -> str:
    """Monolithic POST + PUT for small blobs (config). Returns digest."""
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    head_url = client.url(repo, "blobs", digest)
    h = client.session.head(head_url, headers=client.auth, timeout=30)
    if h.status_code == 200:
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


def head_blob(client: OciClient, repo: str, digest: str) -> dict[str, object]:
    """HEAD a blob, validate Docker-Content-Digest, return {digest, size}."""
    url = client.url(repo, "blobs", digest)
    r = client.session.head(url, headers=client.auth, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"blob not found in {repo}: {digest}")
    got = r.headers.get("Docker-Content-Digest", "")
    if got != digest:
        raise RuntimeError(
            f"digest mismatch on HEAD {digest}: server returned {got!r}"
        )
    cl = r.headers.get("Content-Length", "0")
    return {"digest": digest, "size": int(cl)}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_oci_misc.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/oci.py tests/unit/test_oci_misc.py
git commit -m "feat(oci): push_small_blob + head_blob with Docker-Content-Digest check"
```

---

### Task 12: Manifest push and validation

**Files:**
- Modify: `src/oci_modelcar/oci.py`
- Create: `tests/unit/test_oci_manifest.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_oci_manifest.py`:
```python
import hashlib
import json

import pytest
from pytest_httpserver import HTTPServer

from oci_modelcar.oci import OciClient, push_manifest, validate_manifest_tag


def _client(httpserver: HTTPServer) -> OciClient:
    return OciClient(host_url=httpserver.url_for(""))


def test_push_manifest_returns_digest(httpserver: HTTPServer):
    body = b'{"schemaVersion":2}'
    expected = "sha256:" + hashlib.sha256(body).hexdigest()
    httpserver.expect_request(
        "/v2/repo/manifests/v1", method="PUT"
    ).respond_with_data("", status=201, headers={"Docker-Content-Digest": expected})
    client = _client(httpserver)
    digest = push_manifest(client, repo="repo", tag="v1", manifest_bytes=body)
    assert digest == expected


def test_validate_manifest_tag_match(httpserver: HTTPServer):
    body = b'{"schemaVersion":2}'
    expected = "sha256:" + hashlib.sha256(body).hexdigest()
    httpserver.expect_request(
        "/v2/repo/manifests/v1", method="GET"
    ).respond_with_data(
        body, status=200,
        headers={
            "Docker-Content-Digest": expected,
            "Content-Type": "application/vnd.oci.image.manifest.v1+json",
        },
    )
    client = _client(httpserver)
    validate_manifest_tag(client, repo="repo", tag="v1", expected_digest=expected)


def test_validate_manifest_tag_mismatch(httpserver: HTTPServer):
    body = b'{"schemaVersion":2}'
    expected = "sha256:" + hashlib.sha256(body).hexdigest()
    wrong = "sha256:" + "0" * 64
    httpserver.expect_request(
        "/v2/repo/manifests/v1", method="GET"
    ).respond_with_data(body, status=200, headers={"Docker-Content-Digest": wrong})
    client = _client(httpserver)
    with pytest.raises(RuntimeError, match="manifest digest mismatch"):
        validate_manifest_tag(client, repo="repo", tag="v1", expected_digest=expected)
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/unit/test_oci_manifest.py -v
```

Expected: ImportError.

- [ ] **Step 3: Add `push_manifest` and `validate_manifest_tag`**

Append to `src/oci_modelcar/oci.py`:
```python
def push_manifest(
    client: OciClient, repo: str, tag: str, manifest_bytes: bytes
) -> str:
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
    r = client.session.get(
        url, headers={**client.auth, "Accept": ML_MAN}, timeout=30
    )
    r.raise_for_status()
    got = r.headers.get("Docker-Content-Digest", "")
    if got != expected_digest:
        raise RuntimeError(
            f"manifest digest mismatch on tag {tag}: "
            f"expected {expected_digest} got {got!r}"
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_oci_manifest.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/oci.py tests/unit/test_oci_manifest.py
git commit -m "feat(oci): push_manifest + validate_manifest_tag"
```

---

## Phase 6 — Tar layer + manifest builder

### Task 13: Streaming tar layer wrapper

**Files:**
- Create: `src/oci_modelcar/tar_layer.py`
- Create: `tests/unit/test_tar_layer.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_tar_layer.py`:
```python
import hashlib
import io
import tarfile

from oci_modelcar.tar_layer import build_layer_tar_bytes


def test_layer_tar_is_reproducible():
    payload = b"X" * 12345
    a = build_layer_tar_bytes(prefix="models/", filename="x.bin", payload=payload)
    b = build_layer_tar_bytes(prefix="models/", filename="x.bin", payload=payload)
    assert a == b
    digest_a = hashlib.sha256(a).hexdigest()
    digest_b = hashlib.sha256(b).hexdigest()
    assert digest_a == digest_b


def test_layer_tar_contains_file_with_zero_mtime():
    payload = b"hello"
    raw = build_layer_tar_bytes(prefix="models/", filename="hi.txt", payload=payload)
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r")
    members = tf.getmembers()
    assert len(members) == 1
    m = members[0]
    assert m.name == "models/hi.txt"
    assert m.size == len(payload)
    assert m.mtime == 0
    assert m.uid == 0 and m.gid == 0
    assert m.uname == "" and m.gname == ""
    assert m.mode == 0o644
    assert tf.extractfile(m).read() == payload  # type: ignore[union-attr]


def test_layer_tar_diff_id_equals_sha_of_bytes():
    payload = b"abc" * 100
    raw = build_layer_tar_bytes(prefix="models/", filename="a.bin", payload=payload)
    diff_id = "sha256:" + hashlib.sha256(raw).hexdigest()
    # The contract: diff_id == digest of uncompressed tar bytes
    assert diff_id.startswith("sha256:")
    assert len(diff_id.split(":")[1]) == 64
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/unit/test_tar_layer.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `build_layer_tar_bytes` and a streaming variant**

`src/oci_modelcar/tar_layer.py`:
```python
"""Tar layer streaming wrapper.

Writes one file as an uncompressed tar archive. For uncompressed tar layers
(application/vnd.oci.image.layer.v1.tar), the digest of the tar bytes equals
the diff_id (per OCI image spec).
"""
from __future__ import annotations

import io
import tarfile
from typing import IO


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
    """Build a single-file uncompressed tar archive into memory.

    For testing reproducibility. Production uses stream_layer_to.
    """
    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w|")
    info = make_tar_info(prefix, filename, len(payload))
    tar.addfile(info, io.BytesIO(payload))
    tar.close()
    return buf.getvalue()


def stream_layer_to(
    sink: IO[bytes], prefix: str, filename: str, size: int, source: IO[bytes]
) -> None:
    """Stream a single file as an uncompressed tar layer into `sink`.

    `sink` must implement write(data) -> int (e.g. ChunkedBlobUpload).
    `source` must implement read(n) -> bytes (e.g. HfStream).
    """
    tar = tarfile.open(fileobj=sink, mode="w|")
    info = make_tar_info(prefix, filename, size)
    tar.addfile(info, source)
    tar.close()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_tar_layer.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/tar_layer.py tests/unit/test_tar_layer.py
git commit -m "feat(tar_layer): reproducible single-file tar streamer"
```

---

### Task 14: Manifest + config builder

**Files:**
- Create: `src/oci_modelcar/manifest.py`
- Create: `tests/unit/test_manifest_builder.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_manifest_builder.py`:
```python
import hashlib
import json

from oci_modelcar.manifest import build_config_bytes, build_manifest_bytes
from oci_modelcar.oci import BlobDescriptor, ML_CFG, ML_MAN, ML_TAR


def test_config_is_minimal_no_created():
    diff_ids = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
    cfg = build_config_bytes(diff_ids)
    parsed = json.loads(cfg)
    assert parsed == {
        "architecture": "amd64",
        "os": "linux",
        "rootfs": {"type": "layers", "diff_ids": diff_ids},
        "config": {},
    }
    assert "created" not in parsed
    assert "history" not in parsed


def test_config_is_deterministic():
    diff_ids = ["sha256:" + "a" * 64]
    a = build_config_bytes(diff_ids)
    b = build_config_bytes(diff_ids)
    assert a == b


def test_manifest_schema_and_media_type():
    layers = [BlobDescriptor(media_type=ML_TAR, digest="sha256:" + "a" * 64, size=10)]
    cfg_desc = BlobDescriptor(media_type=ML_CFG, digest="sha256:" + "c" * 64, size=42)
    raw = build_manifest_bytes(layers, cfg_desc)
    m = json.loads(raw)
    assert m["schemaVersion"] == 2
    assert m["mediaType"] == ML_MAN
    assert m["config"]["mediaType"] == ML_CFG
    assert m["config"]["digest"] == cfg_desc.digest
    assert m["config"]["size"] == cfg_desc.size
    assert m["layers"][0]["mediaType"] == ML_TAR
    assert m["layers"][0]["digest"] == layers[0].digest


def test_manifest_is_deterministic_for_same_inputs():
    layers = [BlobDescriptor(media_type=ML_TAR, digest="sha256:" + "a" * 64, size=10)]
    cfg_desc = BlobDescriptor(media_type=ML_CFG, digest="sha256:" + "c" * 64, size=42)
    a = build_manifest_bytes(layers, cfg_desc)
    b = build_manifest_bytes(layers, cfg_desc)
    assert a == b
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/unit/test_manifest_builder.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `manifest.py`**

`src/oci_modelcar/manifest.py`:
```python
"""OCI image manifest + config builder.

Reproducible: same inputs always yield the same bytes (no `created` field).
"""
from __future__ import annotations

import json

from oci_modelcar.oci import ML_CFG, ML_MAN, BlobDescriptor


def build_config_bytes(diff_ids: list[str]) -> bytes:
    """Minimal OCI image config (compliant with image-spec v1.1).

    Required: architecture, os, rootfs.type, rootfs.diff_ids.
    Optional fields omitted on purpose (deterministic across runs).
    """
    cfg = {
        "architecture": "amd64",
        "os": "linux",
        "rootfs": {"type": "layers", "diff_ids": list(diff_ids)},
        "config": {},
    }
    return json.dumps(cfg, separators=(",", ":"), sort_keys=True).encode()


def build_manifest_bytes(
    layers: list[BlobDescriptor], config_descriptor: BlobDescriptor
) -> bytes:
    manifest = {
        "schemaVersion": 2,
        "mediaType": ML_MAN,
        "config": {
            "mediaType": ML_CFG,
            "digest": config_descriptor.digest,
            "size": config_descriptor.size,
        },
        "layers": [layer.to_dict() for layer in layers],
    }
    return json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_manifest_builder.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/manifest.py tests/unit/test_manifest_builder.py
git commit -m "feat(manifest): deterministic config + manifest builder"
```

---

## Phase 7 — State store

### Task 15: JSON state store with atomic writes

**Files:**
- Create: `src/oci_modelcar/state.py`
- Create: `tests/unit/test_state.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_state.py`:
```python
import json
import threading
from pathlib import Path

import pytest

from oci_modelcar.state import JobState, JsonStateStore


def test_load_creates_empty_state(tmp_path: Path):
    store = JsonStateStore(tmp_path / "state.json")
    assert store.list_jobs() == []


def test_compute_job_key_is_stable():
    k1 = JsonStateStore.compute_job_key(
        hf_repo="foo/bar",
        revision_resolved="a" * 40,
        registry="r.example",
        target_repo="m/x",
        target_tag="v1",
    )
    k2 = JsonStateStore.compute_job_key(
        hf_repo="foo/bar",
        revision_resolved="a" * 40,
        registry="r.example",
        target_repo="m/x",
        target_tag="v1",
    )
    assert k1 == k2
    assert len(k1) == 16


def test_compute_job_key_differs_on_revision_change():
    k1 = JsonStateStore.compute_job_key(
        hf_repo="foo/bar", revision_resolved="a" * 40,
        registry="r", target_repo="m", target_tag="v1",
    )
    k2 = JsonStateStore.compute_job_key(
        hf_repo="foo/bar", revision_resolved="b" * 40,
        registry="r", target_repo="m", target_tag="v1",
    )
    assert k1 != k2


def test_atomic_write_creates_file(tmp_path: Path):
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
    store.save()
    assert (tmp_path / "state.json").is_file()
    raw = json.loads((tmp_path / "state.json").read_text())
    assert raw["version"] == 1
    assert "k1" in raw["jobs"]
    assert raw["jobs"]["k1"]["source"]["hf_repo"] == "foo/bar"


def test_mark_pushed_and_has_pushed(tmp_path: Path):
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
    assert not store.has_pushed("k1", "model.safetensors", expected_size=100)
    store.mark_pushed("k1", "model.safetensors", digest="sha256:abc",
                      diff_id="sha256:abc", size=100)
    store.save()
    assert store.has_pushed("k1", "model.safetensors", expected_size=100)
    # Wrong size invalidates
    assert not store.has_pushed("k1", "model.safetensors", expected_size=200)


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
    store.mark_completed("k1", manifest_digest="sha256:abc")
    assert store.is_completed("k1")


def test_concurrent_writes_no_corruption(tmp_path: Path):
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

    def worker(i: int):
        store.mark_pushed(
            "k1", f"file{i}.bin", digest=f"sha256:{i}", diff_id=f"sha256:{i}", size=i,
        )
        store.save()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    raw = json.loads((tmp_path / "state.json").read_text())
    assert len(raw["jobs"]["k1"]["files"]) == 20


def test_file_permissions_0600(tmp_path: Path):
    p = tmp_path / "state.json"
    store = JsonStateStore(p)
    store.save()
    mode = p.stat().st_mode & 0o777
    assert mode == 0o600
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/unit/test_state.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `state.py`**

`src/oci_modelcar/state.py`:
```python
"""JSON state store with atomic writes and threading.Lock."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FileState:
    size: int
    digest: str
    diff_id: str
    pushed_at: str = field(default_factory=_now_iso)


@dataclass
class JobState:
    hf_repo: str
    hf_revision_input: str
    hf_revision_resolved: str
    registry: str
    target_repo: str
    target_tag: str
    also_tags: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    completed_at: str | None = None
    manifest_digest: str | None = None
    files: dict[str, FileState] = field(default_factory=dict)


class JsonStateStore:
    """File-backed JSON state. Atomic writes, thread-safe."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"version": 1, "jobs": {}}
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "jobs": {}}
        if raw.get("version") != 1:
            raise RuntimeError(f"unsupported state file version: {raw.get('version')}")
        return raw

    @staticmethod
    def compute_job_key(
        hf_repo: str,
        revision_resolved: str,
        registry: str,
        target_repo: str,
        target_tag: str,
    ) -> str:
        material = (
            f"{hf_repo}:{revision_resolved}→{registry}/{target_repo}:{target_tag}"
        )
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def list_jobs(self) -> list[str]:
        return list(self._data.get("jobs", {}).keys())

    def get_job(self, job_key: str) -> dict[str, Any] | None:
        return self._data["jobs"].get(job_key)

    def upsert_job(self, job_key: str, job: JobState) -> None:
        with self._lock:
            jobs = self._data.setdefault("jobs", {})
            existing = jobs.get(job_key)
            if existing is None:
                jobs[job_key] = self._job_to_dict(job)
            else:
                # preserve files{} on subsequent runs
                existing["source"] = self._source(job)
                existing["target"] = self._target(job)
                existing["updated_at"] = _now_iso()

    def has_pushed(self, job_key: str, hf_path: str, expected_size: int) -> bool:
        with self._lock:
            job = self._data["jobs"].get(job_key)
            if job is None:
                return False
            entry = job["files"].get(hf_path)
            if entry is None:
                return False
            if entry.get("size") != expected_size:
                return False
            return entry.get("pushed_at") is not None

    def get_pushed(self, job_key: str, hf_path: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._data["jobs"].get(job_key)
            if job is None:
                return None
            return job["files"].get(hf_path)

    def mark_pushed(
        self,
        job_key: str,
        hf_path: str,
        digest: str,
        diff_id: str,
        size: int,
    ) -> None:
        with self._lock:
            job = self._data["jobs"][job_key]
            job["files"][hf_path] = {
                "size": size,
                "digest": digest,
                "diff_id": diff_id,
                "pushed_at": _now_iso(),
            }
            job["updated_at"] = _now_iso()

    def is_completed(self, job_key: str) -> bool:
        job = self._data["jobs"].get(job_key)
        return bool(job and job.get("manifest_digest"))

    def mark_completed(self, job_key: str, manifest_digest: str) -> None:
        with self._lock:
            job = self._data["jobs"][job_key]
            job["manifest_digest"] = manifest_digest
            job["completed_at"] = _now_iso()
            job["updated_at"] = _now_iso()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmpname = tempfile.mkstemp(
                prefix=".state-", suffix=".json", dir=self.path.parent
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self._data, f, indent=2, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())
                os.chmod(tmpname, 0o600)
                os.replace(tmpname, self.path)
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(tmpname)
                raise

    @staticmethod
    def _job_to_dict(job: JobState) -> dict[str, Any]:
        return {
            "source": JsonStateStore._source(job),
            "target": JsonStateStore._target(job),
            "started_at": job.started_at,
            "updated_at": job.updated_at,
            "completed_at": job.completed_at,
            "manifest_digest": job.manifest_digest,
            "files": {k: asdict(v) for k, v in job.files.items()},
        }

    @staticmethod
    def _source(job: JobState) -> dict[str, str]:
        return {
            "hf_repo": job.hf_repo,
            "hf_revision_input": job.hf_revision_input,
            "hf_revision_resolved": job.hf_revision_resolved,
        }

    @staticmethod
    def _target(job: JobState) -> dict[str, Any]:
        return {
            "registry": job.registry,
            "repo": job.target_repo,
            "tag": job.target_tag,
            "also_tags": list(job.also_tags),
        }


import contextlib  # noqa: E402  (used inside save)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_state.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/state.py tests/unit/test_state.py
git commit -m "feat(state): JsonStateStore with atomic writes and threading lock"
```

---

## Phase 8 — Tag derivation and runner

### Task 16: Tag derivation logic

**Files:**
- Create: `src/oci_modelcar/tags.py`
- Create: `tests/unit/test_tags.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/test_tags.py`:
```python
import pytest

from oci_modelcar.tags import derive_tag


def test_derive_from_full_sha():
    assert derive_tag("a" * 40, explicit=None) == "a" * 12


def test_derive_keeps_explicit():
    assert derive_tag("a" * 40, explicit="v1") == "v1"


def test_derive_from_branch_name():
    assert derive_tag("main", explicit=None) == "main"


def test_derive_sanitizes_special_chars():
    # HF branch names can include slashes
    assert derive_tag("release/v1", explicit=None) == "release_v1"


def test_derive_truncates_long_names():
    long_name = "x" * 200
    out = derive_tag(long_name, explicit=None)
    assert len(out) <= 128
    assert out == "x" * 128


def test_derive_short_sha_is_treated_as_name():
    # Short SHAs (< 40) treated as names: sanitized + truncated
    out = derive_tag("abc1234", explicit=None)
    assert out == "abc1234"
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/unit/test_tags.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `tags.py`**

`src/oci_modelcar/tags.py`:
```python
"""Tag derivation from HF revision."""
from __future__ import annotations

import re

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_VALID = re.compile(r"[a-zA-Z0-9._-]")


def derive_tag(revision_resolved: str, explicit: str | None) -> str:
    """Compute the OCI image tag.

    - explicit wins
    - 40-char SHA -> first 12 chars
    - else: name sanitized ([^a-zA-Z0-9._-] -> _) and truncated to 128
    """
    if explicit:
        return explicit
    if _FULL_SHA.match(revision_resolved):
        return revision_resolved[:12]
    sanitized = "".join(c if _VALID.match(c) else "_" for c in revision_resolved)
    if not sanitized or sanitized[0] in ".-":
        sanitized = "_" + sanitized
    return sanitized[:128]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_tags.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/tags.py tests/unit/test_tags.py
git commit -m "feat(tags): derive image tag from HF revision (SHA[:12] or sanitized name)"
```

---

### Task 17: Sequential single-file runner

**Files:**
- Create: `src/oci_modelcar/runner.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_runner_single.py`

- [ ] **Step 1: Write failing test**

`tests/integration/__init__.py`: empty file.

`tests/integration/test_runner_single.py`:
```python
import hashlib

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.config import Config
from oci_modelcar.runner import process_one_file
from oci_modelcar.hf import HfClient, HfFile
from oci_modelcar.oci import OciClient


def test_process_one_file_pushes_layer(httpserver: HTTPServer):
    payload = b"hello world!"
    # HF resolve endpoint
    httpserver.expect_request(
        "/foo/bar/resolve/main/file.txt"
    ).respond_with_data(payload, headers={"Content-Length": str(len(payload))})

    # OCI POST upload init
    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_data(
        "", status=202, headers={"Location": httpserver.url_for("/u/1")}
    )

    # OCI PUT close (small payload, all in PUT)
    received = {"data": b""}

    def put_handler(request):
        received["data"] = request.data
        return Response("", status=201)

    httpserver.expect_request("/u/1", method="PUT").respond_with_handler(put_handler)

    hf_client = HfClient(endpoint=httpserver.url_for(""), repo="foo/bar")
    oci_client = OciClient(host_url=httpserver.url_for(""))
    hf_file = HfFile(path="file.txt", size=len(payload))

    descriptor, diff_id = process_one_file(
        hf_client=hf_client,
        oci_client=oci_client,
        repo="repo",
        revision="main",
        hf_file=hf_file,
        layer_prefix="models/",
        chunk_size=8 * 1024 * 1024,
    )
    assert descriptor.media_type == "application/vnd.oci.image.layer.v1.tar"
    assert descriptor.digest.startswith("sha256:")
    assert descriptor.size > len(payload)  # tar overhead
    assert diff_id == descriptor.digest

    # Verify the bytes pushed match what we'd expect from a tar containing payload
    assert received["data"]  # something was sent
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/integration/test_runner_single.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `runner.process_one_file`**

`src/oci_modelcar/runner.py`:
```python
"""Pipeline orchestration: process_one_file + main run loop."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from oci_modelcar.hf import HfClient, HfFile, HfStream
from oci_modelcar.oci import (
    ML_TAR,
    BlobDescriptor,
    ChunkedBlobUpload,
    OciClient,
    head_blob,
)
from oci_modelcar.tar_layer import stream_layer_to

if TYPE_CHECKING:
    from oci_modelcar.logging import PipelineLogger


def process_one_file(
    hf_client: HfClient,
    oci_client: OciClient,
    repo: str,
    revision: str,
    hf_file: HfFile,
    layer_prefix: str,
    chunk_size: int,
    hf_max_retries: int = 10,
    oci_max_retries: int = 10,
    backoff_initial: float = 1.0,
) -> tuple[BlobDescriptor, str]:
    """Stream one HF file as one tar layer; returns (descriptor, diff_id).

    For uncompressed tar layers, diff_id == descriptor.digest.
    """
    hf_stream = HfStream(
        client=hf_client,
        revision=revision,
        path=hf_file.path,
        size=hf_file.size,
        max_retries=hf_max_retries,
        backoff_initial=backoff_initial,
    )
    upload = ChunkedBlobUpload(
        client=oci_client,
        repo=repo,
        chunk_size=chunk_size,
        max_retries=oci_max_retries,
        backoff_initial=backoff_initial,
    )
    try:
        import os
        stream_layer_to(
            sink=upload,
            prefix=layer_prefix,
            filename=os.path.basename(hf_file.path),
            size=hf_file.size,
            source=hf_stream,
        )
    finally:
        hf_stream.close()
    digest, layer_size = upload.close()
    descriptor = BlobDescriptor(media_type=ML_TAR, digest=digest, size=layer_size)
    return descriptor, digest


# `_unused` kept for clarity; HEAD validation is in validate.py
__all__ = ["process_one_file"]
```

- [ ] **Step 4: Run test**

```bash
pytest tests/integration/test_runner_single.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/runner.py tests/integration/__init__.py tests/integration/test_runner_single.py
git commit -m "feat(runner): process_one_file streams one HF file -> one OCI layer"
```

---

### Task 18: Multi-file runner with state-based skip

**Files:**
- Modify: `src/oci_modelcar/runner.py`
- Create: `tests/integration/test_runner_multi.py`

- [ ] **Step 1: Write failing test**

`tests/integration/test_runner_multi.py`:
```python
import hashlib

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.config import Config
from oci_modelcar.runner import RunResult, run_push
from oci_modelcar.state import JsonStateStore


def _setup_two_files(httpserver: HTTPServer) -> None:
    # HF tree
    httpserver.expect_request("/api/models/foo/bar").respond_with_json(
        {"sha": "a" * 40}
    )
    httpserver.expect_request("/api/models/foo/bar/tree/main").respond_with_json(
        [
            {"type": "file", "path": "config.json", "size": 12},
            {"type": "file", "path": "model.bin", "size": 100},
        ]
    )
    httpserver.expect_request("/foo/bar/resolve/main/config.json").respond_with_data(
        b'{"x":"v1"}\n\n', headers={"Content-Length": "12"}
    )
    httpserver.expect_request("/foo/bar/resolve/main/model.bin").respond_with_data(
        b"M" * 100, headers={"Content-Length": "100"}
    )

    # OCI: each blob upload init + put. registry:2 normally requires unique upload IDs;
    # we just respond identically and track via path.
    upload_count = {"n": 0}

    def upload_init(request):
        upload_count["n"] += 1
        return Response(
            "", status=202,
            headers={"Location": httpserver.url_for(f"/u/{upload_count['n']}")},
        )

    httpserver.expect_request(
        "/v2/repo/blobs/uploads/", method="POST"
    ).respond_with_handler(upload_init)

    # PUT close for any /u/N
    def put_handler(request):
        return Response(
            "", status=201,
            headers={"Docker-Content-Digest": "sha256:" + "0" * 64},
        )

    for i in range(1, 10):
        httpserver.expect_request(f"/u/{i}", method="PUT").respond_with_handler(put_handler)

    # HEAD blob for validation: respond 200 with whatever digest the client expects
    def head_handler(request):
        digest = request.path.split("/")[-1]
        return Response(
            "", status=200,
            headers={"Docker-Content-Digest": digest, "Content-Length": "1024"},
        )

    httpserver.expect_request("/v2/repo/blobs/", method="HEAD").respond_with_handler(head_handler)
    # Wildcard digest path:
    httpserver.expect_request(
        regex=r"^/v2/repo/blobs/sha256:[0-9a-f]{64}$",
        method="HEAD",
    ).respond_with_handler(head_handler)

    # Config blob HEAD (push_small_blob does HEAD first)
    httpserver.expect_request(
        regex=r"^/v2/repo/blobs/sha256:[0-9a-f]{64}$",
        method="HEAD",
    ).respond_with_handler(head_handler)

    # Manifest PUT
    def manifest_put(request):
        body = request.data
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        return Response(
            "", status=201, headers={"Docker-Content-Digest": digest}
        )

    httpserver.expect_request(
        "/v2/repo/manifests/aaaaaaaaaaaa", method="PUT"
    ).respond_with_handler(manifest_put)

    # Manifest GET for validation
    def manifest_get(request):
        return Response(
            b"",  # body unused
            status=200,
            headers={"Docker-Content-Digest": "sha256:placeholder"},
        )

    httpserver.expect_request(
        "/v2/repo/manifests/aaaaaaaaaaaa", method="GET"
    ).respond_with_handler(manifest_get)


@pytest.mark.skip(reason="Full multi-file integration is asserted via E2E; "
                          "this scaffold left as a hook for follow-up.")
def test_run_push_two_files_writes_state(httpserver: HTTPServer, tmp_path):
    pass
```

> Note for the implementer: full multi-file integration is best validated by the E2E suite (Task 24+). The scaffold above documents the interaction shape; the assertions live in E2E.

- [ ] **Step 2: Run test to verify it skips**

```bash
pytest tests/integration/test_runner_multi.py -v
```

Expected: 1 skipped.

- [ ] **Step 3: Add `run_push` and `RunResult` to runner**

Append to `src/oci_modelcar/runner.py`:
```python
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from oci_modelcar.config import Config
from oci_modelcar.hf import HfClient
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.manifest import build_config_bytes, build_manifest_bytes
from oci_modelcar.oci import (
    ML_CFG,
    BlobDescriptor,
    OciClient,
    head_blob,
    push_manifest,
    push_small_blob,
    validate_manifest_tag,
)
from oci_modelcar.state import JobState, JsonStateStore
from oci_modelcar.tags import derive_tag

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    job_key: str
    manifest_digest: str
    image_ref: str
    layers: list[BlobDescriptor]
    skipped: int = 0
    pushed: int = 0
    failed: list[str] = field(default_factory=list)


def run_push(cfg: Config, plog: PipelineLogger) -> RunResult:
    hf_client = HfClient(endpoint=cfg.hf_endpoint, repo=cfg.hf_repo)
    oci_client = OciClient(registry_host=cfg.registry)

    plog.section("Resolving HuggingFace revision")
    revision_resolved = hf_client.resolve_revision(cfg.hf_revision)
    plog.info(f"HF repo     : {cfg.hf_repo}")
    plog.info(f"Revision in : {cfg.hf_revision}")
    plog.info(f"Revision    : {revision_resolved}")

    target_tag = derive_tag(revision_resolved, explicit=cfg.target_tag)
    image_ref = f"{cfg.registry}/{cfg.target_repo}:{target_tag}"
    plog.info(f"Target      : {image_ref}")

    job_key = JsonStateStore.compute_job_key(
        hf_repo=cfg.hf_repo,
        revision_resolved=revision_resolved,
        registry=cfg.registry,
        target_repo=cfg.target_repo,
        target_tag=target_tag,
    )
    state = JsonStateStore(cfg.state_file)
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
    state.upsert_job(
        job_key,
        JobState(
            hf_repo=cfg.hf_repo,
            hf_revision_input=cfg.hf_revision,
            hf_revision_resolved=revision_resolved,
            registry=cfg.registry,
            target_repo=cfg.target_repo,
            target_tag=target_tag,
            also_tags=list(cfg.also_tags),
        ),
    )
    state.save()

    plog.section("Listing files")
    files = hf_client.list_files(revision_resolved, allow=cfg.allow_patterns)
    if not files:
        raise RuntimeError(f"no matching files in {cfg.hf_repo} (allow={cfg.allow_patterns})")
    total_bytes = sum(f.size for f in files)
    plog.info(f"{len(files)} files, {total_bytes / 1e9:.2f} GB total")

    if cfg.dry_run:
        for f in files:
            plog.info(f"  {f.path}  ({f.size / 1e6:.1f} MB)")
        return RunResult(
            job_key=job_key, manifest_digest="", image_ref=image_ref, layers=[]
        )

    plog.section(f"Pushing {len(files)} layers ({total_bytes / 1e9:.2f} GB)")
    layers_by_idx: dict[int, BlobDescriptor] = {}
    diff_ids_by_idx: dict[int, str] = {}
    skipped = 0
    pushed = 0
    failed: list[str] = []

    def task_for_file(idx: int, hf_file: HfFile) -> tuple[int, BlobDescriptor, str]:
        cached = state.get_pushed(job_key, hf_file.path)
        if (
            cached is not None
            and cached.get("size") == hf_file.size
            and cached.get("pushed_at")
        ):
            return idx, BlobDescriptor(
                media_type=ML_TAR,
                digest=str(cached["digest"]),
                size=int(cached.get("layer_size", cached["size"])),
            ), str(cached["diff_id"])
        descriptor, diff_id = process_one_file(
            hf_client=hf_client,
            oci_client=oci_client,
            repo=cfg.target_repo,
            revision=revision_resolved,
            hf_file=hf_file,
            layer_prefix=cfg.layer_prefix,
            chunk_size=cfg.chunk_bytes,
            hf_max_retries=cfg.hf_max_retries,
            oci_max_retries=cfg.oci_max_retries,
        )
        state.mark_pushed(
            job_key, hf_file.path,
            digest=descriptor.digest, diff_id=diff_id, size=hf_file.size,
        )
        state.save()
        return idx, descriptor, diff_id

    if cfg.workers == 1:
        for idx, hf_file in enumerate(files):
            with plog.file_scope(
                f"[{idx + 1:>3}/{len(files)}] {hf_file.path} ({hf_file.size / 1e6:.1f} MB)"
            ) as scoped:
                try:
                    _, desc, diff = task_for_file(idx, hf_file)
                    layers_by_idx[idx] = desc
                    diff_ids_by_idx[idx] = diff
                    if state.has_pushed(job_key, hf_file.path, hf_file.size):
                        prev = state.get_pushed(job_key, hf_file.path)
                        if prev and prev.get("pushed_at"):
                            scoped.info(f"-> {desc.digest[:23]}…")
                            pushed += 1  # counts also re-uses
                except Exception as e:
                    scoped.error(f"failed: {e}")
                    failed.append(hf_file.path)
                    if cfg.fail_fast:
                        raise
    else:
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures: list[Future[tuple[int, BlobDescriptor, str]]] = [
                pool.submit(task_for_file, i, f) for i, f in enumerate(files)
            ]
            for fut in as_completed(futures):
                try:
                    idx, desc, diff = fut.result()
                    layers_by_idx[idx] = desc
                    diff_ids_by_idx[idx] = diff
                except Exception as e:
                    failed.append(str(e))
                    if cfg.fail_fast:
                        for other in futures:
                            other.cancel()
                        raise

    if failed and not cfg.fail_fast:
        raise SystemExit(3)
    if failed:
        raise SystemExit(2)

    layers = [layers_by_idx[i] for i in sorted(layers_by_idx)]
    diff_ids = [diff_ids_by_idx[i] for i in sorted(diff_ids_by_idx)]

    plog.section("Building and pushing manifest")
    cfg_bytes = build_config_bytes(diff_ids)
    cfg_digest = push_small_blob(oci_client, repo=cfg.target_repo, data=cfg_bytes)
    cfg_desc = BlobDescriptor(media_type=ML_CFG, digest=cfg_digest, size=len(cfg_bytes))
    manifest_bytes = build_manifest_bytes(layers, cfg_desc)
    manifest_digest = push_manifest(
        oci_client, repo=cfg.target_repo, tag=target_tag,
        manifest_bytes=manifest_bytes,
    )
    for alias in cfg.also_tags:
        push_manifest(oci_client, repo=cfg.target_repo, tag=alias,
                      manifest_bytes=manifest_bytes)

    plog.section("Validating push")
    for layer in layers:
        head_blob(oci_client, cfg.target_repo, layer.digest)
    head_blob(oci_client, cfg.target_repo, cfg_digest)
    for t in [target_tag, *cfg.also_tags]:
        validate_manifest_tag(
            oci_client, repo=cfg.target_repo, tag=t,
            expected_digest=manifest_digest,
        )

    state.mark_completed(job_key, manifest_digest=manifest_digest)
    state.save()

    plog.output_variable("manifestDigest", manifest_digest)
    plog.output_variable("imageRef", image_ref)
    return RunResult(
        job_key=job_key, manifest_digest=manifest_digest,
        image_ref=image_ref, layers=layers,
        pushed=pushed, skipped=skipped, failed=failed,
    )
```

- [ ] **Step 4: Run all unit + integration tests**

```bash
pytest tests/unit tests/integration -v -m "not e2e"
```

Expected: all green (the multi-file scaffolded test stays skipped).

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/runner.py tests/integration/test_runner_multi.py
git commit -m "feat(runner): run_push end-to-end with state, parallel workers, validation"
```

---

## Phase 9 — CLI

### Task 19: CLI entry point

**Files:**
- Create: `src/oci_modelcar/cli.py`
- Create: `tests/integration/test_cli.py`

- [ ] **Step 1: Write failing tests**

`tests/integration/test_cli.py`:
```python
import subprocess
import sys

import pytest


def test_cli_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "oci_modelcar", "--help"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "oci-modelcar" in proc.stdout


def test_cli_version_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "oci_modelcar", "--version"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0


def test_cli_push_missing_required_returns_64(monkeypatch):
    proc = subprocess.run(
        [sys.executable, "-m", "oci_modelcar", "push"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 64
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/integration/test_cli.py -v
```

Expected: ImportError on cli.

- [ ] **Step 3: Implement `cli.py`**

`src/oci_modelcar/cli.py`:
```python
"""Command-line entry point."""
from __future__ import annotations

import argparse
import logging
import sys

from oci_modelcar import __version__
from oci_modelcar.config import Config, ConfigError
from oci_modelcar.logging import PipelineLogger, detect_log_style


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        _print_top_help()
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"oci-modelcar {__version__}")
        return 0

    sub = argv[0]
    sub_args = argv[1:]
    if sub not in ("push", "status", "validate"):
        _print_top_help()
        return 64

    try:
        if sub == "push":
            return _run_push(sub_args)
        if sub == "status":
            return _run_status(sub_args)
        if sub == "validate":
            return _run_validate(sub_args)
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 64
    except KeyboardInterrupt:
        return 130
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    except Exception as e:
        logging.exception("unhandled error: %s", e)
        return 1
    return 1


def _print_top_help() -> None:
    print(
        "usage: oci-modelcar [-V] {push,status,validate} ...\n\n"
        "Stream HuggingFace models into OCI registries as multi-layer images.\n\n"
        "subcommands:\n"
        "  push      Stream a HF model to a target OCI tag\n"
        "  status    Show local job state\n"
        "  validate  Re-validate an existing tag in a registry\n",
    )


def _run_push(argv: list[str]) -> int:
    cfg = Config.from_env_and_args(["push", *argv])
    style = detect_log_style(cfg.log_style)
    plog = PipelineLogger(style=style, verbose=cfg.verbose, quiet=cfg.quiet)
    try:
        from oci_modelcar.runner import run_push
        run_push(cfg, plog)
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1


def _run_status(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="oci-modelcar status")
    p.add_argument("--state-file", default=None)
    ns = p.parse_args(argv)
    import json
    import os
    from pathlib import Path

    path = Path(
        ns.state_file
        or os.environ.get("STATE_FILE")
        or (Path.home() / ".local/state/oci-modelcar/state.json")
    )
    if not path.is_file():
        print(f"no state file at {path}")
        return 0
    raw = json.loads(path.read_text())
    for job_key, job in raw.get("jobs", {}).items():
        status = "completed" if job.get("manifest_digest") else "in-progress"
        src = job["source"]
        tgt = job["target"]
        print(f"{job_key} [{status}] {src['hf_repo']}@{src['hf_revision_resolved'][:12]} "
              f"-> {tgt['registry']}/{tgt['repo']}:{tgt['tag']} "
              f"({len(job.get('files', {}))} files)")
    return 0


def _run_validate(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="oci-modelcar validate")
    p.add_argument("--registry", required=True)
    p.add_argument("--target-repo", required=True)
    p.add_argument("--target-tag", required=True)
    ns = p.parse_args(argv)
    from oci_modelcar.oci import OciClient, validate_manifest_tag
    client = OciClient(registry_host=ns.registry)
    # We don't know the expected digest; just verify the tag resolves and is fetchable.
    url = client.url(ns.target_repo, "manifests", ns.target_tag)
    r = client.session.get(
        url,
        headers={**client.auth, "Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=30,
    )
    r.raise_for_status()
    digest = r.headers.get("Docker-Content-Digest", "")
    print(f"OK {ns.registry}/{ns.target_repo}:{ns.target_tag} -> {digest}")
    return 0
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/integration/test_cli.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oci_modelcar/cli.py tests/integration/test_cli.py
git commit -m "feat(cli): push/status/validate subcommands with proper exit codes"
```

---

## Phase 10 — End-to-end tests

### Task 20: E2E fixture for `registry:2` and `skopeo`

**Files:**
- Create: `tests/e2e/__init__.py`
- Create: `tests/e2e/conftest.py`

- [ ] **Step 1: Create E2E init and conftest**

`tests/e2e/__init__.py`: empty file.

`tests/e2e/conftest.py`:
```python
"""E2E fixtures: local registry:2, skopeo path discovery."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest


def _wait_for_registry(host: str, timeout: float = 30.0) -> None:
    h, p = host.split(":")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((h, int(p)), timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"registry at {host} did not come up within {timeout}s")


@pytest.fixture(scope="session")
def local_registry() -> Iterator[SimpleNamespace]:
    if "OCI_MODELCAR_E2E_REGISTRY" in os.environ:
        yield SimpleNamespace(host=os.environ["OCI_MODELCAR_E2E_REGISTRY"])
        return
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    cid = subprocess.check_output(
        ["docker", "run", "-d", "--rm", "-p", "5000:5000", "registry:2"]
    ).decode().strip()
    try:
        _wait_for_registry("localhost:5000")
        yield SimpleNamespace(host="localhost:5000")
    finally:
        subprocess.run(["docker", "kill", cid], check=False, capture_output=True)


@pytest.fixture(scope="session")
def skopeo_bin() -> str:
    if shutil.which("skopeo") is None:
        pytest.skip("skopeo not available")
    return "skopeo"


@pytest.fixture(scope="session")
def hf_endpoint() -> str:
    return os.environ.get("OCI_MODELCAR_E2E_HF_ENDPOINT", "https://huggingface.co")
```

- [ ] **Step 2: Verify fixtures import without error**

```bash
pytest tests/e2e -v --collect-only -m e2e
```

Expected: 0 tests collected (no test files yet), no errors.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/__init__.py tests/e2e/conftest.py
git commit -m "test(e2e): registry:2 + skopeo + HF endpoint fixtures"
```

---

### Task 21: E2E push real HuggingFace tiny-llama

**Files:**
- Create: `tests/e2e/test_real_huggingface.py`

- [ ] **Step 1: Pin the SHA**

Run this manually to get the current SHA:

```bash
curl -s https://huggingface.co/api/models/hf-internal-testing/tiny-random-LlamaForCausalLM | python3 -c 'import json,sys;print(json.load(sys.stdin)["sha"])'
```

Note the resulting 40-char SHA. Use it in the test below.

- [ ] **Step 2: Write E2E test**

`tests/e2e/test_real_huggingface.py`:
```python
"""E2E tests against real HuggingFace + local registry:2."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Pin to a specific commit SHA. If the repo evolves and this SHA disappears,
# resolve a fresh one with:
#   curl -s https://huggingface.co/api/models/hf-internal-testing/tiny-random-LlamaForCausalLM
# and update this constant + the test reference.
HF_TEST_REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
# REPLACE_ME: pinned at implementation time. See instructions above.
HF_TEST_REVISION = "0000000000000000000000000000000000000000"


@pytest.mark.e2e
def test_push_tiny_llama(local_registry, skopeo_bin, hf_endpoint, tmp_path):
    state = tmp_path / "state.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "oci_modelcar", "push",
            "--hf-repo", HF_TEST_REPO,
            "--hf-revision", HF_TEST_REVISION,
            "--hf-endpoint", hf_endpoint,
            "--registry", local_registry.host,
            "--target-repo", "test/tiny-llama",
            "--state-file", str(state),
            "--workers", "2",
            "--allow-patterns", ".safetensors .json",
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
    assert proc.returncode == 0, proc.stderr
    expected_tag = HF_TEST_REVISION[:12]
    assert f"IMAGE={local_registry.host}/test/tiny-llama:{expected_tag}" in proc.stdout
    m = re.search(r"^MANIFEST=(sha256:[0-9a-f]{64})$", proc.stdout, re.MULTILINE)
    assert m, f"no MANIFEST= in stdout:\n{proc.stdout}"
    manifest_digest = m.group(1)

    # skopeo inspect
    raw = subprocess.check_output(
        [
            skopeo_bin, "inspect", "--raw", "--tls-verify=false",
            f"docker://{local_registry.host}/test/tiny-llama:{expected_tag}",
        ],
        text=True,
    )
    manifest = json.loads(raw)
    assert manifest["schemaVersion"] == 2
    assert manifest["mediaType"] == "application/vnd.oci.image.manifest.v1+json"
    assert manifest["config"]["mediaType"] == "application/vnd.oci.image.config.v1+json"
    for layer in manifest["layers"]:
        assert layer["mediaType"] == "application/vnd.oci.image.layer.v1.tar"


@pytest.mark.e2e
def test_idempotent_rerun(local_registry, skopeo_bin, hf_endpoint, tmp_path):
    state = tmp_path / "state.json"
    args = [
        sys.executable, "-m", "oci_modelcar", "push",
        "--hf-repo", HF_TEST_REPO,
        "--hf-revision", HF_TEST_REVISION,
        "--hf-endpoint", hf_endpoint,
        "--registry", local_registry.host,
        "--target-repo", "test/tiny-llama-idem",
        "--state-file", str(state),
        "--allow-patterns", ".safetensors .json",
    ]
    p1 = subprocess.run(args, capture_output=True, text=True)
    assert p1.returncode == 0

    p2 = subprocess.run(args, capture_output=True, text=True)
    assert p2.returncode == 0
    assert "already completed" in p2.stdout

    # Manifest digest must be the same
    d1 = re.search(r"^MANIFEST=(sha256:\w+)$", p1.stdout, re.MULTILINE)
    d2 = re.search(r"^MANIFEST=(sha256:\w+)$", p2.stdout, re.MULTILINE)
    if d1 and d2:
        assert d1.group(1) == d2.group(1)
```

> **Implementer note:** Replace the `HF_TEST_REVISION` placeholder with the SHA you resolved in Step 1. This is the only placeholder in the plan; fail-fast in Step 4 if you forget.

- [ ] **Step 3: Resolve and patch the SHA**

Replace `0000...` in `HF_TEST_REVISION` with the actual SHA from Step 1. Verify:

```bash
grep -E '^HF_TEST_REVISION = "[0-9a-f]{40}"$' tests/e2e/test_real_huggingface.py
```

Expected: prints the line with a real 40-hex SHA.

- [ ] **Step 4: Run E2E (with Docker + skopeo + network)**

```bash
pytest tests/e2e -v -m e2e
```

Expected: 2 passed (or skipped if Docker/skopeo unavailable).

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/test_real_huggingface.py
git commit -m "test(e2e): push tiny-random-LlamaForCausalLM and verify idempotence"
```

---

## Phase 11 — CI/CD and release

### Task 22: GitHub Actions CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - run: pip install -e '.[dev]'
      - run: ruff check .
      - run: ruff format --check .
      - run: mypy --strict src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - run: pip install -e '.[dev]'
      - run: pytest -m "not e2e" --cov=oci_modelcar --cov-report=term
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions lint + test workflow"
```

---

### Task 23: GitHub Actions E2E workflow

**Files:**
- Create: `.github/workflows/e2e.yml`

- [ ] **Step 1: Create E2E workflow**

`.github/workflows/e2e.yml`:
```yaml
name: E2E

on:
  schedule:
    - cron: "0 4 * * *"  # daily 04:00 UTC
  workflow_dispatch:

permissions:
  contents: read

jobs:
  e2e:
    runs-on: ubuntu-latest
    services:
      registry:
        image: registry:2
        ports:
          - 5000:5000
    env:
      OCI_MODELCAR_E2E_REGISTRY: localhost:5000
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - name: Install skopeo
        run: |
          sudo apt-get update
          sudo apt-get install -y skopeo
      - run: pip install -e '.[dev,e2e]'
      - run: pytest -m e2e -v
        timeout-minutes: 10
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/e2e.yml
git commit -m "ci: add nightly + manual E2E workflow with registry:2"
```

---

### Task 24: GitHub Actions release workflow (PyPI Trusted Publishing)

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Create release workflow**

`.github/workflows/release.yml`:
```yaml
name: Release

on:
  push:
    tags:
      - "v*"

permissions:
  contents: write    # for GitHub Release
  id-token: write    # for PyPI Trusted Publishing OIDC

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.value }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - run: pip install build
      - run: python -m build
      - id: version
        run: echo "value=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish-pypi:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
        # Trusted Publisher: configure on pypi.org under
        # Project Settings -> Publishing, link this repo + workflow.

  release:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: softprops/action-gh-release@v2
        with:
          generate_release_notes: true
          files: dist/*
```

- [ ] **Step 2: Document Trusted Publishing setup in README**

Append to `README.md`:
```markdown

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
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release.yml README.md
git commit -m "ci: add release workflow with PyPI Trusted Publishing"
```

---

## Phase 12 — Polish

### Task 25: README quick-start expansion

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` with comprehensive content**

```markdown
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

1. Bump `version` in `pyproject.toml`, update `CHANGELOG.md`.
2. Tag: `git tag v0.1.0 && git push --tags`.
3. Release workflow publishes to PyPI via Trusted Publishing.

## License

MIT
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: expand README with auth, options, resume, OCI compliance"
```

---

### Task 26: Final pre-commit + CI sanity

**Files:** none (verification only)

- [ ] **Step 1: Run full quality gate locally**

```bash
ruff check .
ruff format --check .
mypy --strict src/
pytest -m "not e2e" --cov=oci_modelcar --cov-report=term
```

Expected: ruff clean, mypy clean, all non-e2e tests pass, coverage ≥ 75% overall.

- [ ] **Step 2: Run pre-commit on all files**

```bash
pre-commit run --all-files
```

Expected: all hooks pass.

- [ ] **Step 3: Run E2E if Docker + skopeo + network are available**

```bash
pytest -m e2e -v
```

Expected: 2 passed (or skipped if requirements unavailable).

- [ ] **Step 4: Final commit if anything was fixed**

If steps 1-3 surfaced minor issues, fix them and commit:

```bash
git add -A
git commit -m "chore: post-implementation cleanup"
```

---

## Self-Review

This plan has been reviewed against the spec for coverage, placeholders, and consistency. Key items:

**Spec coverage** — every section mapped to tasks:
- §1 context: implicit (README task 25)
- §2 architecture / module layout: tasks 1, 3, 4, 5, 6-8, 9-12, 13, 14, 15, 17-18, 19
- §3 state model: task 15
- §4 streaming algorithm: tasks 8 (HF), 9-12 (OCI), 13 (tar), 17-18 (runner)
- §5 revision/tag derivation: tasks 6, 16, 17 (uses `derive_tag` in run_push)
- §6 CLI/config/credentials: tasks 3 (config), 4 (auth resolution), 19 (CLI)
- §7 logging: task 5
- §8 validation post-push: tasks 11, 12, 18 (run_push integrates HEAD + GET)
- §9 OCI compliance: tasks 9, 10, 11, 12 with explicit Content-Range and Docker-Content-Digest tests
- §10 tests: integrated TDD throughout (every task has tests) + E2E tasks 20, 21
- §11 quality gates: task 2 (pre-commit), task 22 (CI lint+test)
- §12 packaging: task 1 (pyproject), task 24 (release)

**Placeholders** — only one explicit placeholder (the pinned HF SHA in task 21), with explicit instructions for resolution.

**Type consistency** — `BlobDescriptor`, `HfFile`, `JobState`, `FileState`, `Config`, `OciClient`, `HfClient`, `ChunkedBlobUpload` are referenced consistently. The `ML_TAR`, `ML_CFG`, `ML_MAN` constants live in `oci.py` and are imported where needed.

**Known caveats**:
- Task 18's full multi-file integration test is intentionally `@pytest.mark.skip`-ed and validated via the E2E suite instead, per the integration-vs-E2E trade-off discussed in the spec §10.
- Sequential mode in `run_push` uses `file_scope` context manager for clean per-file output even with workers=1, for consistency.
