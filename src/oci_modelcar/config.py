"""Configuration: env vars + CLI args + validation."""

from __future__ import annotations

import argparse
import os
import re
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


def _envstr(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw else default


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
    chunk_mib: int = 32
    workers: int = 1
    state_file: Path = field(
        default_factory=lambda: _xdg_state_home() / "oci-modelcar" / "state.json"
    )
    hf_max_retries: int = 10
    oci_max_retries: int = 10
    fail_fast: bool = True
    force: bool = False
    upload_mode: str = "streaming"
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
            hf_revision=ns.hf_revision or _envstr("HF_REVISION", "main"),
            hf_endpoint=(ns.hf_endpoint or _envstr("HF_ENDPOINT", "https://huggingface.co")),
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
            chunk_mib=int(ns.chunk_mib if ns.chunk_mib is not None else _envstr("CHUNK_MIB", "32")),
            workers=int(ns.workers if ns.workers is not None else _envstr("WORKERS", "1")),
            state_file=Path(
                ns.state_file
                or _envstr(
                    "STATE_FILE",
                    str(_xdg_state_home() / "oci-modelcar" / "state.json"),
                )
            ),
            hf_max_retries=int(
                ns.hf_max_retries
                if ns.hf_max_retries is not None
                else _envstr("HF_MAX_RETRIES", "10")
            ),
            oci_max_retries=int(
                ns.oci_max_retries
                if ns.oci_max_retries is not None
                else _envstr("OCI_MAX_RETRIES", "10")
            ),
            fail_fast=(
                False if ns.continue_on_error else (ns.fail_fast or _envbool("FAIL_FAST", True))
            ),
            force=ns.force or _envbool("FORCE", False),
            upload_mode=(
                ns.upload_mode
                if ns.upload_mode is not None
                else _envstr("OCI_MODELCAR_UPLOAD_MODE", "streaming")
            ),
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
        # Cap is generous (64 GiB) because RAM is the natural limiter, not
        # an arbitrary policy. Large values are sometimes required: registries
        # behind a load balancer without sticky session affinity (Artifactory
        # cluster, Harbor + reverse proxy) reroute each PATCH to a potentially
        # different node, breaking the upload session. Setting chunk_mib >=
        # the largest layer size collapses the whole upload into 1-2 PATCHes,
        # eliminating per-PATCH routing decisions — same approach as
        # containers/image (Podman, Skopeo) and Jib, both of which stream
        # the entire blob in one PATCH.
        if not (1 <= self.chunk_mib <= 65536):
            raise ConfigError(f"chunk_mib must be in [1, 65536], got {self.chunk_mib}")
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
        if self.upload_mode not in ("streaming", "chunked"):
            raise ConfigError(
                f"upload_mode must be 'streaming' or 'chunked', got {self.upload_mode!r}"
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
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--hf-revision", default=None)
    p.add_argument("--hf-endpoint", default=None)
    p.add_argument("--registry", default=None)
    p.add_argument("--target-repo", default=None)
    p.add_argument("--target-tag", default=None)
    p.add_argument("--also-tag", default=None, help="CSV list of additional tags")
    p.add_argument("--allow-patterns", default=None)
    p.add_argument("--layer-prefix", default=None)
    p.add_argument(
        "--chunk-mib",
        default=None,
        type=int,
        help=(
            "OCI upload PATCH chunk size in MiB (default 32, max 65536). "
            "On registries behind a load balancer without sticky session "
            "affinity (Artifactory cluster, Harbor + reverse proxy), set "
            "this to >= the largest layer size to upload each blob in a "
            "single PATCH and avoid per-PATCH routing decisions. "
            "Per-worker peak RAM is ~2x this value."
        ),
    )
    p.add_argument("--workers", default=None, type=int)
    p.add_argument("--state-file", default=None)
    p.add_argument("--hf-max-retries", default=None, type=int)
    p.add_argument("--oci-max-retries", default=None, type=int)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fail-fast", action="store_true", default=False)
    g.add_argument("--continue-on-error", action="store_true", default=False)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument(
        "--upload-mode",
        default=None,
        choices=["streaming", "chunked"],
        help=(
            "OCI blob upload protocol shape (default: streaming). 'streaming' "
            "issues a single PATCH per blob with body=iterator and Content-"
            "Length set upfront — matches containers/image (Podman, Skopeo) "
            "and Jib, and works with registries behind a load balancer "
            "without sticky session affinity. 'chunked' issues multiple "
            "PATCHes (--chunk-mib bytes each) with intra-blob retry on "
            "transient cuts; better for very flaky links but breaks on "
            "Artifactory clusters that reroute PATCHes between nodes."
        ),
    )
    p.add_argument("--log-style", default=None, choices=["text", "azure"])
    g2 = p.add_mutually_exclusive_group()
    g2.add_argument("--verbose", action="store_true", default=False)
    g2.add_argument("--quiet", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    return p
