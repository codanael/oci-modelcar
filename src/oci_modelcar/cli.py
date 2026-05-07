"""CLI entry point: push / status / validate sub-commands."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from oci_modelcar import __version__
from oci_modelcar.config import ConfigError

# Exit codes
_EX_OK = 0
_EX_GENERIC = 1
_EX_FAIL_FAST = 2
_EX_CONTINUE_ON_ERROR = 3
_EX_CONFIG = 64
_EX_AUTH = 65
_EX_SIGINT = 130

_USAGE = f"""\
oci-modelcar {__version__}

Usage: oci-modelcar <sub-command> [options]

Sub-commands:
  push      Stream a HuggingFace model into an OCI registry
  status    Show job summaries from the state file
  validate  Verify that a manifest tag is reachable in the registry

Run `oci-modelcar push --help` for per-subcommand options.
"""


def _print_top_help() -> None:
    sys.stdout.write(_USAGE)


def _run_push(argv: list[str]) -> int:
    from oci_modelcar import runner
    from oci_modelcar.config import Config
    from oci_modelcar.logging import PipelineLogger, detect_log_style

    cfg = Config.from_env_and_args(argv)
    style = detect_log_style(cfg.log_style)
    plog = PipelineLogger(
        style=style,
        verbose=cfg.verbose,
        quiet=cfg.quiet,
    )
    result = runner.run_push(cfg, plog)
    if result.failed:
        return _EX_CONTINUE_ON_ERROR if not cfg.fail_fast else _EX_FAIL_FAST
    return _EX_OK


def _run_status(argv: list[str]) -> int:
    import argparse
    import os
    from pathlib import Path

    from oci_modelcar.state import JsonStateStore

    def _xdg_state_home() -> Path:
        raw = os.environ.get("XDG_STATE_HOME")
        if raw:
            return Path(raw)
        return Path.home() / ".local" / "state"

    p = argparse.ArgumentParser(prog="oci-modelcar status")
    p.add_argument(
        "--state-file",
        default=None,
        help="Path to the state file (default: $XDG_STATE_HOME/oci-modelcar/state.json)",
    )
    ns = p.parse_args(argv)
    state_path = Path(
        ns.state_file
        or os.environ.get("STATE_FILE")
        or str(_xdg_state_home() / "oci-modelcar" / "state.json")
    )
    if not state_path.is_file():
        sys.stderr.write(f"State file not found: {state_path}\n")
        return _EX_GENERIC

    store = JsonStateStore(state_path)
    keys = store.list_jobs()
    if not keys:
        sys.stdout.write("No jobs found.\n")
        return _EX_OK

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


def _run_validate(argv: list[str]) -> int:
    import argparse

    from oci_modelcar.http import build_session, oci_auth_header
    from oci_modelcar.oci import ML_MAN

    p = argparse.ArgumentParser(prog="oci-modelcar validate")
    p.add_argument("--registry", required=True, help="Registry host (e.g. ghcr.io)")
    p.add_argument("--target-repo", required=True, help="Repository name")
    p.add_argument("--target-tag", required=True, help="Tag to validate")
    ns = p.parse_args(argv)

    registry: str = ns.registry
    repo: str = ns.target_repo
    tag: str = ns.target_tag

    session = build_session()
    auth = oci_auth_header(registry)
    url = f"https://{registry}/v2/{repo}/manifests/{tag}"
    r = session.get(url, headers={**auth, "Accept": ML_MAN}, timeout=30)
    if r.status_code == 200:
        digest = r.headers.get("Docker-Content-Digest", "(no digest)")
        sys.stdout.write(f"OK  {registry}/{repo}:{tag}  {digest}\n")
        return _EX_OK
    elif r.status_code in (401, 403):
        sys.stderr.write(f"Auth error {r.status_code}: {registry}/{repo}:{tag}\n")
        return _EX_AUTH
    else:
        sys.stderr.write(f"Manifest not found or error {r.status_code}: {registry}/{repo}:{tag}\n")
        return _EX_GENERIC


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: dispatch to push / status / validate or print help."""
    if argv is None:
        argv = sys.argv[1:]
    args = list(argv)

    if not args or args[0] in ("-h", "--help"):
        _print_top_help()
        return _EX_OK

    if args[0] in ("-V", "--version"):
        sys.stdout.write(f"oci-modelcar {__version__}\n")
        return _EX_OK

    sub = args[0]
    rest = args[1:]

    try:
        if sub == "push":
            return _run_push(rest)
        elif sub == "status":
            return _run_status(rest)
        elif sub == "validate":
            return _run_validate(rest)
        else:
            sys.stderr.write(f"Unknown sub-command: {sub!r}\n\n")
            _print_top_help()
            return _EX_CONFIG
    except ConfigError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        return _EX_CONFIG
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return _EX_SIGINT
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return _EX_GENERIC
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return _EX_GENERIC
