"""CLI entrypoint: argparse dispatch on argv[1] sub-command."""

from __future__ import annotations

import argparse
import logging
import sys

from huggingface_hub import HfApi

from oci_modelcar.config import Config
from oci_modelcar.download import HfDownloader
from oci_modelcar.errors import OciModelcarError, exit_code_for
from oci_modelcar.http import build_session
from oci_modelcar.logging import PipelineLogger
from oci_modelcar.pipeline import Pipeline
from oci_modelcar.registry import OciClient, head_blob
from oci_modelcar.reuse import RegistryReuseStore

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        print("usage: oci-modelcar {push,status,validate} [options]", file=sys.stderr)
        return 1

    sub = argv[1]
    rest = argv[2:]

    if sub in ("-h", "--help"):
        print("usage: oci-modelcar {push,status,validate} [options]")
        print("Run 'oci-modelcar push --help' for sub-command flags.")
        return 0

    if sub == "push":
        return _run_push(rest)
    if sub == "status":
        return _run_status(rest)
    if sub == "validate":
        return _run_validate(rest)

    print(f"unknown sub-command: {sub}", file=sys.stderr)
    return 1


def _run_push(argv: list[str]) -> int:
    try:
        cfg = Config.from_env_and_args(argv)
    except OciModelcarError as e:
        print(f"error: {e}", file=sys.stderr)
        if e.hint:
            print(f"hint: {e.hint}", file=sys.stderr)
        return exit_code_for(e)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2

    plog = PipelineLogger(log_style=cfg.log_style or "text", verbose=cfg.verbose, quiet=cfg.quiet)
    session = build_session()
    api = HfApi(endpoint=cfg.hf_endpoint)
    downloader = HfDownloader(
        api=api,
        session=session,
        spool_dir=cfg.spool_dir,
        stop_event=None,
        max_retries=cfg.hf_max_retries,
    )
    registry_client = OciClient(
        registry_host=cfg.registry,
        target_repo=cfg.target_repo,
        session=session,
    )
    reuse_store = (
        None
        if cfg.no_reuse_records
        else RegistryReuseStore(client=registry_client, repo=cfg.target_repo)
    )
    pipeline = Pipeline(
        cfg=cfg,
        plog=plog,
        downloader=downloader,
        registry_client=registry_client,
        reuse_store=reuse_store,
    )
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
    p = argparse.ArgumentParser(prog="oci-modelcar status")
    p.add_argument("--registry", required=True)
    p.add_argument("--target-repo", required=True)
    p.add_argument("--log-style", default=None, choices=["text", "azure"])
    p.add_argument("--quiet", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true", default=False)
    ns = p.parse_args(argv)

    plog = PipelineLogger(log_style=ns.log_style or "text", verbose=ns.verbose, quiet=ns.quiet)
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
            url,
            headers={**client.auth, "Accept": "application/vnd.oci.image.manifest.v1+json"},
            timeout=30,
        )
        digest = h.headers.get("Docker-Content-Digest", "?")
        plog.info(f"  {tag}  {digest}")
    return 0


def _run_validate(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="oci-modelcar validate")
    p.add_argument("--registry", required=True)
    p.add_argument("--target-repo", required=True)
    p.add_argument("--target-tag", required=True)
    p.add_argument("--log-style", default=None, choices=["text", "azure"])
    p.add_argument("--quiet", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true", default=False)
    ns = p.parse_args(argv)

    plog = PipelineLogger(log_style=ns.log_style or "text", verbose=ns.verbose, quiet=ns.quiet)
    client = OciClient(registry_host=ns.registry, target_repo=ns.target_repo)

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

    if head_blob(client, ns.target_repo, config_digest) is None:
        plog.error(f"config blob missing: {config_digest}")
        return 1
    for layer in layers:
        if head_blob(client, ns.target_repo, layer["digest"]) is None:
            plog.error(f"layer missing: {layer['digest']}")
            return 1
    plog.info(f"manifest at {ns.target_tag} is coherent ({len(layers)} layers)")
    return 0
