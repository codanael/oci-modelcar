"""Integration: HF→CDN redirect must NOT forward Authorization to the CDN.

This test exercises the full HfDownloader.download() code path through a pair
of real HTTP servers, verifying that the flagship v1.0 security guarantee holds
end-to-end (not merely at the _SafeSession.rebuild_auth unit level).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from huggingface_hub import HfApi
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from oci_modelcar.download import HfDownloader, HfFile
from oci_modelcar.http import build_session


def test_authorization_dropped_on_redirect_through_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flagship security guarantee of v1.0: HF Bearer tokens MUST NOT be
    forwarded to redirect targets (S3 / CloudFront / xet CDN).

    Two distinct HTTPServer instances bind to different IPs on loopback so
    their netloc strings differ — which is the discriminant in
    _SafeSession.rebuild_auth. The HF server returns a 302 to the CDN URL;
    we assert the token arrived at HF and was stripped before the CDN leg.
    """
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    # Prevent huggingface_hub from using its own cached token
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    # We inject the token ourselves via the HF_TOKEN env, which huggingface_auth_header reads
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")

    payload = b"X" * 4096

    cdn_seen: list[str | None] = []
    hf_seen: list[str | None] = []

    # CDN server bound to 127.0.0.1 (netloc: "127.0.0.1:<port>")
    cdn = HTTPServer(host="127.0.0.1", port=0)
    cdn.start()

    # HF server bound to localhost (netloc: "localhost:<port>")
    # Note: on most systems localhost also resolves to 127.0.0.1 at TCP level,
    # but the netloc strings differ → rebuild_auth strips Authorization.
    hf = HTTPServer(host="localhost", port=0)
    hf.start()

    try:

        def cdn_handler(request):  # type: ignore[no-untyped-def]
            cdn_seen.append(request.headers.get("Authorization"))
            return Response(
                payload,
                status=200,
                headers={"Content-Length": str(len(payload))},
            )

        cdn.expect_request("/repo/resolve/main/file.bin").respond_with_handler(cdn_handler)

        # CDN URL uses 127.0.0.1 netloc — different from localhost:<hf_port>
        cdn_url = cdn.url_for("/repo/resolve/main/file.bin")

        def hf_handler(request):  # type: ignore[no-untyped-def]
            hf_seen.append(request.headers.get("Authorization"))
            return Response("", status=302, headers={"Location": cdn_url})

        hf.expect_request("/repo/resolve/main/file.bin").respond_with_handler(hf_handler)

        session = build_session()
        api = HfApi(endpoint=hf.url_for("").rstrip("/"))
        downloader = HfDownloader(
            api=api,
            session=session,
            spool_dir=tmp_path / "spool",
            stop_event=None,
            max_retries=2,
            backoff_initial=0.0,
        )
        hf_file = HfFile(path="file.bin", size=len(payload), lfs_sha256=None)
        out = downloader.download("repo", "main", hf_file)
        assert out.read_bytes() == payload

        assert hf_seen == ["Bearer hf_secret_token"], (
            f"HF must receive the Bearer token; got {hf_seen!r}"
        )
        assert cdn_seen == [None], (
            f"CDN must NOT receive the Bearer token (cross-origin redirect); got {cdn_seen!r}"
        )
    finally:
        cdn.clear()
        if cdn.is_running():
            cdn.stop()
        hf.clear()
        if hf.is_running():
            hf.stop()
