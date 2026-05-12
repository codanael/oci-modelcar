import hashlib
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests as _requests

from oci_modelcar.registry import (
    OciClient,
    StreamingBlobUpload,
    head_blob,
    push_manifest,
    push_small_blob,
    validate_manifest_tag,
)


def _make_response(
    status_code: int,
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.headers = headers or {}
    r.content = body
    r.raise_for_status.return_value = None
    return r


def _make_client(fake_session: MagicMock) -> OciClient:
    return OciClient(host_url="http://test", session=fake_session)


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


def test_head_blob_returns_descriptor_when_present():
    digest = "sha256:" + hashlib.sha256(b"x").hexdigest()
    fake_session = MagicMock()
    fake_session.head.return_value = _make_response(
        200, {"Docker-Content-Digest": digest, "Content-Length": "1"}
    )
    info = head_blob(_make_client(fake_session), "repo", digest)
    assert info == {"digest": digest, "size": 1}


def test_head_blob_returns_none_when_404():
    digest = "sha256:" + "a" * 64
    fake_session = MagicMock()
    fake_session.head.return_value = _make_response(404)
    assert head_blob(_make_client(fake_session), "repo", digest) is None


def test_head_blob_raises_on_digest_mismatch():
    digest = "sha256:" + "a" * 64
    other = "sha256:" + "b" * 64
    fake_session = MagicMock()
    fake_session.head.return_value = _make_response(
        200, {"Docker-Content-Digest": other, "Content-Length": "1"}
    )
    with pytest.raises(RuntimeError, match="digest mismatch"):
        head_blob(_make_client(fake_session), "repo", digest)


def test_push_small_blob_skips_when_already_present():
    data = b"config bytes"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    fake_session = MagicMock()
    fake_session.head.return_value = _make_response(
        200, {"Docker-Content-Digest": digest, "Content-Length": str(len(data))}
    )
    out = push_small_blob(_make_client(fake_session), "repo", data)
    assert out == digest
    fake_session.post.assert_not_called()
    fake_session.put.assert_not_called()


def test_push_small_blob_post_then_put():
    data = b"config bytes"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    fake_session = MagicMock()
    fake_session.head.return_value = _make_response(404)
    fake_session.post.return_value = _make_response(202, {"Location": "http://test/u/1"})

    sent_data: list[bytes] = []

    def put_handler(*args: object, **kwargs: object) -> MagicMock:
        sent_data.append(kwargs.get("data", b""))  # type: ignore[arg-type]
        return _make_response(201)

    fake_session.put.side_effect = put_handler

    out = push_small_blob(_make_client(fake_session), "repo", data)
    assert out == digest
    assert sent_data[0] == data


def test_push_small_blob_resolves_relative_location_header():
    """zot returns ``Location: /v2/repo/blobs/uploads/<uuid>?...`` (a path,
    no host). OCI Distribution spec allows this; the client must resolve
    it against the registry base URL before issuing the PUT, otherwise
    ``requests`` rejects the URL with InvalidSchema."""
    data = b"config bytes"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    fake_session = MagicMock()
    fake_session.head.return_value = _make_response(404)
    fake_session.post.return_value = _make_response(
        202, {"Location": "/v2/repo/blobs/uploads/abc-uuid?_state=xyz"}
    )

    put_urls: list[str] = []

    def put_handler(url: str, **kwargs: object) -> MagicMock:
        put_urls.append(url)
        return _make_response(201)

    fake_session.put.side_effect = put_handler

    push_small_blob(_make_client(fake_session), "repo", data)
    # The PUT must have gone to the absolute URL (http://test + Location path).
    assert put_urls[0].startswith("http://test/v2/repo/blobs/uploads/abc-uuid"), (
        f"PUT URL not resolved against base: {put_urls[0]!r}"
    )
    assert f"digest={digest}" in put_urls[0]


def test_push_manifest_returns_digest():
    body = json.dumps({"schemaVersion": 2, "config": {}, "layers": []}).encode()
    expected_digest = "sha256:" + hashlib.sha256(body).hexdigest()

    sent_data: list[bytes] = []

    fake_session = MagicMock()

    def put_handler(*args: object, **kwargs: object) -> MagicMock:
        sent_data.append(kwargs.get("data", b""))  # type: ignore[arg-type]
        return _make_response(201)

    fake_session.put.side_effect = put_handler

    out = push_manifest(_make_client(fake_session), "repo", "v1", body)
    assert out == expected_digest
    assert sent_data[0] == body


def test_validate_manifest_tag_succeeds_on_match():
    digest = "sha256:" + "a" * 64
    fake_session = MagicMock()
    fake_session.get.return_value = _make_response(200, {"Docker-Content-Digest": digest})
    validate_manifest_tag(_make_client(fake_session), "repo", "v1", digest)


def test_validate_manifest_tag_raises_on_mismatch():
    digest = "sha256:" + "a" * 64
    other = "sha256:" + "b" * 64
    fake_session = MagicMock()
    fake_session.get.return_value = _make_response(200, {"Docker-Content-Digest": other})
    with pytest.raises(RuntimeError, match="manifest digest mismatch"):
        validate_manifest_tag(_make_client(fake_session), "repo", "v1", digest)


def test_streaming_push_from_file_happy_path(tmp_path: Path):
    payload = b"X" * (4 * 1024 * 1024 + 17)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    fake_session = MagicMock()
    fake_session.post.return_value = _make_response(202, {"Location": "http://test/u/1"})

    received = bytearray()

    def patch_handler(*args: object, **kwargs: object) -> MagicMock:
        headers = kwargs.get("headers", {})
        cr = headers.get("Content-Range", "")  # type: ignore[union-attr]
        m = re.match(r"^(\d+)-(\d+)$", cr)
        assert m, f"bad Content-Range {cr!r}"
        start, end = int(m.group(1)), int(m.group(2))
        assert start == 0
        assert end == len(payload) - 1
        cl = int(headers.get("Content-Length", "0"))  # type: ignore[union-attr]
        assert cl == len(payload)
        data = kwargs.get("data")
        if hasattr(data, "read"):
            received.extend(data.read())
        return _make_response(202, {"Location": "http://test/u/1"})

    fake_session.patch.side_effect = patch_handler
    fake_session.put.return_value = _make_response(201)

    upload = StreamingBlobUpload(client=_make_client(fake_session), repo="repo")
    out_digest, out_size = upload.push_from_file(f, len(payload), digest)
    assert out_digest == digest
    assert out_size == len(payload)
    assert bytes(received) == payload


def test_streaming_push_from_file_invokes_progress_cb(tmp_path: Path):
    """A multi-chunk body must drive progress_cb monotonically up to total_size."""
    payload = b"P" * (4 * 1024 * 1024 + 3)  # > one read() call's worth
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    fake_session = MagicMock()
    fake_session.post.return_value = _make_response(202, {"Location": "http://test/u/p"})

    def patch_handler(*args: object, **kwargs: object) -> MagicMock:
        # Emulate requests' file-like consumption: chunked read() until EOF.
        data = kwargs["data"]
        while True:
            chunk = data.read(256 * 1024)  # type: ignore[union-attr]
            if not chunk:
                break
        return _make_response(202, {"Location": "http://test/u/p"})

    fake_session.patch.side_effect = patch_handler
    fake_session.put.return_value = _make_response(201)

    progress: list[int] = []
    upload = StreamingBlobUpload(client=_make_client(fake_session), repo="repo")
    upload.push_from_file(f, len(payload), digest, progress_cb=progress.append)

    assert progress, "progress_cb was never invoked during PATCH"
    assert progress == sorted(progress), "progress_cb values must be monotonically non-decreasing"
    assert progress[-1] == len(payload), (
        f"final progress {progress[-1]} should equal payload size {len(payload)}"
    )


@pytest.mark.parametrize("success_status", [200, 201, 202, 204])
def test_streaming_accepts_non_spec_success_codes(tmp_path: Path, success_status: int):
    """Artifactory returns 200/204; Harbor (some setups) returns 204.
    go-containerregistry accepts {201,202,204}; oras-py accepts {200,201,202}.
    Union: {200,201,202,204}."""
    payload = b"Z" * 1024
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    fake_session = MagicMock()
    fake_session.post.return_value = _make_response(202, {"Location": "http://test/u/codes"})
    fake_session.patch.return_value = _make_response(
        success_status, {"Location": "http://test/u/codes"}
    )
    fake_session.put.return_value = _make_response(201)

    upload = StreamingBlobUpload(client=_make_client(fake_session), repo="repo")
    out_digest, _ = upload.push_from_file(f, len(payload), digest)
    assert out_digest == digest


def test_streaming_unhandled_status_raises(tmp_path: Path):
    """299 (no spec meaning) must raise rather than spin or silently succeed."""
    payload = b"A" * 64
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    fake_session = MagicMock()
    fake_session.post.return_value = _make_response(202, {"Location": "http://test/u/odd"})
    fake_session.patch.return_value = _make_response(299)

    upload = StreamingBlobUpload(client=_make_client(fake_session), repo="repo")
    with pytest.raises(RuntimeError, match=r"unexpected.*299|status 299"):
        upload.push_from_file(f, len(payload), digest)


def test_streaming_no_chunked_transfer_encoding(tmp_path: Path):
    """Content-Length must be set explicitly to avoid chunked TE, which
    some registries handle differently from a fixed-size PATCH."""
    payload = b"L" * 512
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    fake_session = MagicMock()
    fake_session.post.return_value = _make_response(202, {"Location": "http://test/u/len"})

    seen_te: list[str | None] = []

    def patch_handler(*args: object, **kwargs: object) -> MagicMock:
        headers = kwargs.get("headers", {})
        seen_te.append(headers.get("Transfer-Encoding"))  # type: ignore[union-attr]
        return _make_response(202, {"Location": "http://test/u/len"})

    fake_session.patch.side_effect = patch_handler
    fake_session.put.return_value = _make_response(201)

    upload = StreamingBlobUpload(client=_make_client(fake_session), repo="repo")
    upload.push_from_file(f, len(payload), digest)

    te = seen_te[0] or ""
    assert "chunked" not in te.lower(), f"Transfer-Encoding leaked chunked: {te!r}"


def test_streaming_retries_on_ssl_eof_with_file_rewound(tmp_path: Path, monkeypatch):
    """First PATCH attempt raises mid-stream SSL EOF; second succeeds.
    File must be reopened/rewound; full body sent again from offset 0."""
    payload = b"R" * 1024
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    monkeypatch.setattr("oci_modelcar.registry.time.sleep", lambda d: None)

    fake_session = MagicMock()
    fake_session.post.return_value = _make_response(202, {"Location": "http://test/u/eof"})

    received = bytearray()
    calls: dict[str, int] = {"n": 0}

    def patch_handler(*args: object, **kwargs: object) -> MagicMock:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _requests.exceptions.SSLError(
                "EOF occurred in violation of protocol (_ssl.c:2437)"
            )
        data = kwargs.get("data")
        if hasattr(data, "read"):
            received.extend(data.read())
        return _make_response(202, {"Location": "http://test/u/eof"})

    fake_session.patch.side_effect = patch_handler
    fake_session.put.return_value = _make_response(201)

    upload = StreamingBlobUpload(
        client=_make_client(fake_session), repo="repo", max_retries=3, backoff_initial=0.0
    )
    out_digest, _out_size = upload.push_from_file(f, len(payload), digest)

    assert out_digest == digest
    assert calls["n"] == 2, "must retry exactly once after SSL EOF"
    assert bytes(received) == payload, "second attempt must re-send full body"


def test_streaming_does_not_retry_on_handshake_ssl(tmp_path: Path):
    """SSL handshake errors are fatal; no retry."""
    payload = b"H" * 64
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    fake_session = MagicMock()
    fake_session.post.return_value = _make_response(202, {"Location": "http://test/u/handshake"})

    calls: dict[str, int] = {"n": 0}

    def patch_handler(*args: object, **kwargs: object) -> MagicMock:
        calls["n"] += 1
        raise _requests.exceptions.SSLError("CERTIFICATE_VERIFY_FAILED")

    fake_session.patch.side_effect = patch_handler

    upload = StreamingBlobUpload(
        client=_make_client(fake_session), repo="repo", max_retries=5, backoff_initial=0.0
    )
    with pytest.raises(_requests.exceptions.SSLError):
        upload.push_from_file(f, len(payload), digest)

    assert calls["n"] == 1, "fatal SSL must not retry"


def test_streaming_max_retries_exhausted_raises_push_error(tmp_path: Path, monkeypatch):
    """All attempts fail with transient SSL EOF → PushError."""
    payload = b"X" * 32
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    monkeypatch.setattr("oci_modelcar.registry.time.sleep", lambda d: None)

    fake_session = MagicMock()
    fake_session.post.return_value = _make_response(202, {"Location": "http://test/u/exhaust"})

    calls: dict[str, int] = {"n": 0}

    def patch_handler(*args: object, **kwargs: object) -> MagicMock:
        calls["n"] += 1
        raise _requests.exceptions.SSLError("EOF occurred in violation of protocol (_ssl.c:2437)")

    fake_session.patch.side_effect = patch_handler

    upload = StreamingBlobUpload(
        client=_make_client(fake_session), repo="repo", max_retries=3, backoff_initial=0.0
    )
    from oci_modelcar.errors import PushError

    with pytest.raises(PushError, match="retries exhausted"):
        upload.push_from_file(f, len(payload), digest)

    assert calls["n"] == 3, "must call PATCH max_retries times before giving up"


def test_streaming_retries_on_5xx(tmp_path: Path, monkeypatch):
    """Server returns 503 then 202 → retry succeeds."""
    payload = b"S" * 16
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    monkeypatch.setattr("oci_modelcar.registry.time.sleep", lambda d: None)

    fake_session = MagicMock()
    fake_session.post.return_value = _make_response(202, {"Location": "http://test/u/5xx"})

    calls: dict[str, int] = {"n": 0}

    def patch_handler(*args: object, **kwargs: object) -> MagicMock:
        calls["n"] += 1
        if calls["n"] == 1:
            return _make_response(503)
        return _make_response(202, {"Location": "http://test/u/5xx"})

    fake_session.patch.side_effect = patch_handler
    fake_session.put.return_value = _make_response(201)

    upload = StreamingBlobUpload(
        client=_make_client(fake_session), repo="repo", max_retries=3, backoff_initial=0.0
    )
    out_digest, _ = upload.push_from_file(f, len(payload), digest)
    assert out_digest == digest
    assert calls["n"] == 2


def test_streaming_re_posts_init_on_each_retry(tmp_path: Path, monkeypatch):
    """v1.0 invariant: each retry attempt starts with a fresh POST init,
    so a TCP cut that invalidates the previous upload session is recovered
    by getting a new Location."""
    payload = b"P" * 256
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    f = tmp_path / "layer.tar"
    f.write_bytes(payload)

    monkeypatch.setattr("oci_modelcar.registry.time.sleep", lambda d: None)

    fake_session = MagicMock()
    post_calls: dict[str, int] = {"n": 0}

    def post_handler(*args: object, **kwargs: object) -> MagicMock:
        post_calls["n"] += 1
        loc = f"http://test/u/{post_calls['n']}"
        return _make_response(202, {"Location": loc})

    fake_session.post.side_effect = post_handler

    # First Location's PATCH "fails" (session invalidated by TCP cut).
    # Second Location's PATCH succeeds.
    patch_calls: dict[str, int] = {"n": 0}

    def patch_handler(*args: object, **kwargs: object) -> MagicMock:
        patch_calls["n"] += 1
        # Determine which location we're patching from positional URL arg
        url = args[0] if args else kwargs.get("url", "")
        if "/u/1" in str(url):
            # 404 → transient (BLOB_UPLOAD_INVALID); registry.py treats this as retryable
            return _make_response(404)
        return _make_response(202, {"Location": str(url)})

    fake_session.patch.side_effect = patch_handler
    fake_session.put.return_value = _make_response(201)

    upload = StreamingBlobUpload(
        client=_make_client(fake_session), repo="repo", max_retries=3, backoff_initial=0.0
    )
    out_digest, _ = upload.push_from_file(f, len(payload), digest)
    assert out_digest == digest
    assert post_calls["n"] >= 2, "must re-POST on each retry attempt"
    assert patch_calls["n"] == 2, "first PATCH 404, second succeeds"
