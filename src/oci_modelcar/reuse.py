"""OCI 1.1 referrer-based reuse store.

Persists `(hf_path, hf_sha256) → layer_digest` mappings as referrer
artifacts anchored on a deterministic stub manifest, so that a re-run
after a crash mid-push can skip HF download for layers whose blobs
are already in the registry — even when `--clean-hf-after-push` has
removed the local sources.

See `docs/superpowers/specs/2026-05-11-reuse-via-oci-referrers-design.md`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading

from oci_modelcar.manifest import (
    ANN_HF_PATH,
    ANN_HF_SHA256,
    ML_MAN,
    ML_TAR,
    BlobDescriptor,
)
from oci_modelcar.registry import OciClient, push_manifest, push_small_blob

ML_INDEX = "application/vnd.oci.image.index.v1+json"

log = logging.getLogger(__name__)

EMPTY_CONFIG_BYTES = b"{}"
EMPTY_CONFIG_SIZE = 2
EMPTY_CONFIG_MEDIA_TYPE = "application/vnd.oci.empty.v1+json"
# OCI-spec-defined canonical empty descriptor; digest of literal b"{}".
EMPTY_CONFIG_DIGEST = "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"

ARTIFACT_TYPE_ANCHOR = "application/vnd.codanael.modelcar.reuse-anchor.v1"
ARTIFACT_TYPE_RECORD = "application/vnd.codanael.modelcar.reuse-record.v1"

ANN_HF_REPO = "io.github.codanael.modelcar.hf-repo"
ANN_HF_REVISION = "io.github.codanael.modelcar.hf-revision"
ANN_ALLOW = "io.github.codanael.modelcar.allow-patterns"
ANN_IGNORE = "io.github.codanael.modelcar.ignore-patterns"
ANN_LAYER_PREFIX = "io.github.codanael.modelcar.layer-prefix"


def _empty_config_descriptor() -> dict[str, object]:
    return {
        "mediaType": EMPTY_CONFIG_MEDIA_TYPE,
        "digest": EMPTY_CONFIG_DIGEST,
        "size": EMPTY_CONFIG_SIZE,
    }


def build_anchor_manifest_bytes(
    hf_repo: str,
    hf_revision: str,
    allow_patterns: tuple[str, ...],
    ignore_patterns: tuple[str, ...],
    layer_prefix: str,
) -> bytes:
    """Build the deterministic stub manifest that anchors reuse records.

    Two runs with identical inputs produce byte-identical output, so the
    SHA-256 of the result is the natural per-configuration anchor key.
    """
    manifest = {
        "schemaVersion": 2,
        "mediaType": ML_MAN,
        "artifactType": ARTIFACT_TYPE_ANCHOR,
        "config": _empty_config_descriptor(),
        "layers": [],
        "annotations": {
            ANN_HF_REPO: hf_repo,
            ANN_HF_REVISION: hf_revision,
            ANN_ALLOW: " ".join(allow_patterns),
            ANN_IGNORE: " ".join(ignore_patterns),
            ANN_LAYER_PREFIX: layer_prefix,
        },
    }
    return json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()


def build_record_manifest_bytes(
    layer: BlobDescriptor,
    anchor_digest: str,
    anchor_size: int,
) -> bytes:
    """Build a per-layer reuse-record artifact.

    The record's `layers[0]` references the already-pushed blob — it
    does not re-upload. `subject` points at the run's anchor manifest
    so a `GET /referrers/<anchor_digest>` returns this record.
    """
    manifest = {
        "schemaVersion": 2,
        "mediaType": ML_MAN,
        "artifactType": ARTIFACT_TYPE_RECORD,
        "config": _empty_config_descriptor(),
        "subject": {
            "mediaType": ML_MAN,
            "digest": anchor_digest,
            "size": anchor_size,
        },
        "layers": [layer.to_dict()],
    }
    return json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()


def fallback_referrers_tag(digest: str) -> str:
    """Per OCI 1.1 spec: when the registry does not support the native
    referrers API, an image index hosted at this tag holds the same
    content. The tag is the digest with ``:`` replaced by ``-``.
    """
    if ":" not in digest:
        raise ValueError(f"digest must be prefixed with algo (e.g. sha256:...), got {digest!r}")
    algo, hex_part = digest.split(":", 1)
    return f"{algo}-{hex_part}"


class RegistryReuseStore:
    """Owns the anchor manifest and reuse-record artifacts for one run.

    Lifecycle:
      1. ``ensure_anchor(anchor_bytes, anchor_digest)`` once at preflight —
         idempotent, leaves the stub manifest in the registry.
      2. ``load_reuse_map(anchor_digest)`` once at preflight — returns
         the union of records currently visible (native API or
         fallback tag).
      3. ``record(layer, anchor_digest, anchor_size)`` after each
         worker's successful push+verify. Concurrent calls are safe.
    """

    def __init__(self, client: OciClient, repo: str) -> None:
        self.client = client
        self.repo = repo
        self._native_referrers: bool | None = None
        self._fallback_lock = threading.Lock()

    def ensure_anchor(self, anchor_bytes: bytes, anchor_digest: str) -> None:
        """HEAD the stub; PUT-by-digest if absent. Idempotent.

        Bootstraps the OCI-spec empty config blob first (``push_small_blob``
        HEAD-skips if already present in the repo).
        """
        push_small_blob(self.client, self.repo, EMPTY_CONFIG_BYTES)
        if self._manifest_exists(anchor_digest):
            log.debug("reuse: anchor %s already in %s", anchor_digest, self.repo)
            return
        push_manifest(self.client, self.repo, anchor_digest, anchor_bytes)
        log.debug("reuse: anchor %s pushed to %s", anchor_digest, self.repo)

    def _manifest_exists(self, reference: str) -> bool:
        url = self.client.url(self.repo, "manifests", reference)
        r = self.client.session.head(
            url,
            headers={**self.client.auth, "Accept": ML_MAN},
            timeout=30,
        )
        if r.status_code == 404:
            return False
        if r.status_code == 200:
            return True
        r.raise_for_status()
        return False

    def load_reuse_map(self, anchor_digest: str) -> dict[tuple[str, str | None], BlobDescriptor]:
        """Return ``(hf_path, hf_sha256) → BlobDescriptor`` from records.

        Reads BOTH the native referrers API and the OCI 1.1 fallback tag
        and unions the descriptors, dedup'd by record digest. The native
        path alone would suffice for the common case (single registry
        consistently supports or doesn't support native referrers), but
        the union covers the upgrade scenario: a registry that didn't
        echo ``OCI-Subject`` during an earlier run (records ended up in
        the fallback tag) and now does (later runs go native). Cost:
        at most 2 GETs per pipeline run, both tiny. Missing or
        malformed records are silently skipped.
        """
        seen_record_digests: set[str] = set()
        descriptors: list[dict[str, object]] = []
        for index in self._fetch_indices(anchor_digest):
            descriptors_raw = index.get("manifests")
            if not isinstance(descriptors_raw, list):
                continue
            for desc in descriptors_raw:
                if not isinstance(desc, dict):
                    continue
                digest = desc.get("digest")
                if isinstance(digest, str) and digest not in seen_record_digests:
                    seen_record_digests.add(digest)
                    descriptors.append(desc)

        out: dict[tuple[str, str | None], BlobDescriptor] = {}
        for desc in descriptors:
            digest = desc.get("digest")
            if not isinstance(digest, str):
                continue
            record = self._fetch_manifest_json(digest)
            if record is None:
                continue
            entry = self._descriptor_from_record(record)
            if entry is None:
                continue
            out[(entry.hf_path, entry.hf_sha256)] = entry
        return out

    def _fetch_indices(self, anchor_digest: str) -> list[dict[str, object]]:
        """Yield every referrer-index source we can find for this anchor.

        Order: native referrers API, then fallback tag. We always check
        both because the registry's PUT echo behavior (whether it sent
        ``OCI-Subject`` during the originating run) determines which
        index actually holds the records, and we can't assume it.
        """
        out: list[dict[str, object]] = []

        # Native referrers API
        url = self.client.url(self.repo, "referrers", anchor_digest)
        url = f"{url}?artifactType={ARTIFACT_TYPE_RECORD}"
        r = self.client.session.get(
            url,
            headers={**self.client.auth, "Accept": ML_INDEX},
            timeout=30,
        )
        if r.status_code == 200:
            body: dict[str, object] = r.json()
            out.append(body)
        elif r.status_code != 404:
            r.raise_for_status()

        # Fallback tag
        tag = fallback_referrers_tag(anchor_digest)
        url = self.client.url(self.repo, "manifests", tag)
        r = self.client.session.get(
            url,
            headers={**self.client.auth, "Accept": ML_INDEX},
            timeout=30,
        )
        if r.status_code == 200:
            out.append(r.json())
        elif r.status_code != 404:
            r.raise_for_status()

        return out

    def _fetch_manifest_json(self, reference: str) -> dict[str, object] | None:
        url = self.client.url(self.repo, "manifests", reference)
        r = self.client.session.get(
            url,
            headers={**self.client.auth, "Accept": ML_MAN},
            timeout=30,
        )
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            r.raise_for_status()
        body: dict[str, object] = r.json()
        return body

    def record(
        self,
        layer: BlobDescriptor,
        anchor_digest: str,
        anchor_size: int,
    ) -> None:
        """Push one reuse-record artifact and, if needed, mirror it into
        the fallback ``sha256-<hex>`` tag index.

        Idempotent: identical layer + anchor → identical record bytes →
        identical record digest → no-op on second push.
        """
        body = build_record_manifest_bytes(layer, anchor_digest, anchor_size)
        record_digest = "sha256:" + hashlib.sha256(body).hexdigest()
        url = self.client.url(self.repo, "manifests", record_digest)
        r = self.client.session.put(
            url,
            data=body,
            headers={**self.client.auth, "Content-Type": ML_MAN},
            timeout=60,
        )
        if r.status_code not in (200, 201):
            r.raise_for_status()
            raise RuntimeError(f"unexpected status {r.status_code} on record PUT")

        native = "OCI-Subject" in r.headers
        if self._native_referrers is None:
            self._native_referrers = native
            log.debug(
                "reuse: native referrers %s in %s",
                "supported" if native else "not detected — using fallback tag",
                self.repo,
            )

        if self._native_referrers:
            return

        # Fallback path: maintain the sha256-<hex> tag image index.
        self._fallback_append(
            anchor_digest=anchor_digest,
            record_digest=record_digest,
            record_size=len(body),
        )

    def _fallback_append(
        self,
        anchor_digest: str,
        record_digest: str,
        record_size: int,
    ) -> None:
        tag = fallback_referrers_tag(anchor_digest)
        with self._fallback_lock:
            existing = self._fetch_fallback_index(tag)
            manifests_raw = existing.get("manifests") if existing else None
            manifests: list[dict[str, object]] = []
            if isinstance(manifests_raw, list):
                manifests = [m for m in manifests_raw if isinstance(m, dict)]
            # Idempotent within the run: don't duplicate the same record.
            if any(m.get("digest") == record_digest for m in manifests):
                return
            manifests.append(
                {
                    "mediaType": ML_MAN,
                    "artifactType": ARTIFACT_TYPE_RECORD,
                    "digest": record_digest,
                    "size": record_size,
                }
            )
            index = {
                "schemaVersion": 2,
                "mediaType": ML_INDEX,
                "manifests": manifests,
            }
            body = json.dumps(index, separators=(",", ":"), sort_keys=True).encode()
            url = self.client.url(self.repo, "manifests", tag)
            r = self.client.session.put(
                url,
                data=body,
                headers={**self.client.auth, "Content-Type": ML_INDEX},
                timeout=60,
            )
            if r.status_code not in (200, 201):
                r.raise_for_status()
                raise RuntimeError(f"unexpected status {r.status_code} on fallback index PUT")

    def _fetch_fallback_index(self, tag: str) -> dict[str, object] | None:
        url = self.client.url(self.repo, "manifests", tag)
        r = self.client.session.get(
            url,
            headers={**self.client.auth, "Accept": ML_INDEX},
            timeout=30,
        )
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            r.raise_for_status()
        body: dict[str, object] = r.json()
        return body

    @staticmethod
    def _descriptor_from_record(record: dict[str, object]) -> BlobDescriptor | None:
        layers = record.get("layers")
        if not isinstance(layers, list) or not layers:
            return None
        layer = layers[0]
        if not isinstance(layer, dict):
            return None
        annotations = layer.get("annotations")
        if not isinstance(annotations, dict):
            return None
        path = annotations.get(ANN_HF_PATH)
        if not path:
            return None
        sha = annotations.get(ANN_HF_SHA256)
        digest = layer.get("digest")
        size = layer.get("size")
        media = layer.get("mediaType", ML_TAR)
        if not isinstance(digest, str) or not isinstance(size, int):
            return None
        return BlobDescriptor(
            media_type=str(media),
            digest=digest,
            size=size,
            hf_path=str(path),
            hf_sha256=str(sha) if sha else None,
        )
