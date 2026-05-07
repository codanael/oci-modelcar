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
