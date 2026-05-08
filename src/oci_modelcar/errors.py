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


class PartialFailureError(OciModelcarError):
    exit_code = 7


def exit_code_for(exc: BaseException) -> int:
    """Map any exception to a CLI exit code. Non-OciModelcarError → 1."""
    if isinstance(exc, OciModelcarError):
        return exc.exit_code
    return 1
