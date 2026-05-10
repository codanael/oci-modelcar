import pytest

from oci_modelcar.errors import (
    ConfigError,
    DiskSpaceError,
    DownloadError,
    EntryNotFoundError,
    GatedRepoError,
    OciModelcarError,
    PartialFailureError,
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
        (PartialFailureError, 7),
    ],
)
def test_exit_codes(exc_cls, expected_code):
    assert exc_cls.exit_code == expected_code


def test_exit_code_for_returns_class_code():
    assert exit_code_for(GatedRepoError("x")) == 3
    assert exit_code_for(ValueError("x")) == 1  # non-OciModelcarError → 1
