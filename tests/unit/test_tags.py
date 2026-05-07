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
