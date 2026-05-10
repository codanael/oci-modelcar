"""Integration test scaffold. Full happy-path coverage requires a stateful
registry mock; deferred to e2e (Task 10.3) against docker registry:2."""

from __future__ import annotations

import pytest


def test_full_pipeline_push_two_files():
    pytest.skip(
        "Full integration requires a stateful registry mock; "
        "happy-path coverage is in tests/e2e/test_real_huggingface.py "
        "via docker registry:2."
    )
