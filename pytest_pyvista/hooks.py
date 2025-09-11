"""Functions users can hook into."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from .doc_mode import _DocVerifyImageCache


@pytest.hookspec
def pytest_pyvista_doc_mode_hook(doc_verify_image_cache: _DocVerifyImageCache, request: pytest.FixtureRequest) -> None:
    """
    Function called for each generated test before it executes.

    Users can mutate `doc_verify_image_cache` in-place.
    """  # noqa: D401
