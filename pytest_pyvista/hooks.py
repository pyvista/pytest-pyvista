"""Functions users can hook into."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import xdist

    from .doc_mode import _DocVerifyImageCache
    from .doc_mode import _VtkszFileSizeTestCase


@pytest.hookspec
def pytest_pyvista_doc_mode_hook(doc_verify_image_cache: _DocVerifyImageCache, request: pytest.FixtureRequest) -> _DocVerifyImageCache:  # type:ignore[empty-body]
    """
    Function called for each generated test before it executes.

    Users can mutate ``doc_verify_image_cache`` in-place.
    """  # noqa: D401


@pytest.hookspec
def pytest_pyvista_max_vtksz_file_size_hook(test_case: _VtkszFileSizeTestCase, request: pytest.FixtureRequest) -> _VtkszFileSizeTestCase:  # type:ignore[empty-body]
    """
    Function called for each generated test before it executes.

    Users can mutate ``test_case`` in-place.
    """  # noqa: D401


if not importlib.util.find_spec("xdist"):  # type: ignore[attr-defined]

    @pytest.hookspec(firstresult=True)
    def pytest_configure_node(node: xdist.workermanage.WorkerController) -> None:
        """Register a placeholder hook for pytest_configure_node in case xdist is not installed."""
