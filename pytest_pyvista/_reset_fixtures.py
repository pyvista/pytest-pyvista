"""Autouse fixtures that reset PyVista global state and theme between tests."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest
import pyvista

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Generator


def _restore_default_pyvista_state() -> None:
    """Reset PyVista global state to defaults, skipping APIs absent on older pyvista."""
    with contextlib.suppress(AttributeError):
        pyvista.vtk_snake_case("error")
    with contextlib.suppress(AttributeError):
        pyvista.vtk_verbosity("info")
    with contextlib.suppress(AttributeError):
        pyvista.allow_new_attributes(False)  # noqa: FBT003
    if hasattr(pyvista, "PICKLE_FORMAT"):
        pyvista.PICKLE_FORMAT = "vtk"


@pytest.fixture(autouse=True)
def _reset_pyvista_state(pytestconfig: pytest.Config) -> Generator[None, None, None]:
    """
    Reset PyVista global state to defaults after each test.

    Gated on the ``pyvista_reset_global_state`` ini option (default: ``True``).
    Each reset is individually guarded so the fixture degrades gracefully on
    older pyvista where some of these APIs do not exist.
    """
    yield

    if pytestconfig.getini("pyvista_reset_global_state"):
        _restore_default_pyvista_state()


@pytest.fixture(autouse=True)
def _set_default_theme(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """
    Reset the plotting theme to the testing theme before and after each test.

    Only active when the test requests the ``verify_image_cache`` fixture; for
    non-plotting tests the fixture short-circuits to keep them fast. The
    ``_TestingTheme`` lookup is guarded so this degrades gracefully on older
    pyvista.
    """
    if "verify_image_cache" not in request.fixturenames:
        yield
        return

    try:
        from pyvista.plotting.themes import _TestingTheme as testing_theme  # noqa: PLC0415
    except ImportError:
        # `_TestingTheme` is not available on older pyvista; degrade gracefully.
        yield
        return

    pyvista.global_theme.load_theme(testing_theme())
    yield
    pyvista.global_theme.load_theme(testing_theme())
