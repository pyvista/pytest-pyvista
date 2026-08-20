"""Autouse fixture that resets PyVista global state between tests."""

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
        pyvista.allow_new_attributes("private")
    if hasattr(pyvista, "PICKLE_FORMAT"):
        pyvista.PICKLE_FORMAT = "vtk" if pyvista.vtk_version_info >= (9, 3) else "xml"


@pytest.fixture(autouse=True)
def _reset_pyvista_state(pytestconfig: pytest.Config) -> Generator[None, None, None]:
    """Reset PyVista global state to defaults after each test, gated on the ``pyvista_reset_global_state`` ini option."""
    yield

    if pytestconfig.getini("pyvista_reset_global_state"):
        _restore_default_pyvista_state()
