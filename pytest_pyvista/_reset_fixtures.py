"""Autouse fixture that resets PyVista global state between tests."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from typing import Any

import pytest
import pyvista

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from collections.abc import Generator


def _capture_default(getter: Callable[[], Any]) -> Any | None:  # noqa: ANN401
    """Return `getter()` now, or None if the API is absent on this pyvista."""
    with contextlib.suppress(AttributeError):
        return getter()
    return None


# Captured once at import time (before any test can mutate them), rather than
# hardcoded, so this stays correct however pyvista computes its own defaults.
_DEFAULT_VTK_SNAKE_CASE = _capture_default(pyvista.vtk_snake_case)
_DEFAULT_VTK_VERBOSITY = _capture_default(pyvista.vtk_verbosity)
_DEFAULT_ALLOW_NEW_ATTRIBUTES = _capture_default(pyvista.allow_new_attributes)
_DEFAULT_PICKLE_FORMAT = getattr(pyvista, "PICKLE_FORMAT", None)


def _restore_default_pyvista_state() -> None:
    """Restore PyVista global state to the values captured at import time."""
    if _DEFAULT_VTK_SNAKE_CASE is not None:
        with contextlib.suppress(AttributeError):
            pyvista.vtk_snake_case(_DEFAULT_VTK_SNAKE_CASE)
    if _DEFAULT_VTK_VERBOSITY is not None:
        with contextlib.suppress(AttributeError):
            pyvista.vtk_verbosity(_DEFAULT_VTK_VERBOSITY)
    if _DEFAULT_ALLOW_NEW_ATTRIBUTES is not None:
        with contextlib.suppress(AttributeError):
            pyvista.allow_new_attributes(_DEFAULT_ALLOW_NEW_ATTRIBUTES)
    if _DEFAULT_PICKLE_FORMAT is not None:
        pyvista.PICKLE_FORMAT = _DEFAULT_PICKLE_FORMAT


@pytest.fixture(autouse=True)
def _reset_pyvista_state(pytestconfig: pytest.Config) -> Generator[None, None, None]:
    """Reset PyVista global state to defaults after each test, gated on the ``reset_global_state`` ini option."""
    yield

    if pytestconfig.getini("reset_global_state"):
        _restore_default_pyvista_state()
