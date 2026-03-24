from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

import pytest
import pyvista as pv

if TYPE_CHECKING:
    from collections.abc import Generator

pytest_plugins = "pytester"


@pytest.fixture(autouse=True)
def close_all_plotters() -> Generator[None]:
    """Close all."""
    yield
    pv.close_all()


@pytest.fixture(autouse=True)
def reset_global_theme() -> Generator[None]:
    """Reset theme."""
    pv.set_plot_theme("document_build")
    yield
    pv.set_plot_theme("document_build")
