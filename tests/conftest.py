from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

import pytest
import pyvista as pv

if TYPE_CHECKING:
    from collections.abc import Generator

pytest_plugins = "pytester"


# NOTE: close_all_plotters is no longer needed here because the plugin
# provides the _close_plotters_clear_trame_servers autouse fixture (pv.close_all + gc.collect).


@pytest.fixture(autouse=True)
def reset_global_theme() -> Generator[None]:
    """Reset theme."""
    pv.set_plot_theme("document_build")
    yield
    pv.set_plot_theme("document_build")
