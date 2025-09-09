from __future__ import annotations  # noqa: D100

import pytest
import pyvista as pv

pytest_plugins = "pytester"


@pytest.fixture(autouse=True)
def close_all_plotters() -> None:
    """Close all."""
    pv.close_all()
