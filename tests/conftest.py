from __future__ import annotations  # noqa: D100

import os
import pathlib
import sys
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


@pytest.fixture(autouse=True, scope="session")
def set_playwright_browsers_path() -> None:
    """Set env var so pytester can find the existing browser(s) installed with `playwright install`."""
    home = pathlib.Path.home()

    if sys.platform == "darwin":
        browsers_path = home / "Library" / "Caches" / "ms-playwright"
    elif sys.platform.startswith("linux"):
        browsers_path = home / ".cache" / "ms-playwright"
    elif sys.platform.startswith("win"):
        browsers_path = pathlib.Path(os.environ["LOCALAPPDATA"]) / "ms-playwright"
    else:
        msg = RuntimeError(f"Unsupported platform: {sys.platform}")
        raise msg

    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers_path)
