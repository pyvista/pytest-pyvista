from __future__ import annotations  # noqa: D100

import pytest

from pytest_pyvista.pytest_pyvista import _SystemProperties

pytest_plugins = "pytester"


@pytest.fixture(autouse=True)
def reset_system_properties(monkeypatch) -> None:
    """Reset system properties."""
    # Replace the module-level _SYSTEM_PROPERTIES with a fresh instance
    monkeypatch.setattr("pytest_pyvista.pytest_pyvista._SYSTEM_PROPERTIES", _SystemProperties())
