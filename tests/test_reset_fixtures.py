"""Tests for the autouse fixture that resets PyVista global state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pytest_pyvista import _reset_fixtures
from pytest_pyvista._reset_fixtures import _capture_default

if TYPE_CHECKING:
    import pytest


def test_reset_pyvista_state_restores_defaults(pytester: pytest.Pytester) -> None:
    """The autouse state-reset fixture restores defaults between tests."""
    pytester.makepyfile(
        """
        import pyvista as pv

        def test_a_mutates_state():
            pv.vtk_verbosity("error")
            assert pv.vtk_verbosity() == "error"

        def test_b_sees_default():
            assert pv.vtk_verbosity() == "info"
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=2)


def test_reset_pyvista_state_disabled_is_noop(pytester: pytest.Pytester) -> None:
    """With ``reset_global_state = false`` the mutation persists."""
    pytester.makeini(
        """
        [pytest]
        reset_global_state = false
        """
    )
    pytester.makepyfile(
        """
        import pyvista as pv

        def test_a_mutates_state():
            pv.vtk_verbosity("error")
            assert pv.vtk_verbosity() == "error"

        def test_b_state_persists():
            assert pv.vtk_verbosity() == "error"
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=2)


def test_reset_pyvista_state_enabled_via_cli(pytester: pytest.Pytester) -> None:
    """Passing ``--reset_global_state=true`` resets state even though the ini default is disabled."""
    pytester.makeini(
        """
        [pytest]
        reset_global_state = false
        """
    )
    pytester.makepyfile(
        """
        import pyvista as pv

        def test_a_mutates_state():
            pv.vtk_verbosity("error")
            assert pv.vtk_verbosity() == "error"

        def test_b_sees_default():
            assert pv.vtk_verbosity() == "info"
        """
    )
    result = pytester.runpytest("--reset_global_state=true")
    result.assert_outcomes(passed=2)


def test_reset_pyvista_state_disabled_via_cli(pytester: pytest.Pytester) -> None:
    """Passing ``--reset_global_state=false`` skips the reset even though the ini default is enabled."""
    pytester.makepyfile(
        """
        import pyvista as pv

        def test_a_mutates_state():
            pv.vtk_verbosity("error")
            assert pv.vtk_verbosity() == "error"

        def test_b_state_persists():
            assert pv.vtk_verbosity() == "error"
        """
    )
    result = pytester.runpytest("--reset_global_state=false")
    result.assert_outcomes(passed=2)


def test_reset_pyvista_state_restores_deleted_attribute(pytester: pytest.Pytester) -> None:
    """The fixture recreates an attribute a test deleted, using the value captured at import time."""
    pytester.makepyfile(
        """
        import pyvista as pv

        _original_pickle_format = pv.PICKLE_FORMAT

        def test_delete_pickle_format():
            del pv.PICKLE_FORMAT
            assert not hasattr(pv, "PICKLE_FORMAT")

        def test_attribute_was_restored():
            assert pv.PICKLE_FORMAT == _original_pickle_format
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=2)


def test_reset_pyvista_state_suppresses_attribute_error(pytester: pytest.Pytester) -> None:
    """``_restore_default_pyvista_state`` swallows ``AttributeError`` from missing APIs; runs in a subprocess for isolation."""
    pytester.makeconftest(
        """
        import pyvista

        del pyvista.vtk_snake_case
        assert not hasattr(pyvista, "vtk_snake_case")
        """
    )
    pytester.makepyfile(
        """
        import pyvista as pv

        def test_a_triggers_teardown():
            # The autouse teardown calls pyvista.vtk_snake_case(...), which is
            # now absent; the suppress(AttributeError) guard must absorb it.
            assert not hasattr(pv, "vtk_snake_case")

        def test_b_teardown_did_not_raise():
            assert not hasattr(pv, "vtk_snake_case")
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=2)


def test_capture_default_returns_none_on_attribute_error() -> None:
    """`_capture_default` returns None when the getter raises `AttributeError`."""

    def _raises() -> None:
        raise AttributeError

    assert _capture_default(_raises) is None


def test_restore_default_pyvista_state_skips_uncaptured_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_restore_default_pyvista_state` does nothing for a setting that was `None` at capture time."""
    for name in (
        "_DEFAULT_VTK_SNAKE_CASE",
        "_DEFAULT_VTK_VERBOSITY",
        "_DEFAULT_ALLOW_NEW_ATTRIBUTES",
        "_DEFAULT_PICKLE_FORMAT",
    ):
        monkeypatch.setattr(_reset_fixtures, name, None)

    _reset_fixtures._restore_default_pyvista_state()  # noqa: SLF001
