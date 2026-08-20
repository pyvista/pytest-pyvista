"""Tests for the autouse fixtures that reset PyVista global state and theme."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.test_pyvista import make_cached_images

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
    """With ``pyvista_reset_global_state = false`` the mutation persists."""
    pytester.makeini(
        """
        [pytest]
        pyvista_reset_global_state = false
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


def test_reset_pyvista_state_survives_missing_attribute(pytester: pytest.Pytester) -> None:
    """The fixture must not raise if a reset API is absent on older pyvista."""
    pytester.makepyfile(
        """
        import pyvista as pv

        def test_delete_pickle_format():
            # Permanently remove the attribute so the autouse fixture teardown
            # must run with PICKLE_FORMAT absent and rely on its hasattr guard.
            del pv.PICKLE_FORMAT
            assert not hasattr(pv, "PICKLE_FORMAT")

        def test_fixture_did_not_raise():
            # If the previous teardown had raised, this test would error out.
            assert not hasattr(pv, "PICKLE_FORMAT")
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=2)


def test_reset_pyvista_state_suppresses_attribute_error(pytester: pytest.Pytester) -> None:
    """
    ``_restore_default_pyvista_state`` swallows ``AttributeError`` from missing APIs.

    An inner conftest deletes ``pyvista.vtk_snake_case`` so the
    ``contextlib.suppress(AttributeError)`` guard around it is the branch under
    test (older pyvista lacks this API). This must run in a subprocess so the
    attribute deletion cannot leak into the xdist worker running this suite.
    """
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


def test_set_default_theme_survives_testing_theme_import_error(pytester: pytest.Pytester) -> None:
    """
    ``_set_default_theme`` degrades gracefully when ``_TestingTheme`` import fails.

    An inner conftest removes ``_TestingTheme`` from ``pyvista.plotting.themes``
    so the fixture's ``except ImportError`` branch is exercised. Subprocess
    isolation keeps the module surgery out of the xdist worker.
    """
    pytester.makeconftest(
        """
        import pyvista.plotting.themes as themes

        del themes._TestingTheme
        assert not hasattr(themes, "_TestingTheme")
        """
    )
    pytester.makepyfile(
        """
        import pyvista as pv

        def test_theme_fixture_short_circuits(verify_image_cache):
            # `_set_default_theme` cannot import `_TestingTheme`; it must yield
            # without loading a theme and without raising.
            verify_image_cache.allow_useless_fixture = True
            pv.global_theme.background = "purple"

        def test_theme_was_not_reset(verify_image_cache):
            # Because the import failed, the fixture never reloaded the testing
            # theme, so test_a's mutation is still visible here.
            verify_image_cache.allow_useless_fixture = True
            assert pv.global_theme.background == pv.Color("purple")
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=2)


def test_set_default_theme_resets_for_verify_image_cache(pytester: pytest.Pytester) -> None:
    """Theme is restored between ``verify_image_cache`` tests, before and after."""
    make_cached_images(pytester.path)
    pytester.makepyfile(
        """
        import pyvista as pv

        def test_a_mutates_theme(verify_image_cache):
            # The fixture resets to the testing theme before the test runs.
            assert pv.global_theme.background != pv.Color("purple")
            pv.global_theme.background = "purple"
            verify_image_cache.allow_useless_fixture = True

        def test_b_sees_default_theme(verify_image_cache):
            # The fixture restored the theme after test_a's mutation.
            assert pv.global_theme.background != pv.Color("purple")
            pl = pv.Plotter()
            pl.add_mesh(pv.Sphere(), color="red")
            pl.show()
        """
    )
    result = pytester.runpytest("--add_missing_images")
    result.assert_outcomes(passed=2)


def test_set_default_theme_short_circuits_without_verify_image_cache(pytester: pytest.Pytester) -> None:
    """Without ``verify_image_cache`` the theme fixture must not reset the theme."""
    pytester.makepyfile(
        """
        import pyvista as pv

        def test_a_mutates_theme():
            pv.global_theme.background = "purple"

        def test_b_theme_persists():
            assert pv.global_theme.background == pv.Color("purple")
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=2)
