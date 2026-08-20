"""Tests for the reusable platform and VTK version conditional skip markers."""

from __future__ import annotations

import pytest
import pyvista


def test_needs_vtk_version_skips_when_higher_required(pytester: pytest.Pytester) -> None:
    """needs_vtk_version skips when requiring a version higher than installed."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(9, 9)
        def test_positional_higher():
            pass

        @pytest.mark.needs_vtk_version(at_least=(99, 0))
        def test_at_least_higher():
            pass

        # A `less_than` bound must stay at or above the plugin's own supported
        # VTK floor (9.2.2, see README) -- anything lower is an obsolete
        # constraint that `pyvista.vtk_version_info`'s own comparison raises
        # on for free (see test_needs_vtk_version_obsolete_constraint_raises).
        @pytest.mark.needs_vtk_version(less_than=(9, 2, 2))
        def test_less_than_lower():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(skipped=3)


def test_needs_vtk_version_runs_when_satisfied(pytester: pytest.Pytester) -> None:
    """needs_vtk_version runs when the installed version satisfies the bound."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(9, 6)
        def test_positional_satisfied():
            pass

        # `at_least` must stay at or above the plugin's own supported VTK
        # floor (9.2.2, see README) to avoid the obsolete-constraint raise
        # that `pyvista.vtk_version_info` triggers for free on older bounds.
        @pytest.mark.needs_vtk_version(at_least=(9, 2, 2))
        def test_at_least_satisfied():
            pass

        @pytest.mark.needs_vtk_version(less_than=(99, 0))
        def test_less_than_satisfied():
            pass

        @pytest.mark.needs_vtk_version(at_least=(9, 2, 2), less_than=(99, 0))
        def test_range_satisfied():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=4)


def test_needs_vtk_version_tuple_padding(pytester: pytest.Pytester) -> None:
    """
    A short version tuple is padded so (9, 6) compares against (9, 6, 1).

    Pins ``pyvista.vtk_version_info`` at module scope, so this must run in a
    subprocess -- an in-process run would leak the pinned plain tuple into
    the outer suite, since pytester's in-process runner shares the same
    ``pyvista`` module object as the tests that follow it in this file.
    """
    pytester.makepyfile(
        """
        import pyvista as pv
        import pytest

        # Pin the installed version so this test does not depend on the VTK
        # actually installed in the environment.
        pv.vtk_version_info = (9, 6, 1)

        # (9, 6) -> (9, 6, 0) which is <= pinned (9, 6, 1), so it runs.
        @pytest.mark.needs_vtk_version(9, 6)
        def test_padded_runs():
            pass

        # (9, 6, 2) > pinned (9, 6, 1), so it skips.
        @pytest.mark.needs_vtk_version(9, 6, 2)
        def test_padded_skips():
            pass

        # less_than=(9, 6) -> (9, 6, 0); pinned (9, 6, 1) >= that, so it skips.
        @pytest.mark.needs_vtk_version(less_than=(9, 6))
        def test_padded_less_than_skips():
            pass
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=1, skipped=2)


def test_skip_windows_and_mac_run_on_linux(pytester: pytest.Pytester) -> None:
    """On this Linux box skip_windows and skip_mac do not skip."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.skip_windows
        def test_not_windows():
            pass

        @pytest.mark.skip_mac
        def test_not_mac():
            pass

        @pytest.mark.skip_mac(machine="arm64")
        def test_not_mac_arm64():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=3)


def test_skip_windows_skips_when_windows(pytester: pytest.Pytester) -> None:
    """skip_windows skips when platform.system() reports Windows."""
    pytester.makepyfile(
        """
        import pytest
        from pytest_pyvista import _markers as plugin

        # pytest_runtest_setup runs before fixtures, so patch the plugin's
        # own `os` reference at import time (avoids breaking pathlib).
        class _FakeOS:
            name = "nt"

        plugin.os = _FakeOS()

        @pytest.mark.skip_windows(reason="no windows")
        def test_windows_skips():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(skipped=1)


def test_skip_mac_skips_when_darwin(pytester: pytest.Pytester) -> None:
    """skip_mac skips on Darwin, honoring the machine filter."""
    pytester.makepyfile(
        """
        import pytest
        from pytest_pyvista import _markers as plugin

        # pytest_runtest_setup runs before fixtures, so patch the plugin's
        # own `platform` reference at import time.
        class _FakePlatform:
            @staticmethod
            def system():
                return "Darwin"

            @staticmethod
            def machine():
                return "arm64"

        plugin.platform = _FakePlatform()

        @pytest.mark.skip_mac
        def test_mac_skips():
            pass

        @pytest.mark.skip_mac(machine="arm64")
        def test_mac_arm64_skips():
            pass

        @pytest.mark.skip_mac(machine="x86_64")
        def test_mac_other_machine_runs():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(skipped=2, passed=1)


def test_skip_egl_registered_and_skips_when_detected(pytester: pytest.Pytester) -> None:
    """skip_egl is registered and skips when uses_egl() reports an EGL build."""
    pytester.makepyfile(
        """
        import pytest
        from pyvista.plotting.utilities import gl_checks

        # pytest_runtest_setup runs before fixtures, so patch at import time.
        gl_checks.uses_egl = lambda: True

        @pytest.mark.skip_egl
        def test_egl_skips():
            pass

        @pytest.mark.skip_egl(reason="osmesa")
        def test_egl_skips_with_reason():
            pass
        """
    )
    result = pytester.runpytest("-v", "-W", "error::pytest.PytestUnknownMarkWarning")
    # No unknown-mark warnings means skip_egl is registered.
    result.assert_outcomes(skipped=2)


def test_needs_vtk_version_no_args_errors(pytester: pytest.Pytester) -> None:
    """needs_vtk_version with no args or kwargs surfaces an error."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version
        def test_no_bound():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*at_least*less_than*"])


def test_needs_vtk_version_args_and_at_least_errors(pytester: pytest.Pytester) -> None:
    """Mixing positional args with at_least= surfaces an error."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(9, 0, at_least=(9, 0))
        def test_conflicting():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*Cannot specify both*at_least*"])


def test_needs_vtk_version_min_greater_than_max_errors(pytester: pytest.Pytester) -> None:
    """at_least greater than less_than surfaces an error."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(at_least=(10, 0), less_than=(9, 0))
        def test_inverted_range():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*minimum version greater than the maximum*"])


def test_needs_vtk_version_too_many_components_errors(pytester: pytest.Pytester) -> None:
    """A four-component version trips the _pad_version length guard."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(9, 1, 2, 3)
        def test_too_long():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*incorrect length*"])


def test_needs_vtk_version_string_component_errors(pytester: pytest.Pytester) -> None:
    """A string version component raises a clear TypeError, not an opaque one."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version("9.6")
        def test_string_version():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*must be a tuple of integers*"])


def test_needs_vtk_version_single_component(pytester: pytest.Pytester) -> None:
    """A single-component positional version pads to (N, 0, 0) and skips correctly."""
    pytester.makepyfile(
        """
        import pytest

        # (99,) -> (99, 0, 0) far exceeds installed -> skips.
        @pytest.mark.needs_vtk_version(99)
        def test_single_skips():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(skipped=1)


def test_needs_vtk_version_obsolete_constraint_raises_for_free(pytester: pytest.Pytester) -> None:
    """
    A single-component ``at_least`` below the plugin's VTK floor is obsolete and errors.

    Because the padded minor/micro are always zero, ``needs_vtk_version(9)`` pads to
    ``(9, 0, 0)`` -- below the 9.2.2 floor this plugin documents as its minimum
    supported VTK version (see README). The plugin does not check for this itself:
    the error comes straight out of comparing against ``pyvista.vtk_version_info``,
    which raises on this comparison on its own on pyvista versions where it is
    version-aware.
    """
    if not hasattr(pyvista, "_MIN_SUPPORTED_VTK_VERSION"):
        pytest.skip("requires a pyvista with a version-aware vtk_version_info")

    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(9)
        def test_obsolete_constraint():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*VTKVersionError*unsupported VTK version*"])


def test_needs_vtk_version_custom_reason_in_report(pytester: pytest.Pytester) -> None:
    """A custom reason= is shown in the skip report (guards against a hardcoded default)."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(at_least=(99, 0), reason="needs a future VTK")
        def test_custom_reason():
            pass
        """
    )
    result = pytester.runpytest("-v", "-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*needs a future VTK*"])
