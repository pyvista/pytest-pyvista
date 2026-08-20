"""Tests for the reusable platform and VTK version conditional skip markers."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING

import pyvista
from pyvista.plotting.utilities import gl_checks

if TYPE_CHECKING:
    import pytest


def test_needs_vtk_version_skips_when_higher_required(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    needs_vtk_version skips when requiring a version higher than installed.

    Pins ``pyvista.vtk_version_info`` and ``pyvista._MIN_SUPPORTED_VTK_VERSION`` via
    ``monkeypatch`` so this does not depend on the VTK actually installed in the
    environment, nor on pyvista's own supported floor (which would otherwise make a
    bound like ``less_than=(9, 3, 0)`` obsolete and raise instead of skip, see
    test_needs_vtk_version_obsolete_constraint_raises). The patch is visible to
    pytester's in-process run since it shares the same ``pyvista`` module, and is
    reverted automatically at teardown.
    """
    monkeypatch.setattr(pyvista, "vtk_version_info", (9, 6, 1))
    monkeypatch.setattr(pyvista, "_MIN_SUPPORTED_VTK_VERSION", (9, 2, 2), raising=False)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(9, 9)
        def test_positional_higher():
            pass

        @pytest.mark.needs_vtk_version(at_least=(99, 0))
        def test_at_least_higher():
            pass

        @pytest.mark.needs_vtk_version(less_than=(9, 3, 0))
        def test_less_than_lower():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(skipped=3)


def test_needs_vtk_version_runs_when_satisfied(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    needs_vtk_version runs when the installed version satisfies the bound.

    Pins ``pyvista.vtk_version_info`` and ``pyvista._MIN_SUPPORTED_VTK_VERSION`` via
    ``monkeypatch`` so this does not depend on the VTK actually installed in the
    environment, nor on pyvista's own supported floor (which would otherwise make a
    bound like ``at_least=(9, 3, 0)`` obsolete and raise instead of pass, see
    test_needs_vtk_version_obsolete_constraint_raises).
    """
    monkeypatch.setattr(pyvista, "vtk_version_info", (9, 6, 1))
    monkeypatch.setattr(pyvista, "_MIN_SUPPORTED_VTK_VERSION", (9, 2, 2), raising=False)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(9, 6)
        def test_positional_satisfied():
            pass

        @pytest.mark.needs_vtk_version(at_least=(9, 3, 0))
        def test_at_least_satisfied():
            pass

        @pytest.mark.needs_vtk_version(less_than=(99, 0))
        def test_less_than_satisfied():
            pass

        @pytest.mark.needs_vtk_version(at_least=(9, 3, 0), less_than=(99, 0))
        def test_range_satisfied():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=4)


def test_needs_vtk_version_tuple_padding(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A short version tuple is padded so (9, 6) compares against (9, 6, 1).

    Pins ``pyvista.vtk_version_info`` and ``pyvista._MIN_SUPPORTED_VTK_VERSION`` via
    ``monkeypatch`` so this does not depend on the VTK actually installed in the
    environment, nor on pyvista's own supported floor.
    """
    monkeypatch.setattr(pyvista, "vtk_version_info", (9, 6, 1))
    monkeypatch.setattr(pyvista, "_MIN_SUPPORTED_VTK_VERSION", (9, 2, 2), raising=False)
    pytester.makepyfile(
        """
        import pytest

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
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1, skipped=2)


def test_needs_vtk_version_range_skips_when_only_max_violated(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A range bound skips when only the `less_than` side is violated.

    The other range test (test_needs_vtk_version_skips_when_higher_required) only
    exercises a violated `at_least` side; this covers the other half of the ``or``
    in the version comparison. Pins ``pyvista.vtk_version_info`` and
    ``pyvista._MIN_SUPPORTED_VTK_VERSION`` for determinism.
    """
    monkeypatch.setattr(pyvista, "vtk_version_info", (9, 6, 1))
    monkeypatch.setattr(pyvista, "_MIN_SUPPORTED_VTK_VERSION", (9, 2, 2), raising=False)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(at_least=(9, 3, 0), less_than=(9, 6, 0))
        def test_max_violated():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(skipped=1)


def test_needs_vtk_version_default_reason_messages(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The auto-generated skip reason covers all three ``needs_vtk_version`` bound shapes.

    Pins ``pyvista.vtk_version_info`` below ``pyvista._MIN_SUPPORTED_VTK_VERSION``'s
    installed value so an ``at_least``-only, a ``less_than``-only, and a range bound
    can each be genuinely unsatisfied (and skip with their default message) without
    tripping the obsolete-constraint check.
    """
    monkeypatch.setattr(pyvista, "vtk_version_info", (9, 0, 0))
    monkeypatch.setattr(pyvista, "_MIN_SUPPORTED_VTK_VERSION", (8, 0, 0), raising=False)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(at_least=(9, 5, 0))
        def test_at_least_only():
            pass

        @pytest.mark.needs_vtk_version(less_than=(8, 5, 0))
        def test_less_than_only():
            pass

        @pytest.mark.needs_vtk_version(at_least=(9, 5, 0), less_than=(9, 8, 0))
        def test_range():
            pass
        """
    )
    result = pytester.runpytest("-v", "-rs")
    result.assert_outcomes(skipped=3)
    result.stdout.fnmatch_lines(["*Test needs VTK version 9.5.0 or greater*"])
    result.stdout.fnmatch_lines(["*Test needs a VTK version less than 8.5.0*"])
    result.stdout.fnmatch_lines(["*Test needs a VTK version of at least 9.5.0 and less than 9.8.0*"])


def test_skip_windows_and_mac_only_skip_on_their_own_platform(pytester: pytest.Pytester) -> None:
    """
    skip_windows and skip_mac only skip on their own platform, whatever this runs on.

    CI runs this suite on Linux, macOS, and Windows, so the expected outcome is
    computed from the current platform instead of assuming Linux (previously this
    hardcoded passed=3, which only holds on Linux and failed on the other two).
    """
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.skip_windows
        def test_windows_marker():
            pass

        @pytest.mark.skip_mac
        def test_mac_marker():
            pass

        @pytest.mark.skip_mac(machine="arm64")
        def test_mac_arm64_marker():
            pass
        """
    )
    result = pytester.runpytest("-v")

    system = platform.system()
    skipped = int(system == "Windows") + int(system == "Darwin") + int(system == "Darwin" and platform.machine() == "arm64")
    result.assert_outcomes(passed=3 - skipped, skipped=skipped)


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


def test_skip_egl_runs_when_not_detected(pytester: pytest.Pytester) -> None:
    """skip_egl does not skip when uses_egl() reports a non-EGL build."""
    pytester.makepyfile(
        """
        import pytest
        from pyvista.plotting.utilities import gl_checks

        # pytest_runtest_setup runs before fixtures, so patch at import time.
        gl_checks.uses_egl = lambda: False

        @pytest.mark.skip_egl
        def test_egl_runs():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_skip_egl_survives_uses_egl_import_error(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `_uses_egl` degrades gracefully to `False` when the private `uses_egl` helper is absent.

    Deletes it from `gl_checks` via ``monkeypatch`` so the `except ImportError` guard is
    the branch under test (older pyvista lacks this helper); reverted automatically at
    teardown.
    """
    monkeypatch.delattr(gl_checks, "uses_egl")
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.skip_egl
        def test_runs_when_uses_egl_unavailable():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


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


def test_needs_vtk_version_obsolete_constraint_raises(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    An `at_least`/`less_than` bound at or below pyvista's VTK floor raises by default.

    Such a bound is guaranteed to always be satisfied (`at_least`) or never satisfied
    (`less_than`), so it is a stale check a maintainer can safely delete -- and the
    default (`raise_obsolete_vtk = true`) surfaces that instead of silently skipping
    or running the guarded test forever. The message names the exact ini setting to
    flip if a project wants to keep such checks anyway (e.g. while still supporting an
    older pyvista). Pins `pyvista._MIN_SUPPORTED_VTK_VERSION` so this does not depend
    on which floor the installed pyvista happens to use.
    """
    monkeypatch.setattr(pyvista, "_MIN_SUPPORTED_VTK_VERSION", (9, 2, 2), raising=False)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(at_least=(9, 0))
        def test_obsolete_at_least():
            pass

        @pytest.mark.needs_vtk_version(less_than=(9, 0))
        def test_obsolete_less_than():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(errors=2)
    result.stdout.fnmatch_lines(["*at_least=9.0.0*is obsolete*always passes*safely removed*"])
    result.stdout.fnmatch_lines(["*less_than=9.0.0*is obsolete*can now never pass*safely removed*"])
    result.stdout.fnmatch_lines(["*raise_obsolete_vtk = false*"])


def test_needs_vtk_version_obsolete_check_disabled_via_ini(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Setting `raise_obsolete_vtk = false` restores the plain check.

    With the obsolete check off, a bound below pyvista's VTK floor is evaluated
    normally against the (pinned) installed version instead of raising.
    """
    monkeypatch.setattr(pyvista, "vtk_version_info", (9, 6, 1))
    monkeypatch.setattr(pyvista, "_MIN_SUPPORTED_VTK_VERSION", (9, 2, 2), raising=False)
    pytester.makeini(
        """
        [pytest]
        raise_obsolete_vtk = false
        """
    )
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(at_least=(9, 0))
        def test_obsolete_but_disabled():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_needs_vtk_version_obsolete_raise_falls_back_without_vtk_version_error(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The obsolete-constraint raise falls back to `RuntimeError` if pyvista lacks `VTKVersionError`.

    Deletes `pyvista.VTKVersionError` via ``monkeypatch`` so the `getattr(...,
    RuntimeError)` fallback is the branch under test (older pyvista lacks this error
    class); reverted automatically at teardown.
    """
    monkeypatch.delattr(pyvista, "VTKVersionError")
    monkeypatch.setattr(pyvista, "_MIN_SUPPORTED_VTK_VERSION", (9, 2, 2), raising=False)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.needs_vtk_version(at_least=(9, 0))
        def test_obsolete():
            pass
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*RuntimeError*is obsolete*"])


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
