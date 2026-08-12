"""Tests for the summary report's pytest options."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

SIMPLE_TEST = """
    import pyvista as pv
    pv.OFF_SCREEN = True

    def test_sphere(verify_image_cache):
        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere())
        pl.show()
"""


def test_summary_html_is_rejected_under_doc_mode(pytester: pytest.Pytester) -> None:
    """--summary_html is a unit-test-only flag and must be rejected under --doc_mode."""
    pytester.makepyfile(SIMPLE_TEST)

    result = pytester.runpytest("--doc_mode", "--summary_html")

    result.stderr.fnmatch_lines(["*--summary_html cannot be used with --doc_mode enabled*"])


def test_summary_html_include_rejects_an_unknown_status(pytester: pytest.Pytester) -> None:
    """An unknown status passed to --summary_html_include fails configuration eagerly."""
    pytester.makepyfile(SIMPLE_TEST)

    result = pytester.runpytest("--summary_html", "--summary_html_include", "passed,bogus")

    result.stderr.fnmatch_lines(["*unknown status*bogus*"])


def test_summary_html_full_size_rejects_an_unknown_mode(pytester: pytest.Pytester) -> None:
    """An unknown value for --summary_html_full_size is rejected by argparse's choices."""
    pytester.makepyfile(SIMPLE_TEST)

    result = pytester.runpytest("--summary_html", "--summary_html_full_size", "bogus")

    assert result.ret != 0


def test_summary_html_dir_implies_enablement(pytester: pytest.Pytester) -> None:
    """Passing --summary_html_dir enables the report even without --summary_html."""
    pytester.makepyfile(
        """
        from pytest_pyvista.pytest_pyvista import _summary_html_enabled

        def test_enabled(pytestconfig):
            assert _summary_html_enabled(pytestconfig)
        """
    )

    result = pytester.runpytest("--summary_html_dir", "somewhere")

    result.assert_outcomes(passed=1)


def test_summary_html_is_off_by_default(pytester: pytest.Pytester) -> None:
    """With no flags or ini options set, the summary report is disabled."""
    pytester.makepyfile(
        """
        from pytest_pyvista.pytest_pyvista import _summary_html_enabled

        def test_disabled(pytestconfig):
            assert not _summary_html_enabled(pytestconfig)
        """
    )

    result = pytester.runpytest()

    result.assert_outcomes(passed=1)


def test_ini_can_enable_the_report(pytester: pytest.Pytester) -> None:
    """The summary_html ini option can enable the report without a CLI flag."""
    pytester.makeini(
        """
        [pytest]
        summary_html = true
        """
    )
    pytester.makepyfile(
        """
        from pytest_pyvista.pytest_pyvista import _summary_html_enabled

        def test_enabled(pytestconfig):
            assert _summary_html_enabled(pytestconfig)
        """
    )

    result = pytester.runpytest()

    result.assert_outcomes(passed=1)
