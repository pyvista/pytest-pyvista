"""End-to-end tests for the image summary report."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    import pytest

CACHE_DIR = "image_cache_dir"
REPORT_DIR = "image_test_report"

TEST_FILE = """
    import pyvista as pv
    pv.OFF_SCREEN = True

    def test_sphere(verify_image_cache):
        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere(), color="red")
        pl.show()
"""


def _report(pytester: pytest.Pytester) -> str:
    """Read the rendered report page."""
    return Path(pytester.path, REPORT_DIR, "index.html").read_text(encoding="utf-8")


def test_report_is_generated_with_no_other_directories_configured(pytester: pytest.Pytester) -> None:
    """A bare run with only the report flag produces a populated report."""
    pytester.makepyfile(TEST_FILE)

    result = pytester.runpytest("--summary_html", "--add_missing_images")

    result.assert_outcomes(passed=1)
    assert Path(pytester.path, REPORT_DIR, "index.html").is_file()
    assert "test_sphere" in _report(pytester)


def test_report_path_is_printed_to_the_terminal(pytester: pytest.Pytester) -> None:
    """The path of the written report is named on the terminal."""
    pytester.makepyfile(TEST_FILE)

    result = pytester.runpytest("--summary_html", "--add_missing_images")

    result.stdout.fnmatch_lines(["*image summary report*index.html*"])


def test_new_image_is_reported_as_new(pytester: pytest.Pytester) -> None:
    """An image added to the cache this run is reported as new, naming the policy."""
    pytester.makepyfile(TEST_FILE)

    pytester.runpytest("--summary_html", "--add_missing_images")

    assert 'data-status="new"' in _report(pytester)
    assert "add_missing_images" in _report(pytester)


def test_matching_image_is_reported_as_passed(pytester: pytest.Pytester) -> None:
    """An image matching its baseline is reported as passed."""
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest("--add_missing_images")

    pytester.runpytest("--summary_html")

    assert 'data-status="passed"' in _report(pytester)


def test_reset_image_cache_is_reported_as_reset_with_a_real_diff(pytester: pytest.Pytester) -> None:
    """A reset baseline is diffed against the preserved prior baseline, not against itself."""
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest("--add_missing_images")
    pytester.makepyfile(TEST_FILE.replace('color="red"', 'color="blue"'))

    pytester.runpytest("--summary_html", "--reset_image_cache")

    html = _report(pytester)
    assert 'data-status="reset"' in html
    assert "reset_image_cache" in html
    assert "sphere.diff.png" in html


def test_report_is_generated_when_a_test_fails(pytester: pytest.Pytester) -> None:
    """A failing run still produces a report, with the failure recorded."""
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest("--add_missing_images")
    pytester.makepyfile(TEST_FILE.replace('color="red"', 'color="blue"'))

    result = pytester.runpytest("--summary_html")

    result.assert_outcomes(failed=1)
    assert 'data-status="failed"' in _report(pytester)


def test_skipped_test_is_reported_as_skipped(pytester: pytest.Pytester) -> None:
    """A skipped comparison is reported as skipped."""
    pytester.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True

        def test_sphere(verify_image_cache):
            verify_image_cache.skip = True
            pl = pv.Plotter()
            pl.add_mesh(pv.Sphere())
            pl.show()
        """
    )

    pytester.runpytest("--summary_html")

    assert 'data-status="skipped"' in _report(pytester)


def test_include_filter_restricts_what_is_written(pytester: pytest.Pytester) -> None:
    """``--summary_html_include`` filters what reaches the report at all."""
    pytester.makepyfile(TEST_FILE)

    pytester.runpytest("--summary_html", "--add_missing_images", "--summary_html_include", "failed")

    assert "No image tests" in _report(pytester)


def test_full_size_modes_control_retained_copies(pytester: pytest.Pytester) -> None:
    """``--summary_html_full_size`` decides whether full-resolution copies are retained."""
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest("--add_missing_images")

    pytester.runpytest("--summary_html", "--summary_html_full_size", "none")
    assert not list(Path(pytester.path, REPORT_DIR, "images").glob("*.full.png"))

    pytester.runpytest("--summary_html", "--summary_html_full_size", "all")
    assert list(Path(pytester.path, REPORT_DIR, "images").glob("*.full.png"))


def test_custom_report_directory_is_honoured(pytester: pytest.Pytester) -> None:
    """``--summary_html_dir`` relocates the report and implies ``--summary_html``."""
    pytester.makepyfile(TEST_FILE)

    pytester.runpytest("--summary_html_dir", "reports/images", "--add_missing_images")

    assert Path(pytester.path, "reports", "images", "index.html").is_file()


def test_one_report_is_produced_under_xdist(pytester: pytest.Pytester) -> None:
    """Under xdist only the master writes, and its report holds every worker's records."""
    pytester.makepyfile(
        test_a="""
        import pyvista as pv
        pv.OFF_SCREEN = True

        def test_a(verify_image_cache):
            pl = pv.Plotter(); pl.add_mesh(pv.Sphere()); pl.show()
        """,
        test_b="""
        import pyvista as pv
        pv.OFF_SCREEN = True

        def test_b(verify_image_cache):
            pl = pv.Plotter(); pl.add_mesh(pv.Cube()); pl.show()
        """,
    )

    pytester.runpytest("-n", "2", "--summary_html", "--add_missing_images")

    html = _report(pytester)
    assert "test_a" in html
    assert "test_b" in html
    assert len(list(Path(pytester.path).rglob("index.html"))) == 1


def test_embed_mode_produces_a_single_file(pytester: pytest.Pytester) -> None:
    """``--summary_html_embed`` inlines every image as a data URI."""
    pytester.makepyfile(TEST_FILE)

    pytester.runpytest("--summary_html", "--summary_html_embed", "--add_missing_images")

    assert "data:image/png;base64," in _report(pytester)


# --- the report must never break the run --------------------------------------------------


def test_a_report_that_cannot_be_written_warns_without_failing_the_run(pytester: pytest.Pytester) -> None:
    """An unwritable report directory costs a warning, not the exit status or a traceback."""
    pytester.makepyfile(TEST_FILE)
    # A plain file where the report directory belongs: every mkdir against it raises, both
    # while images are stored during the run and while the page is written at the end.
    Path(pytester.path, REPORT_DIR).write_text("not a directory", encoding="utf-8")

    result = pytester.runpytest("--summary_html", "--add_missing_images")

    result.assert_outcomes(passed=1)
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*could not write the image summary report*"])


def test_the_report_survives_the_unused_cache_image_abort(pytester: pytest.Pytester) -> None:
    """``--disallow_unused_cache`` exits from the terminal summary; the report is written anyway."""
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest("--add_missing_images")
    shutil.copy(Path(pytester.path, CACHE_DIR, "sphere.png"), Path(pytester.path, CACHE_DIR, "unused.png"))

    result = pytester.runpytest("--summary_html", "--disallow_unused_cache")

    assert result.ret != 0
    result.stdout.fnmatch_lines(["*Unused cached image file(s) detected*"])
    assert 'data-status="passed"' in _report(pytester)


# --- graceful degradation -----------------------------------------------------------------


def test_an_unused_generated_image_is_reported_as_new(pytester: pytest.Pytester) -> None:
    """``--allow_unused_generated`` renders an image with no baseline; it belongs in the report."""
    pytester.makepyfile(TEST_FILE)

    result = pytester.runpytest("--summary_html", "--allow_unused_generated")

    result.assert_outcomes(passed=1)
    html = _report(pytester)
    assert 'data-status="new"' in html
    assert "test_sphere" in html
    # No cache-writing policy ran, so the image is offered for approval.
    assert "Approve this image" in html


def test_a_skipped_test_shows_a_baseline_held_in_a_subdirectory(pytester: pytest.Pytester) -> None:
    """A skipped test whose baselines live in a subdirectory still shows one of them."""
    cache = Path(pytester.path, CACHE_DIR, "sphere")
    cache.mkdir(parents=True)
    # Any readable image will do: the comparison is skipped, so the baseline is only ever
    # copied into the report. Drawing one here is far cheaper than rendering a real sphere.
    Image.new("RGB", (64, 48), "red").save(cache / "one.png")
    pytester.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True

        def test_sphere(verify_image_cache):
            verify_image_cache.skip = True
            pl = pv.Plotter()
            pl.add_mesh(pv.Sphere(), color="red")
            pl.show()
        """
    )

    pytester.runpytest("--summary_html")

    assert 'data-status="skipped"' in _report(pytester)
    assert Path(pytester.path, REPORT_DIR, "images", "test_sphere.baseline.png").is_file()
