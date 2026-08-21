"""End-to-end tests for the image summary report."""

from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING

from PIL import Image

from pytest_pyvista.summary.approve import main as approve_main

if TYPE_CHECKING:
    import pytest

CACHE_DIR = "image_cache_dir"
REPORT_DIR = "image_test_report"
GENERATED_DIR = "generated"

TEST_FILE = """
    import pyvista as pv
    pv.OFF_SCREEN = True

    def test_sphere(verify_image_cache):
        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere(), color="red")
        pl.show()
"""

SKIPPED_TEST_FILE = """
    import pyvista as pv
    pv.OFF_SCREEN = True

    def test_sphere(verify_image_cache):
        verify_image_cache.skip = True
        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere(), color="red")
        pl.show()
"""


WARNING_TEST_FILE = """
    import pyvista as pv
    pv.OFF_SCREEN = True

    def test_sphere(verify_image_cache):
        # Far above any difference this test can produce, so the comparison lands between the
        # warning threshold (200 by default) and the error threshold: a warning, not a failure.
        verify_image_cache.error_value = 1e9
        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere(), color="blue")
        pl.show()
"""


def _report(pytester: pytest.Pytester) -> str:
    """Read the rendered report page."""
    return Path(pytester.path, REPORT_DIR, "index.html").read_text(encoding="utf-8")


def _embedded_manifest(pytester: pytest.Pytester) -> dict:
    """Read back the approval manifest the page carries, the way the browser does."""
    match = re.search(r'<script id="manifest" type="application/json">(.*?)</script>', _report(pytester), re.DOTALL)
    assert match is not None, "the report page carries no embedded manifest"
    return json.loads(match.group(1))


def _exported_entry_keys() -> list[tuple[str, str]]:
    """
    Read the ``(payload key, record key)`` pairs ``exportApprovals`` writes, out of report.js itself.

    The manifest contract is spelled out in four places in two languages - ImageRecord's fields,
    render.py's ``_manifest``, this mapping in report.js, and approve.py's ``_REQUIRED_KEYS`` -
    and only SCHEMA_VERSION is genuinely single-sourced. Deriving the export shape from the
    script rather than restating it here is what makes the round-trip test below fail when
    either side's key names drift, instead of silently testing a third, hand-written copy.
    """
    script = resources.files("pytest_pyvista.summary.assets").joinpath("report.js").read_text(encoding="utf-8")
    block = re.search(r"approved: selectedKeys\(\)\.map\((.*?)\}\),", script, re.DOTALL)
    assert block is not None, "exportApprovals no longer builds its entries in a recognisable shape"
    pairs = re.findall(r"(\w+): record\.(\w+),", block.group(1))
    assert pairs, "exportApprovals no longer copies any field off the manifest record"
    return pairs


def _export_approvals(pytester: pytest.Pytester) -> Path:
    """Write the approvals.json the report's Export button would download, and return its path."""
    manifest = _embedded_manifest(pytester)
    keys = _exported_entry_keys()
    payload = {
        "schema_version": manifest["schema_version"],
        "run_id": manifest["run_id"],
        "exported_at": "2026-08-12T14:05:00Z",
        "cache_dir": manifest["cache_dir"],
        "image_format": manifest["image_format"],
        "approved": [{payload_key: record[record_key] for payload_key, record_key in keys} for record in manifest["records"]],
    }
    path = Path(pytester.path, "approvals.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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


def test_a_bare_run_with_no_baselines_at_all_reports_every_render_as_new(pytester: pytest.Pytester) -> None:
    """
    The report's first-run case: no baselines, no other flags, and every test fails.

    The failure is the point - a missing baseline is a hard error and must stay one - but the
    reader still needs to see what was rendered, and the renders that back those cards have to
    survive the run so an exported approvals.json can name them as its sources.
    """
    pytester.makepyfile(TEST_FILE)

    result = pytester.runpytest("--summary_html")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*does not exist in image cache*"])
    html = _report(pytester)
    assert 'data-status="new"' in html
    assert "No image tests were recorded" not in html
    assert Path(pytester.path, REPORT_DIR, GENERATED_DIR, "sphere.png").is_file()


def test_the_exported_manifest_round_trips_into_the_image_cache(pytester: pytest.Pytester) -> None:
    """
    Render, export, apply: the whole approval loop, across the report's Python and JavaScript halves.

    The export payload is assembled from the key names report.js actually uses, so a rename on
    either side of the manifest contract - the renderer's or the CLI's - fails here rather than
    shipping as an approvals.json the CLI silently rejects. Applying it and re-running the suite
    is what proves the sources outlive the run that wrote them.
    """
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest("--summary_html")

    manifest_path = _export_approvals(pytester)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["approved"], "nothing was offered for approval"

    assert approve_main([str(manifest_path), "--target", "cache"]) == 0

    assert Path(pytester.path, CACHE_DIR, "sphere.png").is_file()
    result = pytester.runpytest("--summary_html")
    result.assert_outcomes(passed=1)
    assert 'data-status="passed"' in _report(pytester)


def test_a_warning_level_difference_is_reported_as_warned(pytester: pytest.Pytester) -> None:
    """A difference above the warning threshold but below the error one is reported as warned."""
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest("--add_missing_images")
    pytester.makepyfile(WARNING_TEST_FILE)

    result = pytester.runpytest("--summary_html")

    result.assert_outcomes(passed=1)
    html = _report(pytester)
    assert 'data-status="warned"' in html
    # A warned card is approvable: the whole point of showing it is to let the reader decide.
    assert "Approve this image" in html


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
    pytester.makepyfile(SKIPPED_TEST_FILE)

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
    # Assert the report exists first: globbing a directory that was never created also
    # returns nothing, which would make the assertion below unable to fail.
    assert Path(pytester.path, REPORT_DIR, "index.html").is_file()
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


def test_a_generated_directory_that_cannot_be_created_says_so_loudly(pytester: pytest.Pytester) -> None:
    """
    A report directory that is fine except for a plain file where ``generated/`` belongs.

    The dangerous shape of this failure is how healthy it looks: the report is written, every
    card is there, and the run says nothing - while the renders behind those cards go to the
    pytest cache and are deleted on the way out, so an approvals.json exported from that
    perfectly normal-looking page cannot be applied to anything. Nothing downstream can explain
    that (the approve CLI sees only a missing file), so the run itself has to.
    """
    Path(pytester.path, REPORT_DIR).mkdir()
    Path(pytester.path, REPORT_DIR, GENERATED_DIR).write_text("not a directory", encoding="utf-8")
    pytester.makepyfile(TEST_FILE)

    result = pytester.runpytest("--summary_html")

    # The degrade is announced, not raised: the run reports exactly what it would have anyway.
    result.assert_outcomes(failed=1)
    assert 'data-status="new"' in _report(pytester)
    result.stdout.fnmatch_lines(["*could not create*generated*"])
    result.stdout.fnmatch_lines(["*approvals exported from it cannot be applied*"])


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
    subdirectory = Path(pytester.path, CACHE_DIR, "sphere")
    subdirectory.mkdir(parents=True)
    # Any readable image will do: the comparison is skipped, so the baseline is only ever
    # copied into the report. Drawing one here is far cheaper than rendering a real sphere.
    Image.new("RGB", (64, 48), "red").save(subdirectory / "one.png")
    pytester.makepyfile(SKIPPED_TEST_FILE)

    pytester.runpytest("--summary_html")

    assert 'data-status="skipped"' in _report(pytester)
    assert Path(pytester.path, REPORT_DIR, "images", "test_sphere.baseline.png").is_file()
