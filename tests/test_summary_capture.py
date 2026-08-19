"""Tests for record capture during image comparison."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
import pyvista as pv

from pytest_pyvista.summary.record import ALL_STATUSES
from pytest_pyvista.summary.record import ImageRecord
from pytest_pyvista.summary.record import read_records
from pytest_pyvista.summary.session import SummarySession
from pytest_pyvista.summary.store import ReportImageStore

if TYPE_CHECKING:
    import pytest

pv.OFF_SCREEN = True


def _write(path: Path, color: tuple[int, int, int]) -> Path:
    """Write a solid-color RGB image to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color).save(path)
    return path


def _session(tmp_path: Path, statuses: tuple[str, ...] = ()) -> SummarySession:
    """Build a SummarySession rooted at ``tmp_path`` for testing."""
    return SummarySession(
        run_id="run-1",
        records_dir=tmp_path / "records",
        store=ReportImageStore(tmp_path / "report"),
        worker_id="master",
        statuses=statuses or ALL_STATUSES,
        cache_dir=tmp_path / "cache",
    )


def _capture(session: SummarySession, **overrides: object) -> ImageRecord | None:
    """Call ``session.capture`` with sensible defaults, overridden by ``overrides``."""
    kwargs = {
        "test_name": "test_sphere",
        "image_name": "sphere.png",
        "call_index": 0,
        "baseline_source": None,
        "generated_source": None,
        "cache_destination": None,
        "skipped": False,
        "skip_reason": None,
        "baseline_existed": True,
        "cache_write_reason": None,
        "error": None,
        "error_threshold": 500.0,
        "warning_threshold": 200.0,
        "high_variance_test": False,
        "matched_alternate": False,
        "matched_baseline": None,
        "candidate_baselines": [],
        "image_format": "png",
        "env_info": "env",
    }
    kwargs.update(overrides)
    return session.capture(**kwargs)


def test_capture_writes_a_record_and_all_three_images(tmp_path: Path) -> None:
    """A comparison with both a baseline and a generated image writes all three images."""
    session = _session(tmp_path)
    baseline = _write(tmp_path / "cache" / "sphere.png", (0, 0, 0))
    generated = _write(tmp_path / "gen" / "sphere.png", (255, 255, 255))

    record = _capture(session, baseline_source=baseline, generated_source=generated)

    assert record.status == "failed"
    assert (tmp_path / "report" / record.baseline_image).is_file()
    assert (tmp_path / "report" / record.generated_image).is_file()
    assert (tmp_path / "report" / record.diff_image).is_file()
    assert read_records(tmp_path / "records") == [record]


def test_capture_computes_its_own_error_ignoring_the_caller(tmp_path: Path) -> None:
    """The recorded error is computed by the session, not taken from the caller's value."""
    session = _session(tmp_path)
    baseline = _write(tmp_path / "cache" / "sphere.png", (0, 0, 0))
    generated = _write(tmp_path / "gen" / "sphere.png", (255, 255, 255))

    record = _capture(session, baseline_source=baseline, generated_source=generated, error=0.0)

    assert record.error > 0.0


def test_identical_images_pass(tmp_path: Path) -> None:
    """Identical baseline and generated images produce a passed record."""
    session = _session(tmp_path)
    baseline = _write(tmp_path / "cache" / "sphere.png", (10, 20, 30))
    generated = _write(tmp_path / "gen" / "sphere.png", (10, 20, 30))

    record = _capture(session, baseline_source=baseline, generated_source=generated)

    assert record.status == "passed"


def test_new_image_records_no_baseline_and_no_diff(tmp_path: Path) -> None:
    """A newly added image (no prior baseline) records only the generated image."""
    session = _session(tmp_path)
    generated = _write(tmp_path / "gen" / "sphere.png", (255, 0, 0))

    record = _capture(session, baseline_existed=False, generated_source=generated, cache_write_reason="add_missing_images")

    assert record.status == "new"
    assert record.baseline_image is None
    assert record.diff_image is None
    assert record.generated_image is not None
    assert record.cache_written is True


def test_skipped_image_records_the_baseline_only(tmp_path: Path) -> None:
    """A skipped comparison records the baseline image and the skip reason."""
    session = _session(tmp_path)
    baseline = _write(tmp_path / "cache" / "sphere.png", (0, 0, 0))

    record = _capture(session, skipped=True, skip_reason="windows_skip_image_cache", baseline_source=baseline)

    assert record.status == "skipped"
    assert record.baseline_image is not None
    assert record.generated_image is None
    assert record.skip_reason == "windows_skip_image_cache"


def test_size_mismatch_is_flagged_without_a_diff(tmp_path: Path) -> None:
    """A size mismatch between baseline and generated images is flagged with no diff."""
    session = _session(tmp_path)
    baseline = _write(tmp_path / "cache" / "sphere.png", (0, 0, 0))
    generated = tmp_path / "gen" / "sphere.png"
    generated.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (41, 30), (0, 0, 0)).save(generated)

    record = _capture(session, baseline_source=baseline, generated_source=generated)

    assert record.size_mismatch is True
    assert record.diff_image is None
    assert record.error is None


def test_capture_records_the_dimensions_of_both_images(tmp_path: Path) -> None:
    """Both image sizes are recorded so the report can name them in a size-mismatch notice."""
    session = _session(tmp_path)
    baseline = _write(tmp_path / "cache" / "sphere.png", (0, 0, 0))
    generated = tmp_path / "gen" / "sphere.png"
    generated.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (41, 30), (0, 0, 0)).save(generated)

    record = _capture(session, baseline_source=baseline, generated_source=generated)

    assert (record.baseline_width, record.baseline_height) == (40, 30)
    assert (record.generated_width, record.generated_height) == (41, 30)


def test_excluded_statuses_are_not_recorded(tmp_path: Path) -> None:
    """A status excluded via ``statuses`` is not written or returned."""
    session = _session(tmp_path, statuses=("failed",))
    baseline = _write(tmp_path / "cache" / "sphere.png", (10, 20, 30))
    generated = _write(tmp_path / "gen" / "sphere.png", (10, 20, 30))

    record = _capture(session, baseline_source=baseline, generated_source=generated)

    assert record is None
    assert read_records(tmp_path / "records") == []


# --- end-to-end capture through the plugin ------------------------------------------------

SPHERE_TEST = """
    import pyvista as pv
    pv.OFF_SCREEN = True

    def test_imcache(verify_image_cache):
        plotter = pv.Plotter()
        plotter.add_mesh(pv.Sphere(), color={color})
        plotter.show()
"""


def _render_sphere(path: Path, color: str | list[int]) -> Path:
    """Render a sphere of ``color`` to ``path`` to serve as a cached baseline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv.Sphere(), color=color)
    plotter.screenshot(path)
    return path


RECORDS_COPY = "records_copy"

# ``pytest_unconfigure`` wipes the records directory at the end of a run, so a run whose
# records the outer test wants to inspect must copy them aside while the session is alive.
SAVE_RECORDS_CONFTEST = f"""
    from pathlib import Path
    import shutil

    from pytest_pyvista.pytest_pyvista import PYVISTA_SUMMARY_RECORDS_DIRNAME


    def pytest_sessionfinish(session):
        records_dir = getattr(session.config, PYVISTA_SUMMARY_RECORDS_DIRNAME, None)
        if records_dir is not None:
            shutil.copytree(records_dir, Path(session.config.rootpath) / "{RECORDS_COPY}", dirs_exist_ok=True)
"""


def _records_of(pytester: pytest.Pytester) -> list[ImageRecord]:
    """Read the records saved aside by ``SAVE_RECORDS_CONFTEST`` during a run of ``pytester``."""
    return read_records(pytester.path / RECORDS_COPY)


def _explode(self: SummarySession, **kwargs: object) -> None:  # noqa: ARG001
    """Stand in for ``SummarySession.capture`` when the report store is unusable."""
    msg = "report store is unwritable"
    raise OSError(msg)


def test_an_alternate_baseline_match_is_recorded_against_the_baseline_that_matched(pytester: pytest.Pytester) -> None:
    """A test that fails its primary baseline but matches another records the one that matched."""
    cache = pytester.path / "image_cache_dir" / "imcache"
    _render_sphere(cache / "im1.png", "red")
    blue = _render_sphere(cache / "im2.png", "blue")
    pytester.makeconftest(SAVE_RECORDS_CONFTEST)
    pytester.makepyfile(SPHERE_TEST.format(color=[0, 0, 254]))

    result = pytester.runpytest("--summary_html")

    result.assert_outcomes(passed=1)
    (record,) = _records_of(pytester)
    assert record.status == "warned"
    assert record.matched_baseline is not None
    assert Path(record.matched_baseline).name == "im2.png"
    assert [Path(candidate).name for candidate in record.candidate_baselines] == ["im1.png", "im2.png"]
    # The error must be measured against the matched baseline, not against candidate 0.
    assert record.error is not None
    assert record.error_threshold is not None
    assert record.error < record.error_threshold
    # ...and the stored baseline image must be that same matched file.
    assert record.baseline_image_full is not None
    stored = pytester.path / "image_test_report" / record.baseline_image_full
    assert pv.compare_images(str(stored), str(blue)) < record.error_threshold


def test_a_reporting_failure_does_not_replace_the_regression_error(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception raised while recording must not hide the failure nor skip the cleanup."""
    _render_sphere(pytester.path / "image_cache_dir" / "imcache.png", "red")
    pytester.makepyfile(SPHERE_TEST.format(color="'green'"))
    monkeypatch.setattr(SummarySession, "capture", _explode)

    result = pytester.runpytest("--summary_html")

    # errors=0 is the point: the close callback is still removed, so teardown does not
    # re-enter the comparison and error against an image that was never rendered.
    result.assert_outcomes(failed=1)
    result.stdout.re_match_lines([r".*RegressionError: .*"])
    result.stdout.no_re_match_line(r".*OSError: report store is unwritable.*")


def test_a_reporting_failure_in_the_skip_branch_does_not_fail_the_test(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """The skipped-comparison capture is guarded in the same way as the main one."""
    _render_sphere(pytester.path / "image_cache_dir" / "imcache.png", "red")
    pytester.makepyfile(SPHERE_TEST.format(color="'red'"))
    monkeypatch.setattr(SummarySession, "capture", _explode)

    result = pytester.runpytest("--summary_html", "--ignore_image_cache")

    result.assert_outcomes(passed=1)


def test_records_carry_the_run_id_stored_on_the_config(pytester: pytest.Pytester) -> None:
    """Without xdist the master must not mint a second run id for its own records."""
    _render_sphere(pytester.path / "image_cache_dir" / "imcache.png", "red")
    pytester.makeconftest(SAVE_RECORDS_CONFTEST)
    pytester.makepyfile(
        """
        from pathlib import Path

        import pyvista as pv
        pv.OFF_SCREEN = True

        def test_imcache(verify_image_cache, pytestconfig):
            Path("run_id.txt").write_text(pytestconfig.pyvista_run_id)
            plotter = pv.Plotter()
            plotter.add_mesh(pv.Sphere(), color='red')
            plotter.show()
        """
    )

    result = pytester.runpytest("--summary_html")

    result.assert_outcomes(passed=1)
    (record,) = _records_of(pytester)
    assert record.run_id == (pytester.path / "run_id.txt").read_text()
