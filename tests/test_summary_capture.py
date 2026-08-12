"""Tests for record capture during image comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from pytest_pyvista.summary.record import ALL_STATUSES
from pytest_pyvista.summary.record import ImageRecord
from pytest_pyvista.summary.record import read_records
from pytest_pyvista.summary.session import SummarySession
from pytest_pyvista.summary.store import ReportImageStore

if TYPE_CHECKING:
    from pathlib import Path


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


def test_excluded_statuses_are_not_recorded(tmp_path: Path) -> None:
    """A status excluded via ``statuses`` is not written or returned."""
    session = _session(tmp_path, statuses=("failed",))
    baseline = _write(tmp_path / "cache" / "sphere.png", (10, 20, 30))
    generated = _write(tmp_path / "gen" / "sphere.png", (10, 20, 30))

    record = _capture(session, baseline_source=baseline, generated_source=generated)

    assert record is None
    assert read_records(tmp_path / "records") == []
