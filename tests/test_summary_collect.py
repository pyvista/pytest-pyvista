"""Tests for pytest_pyvista.summary.collect."""

from __future__ import annotations

import pytest

from pytest_pyvista.summary.collect import determine_status


def _status(**overrides: object) -> str:
    """Call determine_status with default values and overrides."""
    kwargs = {
        "skipped": False,
        "baseline_existed": True,
        "cache_write_reason": None,
        "error": 0.0,
        "error_threshold": 500.0,
        "warning_threshold": 200.0,
        "matched_alternate": False,
    }
    kwargs.update(overrides)
    return determine_status(**kwargs)  # type: ignore[arg-type]


def test_skipped_wins_over_everything() -> None:
    """Skipped status overrides all other conditions."""
    assert _status(skipped=True, baseline_existed=False, error=9999.0) == "skipped"


def test_absent_baseline_is_new() -> None:
    """Missing baseline classifies as new."""
    assert _status(baseline_existed=False) == "new"


def test_absent_baseline_written_by_add_missing_images_is_still_new() -> None:
    """Missing baseline written by add_missing_images policy is still new."""
    assert _status(baseline_existed=False, cache_write_reason="add_missing_images") == "new"


@pytest.mark.parametrize("reason", ["reset_image_cache", "reset_only_failed"])
def test_overwritten_existing_baseline_is_reset(reason: str) -> None:
    """Baseline overwritten by reset policy classifies as reset."""
    assert _status(cache_write_reason=reason, error=9999.0) == "reset"


def test_error_below_warning_threshold_passes() -> None:
    """Error below warning threshold passes."""
    assert _status(error=199.0) == "passed"


def test_error_above_warning_threshold_warns() -> None:
    """Error above warning threshold warns."""
    assert _status(error=201.0) == "warned"


def test_error_exactly_on_warning_threshold_passes() -> None:
    """Error exactly on warning threshold does not trip it."""
    assert _status(error=200.0) == "passed"


def test_error_above_error_threshold_fails() -> None:
    """Error above error threshold fails."""
    assert _status(error=501.0) == "failed"


def test_error_exactly_on_error_threshold_warns() -> None:
    """Error exactly on error threshold does not trip it."""
    assert _status(error=500.0) == "warned"


def test_matching_an_alternate_baseline_warns_however_large_the_primary_error() -> None:
    """Matched alternate baseline always warns regardless of primary error."""
    assert _status(error=9999.0, matched_alternate=True) == "warned"


def test_uncomputable_error_fails() -> None:
    """Error that cannot be computed fails."""
    assert _status(error=None) == "failed"
