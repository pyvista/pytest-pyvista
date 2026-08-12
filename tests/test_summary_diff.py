"""Tests for pytest_pyvista.summary.diff."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from pytest_pyvista.summary.diff import DIFF_COLOR
from pytest_pyvista.summary.diff import compute_diff_image

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_identical_images_produce_no_highlighted_pixels(tmp_path: Path) -> None:
    """Test that identical images produce no highlighted pixels in diff."""
    baseline = _write(tmp_path / "a.png", (10, 20, 30))
    generated = _write(tmp_path / "b.png", (10, 20, 30))

    diff, size_mismatch = compute_diff_image(baseline, generated)

    assert size_mismatch is False
    assert not np.any(np.all(np.asarray(diff) == DIFF_COLOR, axis=2))


def test_fully_different_images_highlight_every_pixel(tmp_path: Path) -> None:
    """Test that fully different images highlight every pixel in diff."""
    baseline = _write(tmp_path / "a.png", (0, 0, 0))
    generated = _write(tmp_path / "b.png", (255, 255, 255))

    diff, size_mismatch = compute_diff_image(baseline, generated)

    assert size_mismatch is False
    assert np.all(np.all(np.asarray(diff) == DIFF_COLOR, axis=2))


def test_diff_keeps_the_baseline_dimensions(tmp_path: Path) -> None:
    """Test that diff image has the same dimensions as the baseline."""
    baseline = _write(tmp_path / "a.png", (0, 0, 0), size=(12, 5))
    generated = _write(tmp_path / "b.png", (0, 0, 255), size=(12, 5))

    diff, _ = compute_diff_image(baseline, generated)

    assert diff.size == (12, 5)


def test_mismatched_sizes_report_a_mismatch_and_no_image(tmp_path: Path) -> None:
    """Test that mismatched image sizes report mismatch with no diff image."""
    baseline = _write(tmp_path / "a.png", (0, 0, 0), size=(8, 8))
    generated = _write(tmp_path / "b.png", (0, 0, 0), size=(9, 8))

    diff, size_mismatch = compute_diff_image(baseline, generated)

    assert size_mismatch is True
    assert diff is None
