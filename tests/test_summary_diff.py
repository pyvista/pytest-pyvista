"""Tests for pytest_pyvista.summary.diff."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from pytest_pyvista.summary.diff import DIFF_COLOR
from pytest_pyvista.summary.diff import compute_diff_image

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# compute_diff_image reads exactly one baseline and one generated image.
_SOURCES_PER_DIFF = 2


def _write(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


class _TrackedImage:
    """Stands in for an opened image and records whether it was closed."""

    def __init__(self, image: Image.Image) -> None:
        """Wrap ``image``."""
        self.image = image
        self.closed = False

    def __enter__(self) -> Image.Image:
        """Hand the wrapped image to the ``with`` body."""
        return self.image

    def __exit__(self, *_exc: object) -> None:
        """Record the close and release the wrapped image."""
        self.closed = True
        self.image.close()

    def convert(self, mode: str) -> Image.Image:
        """Delegate conversion to the wrapped image."""
        return self.image.convert(mode)


def _track_opened_images(monkeypatch: pytest.MonkeyPatch) -> list[_TrackedImage]:
    opened: list[_TrackedImage] = []
    real_open = Image.open

    def tracking_open(path: Path, *args: object, **kwargs: object) -> _TrackedImage:
        tracked = _TrackedImage(real_open(path, *args, **kwargs))
        opened.append(tracked)
        return tracked

    monkeypatch.setattr(Image, "open", tracking_open)
    return opened


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


def test_source_images_are_closed_after_diffing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that both source images are closed rather than left for the garbage collector."""
    baseline = _write(tmp_path / "a.png", (0, 0, 0))
    generated = _write(tmp_path / "b.png", (255, 255, 255))
    opened = _track_opened_images(monkeypatch)

    compute_diff_image(baseline, generated)

    assert len(opened) == _SOURCES_PER_DIFF
    assert all(image.closed for image in opened)


def test_source_images_are_closed_when_sizes_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the early size-mismatch return still closes both source images."""
    baseline = _write(tmp_path / "a.png", (0, 0, 0), size=(8, 8))
    generated = _write(tmp_path / "b.png", (0, 0, 0), size=(9, 8))
    opened = _track_opened_images(monkeypatch)

    compute_diff_image(baseline, generated)

    assert len(opened) == _SOURCES_PER_DIFF
    assert all(image.closed for image in opened)
