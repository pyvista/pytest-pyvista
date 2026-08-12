"""Tests for pytest_pyvista.summary.store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image
import pytest

from pytest_pyvista.summary.store import ReportImageStore
from pytest_pyvista.summary.store import slugify

if TYPE_CHECKING:
    from pathlib import Path

_TEST_MAX_SIZE = 100


def _write(path: Path, size: tuple[int, int] = (800, 600)) -> Path:
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sphere", "sphere"),
        ("test_foo[1-2]", "test_foo_1-2"),
        ("a/b\\c", "a_b_c"),
        ("spaces here", "spaces_here"),
    ],
)
def test_slugify_makes_names_filesystem_safe(name: str, expected: str) -> None:
    """Test that slugify makes names filesystem safe."""
    assert slugify(name) == expected


def test_save_file_downscales_to_the_max_edge(tmp_path: Path) -> None:
    """Test that save_file downscales images to the max edge."""
    store = ReportImageStore(tmp_path / "report", max_image_size=_TEST_MAX_SIZE)
    source = _write(tmp_path / "src.png", size=(800, 600))

    thumb, _ = store.save_file(source, "sphere", "baseline", status="passed")

    assert max(Image.open(tmp_path / "report" / thumb).size) == _TEST_MAX_SIZE


def test_save_file_returns_a_posix_relative_path(tmp_path: Path) -> None:
    """Test that save_file returns a POSIX relative path."""
    store = ReportImageStore(tmp_path / "report")
    source = _write(tmp_path / "src.png")

    thumb, _ = store.save_file(source, "sphere", "baseline", status="passed")

    assert thumb == "images/sphere.baseline.png"
    assert (tmp_path / "report" / thumb).is_file()


def test_failing_mode_keeps_full_size_for_non_passed_only(tmp_path: Path) -> None:
    """Test that failing mode keeps full-size images only for non-passed statuses."""
    store = ReportImageStore(tmp_path / "report", max_image_size=_TEST_MAX_SIZE, full_size="failing")
    source = _write(tmp_path / "src.png")

    _, passed_full = store.save_file(source, "a", "generated", status="passed")
    _, failed_full = store.save_file(source, "b", "generated", status="failed")

    assert passed_full is None
    assert failed_full == "images/b.generated.full.png"
    assert Image.open(tmp_path / "report" / failed_full).size == (800, 600)


def test_none_mode_never_keeps_full_size(tmp_path: Path) -> None:
    """Test that none mode never keeps full-size images."""
    store = ReportImageStore(tmp_path / "report", full_size="none")
    source = _write(tmp_path / "src.png")

    assert store.save_file(source, "b", "generated", status="failed")[1] is None


def test_all_mode_keeps_full_size_even_for_passed(tmp_path: Path) -> None:
    """Test that all mode keeps full-size images even for passed status."""
    store = ReportImageStore(tmp_path / "report", full_size="all")
    source = _write(tmp_path / "src.png")

    assert store.save_file(source, "a", "generated", status="passed")[1] is not None


def test_smaller_images_are_not_enlarged(tmp_path: Path) -> None:
    """Test that smaller images are not enlarged."""
    store = ReportImageStore(tmp_path / "report", max_image_size=1000)
    source = _write(tmp_path / "src.png", size=(40, 30))

    thumb, _ = store.save_file(source, "sphere", "baseline", status="passed")

    assert Image.open(tmp_path / "report" / thumb).size == (40, 30)


def test_save_image_accepts_an_in_memory_image(tmp_path: Path) -> None:
    """Test that save_image accepts an in-memory PIL Image."""
    store = ReportImageStore(tmp_path / "report", max_image_size=_TEST_MAX_SIZE)

    thumb, _ = store.save_image(Image.new("RGB", (800, 600)), "sphere", "diff", status="failed")

    assert (tmp_path / "report" / thumb).is_file()
