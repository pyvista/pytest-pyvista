"""Test the --doc_mode option."""

from __future__ import annotations

from PIL import Image
import pytest

from pytest_pyvista.doc_mode import _preprocess_build_images
from tests.test_pyvista import make_cached_images


def test_doc_mode(pytester: pytest.Pytester) -> None:
    """Test regular usage of the --doc_mode."""
    cache = "cache"
    images = "images"
    make_cached_images(pytester.path, cache)
    make_cached_images(pytester.path, images)
    _preprocess_build_images(str(pytester.path / cache), str(pytester.path / cache))

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache)
    assert result.ret == pytest.ExitCode.OK


def test_cli_errors(pytester: pytest.Pytester) -> None:
    """Test errors generated when using CLI."""
    result = pytester.runpytest("--doc_mode")
    assert result.ret == pytest.ExitCode.INTERNAL_ERROR
    result.stderr.fnmatch_lines(["*ValueError: 'doc_images_dir' must be specified when using --doc_mode"])

    images_path = pytester.path / "images"
    result = pytester.runpytest("--doc_mode", "--doc_images_dir", str(images_path))
    assert result.ret == pytest.ExitCode.INTERNAL_ERROR
    result.stderr.fnmatch_lines(["*ValueError: 'doc_images_dir' must be a valid directory. Got:", "*/images."])

    images_path.mkdir()
    result = pytester.runpytest("--doc_mode", "--doc_images_dir", str(images_path))
    assert result.ret == pytest.ExitCode.INTERNAL_ERROR
    result.stderr.fnmatch_lines(["*ValueError: 'doc_image_cache_dir' must be specified when using --doc_mode"])

    cache_path = pytester.path / "cache"
    result = pytester.runpytest("--doc_mode", "--doc_images_dir", str(images_path), "--doc_image_cache_dir", str(cache_path))
    assert result.ret == pytest.ExitCode.INTERNAL_ERROR
    result.stderr.fnmatch_lines(["*ValueError: 'doc_image_cache_dir' must be a valid directory. Got:", "*/cache."])


@pytest.mark.parametrize("missing", ["build", "cache"])
def test_both_images_exist(pytester: pytest.Pytester, missing) -> None:
    """Test when either the cache or build image is missing for the test."""
    images_path = pytester.path / "images"
    cache_path = pytester.path / "cache"
    if missing == "build":
        make_cached_images(cache_path.parent, cache_path.name)
        _preprocess_build_images(str(cache_path), str(cache_path))
        expected_lines = [
            "*The image exists in the cache directory:",
            f"*{cache_path.name}/imcache.jpg",
            "*but is missing from the docs build directory:",
            f"*{images_path.name}",
        ]
    else:
        make_cached_images(images_path.parent, images_path.name)
        expected_lines = [
            "*The image exists in the docs build directory:",
            f"*{images_path.name}",
            "*but is missing from the cache directory:",
            f"*{cache_path.name}",
        ]

    images_path.mkdir(exist_ok=True)
    cache_path.mkdir(exist_ok=True)
    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images_path, "--doc_image_cache_dir", cache_path)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Failed: Test setup failed for test image:*", *expected_lines])


def test_compare_images_with_different_sizes(pytester: pytest.Pytester) -> None:
    """Test error is raised when there is a mismatch in image size."""
    cache = "cache"
    images = "images"
    make_cached_images(pytester.path, cache)
    make_cached_images(pytester.path, images)

    file = pytester.path / cache / "imcache.png"
    with Image.open(file) as im:
        im = im.convert("RGB") if im.mode != "RGB" else im  # noqa: PLW2901
        im.save(file.with_suffix(".jpg"))

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*Failed: RuntimeError('Input images are not the same size.')"])


def test_compare_images_error(pytester: pytest.Pytester) -> None:
    """Test regression error is raised."""
    cache = "cache"
    images = "images"
    make_cached_images(pytester.path, cache, color="red")
    make_cached_images(pytester.path, images, color="blue")
    _preprocess_build_images(str(pytester.path / cache), str(pytester.path / cache))

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache)
    assert result.ret == pytest.ExitCode.TESTS_FAILED

    result.stdout.re_match_lines([r".*Failed: imcache Exceeded image regression error of 500\.0 with an image error equal to: [0-9]+\.[0-9]+"])


def test_compare_images_warning(pytester: pytest.Pytester) -> None:
    """Test regression warning is issued."""
    cache = "cache"
    images = "images"
    make_cached_images(pytester.path, cache, color=[255, 0, 0])
    make_cached_images(pytester.path, images, color=[240, 0, 0])
    _preprocess_build_images(str(pytester.path / cache), str(pytester.path / cache))

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache)
    assert result.ret == pytest.ExitCode.OK

    result.stdout.re_match_lines([r".*UserWarning: imcache Exceeded image regression warning of 200\.0 with an image error of [0-9]+\.[0-9]+"])


@pytest.mark.parametrize("build_color", ["red", "blue"])
def test_multiple_valid_images(pytester: pytest.Pytester, build_color) -> None:
    """Test regression warning is issued."""
    cache = "cache"
    images = "images"
    make_cached_images(pytester.path / cache, "imcache", name="im1.png", color="red")
    make_cached_images(pytester.path / cache, "imcache", name="im2.png", color="blue")
    make_cached_images(pytester.path, images, color=build_color)
    _preprocess_build_images(str(pytester.path / cache / "imcache"), str(pytester.path / cache / "imcache"))

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache)
    assert result.ret == pytest.ExitCode.OK

    match = r".*UserWarning: imcache Exceeded image regression error of 500\.0 with an image error equal to: [0-9]+\.[0-9]+"
    if build_color == "red":
        # Comparison with first image succeeds without issue
        result.stdout.no_re_match_line(match)
    else:
        # Comparison with first image fails
        # Expect error was converted to a warning
        result.stdout.re_match_lines([match])
