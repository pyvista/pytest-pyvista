"""Test the --doc_mode option."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from pytest_pyvista.doc_mode import _preprocess_build_images
from tests.test_pyvista import file_has_changed
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


@pytest.mark.parametrize("failed_image_dir", [True, False])
def test_compare_images_warning(pytester: pytest.Pytester, failed_image_dir) -> None:
    """Test regression warning is issued."""
    cache = "cache"
    images = "images"
    name = "im.png"
    make_cached_images(pytester.path, cache, name=name, color=[255, 0, 0])
    make_cached_images(pytester.path, images, name=name, color=[240, 0, 0])
    _preprocess_build_images(str(pytester.path / cache), str(pytester.path / cache))

    args = ["--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache]
    failed = "failed"
    if failed_image_dir:
        args.extend(["--doc_failed_image_dir", failed])
    result = pytester.runpytest(*args)
    assert result.ret == pytest.ExitCode.OK

    # Check images saved to the failed image dir
    assert Path(failed).is_dir() == failed_image_dir
    assert Path(failed, "warnings").is_dir() == failed_image_dir
    if failed_image_dir:
        name = str(Path(name).with_suffix(".jpg"))
        original = Path(cache, name)
        from_cache = Path(failed, "warnings", "from_cache", name)
        assert from_cache.is_file()
        assert not file_has_changed(str(from_cache), str(original))

        from_build = Path(failed, "warnings", "from_build", name)
        assert from_build.is_file()
        assert file_has_changed(str(from_build), str(from_cache))

    result.stdout.re_match_lines(
        [rf".*UserWarning: {Path(name).stem} Exceeded image regression warning of 200\.0 with an image error of [0-9]+\.[0-9]+"]
    )


ALMOST_BLUE = [0, 0, 254]
ALMOST_RED = [254, 0, 0]


@pytest.mark.parametrize("failed_image_dir", [True, False])
@pytest.mark.parametrize("nested_subdir", [True, False])
@pytest.mark.parametrize(
    ("build_color", "return_code"), [(ALMOST_RED, pytest.ExitCode.OK), (ALMOST_BLUE, pytest.ExitCode.OK), ("green", pytest.ExitCode.TESTS_FAILED)]
)
def test_multiple_cache_images(pytester: pytest.Pytester, build_color, return_code, nested_subdir, failed_image_dir) -> None:
    """Test regression warning is issued."""
    cache = "cache"
    images = "images"
    name = "imcache.png"
    subdir = Path(name).stem
    cache_parent = pytester.path / cache
    cache_parent = cache_parent / subdir if nested_subdir else cache_parent
    red_filename = make_cached_images(cache_parent, subdir, name="im1.png", color="red")
    blue_filename = make_cached_images(cache_parent, subdir, name="im2.png", color="blue")
    build_filename = make_cached_images(pytester.path, images, name=name, color=build_color)
    _preprocess_build_images(str(cache_parent / subdir), str(cache_parent / subdir))

    args = ["--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache]
    failed = "failed"
    if failed_image_dir:
        args.extend(["--doc_failed_image_dir", failed])
    result = pytester.runpytest(*args)
    assert result.ret == return_code

    partial_match = r"imcache Exceeded image regression error of 500\.0 with an image error equal to: [0-9]+\.[0-9]+"
    rel_subdirs = Path(subdir) / subdir if nested_subdir else subdir

    if build_color == ALMOST_RED:
        # Comparison with first image succeeds without issue
        result.stdout.no_re_match_line(rf".*UserWarning: {partial_match}")

        # Test no images are saved
        assert not Path(failed).is_dir()

    elif build_color == ALMOST_BLUE:
        # Comparison with first image fails
        # Expect error was converted to a warning
        result.stdout.re_match_lines(
            [
                rf".*UserWarning: {partial_match}",
                r".*This test has multiple cached images. It initially failed \(as above\) but passed when compared to:",
                ".*im2.jpg",
            ]
        )
        # Test failed images are saved
        cached_original = blue_filename.with_suffix(".jpg")
        from_cache = Path(failed) / "errors_as_warnings" / "from_cache" / rel_subdirs / cached_original.name
        assert from_cache.is_file() == failed_image_dir
        if failed_image_dir:
            assert not file_has_changed(str(from_cache), str(cached_original))

        from_build = Path(failed, "errors_as_warnings", "from_build", build_filename.with_suffix(".jpg").name)
        assert from_build.is_file() == failed_image_dir
        if failed_image_dir:
            assert file_has_changed(str(from_build), str(from_cache))

    else:  # 'green'
        # Comparison with all cached images fails
        result.stdout.re_match_lines(
            [
                rf".*Failed: {partial_match}",
                r".*This test has multiple cached images. It initially failed \(as above\) and failed again for all images in:",
                ".*cache/imcache",
            ]
        )

        # Test failed images are saved
        # Expect both red and blue cached images saved
        for filename in [blue_filename, red_filename]:
            cached_original = filename.with_suffix(".jpg")
            from_cache = Path(failed) / "errors" / "from_cache" / rel_subdirs / cached_original.name
            assert from_cache.is_file() == failed_image_dir
            if failed_image_dir:
                assert not file_has_changed(str(from_cache), str(cached_original))

        from_build = Path(failed, "errors", "from_build", build_filename.with_suffix(".jpg").name)
        assert from_build.is_file() == failed_image_dir
        if failed_image_dir:
            assert file_has_changed(str(from_build), str(from_cache))


def test_single_cache_image_in_subdir(pytester: pytest.Pytester) -> None:
    """Test that a warning is emitting for a cache subdir with only one image."""
    cache = "cache"
    images = "images"
    subdir = "imcache"
    make_cached_images(pytester.path / cache, subdir)
    make_cached_images(pytester.path, images)
    _preprocess_build_images(str(pytester.path / cache / subdir), str(pytester.path / cache / subdir))

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache)
    assert result.ret == pytest.ExitCode.OK
    match = [
        ".*UserWarning: Cached image sub-directory only contains a single image.",
        ".*Move the cached image 'cache/imcache/imcache.jpg' directly to the cached image dir 'cache'",
        ".*or include more than one image in the sub-directory.",
    ]
    result.stdout.re_match_lines(match)
