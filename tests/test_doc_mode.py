"""Test the --doc_mode option."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
import pytest
import pyvista as pv

from pytest_pyvista.doc_mode import _preprocess_build_images
from pytest_pyvista.pytest_pyvista import _EnvInfo
from pytest_pyvista.pytest_pyvista import _get_file_paths
from tests.test_pyvista import file_has_changed
from tests.test_pyvista import make_cached_images
from tests.test_pyvista import make_multiple_cached_images


@pytest.mark.parametrize("generate_subdirs", [True, False])
@pytest.mark.parametrize("generated_image_dir", [True, False])
@pytest.mark.parametrize("image_format", ["png", "jpg"])
def test_doc_mode(pytester: pytest.Pytester, *, generated_image_dir: bool, generate_subdirs: bool, image_format: str) -> None:
    """Test regular usage of the --doc_mode."""
    cache = "cache"
    images = "images"
    name = f"imcache.{image_format}"
    make_cached_images(pytester.path, cache, name=name)
    make_cached_images(pytester.path, images, name=name)
    _preprocess_build_images(pytester.path / cache, pytester.path / cache, image_format=image_format)

    args = ["--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache, "--image_format", image_format]
    generated = "generated"
    if generated_image_dir:
        args.extend(["--doc_generated_image_dir", generated])
    if generate_subdirs:
        args.append("--doc_generate_subdirs")
    result = pytester.runpytest(*args)
    assert result.ret == pytest.ExitCode.OK

    if generated_image_dir:
        expected_name = f"imcache.{image_format}"
        assert Path(generated).is_dir()
        if generate_subdirs:
            subdir = Path(generated) / Path(expected_name).stem
            assert subdir.is_dir()
            assert os.listdir(subdir) == [f"{_EnvInfo()}.{image_format}"]  # noqa: PTH208
        else:
            assert os.listdir(generated) == [expected_name]  # noqa: PTH208


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


@pytest.mark.parametrize("generated_image_dir", [True, False])
@pytest.mark.parametrize("failed_image_dir", [True, False])
@pytest.mark.parametrize("generate_subdirs", [True, False])
@pytest.mark.parametrize("missing_is_empty_dir", [True, False])
@pytest.mark.parametrize("missing", ["build", "cache"])
@pytest.mark.parametrize("image_format", ["png", "jpg"])
def test_both_images_exist(  # noqa: PLR0913
    *,
    pytester: pytest.Pytester,
    missing: str,
    image_format: str,
    generate_subdirs: bool,
    failed_image_dir: bool,
    generated_image_dir: bool,
    missing_is_empty_dir: bool,
) -> None:
    """Test when either the cache or build image is missing for the test."""
    name = f"imcache.{image_format}"
    images_path = pytester.path / "images"
    cache_path = pytester.path / "cache"
    if missing == "build":
        make_cached_images(cache_path.parent, cache_path.name, name=name)
        _preprocess_build_images(cache_path, cache_path, image_format=image_format)
        expected_lines = [
            "*The image exists in the cache directory:",
            f"*{cache_path.name}/imcache.{image_format}",
            "*but is missing from the docs build directory:",
            f"*{images_path.name}",
        ]
        if missing_is_empty_dir:
            (images_path / Path(name).stem).mkdir(parents=True)
    else:
        make_cached_images(images_path.parent, images_path.name)
        expected_lines = [
            "*The image exists in the docs build directory:",
            f"*{images_path.name}",
            "*but is missing from the cache directory:",
            f"*{cache_path.name}",
        ]
        if missing_is_empty_dir:
            (cache_path / Path(name).stem).mkdir(parents=True)

    images_path.mkdir(exist_ok=True)
    cache_path.mkdir(exist_ok=True)
    args = ["--doc_mode", "--doc_images_dir", images_path, "--doc_image_cache_dir", cache_path, "--image_format", image_format]
    if generate_subdirs:
        args.append("--doc_generate_subdirs")
    failed = "failed"
    if failed_image_dir:
        args.extend(["--doc_failed_image_dir", failed])
    generated = "generated"
    if generated_image_dir:
        args.extend(["--doc_generated_image_dir", generated])
    result = pytester.runpytest(*args)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Failed: Test setup failed for test image:*", *expected_lines])

    assert Path(failed).is_dir() == failed_image_dir
    assert Path(generated).is_dir() == (generated_image_dir and missing == "cache")

    if (path := Path(failed)).is_dir():
        assert os.listdir(path) == ["errors"]  # noqa: PTH208
        errors_dir = path / "errors"
        expected_from = "from_build" if missing == "cache" else "from_cache"
        assert os.listdir(errors_dir) == [expected_from]  # noqa: PTH208
        from_dir = errors_dir / expected_from
        expected_image = from_dir / Path(name).stem / f"{_EnvInfo()}.{image_format}" if generate_subdirs and missing == "cache" else from_dir / name
        assert expected_image.is_file()

    if (path := Path(generated)).is_dir():
        expected_image = path / Path(name).stem / f"{_EnvInfo()}.{image_format}" if generate_subdirs else path / name
        assert expected_image.is_file()


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
    _preprocess_build_images(pytester.path / cache, pytester.path / cache)

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache)
    assert result.ret == pytest.ExitCode.TESTS_FAILED

    result.stdout.re_match_lines([r".*Failed: imcache Exceeded image regression error of 500\.0 with an image error equal to: [0-9]+\.[0-9]+"])


@pytest.mark.parametrize(("failed_image_dir", "generate_subdirs"), [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("image_format", ["png", "jpg"])
def test_compare_images_warning(pytester: pytest.Pytester, *, failed_image_dir: bool, image_format: str, generate_subdirs: bool) -> None:
    """Test regression warning is issued."""
    cache = "cache"
    images = "images"
    name = f"im.{image_format}"
    make_cached_images(pytester.path, cache, name=name, color=[255, 0, 0])
    make_cached_images(pytester.path, images, name=name, color=[240, 0, 0])
    _preprocess_build_images(pytester.path / cache, pytester.path / cache, image_format=image_format)

    args = ["--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache, "--image_format", image_format]
    failed = "failed"
    if failed_image_dir:
        args.extend(["--doc_failed_image_dir", failed])
    if generate_subdirs:
        args.append("--doc_generate_subdirs")
    result = pytester.runpytest(*args)
    assert result.ret == pytest.ExitCode.OK

    # Check images saved to the failed image dir
    assert Path(failed).is_dir() == failed_image_dir
    assert Path(failed, "warnings").is_dir() == failed_image_dir
    if failed_image_dir:
        original = Path(cache, name)
        from_cache_file = Path(failed) / "warnings" / "from_cache" / name
        assert from_cache_file.is_file()
        assert not file_has_changed(str(from_cache_file), str(original))

        from_build = Path(failed) / "warnings" / "from_build"
        from_build_file = from_build / Path(name).stem / f"{_EnvInfo()}.{image_format}" if generate_subdirs else from_build / name
        assert from_build_file.is_file()
        assert file_has_changed(str(from_build_file), str(from_cache_file))

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
@pytest.mark.parametrize("image_format", ["png", "jpg"])
def test_multiple_cache_images(pytester: pytest.Pytester, build_color, return_code, nested_subdir, failed_image_dir, image_format) -> None:  # noqa: PLR0913
    """Test when cache is a subdir with multiple images."""
    cache = "cache"
    images = "images"
    name = f"imcache.{image_format}"
    subdir = Path(name).stem
    cache_parent = pytester.path / cache
    cache_parent = cache_parent / subdir if nested_subdir else cache_parent
    red_filename = make_cached_images(cache_parent, subdir, name=f"im1.{image_format}", color="red")
    blue_filename = make_cached_images(cache_parent, subdir, name=f"im2.{image_format}", color="blue")
    build_filename = make_cached_images(pytester.path, images, name=name, color=build_color)
    _preprocess_build_images(cache_parent / subdir, cache_parent / subdir, image_format=image_format)

    args = ["--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache, "--image_format", image_format]
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
                f".*im2.{image_format}",
            ]
        )
        # Test failed images are saved
        from_cache = Path(failed) / "errors_as_warnings" / "from_cache" / rel_subdirs / blue_filename.name
        assert from_cache.is_file() == failed_image_dir
        if failed_image_dir:
            assert not file_has_changed(str(from_cache), str(blue_filename))

        from_build = Path(failed, "errors_as_warnings", "from_build", build_filename.name)
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
            from_cache = Path(failed) / "errors" / "from_cache" / rel_subdirs / filename.name
            assert from_cache.is_file() == failed_image_dir
            if failed_image_dir:
                assert not file_has_changed(str(from_cache), str(filename))

        from_build = Path(failed, "errors", "from_build", build_filename.name)
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
    _preprocess_build_images(pytester.path / cache / subdir, pytester.path / cache / subdir)

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache)
    assert result.ret == pytest.ExitCode.OK
    match = [
        ".*UserWarning: Cached image sub-directory only contains a single image.",
        ".*Move the cached image 'cache/imcache/imcache.png' directly to the cached image dir 'cache'",
        ".*or include more than one image in the sub-directory.",
    ]
    result.stdout.re_match_lines(match)


def test_multiple_cache_images_parallel(pytester: pytest.Pytester) -> None:
    """Ensure that doc_mode works with multiple workers."""
    cache = "cache"
    images = "images"

    n_images = 50
    make_multiple_cached_images(pytester.path, cache, n_images=n_images)
    image_filenames = make_multiple_cached_images(pytester.path, images, n_images=n_images)

    _preprocess_build_images(pytester.path / cache, pytester.path / cache)

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache, "-n2")
    assert result.ret == pytest.ExitCode.OK

    # replace a single image with a different image
    img_idx = 34
    pv.Cube().plot(screenshot=image_filenames[img_idx])

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache, "-n2")
    assert result.ret == pytest.ExitCode.TESTS_FAILED

    assert f"imcache{img_idx} Exceeded image regression error" in str(result.stdout)


@pytest.mark.parametrize("cli", [True, False])
def test_ini(*, pytester: pytest.Pytester, cli: bool) -> None:
    """Test regular usage of the --doc_mode."""
    cache = "cache"
    cache_ini = cache + "ini"
    cache_cli = cache + "cli"

    images = "images"
    images_ini = images + "ini"
    images_cli = images + "cli"

    image_format_ini = "jpg"
    image_format_cli = "png"

    name = "imcache"
    name_ini = f"{name}.{image_format_ini}"
    name_cli = f"{name}.{image_format_cli}"
    if cli:
        make_cached_images(pytester.path, cache_cli, name=name_cli, color="red")
        make_cached_images(pytester.path, images_cli, name=name_cli, color="blue")
    else:
        make_cached_images(pytester.path, cache_ini, name=name_ini, color="red")
        make_cached_images(pytester.path, images_ini, name=name_ini, color="blue")

    generated = "generated"
    generated_ini = generated + "ini"
    generated_cli = generated + "cli"

    failed = "failed"
    failed_ini = failed + "ini"
    failed_cli = failed + "cli"

    pytester.makeini(
        f"""
        [pytest]
        image_format = {image_format_ini}
        doc_failed_image_dir = {failed_ini}
        doc_generated_image_dir = {generated_ini}
        doc_image_cache_dir = {cache_ini}
        doc_images_dir = {images_ini}
        doc_generate_subdirs = True
        """
    )

    args = ["--doc_mode"]
    if cli:
        args.extend(
            [
                "--doc_images_dir",
                images_cli,
                "--doc_image_cache_dir",
                cache_cli,
                "--doc_failed_image_dir",
                failed_cli,
                "--doc_generated_image_dir",
                generated_cli,
                "--image_format",
                image_format_cli,
                "--doc_generate_subdirs",
            ]
        )

    result = pytester.runpytest(*args)
    assert result.ret == pytest.ExitCode.TESTS_FAILED

    assert Path(generated_cli).is_dir() is cli
    assert Path(cache_cli).is_dir() is cli
    assert Path(generated_cli).is_dir() is cli
    assert Path(failed_cli).is_dir() is cli

    assert Path(generated_ini).is_dir() is not cli
    assert Path(cache_ini).is_dir() is not cli
    assert Path(generated_ini).is_dir() is not cli
    assert Path(failed_ini).is_dir() is not cli

    paths_cli = _get_file_paths(pytester.path, ext=image_format_cli)
    paths_ini = _get_file_paths(pytester.path, ext=image_format_ini)

    num_files = 5
    assert len(paths_cli) == (num_files if cli else 0)
    assert len(paths_ini) == (0 if cli else num_files)
