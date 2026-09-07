"""Test the --doc_mode option."""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
import re
import subprocess
import sys

from PIL import Image
import pytest
import pyvista as pv

from pytest_pyvista import doc_mode
from pytest_pyvista.doc_mode import _DocVerifyImageCache
from pytest_pyvista.doc_mode import _html_screenshots
from pytest_pyvista.doc_mode import _parse_window_size
from pytest_pyvista.doc_mode import _vtksz_to_html_files
from pytest_pyvista.doc_mode import _vtksz_window_sizes
from pytest_pyvista.doc_mode import _VtkszFileSizeTestCase
from pytest_pyvista.pytest_pyvista import _EnvInfo
from pytest_pyvista.pytest_pyvista import _get_file_paths
from tests.test_pyvista import file_has_changed
from tests.test_pyvista import make_cached_images as _make_cached_images
from tests.test_pyvista import make_multiple_cached_images as _make_multiple_cached_images

# Doc-mode tests do not use the `verify_image_cache` fixture, so their rendering
# uses the default theme, not the testing theme. Generate baselines to match.
make_cached_images = functools.partial(_make_cached_images, use_testing_theme=False)
make_multiple_cached_images = functools.partial(_make_multiple_cached_images, use_testing_theme=False)


@pytest.fixture(autouse=True, scope="module")
def check_playwright_config() -> None:
    """
    Set env var so pytester can find the existing browser(s) installed with `playwright install`.

    See https://playwright.dev/python/docs/browsers#managing-browser-binaries.

    Also check that playwright browsers have been installed.
    """
    home = Path.home()

    if sys.platform == "darwin":
        browsers_path = home / "Library" / "Caches" / "ms-playwright"
    elif sys.platform.startswith("linux"):
        browsers_path = home / ".cache" / "ms-playwright"
    elif sys.platform.startswith("win"):
        browsers_path = Path(os.environ["LOCALAPPDATA"]) / "ms-playwright"
    else:
        msg = RuntimeError(f"Unsupported platform: {sys.platform}")
        raise msg

    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers_path)

    # Check non-empty browsers list
    out = subprocess.run(["playwright", "install", "--list"], check=True, capture_output=True)  # noqa: S607
    if len(out.stdout.splitlines()) == 0:
        pytest.fail("`playwright` browsers have not been installed. Run `playwright install` before executing the test suite.")


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

    args = ["--doc_mode", "--doc_images_dir", images, "--image_cache_dir", cache, "--image_format", image_format]
    generated = "generated"
    if generated_image_dir:
        args.extend(["--generated_image_dir", generated])
    if generate_subdirs:
        args.append("--generate_subdirs")
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
    result.stdout.fnmatch_lines(["*ValueError: 'doc_images_dir' must be specified when using --doc_mode"])

    images_path = pytester.path / "images"
    result = pytester.runpytest("--doc_mode", "--doc_images_dir", str(images_path))
    assert result.ret == pytest.ExitCode.INTERNAL_ERROR
    result.stdout.fnmatch_lines(["*ValueError: 'doc_images_dir' must be a valid directory. Got:", "*/images."])

    images_path.mkdir()
    result = pytester.runpytest("--doc_mode", "--doc_images_dir", str(images_path))
    assert result.ret == pytest.ExitCode.INTERNAL_ERROR
    result.stdout.fnmatch_lines("INTERNALERROR> RuntimeError: No doc images or cache images found. The doc images dir:")
    result.stdout.fnmatch_lines("INTERNALERROR>   */images")
    result.stdout.fnmatch_lines("INTERNALERROR> and image cache dir:")
    result.stdout.fnmatch_lines("INTERNALERROR>   */doc_image_cache_dir")
    result.stdout.fnmatch_lines("INTERNALERROR> cannot both be empty.")


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
    args = ["--doc_mode", "--doc_images_dir", images_path, "--image_cache_dir", cache_path, "--image_format", image_format]
    if generate_subdirs:
        args.append("--generate_subdirs")
    failed = "failed"
    if failed_image_dir:
        args.extend(["--failed_image_dir", failed])
    generated = "generated"
    if generated_image_dir:
        args.extend(["--generated_image_dir", generated])
    result = pytester.runpytest(*args)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Failed: Test setup failed for test image:*", *expected_lines])

    assert Path(failed).is_dir() == failed_image_dir
    assert Path(generated).is_dir() == (generated_image_dir and missing == "cache")

    if (path := Path(failed)).is_dir():
        assert os.listdir(path) == ["errors"]  # noqa: PTH208
        errors_dir = path / "errors"
        expected_from = "from_test" if missing == "cache" else "from_cache"
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
    make_cached_images(pytester.path, cache, window_size=(400, 300))
    make_cached_images(pytester.path, images, window_size=(401, 300))

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--image_cache_dir", cache)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*Failed: RuntimeError('Input images are not the same size.')"])


def test_compare_images_error(pytester: pytest.Pytester) -> None:
    """Test regression error is raised."""
    cache = "cache"
    images = "images"
    make_cached_images(pytester.path, cache, color="red")
    make_cached_images(pytester.path, images, color="blue")

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--image_cache_dir", cache)
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
    make_cached_images(pytester.path, images, name=name, color=[251, 0, 0])

    args = ["--doc_mode", "--doc_images_dir", images, "--image_cache_dir", cache, "--image_format", image_format]
    failed = "failed"
    if failed_image_dir:
        args.extend(["--failed_image_dir", failed])
    if generate_subdirs:
        args.append("--generate_subdirs")
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

        from_test = Path(failed) / "warnings" / "from_test"
        from_test_file = from_test / Path(name).stem / f"{_EnvInfo()}.{image_format}" if generate_subdirs else from_test / name
        assert from_test_file.is_file()
        assert file_has_changed(str(from_test_file), str(from_cache_file))

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
def test_multiple_cache_images(  # noqa: PLR0913, PLR0917
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, build_color, return_code, nested_subdir, failed_image_dir, image_format
) -> None:
    """Test when cache is a subdir with multiple images."""
    if build_color == ALMOST_BLUE:
        # Lower the warning threshold so that the matching second image is not a
        # clean match and a warning is emitted for it instead
        monkeypatch.setattr(doc_mode, "DEFAULT_WARNING_THRESHOLD", -1.0)

    cache = "cache"
    images = "images"
    name = f"imcache.{image_format}"
    subdir = Path(name).stem
    cache_parent = pytester.path / cache
    cache_parent = cache_parent / subdir if nested_subdir else cache_parent
    red_filename = make_cached_images(cache_parent, subdir, name=f"im1.{image_format}", color="red")
    blue_filename = make_cached_images(cache_parent, subdir, name=f"im2.{image_format}", color="blue")
    build_filename = make_cached_images(pytester.path, images, name=name, color=build_color)

    args = ["--doc_mode", "--doc_images_dir", images, "--image_cache_dir", cache, "--image_format", image_format]
    failed = "failed"
    if failed_image_dir:
        args.extend(["--failed_image_dir", failed])
    result = pytester.runpytest(*args)
    assert result.ret == return_code

    partial_match = r"imcache Exceeded image regression error of 500\.0 with an image error equal to: [0-9]+\.[0-9]+"
    rel_subdirs = Path(subdir) / subdir if nested_subdir else subdir

    if build_color == ALMOST_RED:
        # Comparison with first image succeeds without issue
        result.stdout.no_re_match_line(rf".*UserWarning: {partial_match}")
        result.stdout.no_re_match_line(r".*This test has multiple cached images.*")

        # Test no images are saved
        assert not Path(failed).is_dir()

    elif build_color == ALMOST_BLUE:
        # Comparison with first image errors, but the second image is the closest
        # match and is only above the warning threshold
        result.stdout.re_match_lines(
            [
                r".*UserWarning: imcache Exceeded image regression warning of -1\.0 with an image error of [0-9]+\.[0-9]+",
                r".*This test has multiple cached images. The closest match was:",
                f".*im2.{image_format}",
            ]
        )
        # Test failed images are saved
        from_cache = Path(failed) / "errors_as_warnings" / "from_cache" / rel_subdirs / blue_filename.name
        assert from_cache.is_file() == failed_image_dir
        if failed_image_dir:
            assert not file_has_changed(str(from_cache), str(blue_filename))

        from_test = Path(failed, "errors_as_warnings", "from_test", build_filename.name)
        assert from_test.is_file() == failed_image_dir
        if failed_image_dir:
            assert file_has_changed(str(from_test), str(from_cache))

    else:  # 'green'
        # Comparison with all cached images fails
        result.stdout.re_match_lines(
            [
                rf".*Failed: {partial_match}",
                r".*This test has multiple cached images. The closest match was:",
                ".*" + re.escape(str(Path("cache/imcache"))),
            ]
        )

        # Test failed images are saved
        # Expect both red and blue cached images saved
        for filename in [blue_filename, red_filename]:
            from_cache = Path(failed) / "errors" / "from_cache" / rel_subdirs / filename.name
            assert from_cache.is_file() == failed_image_dir
            if failed_image_dir:
                assert not file_has_changed(str(from_cache), str(filename))

        from_test = Path(failed, "errors", "from_test", build_filename.name)
        assert from_test.is_file() == failed_image_dir
        if failed_image_dir:
            assert file_has_changed(str(from_test), str(from_cache))


@pytest.mark.parametrize("error_value", [500.0, 100000.0], ids=["initially_errors", "initially_warns"])
def test_multiple_cache_images_clean_match(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, error_value: float) -> None:
    """
    Test that a clean match with any cached image passes silently.

    The first cached image is always compared first and does not match here. Whether
    that mismatch initially exceeds the error threshold or only the warning threshold,
    the second cached image is a clean match and so the test must pass without
    emitting a warning.
    """
    monkeypatch.setattr(doc_mode, "DEFAULT_ERROR_THRESHOLD", error_value)

    cache = "cache"
    images = "images"
    name = "imcache.png"
    subdir = Path(name).stem
    cache_parent = pytester.path / cache
    make_cached_images(cache_parent, subdir, name="im1.png", color="red")
    make_cached_images(cache_parent, subdir, name="im2.png", color="blue")
    make_cached_images(pytester.path, images, name=name, color=ALMOST_BLUE)

    failed = "failed"
    args = ["--doc_mode", "--doc_images_dir", images, "--image_cache_dir", cache, "--failed_image_dir", failed]
    result = pytester.runpytest(*args)

    assert result.ret == pytest.ExitCode.OK
    result.stdout.no_re_match_line(r".*imcache Exceeded image regression.*")
    result.stdout.no_re_match_line(r".*This test has multiple cached images.*")

    # Test no images are saved
    assert not Path(failed).is_dir()


@pytest.mark.parametrize("include_vtksz", [True, False])
def test_multiple_cache_images_parallel(pytester: pytest.Pytester, include_vtksz) -> None:
    """Ensure that doc_mode works with multiple workers."""
    cache = "cache"
    images = "images"

    n_images = 50
    name_cache = "imcache{index}_vtksz.png" if include_vtksz else "imcache{index}.png"
    make_multiple_cached_images(pytester.path, cache, n_images=n_images, name=name_cache)
    name_build = "imcache{index}.vtksz" if include_vtksz else "imcache{index}.png"
    image_filenames = make_multiple_cached_images(pytester.path, images, n_images=n_images, name=name_build)

    args = ["--doc_mode", "--doc_images_dir", images, "--image_cache_dir", cache, "-n2", "-v"]
    if include_vtksz:
        # The vtksz files have no static image to take a size from
        args.extend(["--include_vtksz", "--vtksz_window_size", "1024,768"])
    result = pytester.runpytest(*args)
    assert result.ret == pytest.ExitCode.OK

    preprocessing = "Preprocessing"
    if include_vtksz:
        preprocessing_msg = f"[pyvista] {preprocessing} {n_images} vtksz files. This may take several minutes..."
        result.stdout.fnmatch_lines(preprocessing_msg)
    else:
        assert preprocessing not in result.stdout.str()

    # replace a single image with a different image
    img_idx = 34
    pl = pv.Plotter()
    pl.add_mesh(pv.Cube())
    if include_vtksz:
        pl.export_vtksz(image_filenames[img_idx])
    else:
        pl.screenshot(image_filenames[img_idx])

    result = pytester.runpytest(*args)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    failed_test_name = f"imcache{img_idx}{'_vtksz' if include_vtksz else ''}"
    assert f"{failed_test_name} Exceeded image regression error" in str(result.stdout)


@pytest.mark.parametrize("use_doc_prefix", [True, False])
@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize(("generate_subdirs", "include_vtksz"), [(True, True), (False, False)])
def test_ini(*, pytester: pytest.Pytester, cli: bool, generate_subdirs: bool, include_vtksz: bool, use_doc_prefix: bool) -> None:  # noqa: PLR0915
    """Test regular usage of the --doc_mode."""
    cache = "cache"
    cache_ini = cache + "ini"
    cache_cli = cache + "cli"

    images = "images"
    images_ini = images + "ini"
    images_cli = images + "cli"

    image_format_ini = "jpg"
    image_format_cli = "png"

    max_image_size_ini = 200
    max_image_size_cli = 101

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

    max_vtksz_file_size_cli = 10
    max_vtksz_file_size_ini = 20

    prefix = "doc_" if use_doc_prefix else ""
    pytester.makeini(
        f"""
        [pytest]
        {prefix}image_format = {image_format_ini}
        {prefix}max_image_size = {max_image_size_ini}
        {prefix}failed_image_dir = {failed_ini}
        {prefix}generated_image_dir = {generated_ini}
        {prefix}image_cache_dir = {cache_ini}
        doc_images_dir = {images_ini}
        {prefix}generate_subdirs = {generate_subdirs}
        max_vtksz_file_size = {max_vtksz_file_size_ini}
        include_vtksz = {include_vtksz}
        """
    )

    args = ["--doc_mode"]
    if cli:
        args.extend(
            [
                "--doc_images_dir",
                images_cli,
                "--image_cache_dir",
                cache_cli,
                "--failed_image_dir",
                failed_cli,
                "--generated_image_dir",
                generated_cli,
                "--image_format",
                image_format_cli,
                "--max_image_size",
                max_image_size_cli,
                "--max_vtksz_file_size",
                max_vtksz_file_size_cli,
            ]
        )
        if generate_subdirs:
            args.append("--generate_subdirs")
        if include_vtksz:
            args.append("--include_vtksz")

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

    generated_image_path = Path(generated_cli if cli else generated_ini) / name
    assert generated_image_path.is_dir() is generate_subdirs
    file = (
        next(generated_image_path.iterdir())
        if generated_image_path.is_dir()
        else generated_image_path.with_suffix(f".{image_format_cli if cli else image_format_ini}")
    )
    assert file.is_file()

    expected_max_image_size = max_image_size_cli if cli else max_image_size_ini
    assert max(pv.read(file).dimensions) == expected_max_image_size

    paths_cli = _get_file_paths(pytester.path, ext=image_format_cli)
    paths_ini = _get_file_paths(pytester.path, ext=image_format_ini)

    # Expect five images: 1 in cache dir, 1 in images dir, 1 in generated dir, 2 in failed dir
    num_files = 5
    assert len(paths_cli) == (num_files if cli else 0)
    assert len(paths_ini) == (0 if cli else num_files)

    assert _DocVerifyImageCache.include_vtksz == include_vtksz

    expected_max_size = max_vtksz_file_size_cli if cli else max_vtksz_file_size_ini
    assert _VtkszFileSizeTestCase._max_vtksz_file_size == expected_max_size  # noqa: SLF001


def test_customizing_tests(pytester: pytest.Pytester) -> None:
    """Test that individual test cases can be customized."""
    cache = "cache"
    images = "images"
    name = "imcache.png"
    make_cached_images(pytester.path, cache, name=name, color="blue")
    make_cached_images(pytester.path, images, name=name, color="red")

    custom_string = "custom_string"
    pytester.makeconftest(
        f"""
        def pytest_pyvista_doc_mode_hook(doc_verify_image_cache, request):
            if doc_verify_image_cache.test_name == {Path(name).stem!r}:
                doc_verify_image_cache.env_info = {custom_string!r}
            return doc_verify_image_cache
    """
    )
    generated = "generated"
    failed = "failed"
    result = pytester.runpytest(
        "--doc_mode",
        "--doc_images_dir",
        images,
        "--image_cache_dir",
        cache,
        "--generated_image_dir",
        generated,
        "--failed_image_dir",
        failed,
        "--generate_subdirs",
    )
    result.assert_outcomes(failed=1)

    expected_relpath = Path(Path(name).stem) / f"{custom_string}{Path(name).suffix}"
    assert Path(generated).is_dir()
    expected_file = Path(generated) / expected_relpath
    assert expected_file.is_file()

    assert Path(failed).is_dir()
    expected_file = Path(failed) / "errors" / "from_test" / expected_relpath
    assert expected_file.is_file()


def test_vtksz_screenshot(tmp_path) -> None:
    """Test converting vtksz file to image screenshot."""
    name = "im.vtksz"
    vtksz_file = make_cached_images(tmp_path, name=name)
    html_files = _vtksz_to_html_files([vtksz_file], tmp_path)
    png_files = _html_screenshots(html_files, tmp_path)
    png_file = png_files[0]
    assert png_file.suffix == ".png"

    expected_screenshot = make_cached_images(tmp_path, name=Path(name).with_suffix(".png"), window_size=pv.global_theme.window_size)
    small_error = 90
    actual_error = pv.compare_images(str(expected_screenshot), str(png_file))
    assert actual_error < small_error


@pytest.mark.parametrize("max_image_size", [400, None])
@pytest.mark.parametrize("include_vtksz", [True, False])
def test_include_vtksz(pytester: pytest.Pytester, include_vtksz, max_image_size) -> None:
    """Test that test images are generated from interactive plot files."""
    # Capture logs for testing since logger output is not captured by pytester
    captured_logs = []

    class ListHandler(logging.Handler):
        def emit(self, record):  # noqa: ANN202
            captured_logs.append(record.getMessage())

    handler = ListHandler()
    logger = logging.getLogger("pytest-pyvista")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    stem = "im"
    name_vtksz = f"{stem}.vtksz"
    name_generated = stem + "_vtksz.png"
    cache = "cache"
    images = "images"

    # Make a vtksz file along with a corresponding static image (emulates the plot directive)
    make_cached_images(pytester.path, path=images, name=f"{stem}.png", color="blue")
    make_cached_images(pytester.path, path=images, name=name_vtksz, color="blue")

    # Make a cached image to match the generated image size
    cached_window_size = (max_image_size, int(max_image_size * 3 / 4)) if max_image_size else None
    make_cached_images(pytester.path, path=cache, name=name_generated, color="red", window_size=cached_window_size)

    generated = "generated"
    failed = "failed"
    args = [
        "--doc_mode",
        "--doc_images_dir",
        images,
        "--image_cache_dir",
        cache,
        "--generated_image_dir",
        generated,
        "--failed_image_dir",
        failed,
        "-v",
    ]
    if include_vtksz:
        args.append("--include_vtksz")
    if max_image_size:
        args.extend(["--max_image_size", max_image_size])
    result = pytester.runpytest(*args)

    preprocessing = "Preprocessing"
    expected_logs = [f"Converting {name_vtksz}", f"Rendering {stem}.html"]
    if not include_vtksz:
        result.assert_outcomes(failed=2)
        result.stdout.fnmatch_lines("E           Failed: Test setup failed for test image:")
        result.stdout.fnmatch_lines(f"E           	{Path(name_generated).stem}")
        result.stdout.fnmatch_lines("E           The image exists in the cache directory:")
        result.stdout.fnmatch_lines(f"E           	*cache/{name_generated}")
        result.stdout.fnmatch_lines("E           but is missing from the docs build directory:")
        assert preprocessing not in result.stdout.str()
        assert captured_logs == []
        return

    result.assert_outcomes(failed=2)
    result.stdout.fnmatch_lines(f"E           Failed: {stem}_vtksz Exceeded image regression error*")
    preprocessing_msg = f"[pyvista] {preprocessing} 1 vtksz files. This may take several minutes..."
    result.stdout.fnmatch_lines(preprocessing_msg)
    assert captured_logs == expected_logs

    assert Path(generated).is_dir()
    expected_file = Path(generated) / name_generated
    assert expected_file.is_file()
    expected_max_size = max_image_size or 1024
    actual_max_size = max(pv.read(expected_file).dimensions)
    assert actual_max_size == expected_max_size

    assert Path(failed).is_dir()
    expected_file = Path(failed) / "errors" / "from_test" / name_generated
    assert expected_file.is_file()


@pytest.mark.parametrize("max_size", [1, None, "custom"])
def test_max_vtksz_file_size(pytester: pytest.Pytester, max_size: int | None) -> None:
    """Test --max_vtksz_file_size option."""
    name_vtksz = "im.vtksz"
    cache = "cache"
    Path(cache).mkdir()
    images = "images"
    complex_mesh = pv.Sphere(theta_resolution=100, phi_resolution=1000)
    make_cached_images(pytester.path, path=images, name=name_vtksz, color="blue", mesh=complex_mesh)

    args = [
        "--doc_mode",
        "--doc_images_dir",
        images,
        "--image_cache_dir",
        cache,
        "-v",
    ]
    if max_size == "custom":
        max_size = 2
        pytester.makeconftest(
            f"""
            def pytest_pyvista_max_vtksz_file_size_hook(test_case, request):
                if test_case.test_name == {Path(name_vtksz).stem!r}:
                    test_case.max_vtksz_file_size = {max_size}
                return test_case
        """
        )
    if max_size:
        args.extend(["--max_vtksz_file_size", max_size])
    result = pytester.runpytest(*args)
    if max_size is None:
        result.assert_outcomes(skipped=0, passed=0, failed=0)
        return

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines("E           Failed: The interactive plot file is too large:")
    result.stdout.fnmatch_lines(f"E           	*images/{name_vtksz}")
    result.stdout.fnmatch_lines(f"E           Its size is 2.4 MB, but must be less than {max_size} MB.")
    result.stdout.fnmatch_lines("E           Consider reducing the complexity of the plot or forcing it to be static.")


@pytest.fixture
def gallery_vtksz(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """Make a vtksz file in a sub-dir with its static image in the root dir."""
    images = tmp_path / "images"
    (images / "sub").mkdir(parents=True)
    vtksz_file = make_cached_images(images, path="sub", name="im.vtksz")
    monkeypatch.setattr(_DocVerifyImageCache, "doc_images_dir", images, raising=False)
    monkeypatch.setattr(_DocVerifyImageCache, "vtksz_window_size", None)
    return images, vtksz_file


def test_vtksz_window_size_from_gif(gallery_vtksz) -> None:
    """Test that a gallery vtksz file resolves its size from a GIF in the root dir."""
    images, vtksz_file = gallery_vtksz
    size = (321, 234)
    Image.new("RGB", size).save(images / "im.gif")

    assert _vtksz_window_sizes([vtksz_file]) == [size]


def test_vtksz_window_size_without_static_image(gallery_vtksz) -> None:
    """Test that a vtksz file with no static image raises instead of guessing a size."""
    _images, vtksz_file = gallery_vtksz
    match = "Interactive plot found without a corresponding static image"
    with pytest.raises(RuntimeError, match=match):
        _vtksz_window_sizes([vtksz_file])


def test_vtksz_window_size_option(gallery_vtksz, monkeypatch) -> None:
    """Test that the option pins the size and skips the static image lookup."""
    _images, vtksz_file = gallery_vtksz
    size = (400, 300)
    monkeypatch.setattr(_DocVerifyImageCache, "vtksz_window_size", size)

    assert _vtksz_window_sizes([vtksz_file, vtksz_file]) == [size, size]


@pytest.mark.parametrize("value", ["400", "400,300,200", "400,-300", "400,0", "400,three", ""])
def test_parse_window_size_invalid(value) -> None:
    """Test that a malformed window size option is rejected."""
    with pytest.raises(ValueError, match="must be two positive integers"):
        _parse_window_size(value)


def test_parse_window_size_valid() -> None:
    """Test that surrounding whitespace in the window size option is ignored."""
    assert _parse_window_size(" 400 , 300 ") == (400, 300)


@pytest.mark.parametrize("pin_window_size", [True, False])
def test_vtksz_window_size_end_to_end(*, pytester: pytest.Pytester, pin_window_size: bool) -> None:
    """Test that the option lets a vtksz file render without a static image."""
    images = "images"
    cache = "cache"
    make_cached_images(pytester.path, path=images, name="im.vtksz", color="blue")
    make_cached_images(pytester.path, path=cache, name="im_vtksz.png", color="blue")

    generated = "generated"
    size = (400, 300)
    args = ["--doc_mode", "--doc_images_dir", images, "--image_cache_dir", cache, "--generated_image_dir", generated, "--include_vtksz"]
    if pin_window_size:
        args.extend(["--vtksz_window_size", f"{size[0]},{size[1]}"])
    result = pytester.runpytest(*args)

    if not pin_window_size:
        assert result.ret != pytest.ExitCode.OK
        result.stdout.fnmatch_lines("*Interactive plot found without a corresponding static image:*")
        return

    with Image.open(pytester.path / generated / "im_vtksz.png") as im:
        assert im.size == size
