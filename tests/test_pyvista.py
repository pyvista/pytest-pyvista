"""Tests for pytest-pyvista."""

from __future__ import annotations

from enum import Enum
import filecmp
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from typing import TYPE_CHECKING
from unittest import mock

import matplotlib.pyplot as plt
import pytest
import pyvista as pv

from pytest_pyvista.doc_mode import _preprocess_build_images
from pytest_pyvista.pytest_pyvista import _DOC_MODE_CLI_ARGS
from pytest_pyvista.pytest_pyvista import _SYSTEM_PROPERTIES
from pytest_pyvista.pytest_pyvista import _UNIT_TEST_CLI_ARGS
from pytest_pyvista.pytest_pyvista import PARSER_GROUP_NAME
from pytest_pyvista.pytest_pyvista import _EnvInfo
from pytest_pyvista.pytest_pyvista import _SystemProperties
from pytest_pyvista.pytest_pyvista import pytest_addoption

if TYPE_CHECKING:
    from pytest_mock import MockerFixture
pv.OFF_SCREEN = True

pytest_plugins = "pytester"


def test_arguments(pytester: pytest.Pytester) -> None:
    """Test pytest arguments."""
    pytester.makepyfile(
        """
        def test_args(verify_image_cache):
            assert verify_image_cache.reset_image_cache
            assert verify_image_cache.ignore_image_cache
            assert verify_image_cache.allow_unused_generated == False

        """
    )
    result = pytester.runpytest("--reset_image_cache", "--ignore_image_cache", "--disallow_unused_cache")
    result.assert_outcomes(passed=1)


def make_cached_images(  # noqa: PLR0913
    test_path, path="image_cache_dir", name="imcache.png", color="red", window_size=None, mesh: pv.DataSet | None = None
) -> Path:
    """Make image cache in `test_path/path`."""
    d = Path(test_path, path)
    d.mkdir(exist_ok=True, parents=True)
    if mesh is None:
        mesh = pv.Sphere()
    kwargs = {"window_size": window_size} if window_size else {}
    plotter = pv.Plotter(**kwargs)
    plotter.add_mesh(mesh, color=color)
    filename = d / name
    if filename.suffix == ".vtksz":
        plotter.export_vtksz(filename)
    else:
        plotter.screenshot(filename)
    return filename


def make_multiple_cached_images(test_path, path="image_cache_dir", n_images: int = 10, name: str = "imcache{index}.png") -> list[Path]:
    """Make image cache in `test_path/path` consisting of several images."""
    colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    d = Path(test_path, path)
    d.mkdir(exist_ok=True, parents=True)

    color_to_file = {}
    mesh = pv.Sphere()

    filenames = []
    for ii in range(n_images):
        color = colors[ii % len(colors)]
        filename = d / name.format(index=ii)

        if color in color_to_file:
            # don't regenerate images when that color already exists
            shutil.copy(color_to_file[color], filename)
        else:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(mesh, color=color)
            if filename.suffix == ".vtksz":
                plotter.export_vtksz(filename)
            else:
                plotter.screenshot(filename)
            color_to_file[color] = filename

        filenames.append(filename)

    return filenames


def get_path_inode(path: str | Path) -> int:
    """Return the inode for the given path."""
    return Path(path).stat().st_ino


def file_has_changed(filepath: str, original_contents_path: str | None = None, original_inode: int | None = None) -> bool:
    """
    Return True if a file has changed.

    Specify `original_contents_path` to check if the contents of `filepath` match the contents
    of `original_contents_path`.

    Specify `original_inode` to check if the inode of `filepath` matches `original_inode`.

    Specify both `original_contents_path` and `original_inode` to check both.
    """
    assert original_contents_path or original_inode, "Original contents filepath or original inode must be specified"
    content_changed = False
    if original_contents_path:
        content_changed = not filecmp.cmp(filepath, original_contents_path, shallow=False)

    replaced = False
    if original_inode:
        new_inode = get_path_inode(filepath)
        replaced = new_inode != original_inode

    return content_changed or replaced


@pytest.mark.parametrize("plot_property", [True, False])
def test_verify_image_cache(pytester: pytest.Pytester, plot_property: bool) -> None:  # noqa: FBT001
    """Test regular usage of the `verify_image_cache` fixture."""
    make_cached_images(pytester.path)

    if not plot_property:
        pyfile = """
            import pyvista as pv
            pv.OFF_SCREEN = True
            def test_imcache(verify_image_cache):
                sphere = pv.Sphere()
                plotter = pv.Plotter()
                plotter.add_mesh(sphere, color="red")
                plotter.show()
            """
    else:
        pyfile = """
            import pyvista as pv
            pv.OFF_SCREEN = True
            def test_imcache(verify_image_cache):
                pv.Sphere().plot(color="red")
            """

    pytester.makepyfile(pyfile)

    result = pytester.runpytest()
    result.assert_outcomes(passed=1)

    assert (pytester.path / "image_cache_dir").is_dir()
    assert not (pytester.path / "generated_image_dir").is_dir()
    assert not (pytester.path / "failed_image_dir").is_dir()


def test_verify_image_cache_fail_regression(pytester: pytest.Pytester) -> None:
    """Test regression of the `verify_image_cache` fixture."""
    make_cached_images(pytester.path)
    pytester.makepyfile(
        """
       import pytest
       import pyvista as pv
       pv.OFF_SCREEN = True
       def test_imcache(verify_image_cache):
           sphere = pv.Sphere()
           plotter = pv.Plotter()
           plotter.add_mesh(sphere, color="blue")
           plotter.show()
       """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines("*Exceeded image regression error*")
    result.stdout.fnmatch_lines("*pytest_pyvista.pytest_pyvista.RegressionError:*")
    result.stdout.fnmatch_lines("*Exceeded image regression error of*")


@pytest.mark.parametrize("use_generated_image_dir", [True, False])
@pytest.mark.parametrize("allow_unused_generated", [True, False])
def test_allow_unused_generated(pytester: pytest.Pytester, allow_unused_generated, use_generated_image_dir) -> None:
    """Test using `--allow_unused_generated` CLI option."""
    pytester.makepyfile(
        """
       import pytest
       import pyvista as pv
       pv.OFF_SCREEN = True
       def test_imcache(verify_image_cache):
           sphere = pv.Sphere()
           plotter = pv.Plotter()
           plotter.add_mesh(sphere, color="red")
           plotter.show()
       """
    )
    if allow_unused_generated:
        args = ["--allow_unused_generated"]
        exit_code = pytest.ExitCode.OK
        match = "*[Pp]assed*"
    else:
        args = []
        exit_code = pytest.ExitCode.TESTS_FAILED
        match = "*RegressionFileNotFoundError*"

    if use_generated_image_dir:
        args.extend(["--generated_image_dir", "gen_dir"])

    result = pytester.runpytest(*args)
    result.stdout.fnmatch_lines(match)
    assert result.ret == exit_code

    assert (pytester.path / "gen_dir" / "imcache.png").is_file() == use_generated_image_dir


@pytest.mark.parametrize("mock_platform_system", ["Darwin", None])
@pytest.mark.parametrize("skip_type", ["skip", "ignore_image_cache", "macos_skip_image_cache"])
def test_skip(pytester: pytest.Pytester, skip_type: str, mock_platform_system: str) -> None:
    """Test all skip flags of `verify_image_cache`."""
    if mock_platform_system:
        # Simulate test for macOS
        patcher = mock.patch("platform.system", return_value=mock_platform_system)
        with patcher:
            _run_skip_test(pytester, skip_type)
    else:
        _run_skip_test(pytester, skip_type)


def _run_skip_test(pytester: pytest.Pytester, skip_type: str) -> None:
    make_cached_images(pytester.path)
    pytester.makepyfile(
        f"""
        import pytest
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            verify_image_cache.{skip_type} = True
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="blue")
            plotter.show()
         """
    )

    result = pytester.runpytest()
    # Expect failure if verification is not skipped
    match = "*RegressionError*" if skip_type == "macos_skip_image_cache" and platform.system() != "Darwin" else "*[Pp]assed*"
    result.stdout.fnmatch_lines(match)


def test_image_cache_dir_commandline(pytester: pytest.Pytester) -> None:
    """Test setting image_cache_dir via CLI option."""
    make_cached_images(pytester.path, "newdir")
    pytester.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="red")
            plotter.show()
        """
    )

    result = pytester.runpytest("--image_cache_dir", "newdir")
    result.assert_outcomes(passed=1)


def test_image_cache_dir_ini(pytester: pytest.Pytester) -> None:
    """Test setting image_cache_dir via config."""
    make_cached_images(pytester.path, "newdir")
    pytester.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="red")
            plotter.show()
        """
    )
    pytester.makepyprojecttoml(
        """
        [tool.pytest.ini_options]
        image_cache_dir = "newdir"
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_high_variance_test(pytester: pytest.Pytester) -> None:
    """Test `skip` flag of `verify_image_cache`."""
    make_cached_images(pytester.path)
    make_cached_images(pytester.path, name="imcache_var.png")

    # First make sure test fails with image regression error
    pytester.makepyfile(
        test_file1="""
        import pytest
        import pyvista as pv
        pv.OFF_SCREEN = True

        def test_imcache(verify_image_cache):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color=[255, 5, 5])
            plotter.show()
        """
    )
    # Next mark as a high_variance_test and check that it passes
    pytester.makepyfile(
        test_file2="""
        import pytest
        import pyvista as pv
        pv.OFF_SCREEN = True

        def test_imcache_var(verify_image_cache):
            verify_image_cache.high_variance_test = True
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color=[255, 5, 5])
            plotter.show()
        """
    )
    result = pytester.runpytest("test_file1.py")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines("*Exceeded image regression error*")

    result = pytester.runpytest("test_file2.py")
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize(
    "env_info",
    [
        "prefix_suffix",
        _EnvInfo(prefix="prefix", os=False, machine=False, gpu=False, python=False, pyvista=False, vtk=False, ci=False, suffix="suffix"),
    ],
)
@pytest.mark.parametrize("generate_subdirs", [True, False])
def test_generated_image_dir_commandline(pytester: pytest.Pytester, generate_subdirs, env_info) -> None:
    """Test setting generated_image_dir via CLI option."""
    make_cached_images(pytester.path)
    pytester.makepyfile(
        f"""
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            verify_image_cache.env_info = '{env_info}'
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="red")
            plotter.show()
        """
    )
    args = ["--generated_image_dir", "gen_dir"]
    if generate_subdirs:
        args.append("--generate_subdirs")

    result = pytester.runpytest(*args)
    assert (pytester.path / "gen_dir").is_dir()
    if generate_subdirs:
        subdir = pytester.path / "gen_dir" / "imcache"
        assert subdir.is_dir()
        paths = list(subdir.iterdir())
        assert len(paths) == 1
        image_path = paths[0]
        assert image_path.name == "prefix_suffix.png"
    else:
        assert (pytester.path / "gen_dir" / "imcache.png").is_file()
    result.assert_outcomes(passed=1)


def test_generated_image_dir_ini(pytester: pytest.Pytester) -> None:
    """Test setting generated_image_dir via config."""
    make_cached_images(pytester.path)
    pytester.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="red")
            plotter.show()
        """
    )
    pytester.makepyprojecttoml(
        """
        [tool.pytest.ini_options]
        generated_image_dir = "gen_dir"
        """
    )
    result = pytester.runpytest()
    assert (pytester.path / "gen_dir").is_dir()
    assert (pytester.path / "gen_dir" / "imcache.png").is_file()
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize("reset_only_failed", [True, False])
@pytest.mark.parametrize("force_regression_error", [True, False])
@pytest.mark.parametrize("add_second_test", [True, False])
def test_add_missing_images_commandline(tmp_path, pytester: pytest.Pytester, reset_only_failed, force_regression_error, add_second_test) -> None:
    """Test setting add_missing_images via CLI option."""
    if force_regression_error:
        # Make a cached image (which has a red sphere) but specify a blue sphere in the test file
        # to generate a regression failure
        make_cached_images(pytester.path)
        color = "blue"
    else:
        color = "red"

    if add_second_test:
        second_color = "lime"
        assert second_color != color
        always_passes_filename = make_cached_images(pytester.path, name="always_passes.png", color=second_color)
        always_passes_ground_truth = make_cached_images(tmp_path, name="always_passes.png", color=second_color)
        always_passes_inode = get_path_inode(always_passes_filename)
        second_test = f"""
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_always_passes(verify_image_cache):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color={second_color!r})
            plotter.show()
        """
    else:
        second_test = ""

    pytester.makepyfile(
        f"""
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color={color!r})
            plotter.show()
        {second_test}
        """
    )
    args = ["--add_missing_images"]
    if reset_only_failed:
        args.append("--reset_only_failed")
    result = pytester.runpytest(*args)

    if force_regression_error and not reset_only_failed:
        result.stdout.fnmatch_lines("*RegressionError*")
        assert result.ret == pytest.ExitCode.TESTS_FAILED
    else:
        expected_file = pytester.path / "image_cache_dir" / "imcache.png"
        assert expected_file.is_file()
        result.assert_outcomes(passed=2 if add_second_test else 1)
        assert result.ret == pytest.ExitCode.OK

        # Make sure the final image in the cache matches the generated test image
        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere(), color=color)
        assert pv.compare_images(pl, str(expected_file)) == 0.0

    if add_second_test:
        # Make sure second test image was not modified
        assert not file_has_changed(always_passes_filename, original_contents_path=always_passes_ground_truth, original_inode=always_passes_inode)


@pytest.mark.parametrize("allow_unused_generated", [True, False])
@pytest.mark.parametrize("make_cache", [True, False])
def test_reset_image_cache(pytester: pytest.Pytester, allow_unused_generated, make_cache) -> None:
    """Test reset_image_cache  via CLI option."""
    dirname = "image_cache_dir"
    test_image_name = "imcache.png"
    filename_test = pytester.path / dirname / test_image_name
    filename_original = make_cached_images(pytester.path, dirname, name="original.png")
    if make_cache:
        filename = make_cached_images(pytester.path)
        assert filecmp.cmp(filename, filename_original, shallow=False)
    else:
        filename = filename_test
        assert not filename_test.is_file()

    pytester.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="blue")
            plotter.show()
        """
    )
    args = ["--reset_image_cache"]
    if allow_unused_generated:
        args.append("--allow_unused_generated")
    result = pytester.runpytest(*args)
    # file was created or overwritten
    assert not filecmp.cmp(filename, filename_original, shallow=False)
    # should pass even if image doesn't match
    result.assert_outcomes(passed=1)


def test_cleanup(pytester: pytest.Pytester) -> None:
    """Test cleanup of the `verify_image_cache` fixture."""
    make_cached_images(pytester.path)
    pytester.makepyfile(
        """
       import pytest
       import pyvista as pv
       pv.OFF_SCREEN = True

       @pytest.fixture()
       def cleanup_tester():
           yield
           assert pv.global_theme.before_close_callback is None

       def test_imcache(cleanup_tester, verify_image_cache):
           sphere = pv.Sphere()
           plotter = pv.Plotter()
           plotter.add_mesh(sphere, color="blue")
           try:
               plotter.show()
           except RuntimeError:
               # continue so the cleanup is tested
               pass
       """
    )

    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize("add_missing_images", [True, False])
@pytest.mark.parametrize("reset_image_cache", [True, False])
def test_reset_only_failed(pytester: pytest.Pytester, reset_image_cache, add_missing_images) -> None:
    """Test usage of the `reset_only_failed` flag."""
    filename = make_cached_images(pytester.path)
    filename_original = make_cached_images(pytester.path, name="original.png")
    assert filecmp.cmp(filename, filename_original, shallow=False)

    pytester.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            sphere = pv.Box()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="blue")
            plotter.show()
        """
    )

    args = ["--reset_only_failed"]
    if add_missing_images:
        args.append("--add_missing_images")
    if reset_image_cache:
        args.append("--reset_image_cache")

    result = pytester.runpytest(*args)
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines("*This image will be reset in the cache.")
    # file was overwritten
    assert not filecmp.cmp(filename, filename_original, shallow=False)


def test_callback_called(pytester: pytest.Pytester) -> None:
    """Test the callback of a Plotter.show is correctly called."""
    make_cached_images(test_path=pytester.path)
    pytester.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache, mocker):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="red")
            plotter.show(before_close_callback=(m:=mocker.MagicMock()))

            m.assert_called_once_with(plotter)
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize("make_subdir", [True, False])
def test_file_not_found(pytester: pytest.Pytester, make_subdir) -> None:
    """Test RegressionFileNotFoundError is correctly raised."""
    test_name = "imcache_num2"
    pytester.makepyfile(
        f"""
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_{test_name}(verify_image_cache):
            sphere = pv.Box()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="blue")
            plotter.show()
        """
    )
    if make_subdir:
        # Test case where no images exist in a subdir
        Path("image_cache_dir", test_name).mkdir(parents=True)

    result = pytester.runpytest()
    result.stdout.fnmatch_lines("*RegressionFileNotFoundError*")
    result.stdout.fnmatch_lines("*does not exist in image cache*")


@pytest.mark.parametrize("image_format", ["png", "jpg"])
@pytest.mark.parametrize(("outcome", "make_cache"), [("error", False), ("error", True), ("warning", True), ("success", True)])
def test_failed_image_dir(pytester: pytest.Pytester, outcome, make_cache, image_format) -> None:
    """Test usage of the `failed_image_dir` option."""
    cached_image_name = f"imcache.{image_format}"
    if make_cache:
        make_cached_images(pytester.path, name=cached_image_name)

    red = [255, 0, 0]
    almost_red = [250, 0, 0]
    definitely_not_red = [0, 0, 0]
    color = definitely_not_red if outcome == "error" else almost_red if outcome == "warning" else red
    pytester.makepyfile(
        f"""
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color={color})
            plotter.show()
        """
    )
    dirname = "failed_image_dir"
    result = pytester.runpytest("--failed_image_dir", dirname, "--image_format", image_format)

    failed_image_dir_path = pytester.path / dirname
    if outcome == "success":
        assert not failed_image_dir_path.is_dir()
    else:
        if make_cache:
            result.stdout.fnmatch_lines(f"*Exceeded image regression {outcome}*")
        else:
            result.stdout.fnmatch_lines("*RegressionFileNotFoundError*")

        if outcome == "error":
            expected_subdir = "errors"
            not_expected_subdir = "warnings"
        else:
            expected_subdir = "warnings"
            not_expected_subdir = "errors"

        assert failed_image_dir_path.is_dir()

        # Test that dir with failed images is only created as needed
        assert (failed_image_dir_path / expected_subdir).is_dir()
        assert not (failed_image_dir_path / not_expected_subdir).is_dir()

        from_test_dir = failed_image_dir_path / expected_subdir / "from_test"
        assert from_test_dir.is_dir()
        file = from_test_dir / cached_image_name
        assert file.is_file()
        assert file.name.endswith(image_format)

        from_cache_dir = failed_image_dir_path / expected_subdir / "from_cache"
        if make_cache:
            assert from_cache_dir.is_dir()
            assert (from_cache_dir / cached_image_name).is_file()
        else:
            assert not from_cache_dir.is_dir()
            assert not (from_cache_dir / cached_image_name).is_file()


@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("call_show", [True, False])
@pytest.mark.parametrize("allow_useless_fixture_cli", [True, False])
@pytest.mark.parametrize("allow_useless_fixture_attr", [True, False, None])
def test_allow_useless_fixture(pytester: pytest.Pytester, call_show, allow_useless_fixture_cli, allow_useless_fixture_attr, skip) -> None:
    """Test error is raised if fixture is used but no images are generated."""
    if call_show:
        # Ensure there is a cached image to compare to the generated image
        make_cached_images(pytester.path)

    allow_attr = "" if allow_useless_fixture_attr is None else f"verify_image_cache.allow_useless_fixture = {allow_useless_fixture_attr}"
    skip_attr = f"verify_image_cache.skip = {skip}"
    pytester.makepyfile(
        f"""
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            {allow_attr}
            {skip_attr}
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="red")
            {"plotter.show()" if call_show else ""}
        """
    )

    result = pytester.runpytest("--allow_useless_fixture") if allow_useless_fixture_cli else pytester.runpytest()

    # Expect local attr to take precedence over CLI value
    allow_useless_fixture = allow_useless_fixture_attr if allow_useless_fixture_attr is not None else allow_useless_fixture_cli
    expect_failure = (not call_show and not allow_useless_fixture) and not skip
    expected_code = pytest.ExitCode.TESTS_FAILED if expect_failure else pytest.ExitCode.OK
    assert result.ret == expected_code
    result.assert_outcomes(passed=1, errors=1 if expect_failure else 0)
    if expect_failure:
        result.stdout.fnmatch_lines(
            [
                "*ERROR at teardown of test_imcache*",
                "*Failed: Fixture `verify_image_cache` is used but no images were generated.",
                "*Did you forget to call `show` or `plot`, or set `verify_image_cache.allow_useless_fixture=True`?.",
            ]
        )
    else:
        assert "ERROR" not in result.stdout.str()


class LiteralStrEnum(str, Enum):  # noqa: D101
    def __str__(self) -> str:  # noqa: D105
        return str(self.value)


class PytestMark(LiteralStrEnum):  # noqa: D101
    NONE = ""
    SKIP = "@pytest.mark.skip"


class SkipVerify(LiteralStrEnum):  # noqa: D101
    NONE = ""
    MACOS = "verify_image_cache.macos_skip_image_cache"
    WINDOWS = "verify_image_cache.windows_skip_image_cache"
    IGNORE = "verify_image_cache.ignore_image_cache"
    SKIP = "verify_image_cache.skip"


class MeshColor(LiteralStrEnum):  # noqa: D101
    OK = "red"
    FAIL = "blue"


class HasUnusedCache(Enum):  # noqa: D101
    TRUE = True
    FALSE = False

    def __bool__(self) -> bool:  # noqa: D105
        return self.value


def _unused_cache_lines(image_name: str) -> list[str]:
    return [
        "*pytest-pyvista ERROR*",
        "Unused cached image file(s) detected (1). The following images are",
        "cached, but were not generated or skipped by any of the tests:",
        f"[{image_name!r}]",
        "These images should either be removed from the cache, or the corresponding",
        "tests should be modified to ensure an image is generated for comparison.",
    ]


@pytest.mark.parametrize(
    ("marker", "skip_verify", "color", "stdout_lines", "exit_code", "has_unused_cache"),
    [
        (PytestMark.SKIP, SkipVerify.NONE, MeshColor.OK, ["*skipped*"], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.NONE, MeshColor.OK, ["*[Pp]assed*"], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.MACOS, MeshColor.OK, ["*[Pp]assed*"], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.WINDOWS, MeshColor.OK, ["*[Pp]assed*"], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.IGNORE, MeshColor.OK, ["*[Pp]assed*"], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.SKIP, MeshColor.OK, ["*[Pp]assed*"], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.NONE, MeshColor.FAIL, ["*FAILED*"], pytest.ExitCode.TESTS_FAILED, HasUnusedCache.FALSE),
        (PytestMark.SKIP, SkipVerify.NONE, MeshColor.OK, [*_unused_cache_lines("imcache.png")], pytest.ExitCode.TESTS_FAILED, HasUnusedCache.TRUE),
        (PytestMark.NONE, SkipVerify.NONE, MeshColor.OK, [*_unused_cache_lines("imcache.png")], pytest.ExitCode.TESTS_FAILED, HasUnusedCache.TRUE),
        (PytestMark.NONE, SkipVerify.NONE, MeshColor.FAIL, [*_unused_cache_lines("imcache.png")], pytest.ExitCode.TESTS_FAILED, HasUnusedCache.TRUE),
    ],
)
def test_disallow_unused_cache(pytester: pytest.Pytester, marker, skip_verify, color, stdout_lines, exit_code, has_unused_cache) -> None:  # noqa: PLR0913
    """Ensure unused cached images are detected correctly."""
    test_name = "foo"
    image_name = test_name + ".png"
    image_cache_dir = "image_cache_dir"

    make_cached_images(pytester.path, image_cache_dir, image_name)
    if has_unused_cache:
        make_cached_images(pytester.path)

    pytester.makepyfile(
        f"""
        import pytest
        import pyvista as pv
        pv.OFF_SCREEN = True
        {marker}
        def test_{test_name}(verify_image_cache):
            {skip_verify}
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="{color}")
            plotter.show()
        """
    )

    result = pytester.runpytest("--disallow_unused_cache")

    assert result.ret == exit_code
    result.stdout.fnmatch_lines(stdout_lines)


@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("args", ["--disallow_unused_cache", []])
def test_disallow_unused_cache_skip_multiple_images(pytester: pytest.Pytester, skip, args) -> None:
    """Test skips when there are multiple calls to show() in a test."""
    make_cached_images(pytester.path, name="imcache.png")
    make_cached_images(pytester.path, name="imcache_1.png")

    marker = "@pytest.mark.skip" if skip else ""
    pytester.makepyfile(
        f"""
        import pytest
        import pyvista as pv
        pv.OFF_SCREEN = True
        {marker}
        def test_imcache(verify_image_cache):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="red")
            plotter.show()
            #
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="red")
            plotter.show()
        """
    )

    result = pytester.runpytest(args)
    expected = "*skipped*" if skip else "*[Pp]assed*"
    result.stdout.fnmatch_lines(expected)
    assert result.ret == pytest.ExitCode.OK


@pytest.mark.parametrize("disallow_unused_cache", [True, False])
def test_disallow_unused_cache_name_mismatch(pytester: pytest.Pytester, disallow_unused_cache) -> None:
    """Test cached image doesn't match test name."""
    image_name = "im_cache.png"
    make_cached_images(pytester.path, name=image_name)
    make_cached_images(pytester.path)

    pytester.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="red")
            plotter.show()
        """
    )
    args = "--disallow_unused_cache" if disallow_unused_cache else []
    result = pytester.runpytest(args)
    if disallow_unused_cache:
        result.stdout.fnmatch_lines([*_unused_cache_lines(image_name)])
        assert result.ret == pytest.ExitCode.TESTS_FAILED
    else:
        result.assert_outcomes(passed=1)
        assert result.ret == pytest.ExitCode.OK


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Needs contextlib.chdir")
def test_cache_generated_dir_relative(pytester: pytest.Pytester) -> None:
    """
    Test that directories (cache and generated) are relative to test root
    even when changing the working directory when calling Plotter.show().
    """  # noqa: D205
    make_cached_images(pytester.path, path=(new_dir := "new_dir"))

    pytester.makepyfile(
        """
        import pyvista as pv
        import pytest

        pv.OFF_SCREEN = True
        import contextlib
        from pathlib import Path

        def test_imcache(verify_image_cache, tmp_path: Path, pytestconfig: pytest.Config):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="red")
            with contextlib.chdir(tmp_path):
                plotter.show()

            assert (pytestconfig.rootpath / "generated/imcache.png").exists()
        """
    )
    args = ["--image_cache_dir", new_dir, "--generated_image_dir", "generated"]
    result = pytester.runpytest(*args)
    result.assert_outcomes(passed=1)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Needs contextlib.chdir")
def test_failed_dir_relative(pytester: pytest.Pytester) -> None:
    """
    Test that failed directory is relative to test root
    even when changing the working directory when calling Plotter.show().
    """  # noqa: D205
    make_cached_images(pytester.path, path=(new_dir := "new_dir"))

    pytester.makepyfile(
        """
        import pyvista as pv
        import pytest
        from pytest_pyvista.pytest_pyvista import RegressionError

        pv.OFF_SCREEN = True
        import contextlib
        from pathlib import Path

        def test_imcache(verify_image_cache, tmp_path: Path, pytestconfig: pytest.Config):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="blue")
            with contextlib.chdir(tmp_path), contextlib.suppress(RegressionError):
                plotter.show()

            assert (pytestconfig.rootpath / "failed/errors/from_test/imcache.png").exists()
        """
    )
    args = ["--image_cache_dir", new_dir, "--failed_image_dir", "failed"]
    result = pytester.runpytest(*args)
    result.assert_outcomes(passed=1)


def test_auto_close(pytester: pytest.Pytester) -> None:
    """Test when using auto_close=False."""
    pytester.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_auto_close(verify_image_cache):
            verify_image_cache.skip = True
            pl = pv.Plotter()
            pl.show(auto_close=False)
            assert pl._before_close_callback is not None
            pl.close()

        def test_auto_close_2(verify_image_cache, mocker):
            verify_image_cache.skip = True
            plotter = pv.Plotter()
            plotter.show(auto_close=False, before_close_callback=(m:=mocker.MagicMock()))
            plotter.close()
            m.assert_called_once_with(plotter)
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(passed=2)


ALMOST_BLUE = [0, 0, 254]
ALMOST_RED = [254, 0, 0]


@pytest.mark.parametrize("failed_image_dir", [True, False])
@pytest.mark.parametrize("nested_subdir", [True, False])
@pytest.mark.parametrize(
    ("build_color", "return_code"), [(ALMOST_RED, pytest.ExitCode.OK), (ALMOST_BLUE, pytest.ExitCode.OK), ("'green'", pytest.ExitCode.TESTS_FAILED)]
)
@pytest.mark.parametrize("image_format", ["png", "jpg"])
def test_multiple_cache_images(  # noqa: PLR0913
    pytester: pytest.Pytester, build_color, return_code, nested_subdir, failed_image_dir, image_format
) -> None:
    """Test when cache is a subdir with multiple images."""
    cache = "cache"
    name = f"imcache.{image_format}"
    subdir = Path(name).stem
    cache_parent = pytester.path / cache
    cache_parent = cache_parent / subdir if nested_subdir else cache_parent
    red_filename = make_cached_images(cache_parent, subdir, name=f"im1.{image_format}", color="red")
    blue_filename = make_cached_images(cache_parent, subdir, name=f"im2.{image_format}", color="blue")

    pyfile = f"""
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color={build_color})
            plotter.show()
        """
    pytester.makepyfile(pyfile)

    args = ["--image_cache_dir", cache, "--image_format", image_format]
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

        from_test = Path(failed, "errors_as_warnings", "from_test", name)
        assert from_test.is_file() == failed_image_dir
        if failed_image_dir:
            assert file_has_changed(str(from_test), str(from_cache))

    else:  # 'green'
        # Comparison with all cached images fails
        result.stdout.re_match_lines(
            [
                rf".*RegressionError: {partial_match}",
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

        from_test = Path(failed, "errors", "from_test", name)
        assert from_test.is_file() == failed_image_dir
        if failed_image_dir:
            assert file_has_changed(str(from_test), str(from_cache))


def test_env_info() -> None:
    """Test env info dataclass."""
    info = str(_EnvInfo())
    assert " " not in info
    assert info.startswith(f"{_SYSTEM_PROPERTIES.os_name}-{_SYSTEM_PROPERTIES.os_version}")
    if platform.system() == "Linux":
        if sys.version_info >= (3, 10):
            assert info.startswith("ubuntu")
        else:
            assert info.startswith("Linux")

    # Generic regex for "_package-#.#.#" with optional suffix (like .dev0, .post1, etc.)
    pattern = r"_[a-zA-Z]+-\d+\.\d+\.\d+(?:[a-zA-Z0-9\.]*)?"
    matches = re.findall(pattern, info)

    assert any(m.startswith("_py-") for m in matches)
    assert any(m.startswith("_pyvista-") for m in matches)
    assert any(m.startswith("_vtk-") for m in matches)

    assert any(f"gpu-{vendor.lower()}" in info.lower() for vendor in ["Apple", "NVIDIA", "Mesa", "AMD", "ATI"])

    if os.environ.get("CI", None):
        assert "no-CI" not in info
        assert "CI" in info
    else:
        assert "no-CI" in info


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("os", _SYSTEM_PROPERTIES.os_name),
        ("machine", platform.machine()),
        ("gpu", "gpu-"),
        ("python", "py-"),
        ("pyvista", "pyvista-"),
        ("vtk", "vtk-"),
        ("ci", "CI"),
    ],
)
def test_env_info_exclude(name: str, value: str) -> None:
    """Test removing parts of the env info."""
    # Sanity check to ensure the value is there ordinarily
    info = str(_EnvInfo())
    assert value in info

    # Test it's excluded
    info = str(_EnvInfo(**{name: False}))
    assert value not in info
    assert "__" not in info
    assert "_-" not in info
    assert "-_" not in info


def test_env_info_prefix_suffix() -> None:
    """Test env info dataclass prefix and suffix."""
    text = "foobar"
    default = str(_EnvInfo())
    sep = "_"
    assert not default.startswith(sep)
    assert not default.endswith(sep)

    with_prefix = str(_EnvInfo(prefix=text))
    assert f"{text}{sep}{default}" == with_prefix

    with_suffix = str(_EnvInfo(suffix=text))
    assert f"{default}{sep}{text}" == with_suffix


@pytest.mark.parametrize(
    ("vendor", "expected"),
    [
        ("NVIDIA Corporation", "NVIDIA"),
        ("AMD Technologies", "AMD"),
        ("ATI Graphics", "ATI"),
        ("Mesa DRI Intel", "Mesa"),
        ('Unknown\\/:*?"<>| \t\n\r\v\f\x00Vendor', "UnknownVendor"),
    ],
)
def test_gpu_vendor(mocker: MockerFixture, vendor: str, expected: str) -> None:
    """Test vendor name processing."""
    mock_gpuinfo = mocker.patch("pyvista.GPUInfo")
    mock_gpuinfo.return_value.vendor = vendor

    result = _SystemProperties._gpu_vendor()  # noqa: SLF001
    assert result == expected


def test_gpu_vendor_unknown(mocker: MockerFixture) -> None:
    """Test exception from GPUInfo is handled properly."""
    mocker.patch("pyvista.GPUInfo", side_effect=RuntimeError("fail"))
    assert _SystemProperties._gpu_vendor() == "unknown"  # noqa: SLF001


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Missing freedesktop_os_release")
def test_os_linux_freedesktop(mocker: MockerFixture) -> None:
    """Test system properties for Linux."""
    mocker.patch("platform.system", return_value="Linux")
    mocker.patch(
        "platform.freedesktop_os_release",
        return_value={
            "ID": "ubuntu",
            "VERSION_ID": "22.04",
        },
    )

    result = _SystemProperties._get_os()  # noqa: SLF001
    assert result == ("ubuntu", "22.04")


@pytest.mark.parametrize(
    ("system", "release", "expected"),
    [
        ("Darwin", "24.6.0", ("macOS", "24.6.0")),
        ("Windows", "10", ("Windows", "10")),
    ],
)
def test_os_darwin_windows(
    mocker: MockerFixture,
    system: str,
    release: str,
    expected: tuple[str, str],
) -> None:
    """Test system properties for macOS and Windows."""
    mocker.patch("platform.system", return_value=system)
    mocker.patch("platform.release", return_value=release)

    result = _SystemProperties._get_os()  # noqa: SLF001
    assert result == expected


@pytest.mark.parametrize("doc_mode", [True, False])
@pytest.mark.parametrize(("valid_format", "invalid_format"), [("png", "jpg"), ("jpg", "png")])
def test_validate_cache_image_format(*, pytester: pytest.Pytester, valid_format, invalid_format, doc_mode) -> None:
    """Test cache validation."""
    test_name = "imcache"
    name = f"{test_name}.{invalid_format}"
    cache = "image_cache_dir"
    make_cached_images(pytester.path, path=cache, name=name)
    make_cached_images(pytester.path / cache, path=test_name, name=name)

    args = ["--image_format", valid_format]
    if doc_mode:
        images = "images"
        _preprocess_build_images(Path(cache), Path(images), image_format=valid_format)
        args.extend(["--doc_mode", "--image_cache_dir", cache, "--doc_images_dir", images])
    else:
        pytester.makepyfile(
            f"""def test_{test_name}(verify_image_cache):
                    ...
            """
        )

    result = pytester.runpytest(*args)
    assert result.ret == pytest.ExitCode.TESTS_FAILED

    result.stdout.fnmatch_lines("E           pytest_pyvista.pytest_pyvista.InvalidCacheError: The image format required by")
    result.stdout.fnmatch_lines(f"E           the image cache directory is {valid_format!r}, but {invalid_format!r} images exist in the cache.")
    result.stdout.fnmatch_lines("E           Cache directory: *")
    result.stdout.fnmatch_lines(f"E           Invalid images: ['imcache/imcache.{invalid_format}', 'imcache.{invalid_format}']")


@pytest.mark.parametrize("doc_mode", [True, False])
def test_validate_cache_unique_names(*, pytester: pytest.Pytester, doc_mode: bool) -> None:
    """Test cache validation."""
    test_name = "imcache"
    name = f"{test_name}.png"
    cache = "image_cache_dir"
    make_cached_images(pytester.path, path=cache, name=name)
    make_cached_images(pytester.path / cache, path=test_name, name=name)

    if doc_mode:
        images = "images"
        _preprocess_build_images(Path(cache), Path(images))
        args = ["--doc_mode", "--image_cache_dir", cache, "--doc_images_dir", images]
    else:
        args = []
        pytester.makepyfile(
            f"""def test_{test_name}(verify_image_cache):
                    ...
            """
        )

    result = pytester.runpytest(*args)
    assert result.ret == pytest.ExitCode.TESTS_FAILED

    result.stdout.fnmatch_lines("E           pytest_pyvista.pytest_pyvista.InvalidCacheError: Non-unique image test names detected in the cache.")
    result.stdout.fnmatch_lines("E           An image's name must not share the same name as a subdirectory. Either the image")
    result.stdout.fnmatch_lines("E           or the subdirectory should be removed for the following test cases:")
    result.stdout.fnmatch_lines(f"E           {{{test_name!r}}}")


def test_cli_args_classified() -> None:
    """Ensure that all CLI options are added to either unit testing or doc mode options sets."""
    # Add all options to a parser just like pytest would
    parser = pytest.Parser()
    pytest_addoption(parser)

    # Get group options
    group = parser.getgroup(PARSER_GROUP_NAME)
    assert group is not None

    defined = {name for opt in group.options for name in opt.names()}
    classified = _UNIT_TEST_CLI_ARGS | _DOC_MODE_CLI_ARGS

    missing = defined - classified
    extra = classified - defined

    set_names = "classification sets _UNIT_TEST_CLI_ARGS, _UNIT_TEST_CLI_ARGS"
    assert not missing, f"Parser options are not included in {set_names}:\n{missing}"
    assert not extra, f"{set_names} contain unknown options:\n{extra}"


@pytest.mark.parametrize("arg", _UNIT_TEST_CLI_ARGS - _DOC_MODE_CLI_ARGS)
def test_unit_test_args_invalid_in_doc_mode(pytester, arg) -> None:
    """Run pytest with --doc_mode and forbidden args."""
    result = pytester.runpytest("--doc_mode", arg)
    result.stderr.fnmatch_lines([f"ERROR: argument {arg} cannot be used with --doc_mode enabled"])
    assert result.ret == pytest.ExitCode.USAGE_ERROR


@pytest.mark.parametrize("arg", _DOC_MODE_CLI_ARGS - _UNIT_TEST_CLI_ARGS - {"--doc_mode"})
def test_doc_mode_args_invalid_in_unit_test_mode(pytester, arg) -> None:
    """Run pytest unit tests with forbidden args."""
    args = [arg]
    if arg == "--max_vtksz_file_size":
        args.append("0")
    elif arg == "--doc_images_dir":
        args.append("foo")
    result = pytester.runpytest(*args)

    result.stderr.fnmatch_lines([f"ERROR: argument {arg} can only be used with --doc_mode enabled"])
    assert result.ret == pytest.ExitCode.USAGE_ERROR
