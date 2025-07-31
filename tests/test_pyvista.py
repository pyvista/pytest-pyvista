"""Tests for pytest-pyvista."""

from __future__ import annotations

from enum import Enum
import filecmp
from pathlib import Path
import platform
import sys
from unittest import mock

import pytest
import pyvista as pv

pv.OFF_SCREEN = True


def test_arguments(testdir) -> None:
    """Test pytest arguments."""
    testdir.makepyfile(
        """
        def test_args(verify_image_cache):
            assert verify_image_cache.reset_image_cache
            assert verify_image_cache.ignore_image_cache
            assert verify_image_cache.allow_unused_generated == False

        """
    )
    result = testdir.runpytest("--reset_image_cache", "--ignore_image_cache", "--disallow_unused_cache")
    result.stdout.fnmatch_lines("*[Pp]assed*")


def make_cached_images(test_path, path="image_cache_dir", name="imcache.png", color="red") -> Path:
    """Make image cache in `test_path/path`."""
    d = Path(test_path, path)
    d.mkdir(exist_ok=True)
    sphere = pv.Sphere()
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, color=color)
    filename = d / name
    plotter.screenshot(filename)
    return filename


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
def test_verify_image_cache(testdir, plot_property: bool) -> None:  # noqa: FBT001
    """Test regular usage of the `verify_image_cache` fixture."""
    make_cached_images(testdir.tmpdir)

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

    testdir.makepyfile(pyfile)

    result = testdir.runpytest()
    result.stdout.fnmatch_lines("*[Pp]assed*")

    assert (testdir.tmpdir / "image_cache_dir").isdir()
    assert not (testdir.tmpdir / "generated_image_dir").isdir()
    assert not (testdir.tmpdir / "failed_image_dir").isdir()


def test_verify_image_cache_fail_regression(testdir) -> None:
    """Test regression of the `verify_image_cache` fixture."""
    make_cached_images(testdir.tmpdir)
    testdir.makepyfile(
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

    result = testdir.runpytest()
    result.stdout.fnmatch_lines("*[Ff]ailed*")
    result.stdout.fnmatch_lines("*Exceeded image regression error*")
    result.stdout.fnmatch_lines("*pytest_pyvista.pytest_pyvista.RegressionError:*")
    result.stdout.fnmatch_lines("*Exceeded image regression error of*")


@pytest.mark.parametrize("use_generated_image_dir", [True, False])
@pytest.mark.parametrize("allow_unused_generated", [True, False])
def test_allow_unused_generated(testdir, allow_unused_generated, use_generated_image_dir) -> None:
    """Test using `--allow_unused_generated` CLI option."""
    testdir.makepyfile(
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

    result = testdir.runpytest(*args)
    result.stdout.fnmatch_lines(match)
    assert result.ret == exit_code

    assert (testdir.tmpdir / "gen_dir" / "imcache.png").isfile() == use_generated_image_dir


@pytest.mark.parametrize("mock_platform_system", ["Darwin", None])
@pytest.mark.parametrize("skip_type", ["skip", "ignore_image_cache", "macos_skip_image_cache"])
def test_skip(testdir, skip_type: str, mock_platform_system: str) -> None:
    """Test all skip flags of `verify_image_cache`."""
    if mock_platform_system:
        # Simulate test for macOS
        patcher = mock.patch("platform.system", return_value=mock_platform_system)
        with patcher:
            _run_skip_test(testdir, skip_type)
    else:
        _run_skip_test(testdir, skip_type)


def _run_skip_test(testdir, skip_type: str) -> None:
    make_cached_images(testdir.tmpdir)
    testdir.makepyfile(
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

    result = testdir.runpytest()
    # Expect failure if verification is not skipped
    match = "*RegressionError*" if skip_type == "macos_skip_image_cache" and platform.system() != "Darwin" else "*[Pp]assed*"
    result.stdout.fnmatch_lines(match)


def test_image_cache_dir_commandline(testdir) -> None:
    """Test setting image_cache_dir via CLI option."""
    make_cached_images(testdir.tmpdir, "newdir")
    testdir.makepyfile(
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

    result = testdir.runpytest("--image_cache_dir", "newdir")
    result.stdout.fnmatch_lines("*[Pp]assed*")


def test_image_cache_dir_ini(testdir) -> None:
    """Test setting image_cache_dir via config."""
    make_cached_images(testdir.tmpdir, "newdir")
    testdir.makepyfile(
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
    testdir.makepyprojecttoml(
        """
        [tool.pytest.ini_options]
        image_cache_dir = "newdir"
        """
    )
    result = testdir.runpytest()
    result.stdout.fnmatch_lines("*[Pp]assed*")


def test_high_variance_test(testdir) -> None:
    """Test `skip` flag of `verify_image_cache`."""
    make_cached_images(testdir.tmpdir)
    make_cached_images(testdir.tmpdir, name="imcache_var.png")

    # First make sure test fails with image regression error
    testdir.makepyfile(
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
    testdir.makepyfile(
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
    result = testdir.runpytest("test_file1.py")
    result.stdout.fnmatch_lines("*[Ff]ailed*")
    result.stdout.fnmatch_lines("*Exceeded image regression error*")

    result = testdir.runpytest("test_file2.py")
    result.stdout.fnmatch_lines("*[Pp]assed*")


def test_generated_image_dir_commandline(testdir) -> None:
    """Test setting generated_image_dir via CLI option."""
    make_cached_images(testdir.tmpdir)
    testdir.makepyfile(
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

    result = testdir.runpytest("--generated_image_dir", "gen_dir")
    assert (testdir.tmpdir / "gen_dir").isdir()
    assert (testdir.tmpdir / "gen_dir" / "imcache.png").isfile()
    result.stdout.fnmatch_lines("*[Pp]assed*")


def test_generated_image_dir_ini(testdir) -> None:
    """Test setting generated_image_dir via config."""
    make_cached_images(testdir.tmpdir)
    testdir.makepyfile(
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
    testdir.makepyprojecttoml(
        """
        [tool.pytest.ini_options]
        generated_image_dir = "gen_dir"
        """
    )
    result = testdir.runpytest()
    assert (testdir.tmpdir / "gen_dir").isdir()
    assert (testdir.tmpdir / "gen_dir" / "imcache.png").isfile()
    result.stdout.fnmatch_lines("*[Pp]assed*")


@pytest.mark.parametrize("reset_only_failed", [True, False])
@pytest.mark.parametrize("force_regression_error", [True, False])
@pytest.mark.parametrize("add_second_test", [True, False])
def test_add_missing_images_commandline(tmp_path, testdir, reset_only_failed, force_regression_error, add_second_test) -> None:
    """Test setting add_missing_images via CLI option."""
    if force_regression_error:
        # Make a cached image (which has a red sphere) but specify a blue sphere in the test file
        # to generate a regression failure
        make_cached_images(testdir.tmpdir)
        color = "blue"
    else:
        color = "red"

    if add_second_test:
        second_color = "lime"
        assert second_color != color
        always_passes_filename = make_cached_images(testdir.tmpdir, name="always_passes.png", color=second_color)
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

    testdir.makepyfile(
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
    result = testdir.runpytest(*args)

    if force_regression_error and not reset_only_failed:
        result.stdout.fnmatch_lines("*RegressionError*")
        assert result.ret == pytest.ExitCode.TESTS_FAILED
    else:
        expected_file = testdir.tmpdir / "image_cache_dir" / "imcache.png"
        assert expected_file.isfile()
        result.stdout.fnmatch_lines("*[Pp]assed*")
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
def test_reset_image_cache(testdir, allow_unused_generated, make_cache) -> None:
    """Test reset_image_cache  via CLI option."""
    dirname = "image_cache_dir"
    test_image_name = "imcache.png"
    filename_test = testdir.tmpdir / dirname / test_image_name
    filename_original = make_cached_images(testdir.tmpdir, dirname, name="original.png")
    if make_cache:
        filename = make_cached_images(testdir.tmpdir)
        assert filecmp.cmp(filename, filename_original, shallow=False)
    else:
        filename = filename_test
        assert not filename_test.isfile()

    testdir.makepyfile(
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
    result = testdir.runpytest(*args)
    # file was created or overwritten
    assert not filecmp.cmp(filename, filename_original, shallow=False)
    # should pass even if image doesn't match
    result.stdout.fnmatch_lines("*[Pp]assed*")


def test_cleanup(testdir) -> None:
    """Test cleanup of the `verify_image_cache` fixture."""
    make_cached_images(testdir.tmpdir)
    testdir.makepyfile(
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

    result = testdir.runpytest()
    result.stdout.fnmatch_lines("*[Pp]assed*")


@pytest.mark.parametrize("add_missing_images", [True, False])
@pytest.mark.parametrize("reset_image_cache", [True, False])
def test_reset_only_failed(testdir, reset_image_cache, add_missing_images) -> None:
    """Test usage of the `reset_only_failed` flag."""
    filename = make_cached_images(testdir.tmpdir)
    filename_original = make_cached_images(testdir.tmpdir, name="original.png")
    assert filecmp.cmp(filename, filename_original, shallow=False)

    testdir.makepyfile(
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

    result = testdir.runpytest(*args)
    result.stdout.fnmatch_lines("*[Pp]assed*")
    result.stdout.fnmatch_lines("*This image will be reset in the cache.")
    # file was overwritten
    assert not filecmp.cmp(filename, filename_original, shallow=False)


def test_callback_called(testdir: pytest.Testdir) -> None:
    """Test the callback of a Plotter.show is correctly called."""
    make_cached_images(test_path=testdir.tmpdir)
    testdir.makepyfile(
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

    result = testdir.runpytest()
    result.assert_outcomes(passed=1)


def test_file_not_found(testdir) -> None:
    """Test RegressionFileNotFoundError is correctly raised."""
    testdir.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache_num2(verify_image_cache):
            sphere = pv.Box()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="blue")
            plotter.show()
        """
    )

    result = testdir.runpytest()
    result.stdout.fnmatch_lines("*RegressionFileNotFoundError*")
    result.stdout.fnmatch_lines("*does not exist in image cache*")


@pytest.mark.parametrize(("outcome", "make_cache"), [("error", False), ("error", True), ("warning", True), ("success", True)])
def test_failed_image_dir(testdir, outcome, make_cache) -> None:
    """Test usage of the `failed_image_dir` option."""
    cached_image_name = "imcache.png"
    if make_cache:
        make_cached_images(testdir.tmpdir)

    red = [255, 0, 0]
    almost_red = [250, 0, 0]
    definitely_not_red = [0, 0, 0]
    color = definitely_not_red if outcome == "error" else almost_red if outcome == "warning" else red
    testdir.makepyfile(
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
    result = testdir.runpytest("--failed_image_dir", dirname)

    failed_image_dir_path = testdir.tmpdir / dirname
    if outcome == "success":
        assert not failed_image_dir_path.isdir()
    else:
        result.stdout.fnmatch_lines("*UserWarning: pyvista test failed image dir: *failed_image_dir does not yet exist.  Creating dir.")
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

        assert failed_image_dir_path.isdir()

        # Test that dir with failed images is only created as needed
        assert (failed_image_dir_path / expected_subdir).isdir()
        assert not (failed_image_dir_path / not_expected_subdir).isdir()

        from_test_dir = failed_image_dir_path / expected_subdir / "from_test"
        assert from_test_dir.isdir()
        assert (from_test_dir / cached_image_name).isfile()

        from_cache_dir = failed_image_dir_path / expected_subdir / "from_cache"
        if make_cache:
            assert from_cache_dir.isdir()
            assert (from_cache_dir / cached_image_name).isfile()
        else:
            assert not from_cache_dir.isdir()
            assert not (from_cache_dir / cached_image_name).isfile()


@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("call_show", [True, False])
@pytest.mark.parametrize("allow_useless_fixture_cli", [True, False])
@pytest.mark.parametrize("allow_useless_fixture_attr", [True, False, None])
def test_allow_useless_fixture(testdir, call_show, allow_useless_fixture_cli, allow_useless_fixture_attr, skip) -> None:
    """Test error is raised if fixture is used but no images are generated."""
    if call_show:
        # Ensure there is a cached image to compare to the generated image
        make_cached_images(testdir.tmpdir)

    allow_attr = "" if allow_useless_fixture_attr is None else f"verify_image_cache.allow_useless_fixture = {allow_useless_fixture_attr}"
    skip_attr = f"verify_image_cache.skip = {skip}"
    testdir.makepyfile(
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

    result = testdir.runpytest("--allow_useless_fixture") if allow_useless_fixture_cli else testdir.runpytest()

    # Expect local attr to take precedence over CLI value
    allow_useless_fixture = allow_useless_fixture_attr if allow_useless_fixture_attr is not None else allow_useless_fixture_cli
    expect_failure = (not call_show and not allow_useless_fixture) and not skip
    expected_code = pytest.ExitCode.TESTS_FAILED if expect_failure else pytest.ExitCode.OK
    assert result.ret == expected_code
    result.stdout.fnmatch_lines("*[Pp]assed*")
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
def test_disallow_unused_cache(testdir, marker, skip_verify, color, stdout_lines, exit_code, has_unused_cache) -> None:  # noqa: PLR0913
    """Ensure unused cached images are detected correctly."""
    test_name = "foo"
    image_name = test_name + ".png"
    image_cache_dir = "image_cache_dir"

    make_cached_images(testdir.tmpdir, image_cache_dir, image_name)
    if has_unused_cache:
        make_cached_images(testdir.tmpdir)

    testdir.makepyfile(
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

    result = testdir.runpytest("--disallow_unused_cache")

    assert result.ret == exit_code
    result.stdout.fnmatch_lines(stdout_lines)


@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("args", ["--disallow_unused_cache", []])
def test_disallow_unused_cache_skip_multiple_images(testdir, skip, args) -> None:
    """Test skips when there are multiple calls to show() in a test."""
    make_cached_images(testdir.tmpdir, name="imcache.png")
    make_cached_images(testdir.tmpdir, name="imcache_1.png")

    marker = "@pytest.mark.skip" if skip else ""
    testdir.makepyfile(
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

    result = testdir.runpytest(args)
    expected = "*skipped*" if skip else "*[Pp]assed*"
    result.stdout.fnmatch_lines(expected)
    assert result.ret == pytest.ExitCode.OK


@pytest.mark.parametrize("disallow_unused_cache", [True, False])
def test_disallow_unused_cache_name_mismatch(testdir, disallow_unused_cache) -> None:
    """Test cached image doesn't match test name."""
    image_name = "im_cache.png"
    make_cached_images(testdir.tmpdir, name=image_name)
    make_cached_images(testdir.tmpdir)

    testdir.makepyfile(
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
    result = testdir.runpytest(args)
    if disallow_unused_cache:
        result.stdout.fnmatch_lines([*_unused_cache_lines(image_name)])
        assert result.ret == pytest.ExitCode.TESTS_FAILED
    else:
        result.stdout.fnmatch_lines("*[Pp]assed*")
        assert result.ret == pytest.ExitCode.OK


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Needs contextlib.chdir")
def test_cache_generated_dir_relative(testdir: pytest.Testdir) -> None:
    """
    Test that directories (cache and generated) are relative to test root
    even when changing the working directory when calling Plotter.show().
    """  # noqa: D205
    make_cached_images(testdir.tmpdir, path=(new_dir := "new_dir"))

    testdir.makepyfile(
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
    result = testdir.runpytest(*args)
    result.assert_outcomes(passed=1)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Needs contextlib.chdir")
def test_failed_dir_relative(testdir: pytest.Testdir) -> None:
    """
    Test that failed directory is relative to test root
    even when changing the working directory when calling Plotter.show().
    """  # noqa: D205
    make_cached_images(testdir.tmpdir, path=(new_dir := "new_dir"))

    testdir.makepyfile(
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
    result = testdir.runpytest(*args)
    result.assert_outcomes(passed=1)
