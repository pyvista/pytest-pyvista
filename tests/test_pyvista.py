from __future__ import annotations  # noqa: D100

from enum import Enum
import filecmp
import os

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
            assert verify_image_cache.fail_extra_image_cache

        """
    )
    result = testdir.runpytest("--reset_image_cache", "--ignore_image_cache", "--fail_extra_image_cache", "--allow_unused_cache")
    result.stdout.fnmatch_lines("*[Pp]assed*")


def make_cached_images(test_path, path="image_cache_dir", name="imcache.png"):  # noqa: ANN201
    """Makes image cache in `test_path/path`."""  # noqa: D401
    d = os.path.join(test_path, path)  # noqa: PTH118
    if not os.path.isdir(d):  # noqa: PTH112
        os.mkdir(d)  # noqa: PTH102
    sphere = pv.Sphere()
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, color="red")
    filename = os.path.join(d, name)  # noqa: PTH118
    plotter.screenshot(filename)
    return filename


def test_verify_image_cache(testdir) -> None:
    """Test regular usage of the `verify_image_cache` fixture."""
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

    result = testdir.runpytest("--fail_extra_image_cache")
    result.stdout.fnmatch_lines("*[Pp]assed*")

    assert (testdir.tmpdir / "image_cache_dir").isdir()
    assert not (testdir.tmpdir / "generated_image_dir").isdir()


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

    result = testdir.runpytest("--fail_extra_image_cache")
    result.stdout.fnmatch_lines("*[Ff]ailed*")
    result.stdout.fnmatch_lines("*Exceeded image regression error*")
    result.stdout.fnmatch_lines("*pytest_pyvista.pytest_pyvista.RegressionError:*")
    result.stdout.fnmatch_lines("*Exceeded image regression error of*")


def test_skip(testdir) -> None:
    """Test `skip` flag of `verify_image_cache`."""
    make_cached_images(testdir.tmpdir)
    testdir.makepyfile(
        """
        import pytest
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            verify_image_cache.skip = True
            sphere = pv.Sphere()
            plotter = pv.Plotter()
            plotter.add_mesh(sphere, color="blue")
            plotter.show()
         """
    )

    result = testdir.runpytest("--fail_extra_image_cache")
    result.stdout.fnmatch_lines("*[Pp]assed*")


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

    result = testdir.runpytest("--fail_extra_image_cache", "--image_cache_dir", "newdir")
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
    result = testdir.runpytest("--fail_extra_image_cache")
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
    result = testdir.runpytest("--fail_extra_image_cache", "test_file1.py")
    result.stdout.fnmatch_lines("*[Ff]ailed*")
    result.stdout.fnmatch_lines("*Exceeded image regression error*")

    result = testdir.runpytest("--fail_extra_image_cache", "test_file2.py")
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

    result = testdir.runpytest("--fail_extra_image_cache", "--generated_image_dir", "gen_dir")
    assert os.path.isdir(os.path.join(testdir.tmpdir, "gen_dir"))  # noqa: PTH112, PTH118
    assert os.path.isfile(os.path.join(testdir.tmpdir, "gen_dir", "imcache.png"))  # noqa: PTH113, PTH118
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
    result = testdir.runpytest("--fail_extra_image_cache")
    assert os.path.isdir(os.path.join(testdir.tmpdir, "gen_dir"))  # noqa: PTH112, PTH118
    assert os.path.isfile(os.path.join(testdir.tmpdir, "gen_dir", "imcache.png"))  # noqa: PTH113, PTH118
    result.stdout.fnmatch_lines("*[Pp]assed*")


def test_add_missing_images_commandline(testdir) -> None:
    """Test setting add_missing_images via CLI option."""
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

    result = testdir.runpytest("--add_missing_images")
    assert os.path.isfile(os.path.join(testdir.tmpdir, "image_cache_dir", "imcache.png"))  # noqa: PTH113, PTH118
    result.stdout.fnmatch_lines("*[Pp]assed*")


def test_reset_image_cache(testdir) -> None:
    """Test reset_image_cache  via CLI option."""
    filename = make_cached_images(testdir.tmpdir)
    filename_original = make_cached_images(testdir.tmpdir, name="original.png")
    assert filecmp.cmp(filename, filename_original, shallow=False)

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
    result = testdir.runpytest("--fail_extra_image_cache", "--reset_image_cache")
    # file was overwritten
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

    result = testdir.runpytest("--fail_extra_image_cache")
    result.stdout.fnmatch_lines("*[Pp]assed*")


def test_reset_only_failed(testdir) -> None:
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

    result = testdir.runpytest("--reset_only_failed")
    result.stdout.fnmatch_lines("*[Pp]assed*")
    result.stdout.fnmatch_lines("*This image will be reset in the cache.")
    # file was overwritten
    assert not filecmp.cmp(filename, filename_original, shallow=False)


def test_file_not_found(testdir) -> None:
    """Test RegressionFileNotFound is correctly raised."""
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

    result = testdir.runpytest("--fail_extra_image_cache")
    result.stdout.fnmatch_lines("*RegressionFileNotFound*")
    result.stdout.fnmatch_lines("*does not exist in image cache*")


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
    SUCCESS = "red"
    FAIL = "blue"


class HasUnusedCache(Enum):  # noqa: D101
    TRUE = True
    FALSE = False

    def __bool__(self) -> bool:  # noqa: D105
        return self.value


TESTS_FAILED_ERROR_LINES = [
    "pytest-pyvista: ERROR: Unused cached image file(s) detected (1).",
    "The following images were not generated or skipped by any of the tests:",
]


# fmt: off
@pytest.mark.parametrize(
    ("marker", "skip_verify", "color", "stdout_lines", "stderr_lines", "exit_code", "has_unused_cache"),
    [
        (PytestMark.SKIP, SkipVerify.NONE, MeshColor.SUCCESS, ["*skipped*"], [], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.NONE, MeshColor.SUCCESS, ["*[Pp]assed*"], [], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.MACOS, MeshColor.SUCCESS, ["*[Pp]assed*"], [], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.WINDOWS, MeshColor.SUCCESS, ["*[Pp]assed*"], [], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.IGNORE, MeshColor.SUCCESS, ["*[Pp]assed*"], [], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.SKIP, MeshColor.SUCCESS, ["*[Pp]assed*"], [], pytest.ExitCode.OK, HasUnusedCache.FALSE),
        (PytestMark.NONE, SkipVerify.NONE, MeshColor.FAIL, ["*FAILED*"], [], pytest.ExitCode.TESTS_FAILED, HasUnusedCache.FALSE),
        (PytestMark.SKIP, SkipVerify.NONE, MeshColor.SUCCESS, [], [*TESTS_FAILED_ERROR_LINES, "['imcache.png']"], pytest.ExitCode.TESTS_FAILED, HasUnusedCache.TRUE),  # noqa: E501
        (PytestMark.NONE, SkipVerify.NONE, MeshColor.SUCCESS, [], [*TESTS_FAILED_ERROR_LINES, "['imcache.png']"], pytest.ExitCode.TESTS_FAILED, HasUnusedCache.TRUE),  # noqa: E501
        (PytestMark.NONE, SkipVerify.NONE, MeshColor.FAIL, [], [*TESTS_FAILED_ERROR_LINES, "['imcache.png']"], pytest.ExitCode.TESTS_FAILED, HasUnusedCache.TRUE),  # noqa: E501
    ],
)# fmt: skip
def test_allow_unused_cache(testdir, marker, skip_verify, color, stdout_lines, stderr_lines, exit_code, has_unused_cache) -> None:  # noqa: PLR0913
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

    result = testdir.runpytest()

    assert result.ret == exit_code
    result.stderr.fnmatch_lines(stderr_lines)
    result.stdout.fnmatch_lines(stdout_lines)


@pytest.mark.parametrize("skip", [True,False])
@pytest.mark.parametrize("args", ["--allow_unused_cache", []])
def test_allow_unused_cache_skip_multiple_images(testdir, skip, args) -> None:
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


@pytest.mark.parametrize("allow_unused_cache", [True, False])
def test_allow_unused_cache_name_mismatch(testdir, allow_unused_cache) -> None:
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
    args = "--allow_unused_cache" if allow_unused_cache else []
    result = testdir.runpytest(args)
    if allow_unused_cache:
        result.stdout.fnmatch_lines("*[Pp]assed*")
        assert result.ret == pytest.ExitCode.OK
    else:
        result.stderr.fnmatch_lines([*TESTS_FAILED_ERROR_LINES, f"[{image_name!r}]"])
        assert result.ret == pytest.ExitCode.TESTS_FAILED
