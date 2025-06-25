from __future__ import annotations  # noqa: D100

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
            assert verify_image_cache.allow_unused_generated == False

        """
    )
    result = testdir.runpytest("--reset_image_cache", "--ignore_image_cache")
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

    result = testdir.runpytest()
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
    result = testdir.runpytest()
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
    result = testdir.runpytest("--reset_image_cache")
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

    result = testdir.runpytest()
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

    result = testdir.runpytest()
    result.stdout.fnmatch_lines("*RegressionFileNotFound*")
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
        result.stdout.fnmatch_lines("*UserWarning: pyvista test failed image dir: failed_image_dir does not yet exist.  Creating dir.")
        if make_cache:
            result.stdout.fnmatch_lines(f"*Exceeded image regression {outcome}*")
        else:
            result.stdout.fnmatch_lines("*RegressionFileNotFound*")

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
