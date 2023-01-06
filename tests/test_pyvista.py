# -*- coding: utf-8 -*-
import os
import pytest

import pyvista as pv

pv.OFF_SCREEN = True
skip_vtk8 = pytest.mark.skipif(pv.vtk_version_info < (9,), reason="vtk8 not supported")
skip_vtk9 = pytest.mark.skipif(pv.vtk_version_info >= (9,), reason="vtk8 only test")


@skip_vtk8
def test_arguments(testdir):
    """Test pytest arguments"""
    testdir.makepyfile(
        """
        def test_args(verify_image_cache):
            assert verify_image_cache.reset_image_cache
            assert verify_image_cache.ignore_image_cache
            assert verify_image_cache.fail_extra_image_cache

        """
    )
    result = testdir.runpytest(
        "--reset_image_cache", "--ignore_image_cache", "--fail_extra_image_cache"
    )
    result.stdout.fnmatch_lines("*[Pp]assed*")


@skip_vtk8
def make_cached_images(test_path, path="image_cache_dir", name="imcache.png"):
    """Makes image cache in `test_path\path`."""
    d = os.path.join(test_path, path)
    if not os.path.isdir(d):
        os.mkdir(d)
    sphere = pv.Sphere()
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, color="red")
    plotter.screenshot(os.path.join(d, name))


@skip_vtk8
def test_verify_image_cache(testdir):
    """Test regular usage of the `verify_image_cache` fixture"""
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


@skip_vtk8
def test_verify_image_cache_fail_regression(testdir):
   """Test regression of the `verify_image_cache` fixture"""
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


@skip_vtk8
def test_skip(testdir):
    """Test `skip` flag of `verify_image_cache`"""
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


@skip_vtk8
def test_image_cache_dir_commandline(testdir):
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


@skip_vtk8
def test_image_cache_dir_ini(testdir):
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


@skip_vtk8
def test_high_variance_test(testdir):
    """Test `skip` flag of `verify_image_cache`"""
    make_cached_images(testdir.tmpdir)
    make_cached_images(testdir.tmpdir, name="imcache_var.png")

    # First make sure test fails with image regression error
    testdir.makepyfile(test_file1=
        """
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
    testdir.makepyfile(test_file2=
        """
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


@skip_vtk8
def test_generated_image_dir_commandline(testdir):
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
    assert os.path.isdir(os.path.join(testdir.tmpdir, "gen_dir"))
    assert os.path.isfile(os.path.join(testdir.tmpdir, "gen_dir", "imcache.png"))
    result.stdout.fnmatch_lines("*[Pp]assed*")


@skip_vtk8
def test_gladd_missing_images_commandline(testdir):
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
    assert os.path.isfile(os.path.join(testdir.tmpdir, "image_cache_dir", "imcache.png"))
    result.stdout.fnmatch_lines("*[Pp]assed*")


@skip_vtk9
def test_skip_vtk8_commandline(testdir):
    """Test skip vtk8 via CLI option."""
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
    result.stdout.fnmatch_lines("*[Ff]ailed*")
    result.stdout.fnmatch_lines("*Image cache is only valid for VTK9+*")

    result = testdir.runpytest("--fail_extra_image_cache", "--skip_image_cache_vtk8")
    result.stdout.fnmatch_lines("*[Pp]assed*")
