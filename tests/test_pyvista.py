# -*- coding: utf-8 -*-
import os
import pytest

import pyvista as pv

pv.OFF_SCREEN = True

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


def make_cached_images(test_path, path="image_cache_dir"):
    """Makes image cache in `test_path\path`."""
    d = os.path.join(test_path, path)
    os.mkdir(d)
    sphere = pv.Sphere()
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, color="red")
    plotter.screenshot(os.path.join(d, "imcache.png"))


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
