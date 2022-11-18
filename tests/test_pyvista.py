# -*- coding: utf-8 -*-


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


def test_verify_image_cache(testdir):
    """Test regular usage of the `verify_image_cache` fixture"""
    testdir.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            sphere = pv.Sphere()
            pv.plot(sphere.points)
            plotter = pv.Plotter()
            plotter.add_points(sphere.points)
            plotter.add_points(sphere.points + 1)
            plotter.show()
        """
    )

    result = testdir.runpytest()
    result.stdout.fnmatch_lines("*[Pp]assed*")


def test_skip(testdir):
    """Test `skip` flag of `verify_image_cache`"""
    testdir.makepyfile(
        """
        import pyvista as pv
        pv.OFF_SCREEN = True
        def test_imcache(verify_image_cache):
            verify_image_cache.skip = True
            sphere = pv.Sphere()
            pv.plot(sphere.points)
            plotter = pv.Plotter()
            plotter.add_points(sphere.points)
            plotter.add_points(sphere.points + 1)
            plotter.show()
        """
    )

    result = testdir.runpytest()
    result.stdout.fnmatch_lines("*[Pp]assed*")
