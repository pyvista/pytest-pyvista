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
        "--reset_image_cache",
        "--ignore_image_cache",
        "--fail_extra_image_cache",
    )
    result.stdout.fnmatch_lines("*Passed*")


def test_verify_image_cache(testdir):
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
    print(result.stdout)
    result.stdout.fnmatch_lines("*Passed*")


def test_skip(testdir):
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
    print(result.stdout)
    result.stdout.fnmatch_lines("*Passed*")
