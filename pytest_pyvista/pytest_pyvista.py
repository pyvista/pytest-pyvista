# -*- coding: utf-8 -*-


import os
import pathlib
import platform
import warnings

import pytest
import pyvista
from pyvista._vtk import VTK9


def pytest_addoption(parser):
    """Adds new flag options to the pyvista plugin."""

    group = parser.getgroup("pyvista")
    group.addoption(
        "--reset_image_cache",
        action="store_true",
        help="Reset the images in the PyVista cache.",
    )
    group.addoption(
        "--ignore_image_cache", action="store_true", help="Ignores the image cache."
    )
    group.addoption(
        "--fail_extra_image_cache",
        action="store_true",
        help="Enables failure if image cache does not exist.",
    )
    group.addoption(
        "--image_cache_dir",
        action="store",
        help="Path to the image cache folder.",
    )


class VerifyImageCache:
    """Control image caching for testing.

    Image cache files are named according to ``test_name``.
    Multiple calls to an instance of this class will append
    `_X` to the name after the first one.  That is, files
    ``{test_name}``, ``{test_name}_1``, and ``{test_name}_2``
    will be saved if called 3 times.

    Parameters
    ----------
    test_name : str
        Name of test to save.  It is used to define the name of image cache file.

    error_value : int
        Threshold value for determining if two images are not similar enough in a test.

    warning_value : int
        Threshold value to warn that two images are different but not enough to fail the test.

    var_error_value : int
        Same as error_value but for high variance tests.

    var_warning_value : int
        same as warning_value but for high variance tests.
    """

    reset_image_cache = False
    ignore_image_cache = False
    fail_extra_image_cache = False

    high_variance_tests = False
    windows_skip_image_cache = False
    macos_skip_image_cache = False

    cache_dir = None

    def __init__(
        self,
        test_name,
        *,
        error_value=500,
        warning_value=200,
        var_error_value=1000,
        var_warning_value=1000,
    ):
        self.test_name = test_name

        if self.cache_dir is None:
            # Reset image cache with new images
            this_path = pathlib.Path(__file__).parent.absolute()
            self.cache_dir = os.path.join(this_path, "image_cache")

            # not working with pytest-pyvista tests for some reason
            # self.cache_dir = appdirs.user_cache_dir(appname="pytest-pyvista", appauthor="pyvista")

        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        self.error_value = error_value
        self.warning_value = warning_value
        self.var_error_value = var_error_value
        self.var_warning_value = var_warning_value

        self.skip = False
        self.n_calls = 0

    def __call__(self, plotter):
        """Either store or validate an image.

        Parameters
        ----------
        plotter : pyvista.Plotter
            The Plotter object that is being closed.

        """
        if self.skip:
            return

        # Image cache is only valid for VTK9+
        if not VTK9:
            return

        if self.ignore_image_cache:
            return

        if self.n_calls > 0:
            test_name = f"{self.test_name}_{self.n_calls}"
        else:
            test_name = self.test_name
        self.n_calls += 1

        if self.high_variance_tests:
            allowed_error = self.var_error_value
            allowed_warning = self.var_warning_value
        else:
            allowed_error = self.error_value
            allowed_warning = self.warning_value

        # some tests fail when on Windows with OSMesa
        if os.name == "nt" and self.windows_skip_image_cache:
            return
        # high variation for MacOS
        if platform.system() == "Darwin" and self.macos_skip_image_cache:
            return

        # cached image name. We remove the first 5 characters of the function name
        # "test_" to get the name for the image.
        image_filename = os.path.join(self.cache_dir, test_name[5:] + ".png")

        if not os.path.isfile(image_filename) and self.fail_extra_image_cache:
            raise RuntimeError(f"{image_filename} does not exist in image cache")
        # simply save the last screenshot if it doesn't exist or the cache
        # is being reset.
        if self.reset_image_cache or not os.path.isfile(image_filename):
            return plotter.screenshot(image_filename)

        # otherwise, compare with the existing cached image
        error = pyvista.compare_images(image_filename, plotter)
        if error > allowed_error:
            raise RuntimeError(
                f"{test_name} Exceeded image regression error of "
                f"{allowed_error} with an image error equal to: {error}"
            )
        if error > allowed_warning:
            warnings.warn(
                f"{test_name} Exceeded image regression warning of "
                f"{allowed_warning} with an image error of "
                f"{error}"
            )


@pytest.fixture()
def verify_image_cache(request, pytestconfig):
    """Checks cached images against test images for PyVista."""

    # Set CMD options in class attributes
    VerifyImageCache.reset_image_cache = pytestconfig.getoption("reset_image_cache")
    VerifyImageCache.ignore_image_cache = pytestconfig.getoption("ignore_image_cache")
    VerifyImageCache.fail_extra_image_cache = pytestconfig.getoption(
        "fail_extra_image_cache"
    )
    VerifyImageCache.cache_dir = pytestconfig.getoption("image_cache_dir")

    verify_image_cache = VerifyImageCache(request.node.name)
    pyvista.global_theme.before_close_callback = verify_image_cache
    return verify_image_cache
