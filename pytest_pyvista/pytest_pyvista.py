"""pytest-pyvista module."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import warnings

import pytest
import pyvista


class RegressionError(RuntimeError):
    """Error when regression does not meet the criteria."""


class RegressionFileNotFound(FileNotFoundError):  # noqa: N818
    """Error when regression file is not found."""


def pytest_addoption(parser) -> None:  # noqa: ANN001
    """Adds new flag options to the pyvista plugin."""  # noqa: D401
    group = parser.getgroup("pyvista")
    group.addoption(
        "--reset_image_cache",
        action="store_true",
        help="Reset the images in the PyVista cache.",
    )
    group.addoption("--ignore_image_cache", action="store_true", help="Ignores the image cache.")
    group.addoption(
        "--allow_unused_generated",
        action="store_true",
        help="Prevent test failure if a generated test image has no use.",
    )
    group.addoption(
        "--generated_image_dir",
        action="store",
        help="Path to dump test images from the current run.",
    )
    parser.addini(
        "generated_image_dir",
        default=None,
        help="Path to dump test images from the current run.",
    )
    group.addoption(
        "--add_missing_images",
        action="store_true",
        help="Adds images to cache if missing.",
    )
    group.addoption(
        "--image_cache_dir",
        action="store",
        help="Path to the image cache folder.",
    )
    parser.addini(
        "image_cache_dir",
        default="image_cache_dir",
        help="Path to the image cache folder.",
    )
    group.addoption(
        "--reset_only_failed",
        action="store_true",
        help="Reset only the failed images in the PyVista cache.",
    )


class VerifyImageCache:
    """
    Control image caching for testing.

    Image cache files are named according to ``test_name``.
    Multiple calls to an instance of this class will append
    `_X` to the name after the first one.  That is, files
    ``{test_name}``, ``{test_name}_1``, and ``{test_name}_2``
    will be saved if called 3 times.

    Parameters
    ----------
    test_name : str
        Name of test to save.  It is used to define the name of image cache
        file.

    cache_dir : str
        Directory for image cache comparisons.

    error_value : float, default: 500
        Threshold value for determining if two images are not similar enough in
        a test.

    warning_value : float, default: 200
        Threshold value to warn that two images are different but not enough to
        fail the test.

    var_error_value : float, default: 1000
        Same as error_value but for high variance tests.

    var_warning_value : float, default 1000
        Same as warning_value but for high variance tests.

    generated_image_dir : str, optional
        Directory to save generated images.  If not specified, no generated
        images are saved.

    Examples
    --------
    Create an image cache directory named image_cache and check a simple
    plotter against it. Since ``image_cache`` doesn't exist, it will be created
    and basic.png will be added to it. Subsequent calls to ``verif`` will
    compare the plotter against the cached image.

    >>> import pyvista as pv
    >>> from pytest_pyvista import VerifyImageCache
    >>> pl = pv.Plotter(off_screen=True)
    >>> pl.add_mesh(pv.Sphere())
    >>> verif = VerifyImageCache("test_basic", "image_cache")
    >>> verif(pl)

    """

    reset_image_cache = False
    ignore_image_cache = False
    allow_unused_generated = False
    add_missing_images = False
    reset_only_failed = False

    def __init__(  # noqa: D107, PLR0913
        self,
        test_name,  # noqa: ANN001
        cache_dir,  # noqa: ANN001
        *,
        error_value=500.0,  # noqa: ANN001
        warning_value=200.0,  # noqa: ANN001
        var_error_value=1000.0,  # noqa: ANN001
        var_warning_value=1000.0,  # noqa: ANN001
        generated_image_dir=None,  # noqa: ANN001
    ) -> None:
        self.test_name = test_name

        self.cache_dir = cache_dir

        if not os.path.isdir(self.cache_dir):  # noqa: PTH112
            warnings.warn(f"pyvista test cache image dir: {self.cache_dir} does not yet exist'  Creating empty cache.")  # noqa: B028
            os.mkdir(self.cache_dir)  # noqa: PTH102

        self.error_value = error_value
        self.warning_value = warning_value
        self.var_error_value = var_error_value
        self.var_warning_value = var_warning_value

        self.generated_image_dir = generated_image_dir
        if self.generated_image_dir is not None and not os.path.isdir(self.generated_image_dir):  # noqa: PTH112
            warnings.warn(f"pyvista test generated image dir: {self.generated_image_dir} does not yet exist.  Creating dir.")  # noqa: B028
            os.makedirs(self.generated_image_dir)  # noqa: PTH103

        self.high_variance_test = False
        self.windows_skip_image_cache = False
        self.macos_skip_image_cache = False

        self.skip = False
        self.n_calls = 0

    def __call__(self, plotter):  # noqa: ANN001, ANN204, C901, PLR0912
        """
        Either store or validate an image.

        Parameters
        ----------
        plotter : pyvista.Plotter
            The Plotter object that is being closed.

        """
        if self.skip:
            return

        if self.ignore_image_cache:
            return

        test_name = f"{self.test_name}_{self.n_calls}" if self.n_calls > 0 else self.test_name
        self.n_calls += 1

        if self.high_variance_test:
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
        image_name = test_name[5:] + ".png"
        image_filename = os.path.join(self.cache_dir, image_name)  # noqa: PTH118

        overwrite_cache = (self.add_missing_images or self.reset_image_cache) and not self.reset_only_failed
        if not Path(image_filename).isfiles() or overwrite_cache:
            if self.allow_unused_generated or overwrite_cache:
                plotter.screenshot(image_filename)
            else:
                # Make sure this doesn't get called again if this plotter doesn't close properly
                plotter._before_close_callback = None  # noqa: SLF001
                msg = f"{image_filename} does not exist in image cache"
                raise RegressionFileNotFound(msg)

        if self.generated_image_dir is not None:
            gen_image_filename = os.path.join(self.generated_image_dir, test_name[5:] + ".png")  # noqa: PTH118
            plotter.screenshot(gen_image_filename)

        error = pyvista.compare_images(image_filename, plotter)

        if error > allowed_error:
            if self.reset_only_failed:
                warnings.warn(  # noqa: B028
                    f"{test_name} Exceeded image regression error of "
                    f"{allowed_error} with an image error equal to: {error}"
                    f"\nThis image will be reset in the cache."
                )
                plotter.screenshot(image_filename)
            else:
                # Make sure this doesn't get called again if this plotter doesn't close properly
                plotter._before_close_callback = None  # noqa: SLF001
                msg = f"{test_name} Exceeded image regression error of {allowed_error} with an image error equal to: {error}"
                raise RegressionError(msg)
        if error > allowed_warning:
            warnings.warn(f"{test_name} Exceeded image regression warning of {allowed_warning} with an image error of {error}")  # noqa: B028


@pytest.fixture
def verify_image_cache(request, pytestconfig):  # noqa: ANN001, ANN201
    """Checks cached images against test images for PyVista."""  # noqa: D401
    # Set CMD options in class attributes
    VerifyImageCache.reset_image_cache = pytestconfig.getoption("reset_image_cache")
    VerifyImageCache.ignore_image_cache = pytestconfig.getoption("ignore_image_cache")
    VerifyImageCache.allow_unused_generated = pytestconfig.getoption("allow_unused_generated")
    VerifyImageCache.add_missing_images = pytestconfig.getoption("add_missing_images")
    VerifyImageCache.reset_only_failed = pytestconfig.getoption("reset_only_failed")

    cache_dir = pytestconfig.getoption("image_cache_dir")
    if cache_dir is None:
        cache_dir = pytestconfig.getini("image_cache_dir")

    gen_dir = pytestconfig.getoption("generated_image_dir")
    if gen_dir is None:
        gen_dir = pytestconfig.getini("generated_image_dir")

    verify_image_cache = VerifyImageCache(request.node.name, cache_dir, generated_image_dir=gen_dir)
    pyvista.global_theme.before_close_callback = verify_image_cache

    def reset() -> None:
        pyvista.global_theme.before_close_callback = None

    request.addfinalizer(reset)  # noqa: PT021
    return verify_image_cache
