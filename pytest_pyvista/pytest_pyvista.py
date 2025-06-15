"""pytest-pyvista module."""

from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import TYPE_CHECKING
from typing import NamedTuple
import warnings

import pytest
import pyvista

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Generator


class RegressionError(RuntimeError):
    """Error when regression does not meet the criteria."""


class RegressionFileNotFound(FileNotFoundError):  # noqa: N818
    """Error when regression file is not found."""


class Outcome(Enum):
    """Outcome of the image verification."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    ERROR = "error"
    WARNING = "warning"


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
        "--fail_extra_image_cache",
        action="store_true",
        help="Enables failure if image cache does not exist.",
    )
    group.addoption(
        "--generated_image_dir",
        action="store",
        help="Path to dump test images from the current run.",
    )
    parser.addini(
        "generated_image_dir",
        default="generated_image_dir",
        help="Path to dump test images from the current run.",
    )
    group.addoption(
        "--failed_image_dir",
        action="store",
        help="Path to dump images from failed tests from the current run.",
    )
    parser.addini(
        "failed_image_dir",
        default="failed_image_dir",
        help="Path to dump images from failed tests from the current run.",
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
    group.addoption(
        "--fail_unused_cache",
        action="store_true",
        help="Report test failure if there are any images in the cache which are not compared to any generated images.",
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
    fail_extra_image_cache = False
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
        if self.generated_image_dir is not None:
            _ensure_dir_exists(self.generated_image_dir, msg_name="generated image dir")

        self.high_variance_test = False
        self.windows_skip_image_cache = False
        self.macos_skip_image_cache = False

        self.skip = False
        self.n_calls = 0

    def __call__(self, plotter):  # noqa: ANN001, ANN204
        """
        Either store or validate an image.

        Parameters
        ----------
        plotter : pyvista.Plotter
            The Plotter object that is being closed.

        """
        test_name = f"{self.test_name}_{self.n_calls}" if self.n_calls > 0 else self.test_name
        self.n_calls += 1

        if self.high_variance_test:
            allowed_error = self.var_error_value
            allowed_warning = self.var_warning_value
        else:
            allowed_error = self.error_value
            allowed_warning = self.warning_value

        # cached image name. We remove the first 5 characters of the function name
        # "test_" to get the name for the image.
        image_name = _image_name_from_test_name(test_name)
        image_filename = os.path.join(self.cache_dir, image_name)  # noqa: PTH118
        gen_image_filename = None

        skip_windows = os.name == "nt" and self.windows_skip_image_cache
        skip_macos = platform.system() == "Darwin" and self.macos_skip_image_cache
        if self.skip or self.ignore_image_cache or skip_windows or skip_macos:
            # Log result as skipped
            _store_result(
                test_name=test_name,
                outcome=Outcome.SKIPPED,
                cached_filename=image_name,
                generated_filename=gen_image_filename,
            )
            return

        if not os.path.isfile(image_filename) and self.fail_extra_image_cache and not self.reset_image_cache:  # noqa: PTH113
            # Make sure this doesn't get called again if this plotter doesn't close properly
            plotter._before_close_callback = None  # noqa: SLF001
            msg = f"{image_filename} does not exist in image cache"
            raise RegressionFileNotFound(msg)

        if ((self.add_missing_images and not os.path.isfile(image_filename)) or self.reset_image_cache) and not self.reset_only_failed:  # noqa: PTH113
            plotter.screenshot(image_filename)

        if self.generated_image_dir is not None:
            gen_image_filename = os.path.join(self.generated_image_dir, test_name[5:] + ".png")  # noqa: PTH118
            plotter.screenshot(gen_image_filename)

        error = pyvista.compare_images(image_filename, plotter)

        if error > allowed_error:
            _store_result(test_name=test_name, outcome=Outcome.ERROR, cached_filename=image_filename, generated_filename=gen_image_filename)
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
            _store_result(test_name=test_name, outcome=Outcome.WARNING, cached_filename=image_filename, generated_filename=gen_image_filename)
            warnings.warn(f"{test_name} Exceeded image regression warning of {allowed_warning} with an image error of {error}")  # noqa: B028
        else:
            _store_result(test_name=test_name, outcome=Outcome.SUCCESS, cached_filename=image_filename, generated_filename=gen_image_filename)


def _ensure_dir_exists(dirpath: str, msg_name: str) -> None:
    if not Path(dirpath).is_dir():
        msg = f"pyvista test {msg_name}: {dirpath} does not yet exist.  Creating dir."
        warnings.warn(msg, stacklevel=2)
        Path(dirpath).mkdir()


def _get_dir_from_config_or_ini(pytestconfig, dirname: str) -> str:  # noqa: ANN001
    gen_dir = pytestconfig.getoption(dirname)
    if gen_dir is None:
        gen_dir = pytestconfig.getini(dirname)
    return gen_dir


def _image_name_from_test_name(test_name: str) -> str:
    return test_name[5:] + ".png"


def _test_name_from_image_name(image_name: str) -> str:
    def remove_suffix(s: str) -> str:
        """Remove integer and png suffix."""
        no_png_ext = s[:-4]
        parts = no_png_ext.split("_")
        if len(parts) > 1:
            try:
                int(parts[-1])
                parts = parts[:-1]  # Remove the integer suffix
            except ValueError:
                pass  # Last part is not an integer; do nothing
        return "_".join(parts)

    return "test_" + remove_suffix(image_name)


class _ResultTuple(NamedTuple):
    outcome: Outcome
    cached_filename: str
    generated_filename: str | None


RESULTS = {}


def _store_result(*, test_name: str, outcome: Outcome, cached_filename: str, generated_filename: str | None = None) -> None:
    result = _ResultTuple(
        outcome=outcome,
        cached_filename=str(Path(cached_filename).name),
        generated_filename=str(Path(generated_filename).name) if generated_filename else None,
    )
    RESULTS[test_name] = result


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call) -> Generator:  # noqa: ANN001, ARG001
    """Store test results for skipped tests."""
    outcome = yield
    if outcome:
        rep = outcome.get_result()

        # Log if test was skipped
        if rep.when in ["call", "setup"] and rep.skipped:
            test_name = item.name
            _store_result(
                test_name=test_name,
                outcome=Outcome.SKIPPED,
                cached_filename=_image_name_from_test_name(test_name),
                generated_filename=None,
            )


@pytest.hookimpl
def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001, ARG001
    """Execute after the whole test run completes."""
    config = session.config

    fail_unused_cache = config.getoption("fail_unused_cache")
    image_cache_dir = _get_dir_from_config_or_ini(config, "image_cache_dir")
    failed_image_dir = _get_dir_from_config_or_ini(config, "failed_image_dir")
    generated_image_dir = _get_dir_from_config_or_ini(config, "generated_image_dir")

    if image_cache_dir and fail_unused_cache:
        cache_path = Path(image_cache_dir)
        cached_files = {f.name for f in cache_path.glob("*.png")}
        tested_files = {result.cached_filename for result in RESULTS.values()}
        unused = cached_files - tested_files

        # Exclude images from skipped tests where multiple images are generated
        unused_skipped = unused.copy()
        for image_name in unused:
            test_name = _test_name_from_image_name(image_name)
            result = RESULTS.get(test_name)
            if result and result.outcome == Outcome.SKIPPED:
                unused_skipped.remove(image_name)

        if unused_skipped:
            msg = (
                f"\npytest-pyvista: ERROR: Unused cached image file(s) detected ({len(unused_skipped)}).\n"
                f"The following images were not generated or skipped by any of the tests:\n"
                f"{sorted(unused_skipped)}\n"
            )
            # Print the message so it appears in the output
            sys.stderr.write(msg)
            sys.stderr.flush()

            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    if failed_image_dir:
        for result in RESULTS.values():
            if result.outcome in [Outcome.WARNING, Outcome.ERROR]:
                cached_image_path = Path(image_cache_dir, result.cached_filename)
                if cached_image_path.is_file():
                    _ensure_dir_exists(failed_image_dir, msg_name="failed image dir")
                    _save_failed_test_image(cached_image_path, result.outcome, image_cache_dir, failed_image_dir)
                if result.generated_filename:
                    _ensure_dir_exists(failed_image_dir, msg_name="failed image dir")
                    generated_image_path = Path(generated_image_dir, result.generated_filename)
                    _save_failed_test_image(generated_image_path, result.outcome, image_cache_dir, failed_image_dir)

    RESULTS.clear()


def _save_failed_test_image(source_image_path: Path, outcome: Outcome, image_cache_dir: str, failed_image_dir: str) -> None:
    """Save test image from cache or test to the failed image dir."""
    parent_dir = Path(outcome.name.lower() + "s")
    dest_dirname = "from_cache" if Path(source_image_path).parent == Path(image_cache_dir) else "from_test"
    Path(failed_image_dir, parent_dir).mkdir(exist_ok=True)
    dest_dir = Path(failed_image_dir, parent_dir, dest_dirname)
    dest_dir.mkdir(exist_ok=True)
    dest_path = Path(dest_dir, Path(source_image_path).name)
    shutil.copy(source_image_path, dest_path)


@pytest.fixture
def verify_image_cache(request, pytestconfig):  # noqa: ANN001, ANN201
    """Checks cached images against test images for PyVista."""  # noqa: D401
    # Set CMD options in class attributes
    VerifyImageCache.reset_image_cache = pytestconfig.getoption("reset_image_cache")
    VerifyImageCache.ignore_image_cache = pytestconfig.getoption("ignore_image_cache")
    VerifyImageCache.fail_extra_image_cache = pytestconfig.getoption("fail_extra_image_cache")
    VerifyImageCache.add_missing_images = pytestconfig.getoption("add_missing_images")
    VerifyImageCache.reset_only_failed = pytestconfig.getoption("reset_only_failed")

    cache_dir = _get_dir_from_config_or_ini(pytestconfig, "image_cache_dir")
    gen_dir = _get_dir_from_config_or_ini(pytestconfig, "generated_image_dir")

    verify_image_cache = VerifyImageCache(request.node.name, cache_dir, generated_image_dir=gen_dir)
    pyvista.global_theme.before_close_callback = verify_image_cache

    def reset() -> None:
        pyvista.global_theme.before_close_callback = None

    request.addfinalizer(reset)  # noqa: PT021
    return verify_image_cache
