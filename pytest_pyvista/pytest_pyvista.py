"""pytest-pyvista module."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal
from typing import cast
import warnings

import pytest
import pyvista
from pyvista import Plotter

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Generator


VISITED_CACHED_IMAGE_NAMES: set[str] = set()
SKIPPED_CACHED_IMAGE_NAMES: set[str] = set()


class RegressionError(RuntimeError):
    """Error when regression does not meet the criteria."""


class RegressionFileNotFound(FileNotFoundError):  # noqa: N818
    """
    Error when regression file is not found.

    DO NOT USE, maintained for backwards-compatibility only.
    Use RegressionFileNotFoundError instead.
    """


class RegressionFileNotFoundError(RegressionFileNotFound):
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
        "--failed_image_dir",
        action="store",
        help="Path to dump images from failed tests from the current run.",
    )
    parser.addini(
        "failed_image_dir",
        default=None,
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
        "--disallow_unused_cache",
        action="store_true",
        help="Report test failure if there are any images in the cache which are not compared to any generated images.",
    )
    group.addoption(
        "--allow_useless_fixture",
        action="store_true",
        help="Prevent test failure if the `verify_image_cache` fixture is used but no images are generated.",
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

    cache_dir : Path
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

    generated_image_dir : Path, optional
        Directory to save generated images.  If not specified, no generated
        images are saved.

    failed_image_dir : Path, optional
        Directory to save failed images.  If not specified, no generated
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

    def __init__(  # noqa: PLR0913
        self,
        test_name: str,
        cache_dir: Path,
        *,
        error_value: float = 500.0,
        warning_value: float = 200.0,
        var_error_value: float = 1000.0,
        var_warning_value: float = 1000.0,
        generated_image_dir: Path | None = None,
        failed_image_dir: Path | None = None,
    ) -> None:
        """Initialize VerifyImageCache."""
        self.test_name = test_name

        # handle paths
        if not cache_dir.is_dir():
            _ensure_dir_exists(cache_dir, msg_name="cache image dir")
        self.cache_dir = cache_dir

        if generated_image_dir is not None:
            _ensure_dir_exists(generated_image_dir, msg_name="generated image dir")
        self.generated_image_dir = generated_image_dir

        self.failed_image_dir = failed_image_dir

        self.error_value = error_value
        self.warning_value = warning_value
        self.var_error_value = var_error_value
        self.var_warning_value = var_warning_value

        self.high_variance_test = False
        self.windows_skip_image_cache = False
        self.macos_skip_image_cache = False

        self.skip = False
        self.n_calls = 0

    @staticmethod
    def _is_skipped(*, skip: bool, windows_skip_image_cache: bool, macos_skip_image_cache: bool, ignore_image_cache: bool) -> bool:
        skip_windows = os.name == "nt" and windows_skip_image_cache
        skip_macos = platform.system() == "Darwin" and macos_skip_image_cache
        return skip or ignore_image_cache or skip_windows or skip_macos

    def __call__(self, plotter: Plotter) -> None:  # noqa: C901, PLR0912
        """
        Either store or validate an image.

        Parameters
        ----------
        plotter : pyvista.Plotter
            The Plotter object that is being closed.

        """

        def remove_plotter_close_callback() -> None:
            # Make sure this doesn't get called again if this plotter doesn't close properly
            # This is typically needed if an error is raised by this function
            plotter._before_close_callback = None  # noqa: SLF001

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
        image_filename = Path(self.cache_dir, image_name)
        gen_image_filename = None if self.generated_image_dir is None else Path(self.generated_image_dir, image_name)

        if VerifyImageCache._is_skipped(
            skip=self.skip,
            windows_skip_image_cache=self.windows_skip_image_cache,
            macos_skip_image_cache=self.macos_skip_image_cache,
            ignore_image_cache=self.ignore_image_cache,
        ):
            SKIPPED_CACHED_IMAGE_NAMES.add(image_name)
            return
        VISITED_CACHED_IMAGE_NAMES.add(image_name)

        if not image_filename.is_file() and not (self.allow_unused_generated or self.add_missing_images or self.reset_image_cache):
            # Raise error since the cached image does not exist and will not be added later

            # Save images as needed before error
            if gen_image_filename is not None:
                plotter.screenshot(gen_image_filename)
            if self.failed_image_dir is not None:
                self._save_failed_test_images("error", plotter, image_name)

            remove_plotter_close_callback()
            msg = f"{image_filename} does not exist in image cache"
            raise RegressionFileNotFoundError(msg)

        if (self.add_missing_images and not image_filename.is_file()) or (self.reset_image_cache and not self.reset_only_failed):
            plotter.screenshot(image_filename)

        if gen_image_filename is not None:
            plotter.screenshot(gen_image_filename)

        if not Path(image_filename).is_file() and self.allow_unused_generated:
            # Test image has been generated, but cached image does not exist
            # The generated image is considered unused, so exit safely before image
            # comparison to avoid a FileNotFoundError
            return

        if self.failed_image_dir is not None and not Path(image_filename).is_file():
            # Image comparison will fail, so save image before error
            self._save_failed_test_images("error", plotter, image_name)
            remove_plotter_close_callback()

        error = pyvista.compare_images(str(image_filename), plotter)

        if error > allowed_error:
            if self.failed_image_dir is not None:
                self._save_failed_test_images("error", plotter, image_name)
            if self.reset_only_failed:
                warnings.warn(
                    f"{test_name} Exceeded image regression error of "
                    f"{allowed_error} with an image error equal to: {error}"
                    f"\nThis image will be reset in the cache.",
                    stacklevel=2,
                )
                plotter.screenshot(image_filename)
            else:
                remove_plotter_close_callback()
                msg = f"{test_name} Exceeded image regression error of {allowed_error} with an image error equal to: {error}"
                raise RegressionError(msg)
        if error > allowed_warning:
            if self.failed_image_dir is not None:
                self._save_failed_test_images("warning", plotter, image_name)
            warnings.warn(f"{test_name} Exceeded image regression warning of {allowed_warning} with an image error of {error}", stacklevel=2)

    def _save_failed_test_images(self, error_or_warning: Literal["error", "warning"], plotter: Plotter, image_name: str) -> None:
        """Save test image from cache and from test to the failed image dir."""

        def _make_failed_test_image_dir(
            errors_or_warnings: Literal["errors", "warnings"], from_cache_or_test: Literal["from_cache", "from_test"]
        ) -> Path:
            # Check was done earlier to verify this is not None
            failed_image_dir = cast("str", self.failed_image_dir)
            _ensure_dir_exists(failed_image_dir, msg_name="failed image dir")
            dest_dir = Path(failed_image_dir, errors_or_warnings, from_cache_or_test)
            dest_dir.mkdir(exist_ok=True, parents=True)
            return dest_dir

        error_dirname = cast("Literal['errors', 'warnings']", error_or_warning + "s")

        from_test_dir = _make_failed_test_image_dir(error_dirname, "from_test")
        plotter.screenshot(from_test_dir / image_name)

        cached_image = Path(self.cache_dir, image_name)
        if cached_image.is_file():
            from_cache_dir = _make_failed_test_image_dir(error_dirname, "from_cache")
            shutil.copy(cached_image, from_cache_dir / image_name)


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


@pytest.hookimpl
def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ANN001, ARG001
    """Execute after the whole test run completes."""
    if config.getoption("disallow_unused_cache"):
        cache_path = Path(_get_option_from_config_or_ini(config, "image_cache_dir"))
        cached_image_names = {f.name for f in cache_path.glob("*.png")}
        unused_cached_image_names = cached_image_names - VISITED_CACHED_IMAGE_NAMES - SKIPPED_CACHED_IMAGE_NAMES

        # Exclude images from skipped tests where multiple images are generated
        unused_skipped = unused_cached_image_names.copy()
        for image_name in unused_cached_image_names:
            base_image_name = _image_name_from_test_name(_test_name_from_image_name(image_name))
            if base_image_name in SKIPPED_CACHED_IMAGE_NAMES:
                unused_skipped.remove(image_name)

        if unused_skipped:
            tr = terminalreporter
            tr.ensure_newline()
            tr.section("pytest-pyvista ERROR", sep="=", red=True, bold=True)
            tr.line(f"Unused cached image file(s) detected ({len(unused_skipped)}). The following images are", red=True)
            tr.line("cached, but were not generated or skipped by any of the tests:", red=True)
            tr.line(f"{sorted(unused_skipped)}", yellow=True)
            tr.line("")
            tr.line("These images should either be removed from the cache, or the corresponding", red=True)
            tr.line("tests should be modified to ensure an image is generated for comparison.", red=True)
            pytest.exit("Unused cache images", returncode=pytest.ExitCode.TESTS_FAILED)

    VISITED_CACHED_IMAGE_NAMES.clear()
    SKIPPED_CACHED_IMAGE_NAMES.clear()


def _ensure_dir_exists(dirpath: str | Path, msg_name: str) -> None:
    if not Path(dirpath).is_dir():
        msg = f"pyvista test {msg_name}: {dirpath} does not yet exist.  Creating dir."
        warnings.warn(msg, stacklevel=2)
        Path(dirpath).mkdir(parents=True)


def _get_option_from_config_or_ini(pytestconfig: pytest.Config, option: str, *, is_dir: bool = False):  # noqa: ANN202
    value = pytestconfig.getoption(option)
    if value is None:
        value = pytestconfig.getini(option)

    if value is None:
        return value

    return pytestconfig.rootpath / value if is_dir else value


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call) -> Generator:  # noqa: ANN001, ARG001
    """Store test results for inspection."""
    outcome = yield
    if outcome:
        rep = outcome.get_result()

        # Mark cached image as skipped if test was skipped during setup or execution
        if rep.when in ["call", "setup"] and rep.skipped:
            SKIPPED_CACHED_IMAGE_NAMES.add(_image_name_from_test_name(item.name))

        # Attach the report to the item so fixtures/finalizers can inspect it
        setattr(item, f"rep_{rep.when}", rep)


class _ChainedCallbacks:
    def __init__(self, *funcs: Callable[[Plotter], None]) -> None:
        """Chainable callbacks for pyvista.Plotter.show method."""
        self.funcs = funcs

    def __call__(self, plotter: Plotter) -> None:
        """Call all input functions in chain for the given Plotter instance."""
        for f in self.funcs:
            f(plotter)


@pytest.fixture
def verify_image_cache(
    request: pytest.FixtureRequest,
    pytestconfig: pytest.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[VerifyImageCache, None, None]:
    """Check cached images against test images for PyVista."""
    # Set CMD options in class attributes
    VerifyImageCache.reset_image_cache = pytestconfig.getoption("reset_image_cache")
    VerifyImageCache.ignore_image_cache = pytestconfig.getoption("ignore_image_cache")
    VerifyImageCache.allow_unused_generated = pytestconfig.getoption("allow_unused_generated")
    VerifyImageCache.add_missing_images = pytestconfig.getoption("add_missing_images")
    VerifyImageCache.reset_only_failed = pytestconfig.getoption("reset_only_failed")

    cache_dir = _get_option_from_config_or_ini(pytestconfig, "image_cache_dir", is_dir=True)
    gen_dir = _get_option_from_config_or_ini(pytestconfig, "generated_image_dir", is_dir=True)
    failed_dir = _get_option_from_config_or_ini(pytestconfig, "failed_image_dir", is_dir=True)

    verify_image_cache = VerifyImageCache(
        test_name=request.node.name,
        cache_dir=cache_dir,
        generated_image_dir=gen_dir,
        failed_image_dir=failed_dir,
    )

    # Wrapping call to `Plotter.show` to inject the image cache callback
    def func_show(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        key = "before_close_callback"
        user_callback = kwargs.get(key, lambda *a: ...)  # noqa: ARG005

        if user_callback is None:  # special case encountered when using the `plot` property of pyvista objects
            user_callback = lambda *a: ...  # noqa: ARG005, E731

        kwargs[key] = _ChainedCallbacks(user_callback, verify_image_cache)

        return old_show(*args, **kwargs)

    old_show = Plotter.show
    monkeypatch.setattr(Plotter, "show", func_show)

    yield verify_image_cache

    # Check if the fixture was not used
    # Value from fixture takes precedence over value set by CLI
    allow_useless_fixture = getattr(verify_image_cache, "allow_useless_fixture", None)
    if allow_useless_fixture is None:
        allow_useless_fixture = pytestconfig.getoption("allow_useless_fixture")

    skipped = VerifyImageCache._is_skipped(  # noqa: SLF001
        skip=verify_image_cache.skip,
        windows_skip_image_cache=verify_image_cache.windows_skip_image_cache,
        macos_skip_image_cache=verify_image_cache.macos_skip_image_cache,
        ignore_image_cache=verify_image_cache.ignore_image_cache,
    )
    if not allow_useless_fixture and not skipped:
        # Retrieve test call report
        rep_call = getattr(request.node, "rep_call", None)

        if rep_call and rep_call.passed and verify_image_cache.n_calls == 0:
            pytest.fail(
                "Fixture `verify_image_cache` is used but no images were generated.\n"
                "Did you forget to call `show` or `plot`, or set `verify_image_cache.allow_useless_fixture=True`?."
            )
