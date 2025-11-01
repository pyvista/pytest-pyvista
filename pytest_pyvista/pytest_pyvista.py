"""pytest-pyvista module."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import cached_property
import importlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal
from typing import Union
from typing import cast
from typing import get_args
from typing import overload
import uuid
import warnings

import numpy as np
from PIL import Image
import pytest
import pyvista
from pyvista import Plotter
import vtkmodules

from pytest_pyvista import hooks

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Generator

    import xdist.workermanage

VISITED_CACHED_IMAGE_NAMES: set[str] = set()
SKIPPED_CACHED_IMAGE_NAMES: set[str] = set()
PYVISTA_IMAGE_NAMES_CACHE_DIRNAME = "pyvista_image_names_dir"
PYVISTA_GENERATED_IMAGE_CACHE_DIRNAME = "pyvista_generated_image_dir"
PYVISTA_FAILED_IMAGE_CACHE_DIRNAME = "pyvista_failed_image_dir"

PARSER_GROUP_NAME = "pyvista"
DEFAULT_ERROR_THRESHOLD: float = 500.0
DEFAULT_WARNING_THRESHOLD: float = 200.0
_DOC_MODE_CLI_ARGS: set[str] = set()
_UNIT_TEST_CLI_ARGS: set[str] = set()

_AllowedImageFormats = Literal["png", "jpg"]
_OriginalImageFormats = Union[_AllowedImageFormats, Literal["gif", "vtksz"]]


@dataclass
class _EnvInfo:
    prefix: str = ""
    os: bool = True
    machine: bool = True
    python: bool = True
    pyvista: bool = True
    vtk: bool = True
    gpu: bool = True
    ci: bool = True
    suffix: str = ""

    def __repr__(self) -> str:
        os_version = f"{_SYSTEM_PROPERTIES.os_name}-{_SYSTEM_PROPERTIES.os_version}" if self.os else ""
        machine = f"{platform.machine()}" if self.machine else ""
        gpu = f"gpu-{_SYSTEM_PROPERTIES.gpu_vendor}" if self.gpu else ""
        python_version = f"py-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}" if self.python else ""
        pyvista_version = f"pyvista-{pyvista.__version__}" if self.pyvista else ""
        vtk_version = f"vtk-{vtkmodules.__version__}" if self.vtk else ""
        ci = f"{'' if os.environ.get('CI', None) else 'no-'}CI" if self.ci else ""

        values = [
            f"{self.prefix}",
            f"{os_version}",
            f"{machine}",
            f"{gpu}",
            f"{python_version}",
            f"{pyvista_version}",
            f"{vtk_version}",
            f"{ci}",
            f"{self.suffix}",
        ]
        return "_".join(val for val in values if val)


class _SystemProperties:
    @cached_property
    def os_name(self) -> str:
        return _SystemProperties._get_os()[0]

    @cached_property
    def os_version(self) -> str:
        return _SystemProperties._get_os()[1]

    @cached_property
    def gpu_vendor(self) -> str:
        return _SystemProperties._gpu_vendor()

    @staticmethod
    def _get_os() -> tuple[str, str]:
        system = platform.system()
        if system == "Linux":
            try:
                name = platform.freedesktop_os_release()["ID"]
                version = platform.freedesktop_os_release()["VERSION_ID"]
            except AttributeError:
                name = system
                version = platform.release()
            return name, version
        name = "macOS" if system == "Darwin" else system
        return name, platform.release()

    @staticmethod
    def _gpu_vendor() -> str:
        try:
            vendor = pyvista.GPUInfo().vendor
        except Exception:  # noqa: BLE001
            return "unknown"

        # Try to shorten vendor string
        lower = vendor.lower()
        if lower.startswith(nv := "nvidia"):
            text = nv
        elif lower.startswith(amd := "amd"):
            text = amd
        elif lower.startswith(ati := "ati"):
            text = ati
        elif lower.startswith(mesa := "mesa"):
            text = mesa
        else:
            text = vendor  # pragma: no cover
        # Shorten original string and remove whitespace
        vendor = vendor[: len(text)].replace(" ", "")
        # Remove all potentially invalid/undesired filename characters
        disallowed = r'[\\/:*?"<>|\s.\x00]'
        return re.sub(disallowed, "", vendor)


_SYSTEM_PROPERTIES = _SystemProperties()


class RegressionError(RuntimeError):
    """Error when regression does not meet the criteria."""


class InvalidCacheError(RuntimeError):
    """Error when validating the cache."""


class RegressionFileNotFound(FileNotFoundError):  # noqa: N818
    """
    Error when regression file is not found.

    DO NOT USE, maintained for backwards-compatibility only.
    Use RegressionFileNotFoundError instead.
    """


class RegressionFileNotFoundError(RegressionFileNotFound):
    """Error when regression file is not found."""


def pytest_addhooks(pluginmanager: pytest.PytestPluginManager) -> None:
    """Add hooks."""
    pluginmanager.add_hookspecs(hooks)


def pytest_addoption(parser: pytest.Parser) -> None:  # noqa: PLR0915
    """Add new flag options to the pyvista plugin."""

    def _add_unit_test_cli_option(option: str, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Add a CLI option reserved for regular unit tests only."""
        group.addoption(option, *args, **kwargs)
        _UNIT_TEST_CLI_ARGS.add(option)

    def _add_doc_cli_option(option: str, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Add a CLI option reserved for documentation tests only."""
        group.addoption(option, *args, **kwargs)
        _DOC_MODE_CLI_ARGS.add(option)

    def _add_common_cli_option(option: str, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Add a CLI option common to regular unit tests and documentation tests."""
        group.addoption(option, *args, **kwargs)
        _UNIT_TEST_CLI_ARGS.add(option)
        _DOC_MODE_CLI_ARGS.add(option)

    def _add_common_ini_option(option: str, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        parser.addini(option, *args, **kwargs)
        parser.addini("doc_" + option, *args, **kwargs)

    def _add_common_cli_and_ini_options() -> None:
        """
        Add CLI and INI options common to both regular unit tests and doc mode.

        The CLI argument name is the same for unit tests and doc mode. For the INI config, a ``doc_``
        prefix is added.

        Important:
            A default value for INI options should *NOT* be set, i.e. the default should be None.
            This is needed because any INI options with a ``doc_`` prefix has priority over the
            non-prefixed version, and should only be set by users that want to explicitly override
            the non-prefixed INI value.

            Non-None default values should be set inside "_get_option_from_config_or_ini" instead.

        """
        option = "image_cache_dir"
        help_ = "Path to the image cache folder."
        _add_common_cli_option(f"--{option}", action="store", help=help_)
        _add_common_ini_option(option, default=None, help=help_)  # Default is set when getting from config or ini

        option = "generated_image_dir"
        help_ = "Path to dump test images from the current run."
        _add_common_cli_option(f"--{option}", action="store", help=help_)
        _add_common_ini_option(option, default=None, help=help_)

        option = "failed_image_dir"
        help_ = "Path to dump images from failed tests from the current run."
        _add_common_cli_option(f"--{option}", action="store", help=help_)
        _add_common_ini_option(option, default=None, help=help_)

        option = "generate_subdirs"
        help_ = "Save generated images to sub-directories. The image names are determined by the environment info."
        _add_common_cli_option(f"--{option}", action="store_const", const=True, default=None, help=help_)
        _add_common_ini_option(option, default=None, help=help_)

        option = "image_format"
        help_ = "Image format to use when generating test images."
        _add_common_cli_option(f"--{option}", action="store", choices=get_args(_AllowedImageFormats), default=None, help=help_)
        _add_common_ini_option(option, default=None, help=help_)  # Default is set when getting from config or ini

        option = "max_image_size"
        help_ = "Saved images are resized so that dimensions will not exceed this value."
        _add_common_cli_option(f"--{option}", default=None, help=help_)
        _add_common_ini_option(option, default=None, help=help_)

    def _add_unit_test_cli_and_ini_options() -> None:
        """Add options specific to regular unit tests."""
        _add_unit_test_cli_option(
            "--reset_image_cache",
            action="store_true",
            help="Reset the images in the PyVista cache.",
        )

        _add_unit_test_cli_option(
            "--ignore_image_cache",
            action="store_true",
            help="Ignores the image cache.",
        )

        _add_unit_test_cli_option(
            "--allow_unused_generated",
            action="store_true",
            help="Prevent test failure if a generated test image has no use.",
        )

        _add_unit_test_cli_option(
            "--add_missing_images",
            action="store_true",
            help="Adds images to cache if missing.",
        )

        _add_unit_test_cli_option(
            "--reset_only_failed",
            action="store_true",
            help="Reset only the failed images in the PyVista cache.",
        )

        _add_unit_test_cli_option(
            "--disallow_unused_cache",
            action="store_true",
            help="Report test failure if there are any images in the cache which are not compared to any generated images.",
        )

        _add_unit_test_cli_option(
            "--allow_useless_fixture",
            action="store_true",
            help="Prevent test failure if the `verify_image_cache` fixture is used but no images are generated.",
        )

    def _add_doc_cli_and_ini_options() -> None:
        """Add options specific to the documentation tests."""
        _add_doc_cli_option(
            "--doc_mode",
            action="store_true",
            help="Enable documentation image testing.",
        )

        _add_doc_cli_option(
            "--doc_images_dir",
            action="store",
            help="Path to the documentation images.",
        )
        parser.addini(
            "doc_images_dir",
            default=None,
            help="Path to the documentation images.",
        )

        _add_doc_cli_option(
            "--include_vtksz",
            action="store_true",
            default=None,
            help="Include tests for interactive images with the .vtksz file format.",
        )
        parser.addini(
            "include_vtksz",
            type="bool",
            default=None,
            help="Include tests for interactive images with the .vtksz file format.",
        )

        _add_doc_cli_option(
            "--max_vtksz_file_size",
            action="store",
            default=None,
            help="Maximum size allowed for vtksz interactive plot files.",
        )
        parser.addini(
            "max_vtksz_file_size",
            default=None,
            type="int",
            help="Maximum size allowed for vtksz interactive plot files.",
        )

    group = parser.getgroup(PARSER_GROUP_NAME)
    _add_common_cli_and_ini_options()
    _add_unit_test_cli_and_ini_options()
    _add_doc_cli_and_ini_options()


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
        file or sub-directory.

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
    generate_subdirs: bool = False
    image_format: _AllowedImageFormats
    max_image_size: int | None

    def __init__(  # noqa: PLR0913
        self,
        test_name: str,
        cache_dir: Path,
        *,
        error_value: float = DEFAULT_ERROR_THRESHOLD,
        warning_value: float = DEFAULT_WARNING_THRESHOLD,
        var_error_value: float = 1000.0,
        var_warning_value: float = 1000.0,
        generated_image_dir: Path | None = None,
        failed_image_dir: Path | None = None,
    ) -> None:
        """Initialize VerifyImageCache."""
        self.test_name = test_name
        self.env_info: str | _EnvInfo = _EnvInfo()

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

    def __call__(self, plotter: Plotter) -> None:  # noqa: C901, PLR0912, PLR0915
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
        image_name = _image_name_from_test_name(test_name, image_format=self.image_format)
        image_filename = Path(self.cache_dir, image_name)
        image_dirname = Path(self.cache_dir, Path(image_name).stem)

        cached_image_paths = _get_file_paths(image_dirname, ext=self.image_format) if image_dirname.is_dir() else [image_filename]
        if not cached_image_paths:
            # Path is an empty dir, append default expected image path
            cached_image_paths.append(image_dirname / f"{self.env_info}.{self.image_format}")
        current_cached_image = cached_image_paths[0]

        if VerifyImageCache._is_skipped(
            skip=self.skip,
            windows_skip_image_cache=self.windows_skip_image_cache,
            macos_skip_image_cache=self.macos_skip_image_cache,
            ignore_image_cache=self.ignore_image_cache,
        ):
            SKIPPED_CACHED_IMAGE_NAMES.add(image_name)
            return
        VISITED_CACHED_IMAGE_NAMES.add(image_name)

        if not current_cached_image.is_file() and not (self.allow_unused_generated or self.add_missing_images or self.reset_image_cache):
            # Raise error since the cached image does not exist and will not be added later

            # Save images as needed before error
            if self.generated_image_dir is not None:
                self._save_generated_image(plotter, image_name=image_name)
            if self.failed_image_dir is not None:
                self._save_failed_test_images("error", plotter, image_name)

            remove_plotter_close_callback()
            msg = f"{current_cached_image} does not exist in image cache"
            raise RegressionFileNotFoundError(msg)

        if (self.add_missing_images and not current_cached_image.is_file()) or (self.reset_image_cache and not self.reset_only_failed):
            _screenshot(plotter, current_cached_image, max_image_size=VerifyImageCache.max_image_size)

        if self.generated_image_dir is not None:
            self._save_generated_image(plotter, image_name=image_name)

        if not Path(current_cached_image).is_file() and self.allow_unused_generated:
            # Test image has been generated, but cached image does not exist
            # The generated image is considered unused, so exit safely before image
            # comparison to avoid a FileNotFoundError
            return

        test_name_no_prefix = test_name.removeprefix("test_")
        warn_msg, fail_msg = _test_compare_images(
            test_name=test_name_no_prefix,
            test_image=plotter,
            cached_image=current_cached_image,
            allowed_error=allowed_error,
            allowed_warning=allowed_warning,
        )

        # Try again and compare with other cached images
        if fail_msg and len(cached_image_paths) > 1:
            # Compare test image to other known valid versions
            msg_start = "This test has multiple cached images. It initially failed (as above)"
            for path in cached_image_paths[1:]:
                error = _compare_images(plotter, path)
                if _check_compare_fail(test_name, error, allowed_error=allowed_error) is None:
                    # Convert failure into a warning
                    warn_msg = fail_msg + (f"\n{msg_start} but passed when compared to:\n\t{path}")
                    fail_msg = None
                    current_cached_image = path
                    break
            else:  # Loop completed - test still fails
                fail_msg += f"\n{msg_start} and failed again for all images in:\n\t{Path(self.cache_dir, test_name_no_prefix)!s}"

        if fail_msg:
            if self.failed_image_dir is not None:
                self._save_failed_test_images("error", plotter, image_name)
            if self.reset_only_failed:
                warnings.warn(
                    f"{fail_msg}\nThis image will be reset in the cache.",
                    stacklevel=2,
                )
                _screenshot(plotter, current_cached_image, max_image_size=VerifyImageCache.max_image_size)
            else:
                remove_plotter_close_callback()
                raise RegressionError(fail_msg)

        if warn_msg:
            parent_dir: Literal["errors_as_warning", "warning"] = "errors_as_warning" if image_dirname.is_dir() else "warning"
            if self.failed_image_dir is not None:
                self._save_failed_test_images(parent_dir, plotter, image_name, cache_image_path=current_cached_image)
            warnings.warn(warn_msg, stacklevel=2)

    def _save_generated_image(self, plotter: pyvista.Plotter, image_name: str, parent_dir: Path | None = None) -> None:
        parent = cast("Path", self.generated_image_dir) if parent_dir is None else parent_dir
        generated_image_path = _get_generated_image_path(
            parent=parent, image_name=image_name, generate_subdirs=self.generate_subdirs, env_info=self.env_info
        )
        _screenshot(plotter, generated_image_path, max_image_size=VerifyImageCache.max_image_size)

    def _save_failed_test_images(
        self,
        error_or_warning: Literal["error", "warning", "errors_as_warning"],
        plotter: Plotter,
        image_name: str,
        cache_image_path: Path | None = None,
    ) -> None:
        """Save test image from cache and from test to the failed image dir."""

        def _make_failed_test_image_dir(
            errors_or_warnings: Literal["errors", "warnings", "errors_as_warnings"], from_cache_or_test: Literal["from_cache", "from_test"]
        ) -> Path:
            # Check was done earlier to verify this is not None
            failed_image_dir = cast("str", self.failed_image_dir)
            dest_dir = Path(failed_image_dir, errors_or_warnings, from_cache_or_test)
            dest_dir.mkdir(exist_ok=True, parents=True)
            return dest_dir

        def _save_single_cache_image(path: Path) -> None:
            rel = Path(path).relative_to(self.cache_dir)
            from_cache_dir = _make_failed_test_image_dir(error_dirname, "from_cache")
            dest_path = from_cache_dir / rel
            dest_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(path, dest_path)

        error_dirname = cast("Literal['errors', 'warnings', 'errors_as_warnings']", error_or_warning + "s")

        from_test_dir = _make_failed_test_image_dir(error_dirname, "from_test")
        self._save_generated_image(plotter, image_name=image_name, parent_dir=from_test_dir)

        cached_image = Path(self.cache_dir, image_name) if cache_image_path is None else cache_image_path
        if cached_image.is_file():
            # Save single cache file
            _save_single_cache_image(cached_image)
        elif (image_dir := cached_image.with_suffix("")).is_dir():
            # Save multiple cached files
            for path in _get_file_paths(image_dir, ext=self.image_format):
                _save_single_cache_image(path)


def _image_name_from_test_name(test_name: str, image_format: str) -> str:
    return f"{test_name.removeprefix('test_')}.{image_format}"


def _test_name_from_image_name(image_name: str) -> str:
    def remove_suffix(s: str) -> str:
        """Remove integer and image format suffix."""
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


def _get_generated_image_path(parent: Path, image_name: Path | str, *, generate_subdirs: bool, env_info: str | _EnvInfo, vtksz: bool = False) -> Path:
    name = Path(image_name)
    name = name.with_stem(name.stem + "_vtksz") if vtksz else name
    generated_image_path = parent / name.with_suffix("") / f"{env_info}{name.suffix}" if generate_subdirs else parent / name
    generated_image_path.parent.mkdir(exist_ok=True, parents=True)
    return generated_image_path.resolve()


def _get_file_paths(dir_: Path, ext: str) -> list[Path]:
    """Get all paths of files with a specific extension inside a directory tree."""
    return sorted(dir_.rglob(f"*.{ext}"))


def _compare_images(test_image: Path | str | pyvista.Plotter, cached_image: Path | str) -> float:
    if isinstance(test_image, pyvista.Plotter) and Path(cached_image).suffix == ".jpg":
        # Need to process image to apply jpg compression

        # Get screenshot as a PIL image
        pl = cast("pyvista.Plotter", test_image)
        arr = _screenshot(pl, return_img=True, max_image_size=VerifyImageCache.max_image_size)
        img = Image.fromarray(arr)

        # Save as JPEG in memory
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        # Reload compressed JPEG back into NumPy
        arr_jpg = np.array(Image.open(buf))
        return pyvista.compare_images(arr_jpg, str(cached_image))
    # Cast Path to str
    test_img = test_image if isinstance(test_image, pyvista.Plotter) else str(test_image)
    return pyvista.compare_images(test_img, str(cached_image))


def _get_thumbnail_size(current_size: tuple[int, int], max_image_size: int) -> tuple[int, int]:
    ref_image = Image.fromarray(np.zeros(current_size).T)
    ref_image.thumbnail(size=(max_image_size, max_image_size))
    return ref_image.size


def _screenshot(plotter: Plotter, *args, max_image_size: int | None, **kwargs) -> pyvista.pyvista_ndarray | None:  # noqa: ANN002, ANN003
    old_window_size = cast("tuple[int, int]", tuple(plotter.window_size))
    do_resize = max_image_size is not None and any(size > max_image_size for size in old_window_size)
    if do_resize:
        plotter.window_size = _get_thumbnail_size(old_window_size, max_image_size=cast("int", max_image_size))

    output = plotter.screenshot(*args, **kwargs)

    if do_resize:
        plotter.window_size = old_window_size

    return output


def _test_compare_images(
    test_name: str, test_image: Path | str | pyvista.Plotter, cached_image: Path | str, allowed_error: float, allowed_warning: float
) -> tuple[str | None, str | None]:
    try:
        # Check if test should fail or warn
        error = _compare_images(test_image, cached_image)
        fail_msg = _check_compare_fail(test_name, error, allowed_error)
        warn_msg = _check_compare_warn(test_name, error, allowed_warning)
    except RuntimeError as e:
        warn_msg = None
        fail_msg = repr(e)
    return warn_msg, fail_msg


def _check_compare_fail(test_name: str, error_: float, allowed_error: float) -> str | None:
    if error_ > allowed_error:
        return f"{test_name} Exceeded image regression error of {allowed_error} with an image error equal to: {error_}"
    return None


def _check_compare_warn(test_name: str, error_: float, allowed_warning: float) -> str | None:
    if error_ > allowed_warning:
        return f"{test_name} Exceeded image regression warning of {allowed_warning} with an image error of {error_}"
    return None


@pytest.hookimpl
def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ANN001, ARG001
    """Execute after the whole test run completes."""
    if hasattr(config, "workerinput"):
        # on an pytest-xdist worker node, exit early
        return

    if config.getoption("disallow_unused_cache") and getattr(VerifyImageCache, "image_format", None):
        value = _get_option_from_config_or_ini(config, "image_cache_dir")
        cache_path = Path(cast("Path", value))
        cached_image_names = {f.name for f in cache_path.glob(f"*.{VerifyImageCache.image_format}")}

        image_names_dir = getattr(config, PYVISTA_IMAGE_NAMES_CACHE_DIRNAME, None)
        if image_names_dir:
            visited_cached_image_names = _combine_temp_jsons(image_names_dir, "visited")
            skipped_cached_image_names = _combine_temp_jsons(image_names_dir, "skipped")
        else:
            visited_cached_image_names = set()
            skipped_cached_image_names = set()

        unused_cached_image_names = cached_image_names - visited_cached_image_names - skipped_cached_image_names

        # Exclude images from skipped tests where multiple images are generated
        unused_skipped = unused_cached_image_names.copy()
        for image_name in unused_cached_image_names:
            base_image_name = _image_name_from_test_name(_test_name_from_image_name(image_name), image_format=VerifyImageCache.image_format)
            if base_image_name in skipped_cached_image_names:
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

        # exist_ok to allow for multi-threading
        Path(dirpath).mkdir(exist_ok=True, parents=True)


@overload
def _get_option_from_config_or_ini(pytestconfig: pytest.Config, option: str, *, is_dir: Literal[False] = False) -> str | int | bool | None: ...
@overload
def _get_option_from_config_or_ini(pytestconfig: pytest.Config, option: str, *, is_dir: Literal[True] = True) -> Path | None: ...
@overload
def _get_option_from_config_or_ini(pytestconfig: pytest.Config, option: str, *, is_dir: bool) -> Path | str | int | bool | None: ...
def _get_option_from_config_or_ini(pytestconfig: pytest.Config, option: str, *, is_dir: bool = False) -> Path | str | int | bool | None:  # noqa: C901
    def _resolve(value: str | int | bool) -> Path | str | int | bool:  # noqa: FBT001
        if is_dir:
            return pytestconfig.rootpath / value
        if str(value).lower() == "true":
            return True
        if str(value).lower() == "false":
            return False
        with contextlib.suppress(ValueError):
            value = int(value)

        return value

    # CLI always wins, and only plain name without additional doc prefix
    value = pytestconfig.getoption(option)
    if value is not None:
        return _resolve(value)

    # Get from ini with `doc_` prefix, if available
    if doc_mode := pytestconfig.getoption("doc_mode"):
        try:
            value = pytestconfig.getini(f"doc_{option}")
        except ValueError:
            # Not defined, continue
            ...
        else:
            if value is not None:
                return _resolve(value)

    # Get from ini
    value = pytestconfig.getini(option)
    if value is not None:
        return _resolve(value)

    # Special cases, set defaults here
    if option == "image_cache_dir":
        value = f"{'doc_' if doc_mode else ''}image_cache_dir"
        return _resolve(value)

    if option == "image_format":
        return _resolve("png")

    return None


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call) -> Generator:  # noqa: ANN001, ARG001
    """Store test results for inspection."""
    outcome = yield
    if outcome and getattr(VerifyImageCache, "image_format", None):
        rep = outcome.get_result()

        # Mark cached image as skipped if test was skipped during setup or execution
        if rep.when in ["call", "setup"] and rep.skipped:
            SKIPPED_CACHED_IMAGE_NAMES.add(_image_name_from_test_name(item.name, image_format=VerifyImageCache.image_format))

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


@pytest.fixture(scope="session")
def _validate_image_cache_dir(pytestconfig: pytest.Config) -> None:
    """
    Validate the contents of the image cache directory.

    A session scope fixture is used since we only need to evaluate this once, and we want
    the error raised during test setup.
    """
    if pytestconfig.getoption("doc_mode"):
        from pytest_pyvista.doc_mode import _DocVerifyImageCache  # noqa: PLC0415

        image_cache_dir = _DocVerifyImageCache.image_cache_dir
        image_format = _DocVerifyImageCache.image_format
    else:
        image_cache_dir = cast("Path", _get_option_from_config_or_ini(pytestconfig, "image_cache_dir", is_dir=True))
        image_format = cast("_AllowedImageFormats", _get_option_from_config_or_ini(pytestconfig, "image_format"))
    __validate_image_cache_dir(image_cache_dir, image_format)


def __validate_image_cache_dir(cache_dir: Path, image_format: _AllowedImageFormats) -> None:
    def check_image_format(format_to_check: _AllowedImageFormats) -> None:
        image_paths = [str(p.relative_to(cache_dir)) for p in _get_file_paths(cache_dir, ext=format_to_check)]
        if image_paths and image_format != format_to_check:
            msg = (
                f"The image format required by\n"
                f"the image cache directory is {image_format!r}, but {format_to_check!r} images exist in the cache.\n"
                f"Cache directory: {str(cache_dir.resolve())!r}\n"
                f"Invalid images: {image_paths}"
            )
            raise InvalidCacheError(msg)

    check_image_format("png")
    check_image_format("jpg")

    subdir_names = {p.name for p in cache_dir.glob("*") if p.is_dir()}
    image_names = {p.stem for p in cache_dir.glob(f"*.{image_format}")}
    if intersection := (subdir_names & image_names):
        msg = (
            "Non-unique image test names detected in the cache.\n"
            "An image's name must not share the same name as a subdirectory. Either the image\n"
            "or the subdirectory should be removed for the following test cases:\n"
            f"{intersection}"
        )
        raise InvalidCacheError(msg)


def _make_config_cache_dir(config: pytest.Config, dirname: str, *, clean: bool = False) -> Path:
    newdir = Path(config.cache.makedir(dirname))
    newdir.mkdir(exist_ok=True)
    if clean:
        with contextlib.suppress(OSError):
            for item in newdir.iterdir():
                item.unlink()
    setattr(config, dirname, newdir)
    return newdir


def _get_num_workers_from_config(config: pytest.Config) -> int:
    """Return number of xdist workers, or 1 if xdist is not installed or not used."""
    try:
        num = config.getoption("numprocesses")  # -n
        if num is None:
            return 1
        return int(num)
    except (AttributeError, ValueError):
        # option doesn't exist → xdist not installed
        return 1


def _is_master(config: pytest.Config) -> bool:
    """Return True if running in a xdist master node or not running xdist at all."""
    return not hasattr(config, "workerinput")


def _strings_from_paths(paths: list[Path]) -> list[str]:
    return [str(p) for p in paths]


def _paths_from_strings(strings: list[str]) -> list[Path]:
    return [Path(s) for s in strings]


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest session."""
    # Validate CLI args
    doc_mode = config.getoption("doc_mode")

    cli_args = config.invocation_params.args
    plugin_options = _UNIT_TEST_CLI_ARGS | _DOC_MODE_CLI_ARGS
    for arg in cli_args:
        if arg in plugin_options:
            if doc_mode and arg not in _DOC_MODE_CLI_ARGS:
                msg = f"argument {arg} cannot be used with --doc_mode enabled"
                raise pytest.UsageError(msg)
            if not doc_mode and arg not in _UNIT_TEST_CLI_ARGS:
                msg = f"argument {arg} can only be used with --doc_mode enabled"
                raise pytest.UsageError(msg)

    is_master = _is_master(config)
    disallow_unused_cache = config.getoption("disallow_unused_cache")
    if is_master and disallow_unused_cache:
        # create a image names directory for individual or multiple workers to write to
        _make_config_cache_dir(config, PYVISTA_IMAGE_NAMES_CACHE_DIRNAME, clean=True)

    if doc_mode:
        from pytest_pyvista.doc_mode import _DocVerifyImageCache  # noqa: PLC0415
        from pytest_pyvista.doc_mode import _preprocess_all_images_for_test_cases  # noqa: PLC0415
        from pytest_pyvista.doc_mode import _VtkszFileSizeTestCase  # noqa: PLC0415

        _VtkszFileSizeTestCase.init_from_config(config)
        _DocVerifyImageCache.init_from_config(config)

        if is_master:
            # Clear any cached test files
            _make_config_cache_dir(config, PYVISTA_GENERATED_IMAGE_CACHE_DIRNAME, clean=True)
            _make_config_cache_dir(config, PYVISTA_FAILED_IMAGE_CACHE_DIRNAME, clean=True)

            # Determine how many processes to use for preprocessing
            num_workers = _get_num_workers_from_config(config)

            # Run preprocessing once in the master
            (
                input_paths,
                test_image_paths,
                vtksz_input_paths,
                vtksz_test_image_paths,
            ) = _preprocess_all_images_for_test_cases(num_workers=num_workers)

            # Store in config for xdist workers
            config.paths = {
                "input_paths": _strings_from_paths(input_paths),
                "test_image_paths": _strings_from_paths(test_image_paths),
                "vtksz_input_paths": _strings_from_paths(vtksz_input_paths),
                "vtksz_test_image_paths": _strings_from_paths(vtksz_test_image_paths),
            }


try:
    import xdist.plugin  # noqa: TC002
except ImportError:
    pass
else:

    def pytest_configure_node(node: xdist.workermanage.WorkerController) -> None:
        """Modify each xdist worker."""
        if paths := getattr(node.config, "paths", None):
            node.workerinput["paths"] = paths


@pytest.fixture
def verify_image_cache(
    request: pytest.FixtureRequest,
    pytestconfig: pytest.Config,
    monkeypatch: pytest.MonkeyPatch,
    _validate_image_cache_dir: None,
) -> Generator[VerifyImageCache, None, None]:
    """Check cached images against test images for PyVista."""
    # Set CMD options in class attributes
    VerifyImageCache.reset_image_cache = pytestconfig.getoption("reset_image_cache")
    VerifyImageCache.ignore_image_cache = pytestconfig.getoption("ignore_image_cache")
    VerifyImageCache.allow_unused_generated = pytestconfig.getoption("allow_unused_generated")
    VerifyImageCache.add_missing_images = pytestconfig.getoption("add_missing_images")
    VerifyImageCache.reset_only_failed = pytestconfig.getoption("reset_only_failed")
    VerifyImageCache.generate_subdirs = pytestconfig.getoption("generate_subdirs")
    VerifyImageCache.image_format = cast("_AllowedImageFormats", _get_option_from_config_or_ini(pytestconfig, "image_format"))
    VerifyImageCache.max_image_size = cast("int | None", _get_option_from_config_or_ini(pytestconfig, "max_image_size"))

    cache_dir = cast("Path", _get_option_from_config_or_ini(pytestconfig, "image_cache_dir", is_dir=True))
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

        # Set kwargs to None in order to get the callback from the
        # global theme one which is patched by the current callback.
        # This is done to make sure that the weak ref `_before_close_callback` is not dead
        # when using `auto_close=False` on the plotter
        # See https://github.com/pyvista/pytest-pyvista/issues/172
        callback = _ChainedCallbacks(user_callback, verify_image_cache)
        kwargs[key] = None

        monkeypatch.setattr(pyvista.global_theme, "before_close_callback", callback)

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


def _combine_temp_jsons(json_dir: Path, prefix: str = "") -> set[str]:
    # Read all JSON files from a directory and combine into single set
    combined_data: set[str] = set()
    if json_dir.exists():
        for json_file in json_dir.glob(f"{prefix}*.json"):
            with json_file.open() as f:
                data = json.load(f)
                combined_data.update(data)

    return combined_data


@pytest.hookimpl
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Write skipped and visited image names to disk."""
    image_names_dir = getattr(session.config, PYVISTA_IMAGE_NAMES_CACHE_DIRNAME, None)
    if image_names_dir:
        test_id = uuid.uuid4()
        visited_file = image_names_dir / f"visited_{test_id}_cache_names.json"
        skipped_file = image_names_dir / f"skipped_{test_id}_cache_names.json"

        # Fixed: Write JSON instead of plain text
        visited_file.write_text(json.dumps(list(VISITED_CACHED_IMAGE_NAMES)))
        skipped_file.write_text(json.dumps(list(SKIPPED_CACHED_IMAGE_NAMES)))


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove temporary files."""
    if _is_master(config):
        for dirname in [PYVISTA_FAILED_IMAGE_CACHE_DIRNAME, PYVISTA_GENERATED_IMAGE_CACHE_DIRNAME, PYVISTA_IMAGE_NAMES_CACHE_DIRNAME]:
            _make_config_cache_dir(config, dirname, clean=True)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:  # noqa: ARG001
    """Block regular file collection entirely when using --doc_mode."""
    if config.getoption("doc_mode"):
        return True
    return None


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Collect tests from doc images when --doc_mode is enabled."""
    if config.getoption("doc_mode"):
        items.clear()  # Clear previously collected items

        # Import the doc images module
        module_name = "pytest_pyvista.doc_mode"
        doc_module = importlib.import_module(module_name)
        module_file = Path(cast("Path", doc_module.__file__))

        # Collect test items from the module
        module_collector = pytest.Module.from_parent(parent=session, path=module_file)
        collected_items = list(module_collector.collect())

        # Remove tests if there are no test cases
        collected_items = [item for item in collected_items if not item.name.endswith("[NOTSET]")]

        items.extend(collected_items)
