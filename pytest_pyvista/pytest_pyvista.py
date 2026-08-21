"""pytest-pyvista module."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from functools import cached_property
import gc
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
from typing import Literal
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
from pytest_pyvista.summary.record import ALL_STATUSES
from pytest_pyvista.summary.record import read_records
from pytest_pyvista.summary.render import write_report

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from collections.abc import Generator

    from _pytest.terminal import TerminalReporter
    import xdist.workermanage

    from pytest_pyvista.summary.record import CacheWriteReason
    from pytest_pyvista.summary.session import SummarySession

VISITED_CACHED_IMAGE_NAMES: set[str] = set()
SKIPPED_CACHED_IMAGE_NAMES: set[str] = set()
PYVISTA_IMAGE_NAMES_CACHE_DIRNAME = "pyvista_image_names_dir"
PYVISTA_GENERATED_IMAGE_CACHE_DIRNAME = "pyvista_generated_image_dir"
PYVISTA_FAILED_IMAGE_CACHE_DIRNAME = "pyvista_failed_image_dir"
PYVISTA_SUMMARY_RECORDS_DIRNAME = "pyvista_summary_records_dir"

PARSER_GROUP_NAME = "pyvista"
DEFAULT_ERROR_THRESHOLD: float = 500.0
DEFAULT_WARNING_THRESHOLD: float = 200.0
_DOC_MODE_CLI_ARGS: set[str] = set()
_UNIT_TEST_CLI_ARGS: set[str] = set()

_AllowedImageFormats = Literal["png", "jpg"]
_OriginalImageFormats = _AllowedImageFormats | Literal["gif", "vtksz"]


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

        _add_unit_test_cli_option(
            "--summary_html",
            action="store_const",
            const=True,
            default=None,
            help="Generate an HTML image summary report at the end of the run.",
        )
        parser.addini("summary_html", type="bool", default=None, help="Generate an HTML image summary report.")

        _add_unit_test_cli_option(
            "--summary_html_dir",
            action="store",
            default=None,
            help="Directory for the HTML image summary report. Implies --summary_html.",
        )
        parser.addini("summary_html_dir", default=None, help="Directory for the HTML image summary report.")

        _add_unit_test_cli_option(
            "--summary_html_include",
            action="store",
            default=None,
            help="Comma-separated statuses to include in the summary report.",
        )
        parser.addini("summary_html_include", default=None, help="Comma-separated statuses to include in the summary report.")

        _add_unit_test_cli_option(
            "--summary_html_max_image_size",
            action="store",
            default=None,
            help="Longest-edge pixel limit for images shown inline in the summary report.",
        )
        parser.addini("summary_html_max_image_size", default=None, help="Longest-edge pixel limit for summary report images.")

        _add_unit_test_cli_option(
            "--summary_html_full_size",
            action="store",
            choices=["none", "failing", "all"],
            default=None,
            help="Which records retain a full-resolution copy in the summary report.",
        )
        parser.addini("summary_html_full_size", default=None, help="Which records retain a full-resolution image copy.")

        _add_unit_test_cli_option(
            "--summary_html_embed",
            action="store_const",
            const=True,
            default=None,
            help="Embed summary report images as data URIs in a single HTML file.",
        )
        parser.addini("summary_html_embed", type="bool", default=None, help="Embed summary report images as data URIs.")

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

    # VTK resource cleanup options
    parser.addini(
        "pyvista_close_all",
        type="bool",
        default=True,
        help="Automatically close all plotters and run gc.collect() after each test (default: True).",
    )


@contextlib.contextmanager
def _summary_capture_failures_are_warnings() -> Generator[None, None, None]:
    """
    Downgrade any failure raised while recording a summary record to a warning.

    The summary report only observes a test run, so it must never change what a test
    reports nor skip the cleanup that follows a comparison. Anything the reporting code
    can raise - an unwritable or full report directory, a truncated screenshot that PIL
    refuses to decode - is caught here and surfaced as a warning instead.
    """
    try:
        yield
    except Exception as error:  # noqa: BLE001 - deliberately total: no reporting failure may reach the test
        # `str` rather than `repr` because an OSError names the offending path only in `str`,
        # and that path is the one thing the reader needs. Matches _write_summary_report's
        # equivalent warning.
        warnings.warn(f"pytest-pyvista could not record an image in the summary report: {type(error).__name__}: {error}", stacklevel=3)


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
    summary_session: SummarySession | None = None
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

        pending_error: RegressionError | None = None

        test_name = f"{self.test_name}_{self.n_calls}" if self.n_calls > 0 else self.test_name
        self.n_calls += 1

        allowed_error, allowed_warning = self._allowed_thresholds()

        # cached image name. We remove the first 5 characters of the function name
        # "test_" to get the name for the image.
        image_name = _image_name_from_test_name(test_name, image_format=self.image_format)

        if VerifyImageCache._is_skipped(
            skip=self.skip,
            windows_skip_image_cache=self.windows_skip_image_cache,
            macos_skip_image_cache=self.macos_skip_image_cache,
            ignore_image_cache=self.ignore_image_cache,
        ):
            SKIPPED_CACHED_IMAGE_NAMES.add(image_name)
            skip_summary = VerifyImageCache.summary_session
            if skip_summary is not None:
                # The comparison never runs, but the card should still name the baseline it
                # skipped over, so resolve the one the comparison would have used.
                skipped_baseline = self._candidate_baselines(image_name)[0]
                # Reporting must never change a test's outcome, so any failure here is
                # downgraded to a warning (see the identical guard on the main capture).
                with _summary_capture_failures_are_warnings():
                    skip_summary.capture(
                        test_name=test_name,
                        image_name=image_name,
                        call_index=self.n_calls - 1,
                        baseline_source=skipped_baseline if skipped_baseline.is_file() else None,
                        generated_source=None,
                        cache_destination=skipped_baseline,
                        skipped=True,
                        skip_reason=self._skip_reason(),
                        baseline_existed=skipped_baseline.is_file(),
                        cache_write_reason=None,
                        error=None,
                        error_threshold=allowed_error,
                        warning_threshold=allowed_warning,
                        high_variance_test=self.high_variance_test,
                        image_format=self.image_format,
                        env_info=str(self.env_info),
                    )
            return

        VISITED_CACHED_IMAGE_NAMES.add(image_name)

        image_dirname = Path(self.cache_dir, Path(image_name).stem)

        cached_image_paths = self._candidate_baselines(image_name)
        current_cached_image = cached_image_paths[0]

        summary = VerifyImageCache.summary_session
        preserved_baseline = None
        if summary is not None and current_cached_image.is_file():
            preserved_baseline = summary.preserve_baseline(current_cached_image, test_name, self.n_calls - 1)

        if not current_cached_image.is_file() and not (self.allow_unused_generated or self.add_missing_images or self.reset_image_cache):
            # Raise error since the cached image does not exist and will not be added later

            # Save images as needed before error
            if self.generated_image_dir is not None:
                self._save_generated_image(plotter, image_name=image_name)
            if self.failed_image_dir is not None:
                self._save_failed_test_images("error", plotter, image_name)

            if summary is not None:
                # Recorded here, after the render above is on disk and before the raise, because
                # this is the *default* path for a suite that has no baselines yet: without it a
                # bare `--summary_html` run on a fresh suite reports nothing at all, which is the
                # most likely first experience of the feature.
                self._capture_new_image(
                    summary,
                    test_name=test_name,
                    image_name=image_name,
                    cache_destination=current_cached_image,
                )

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
            if summary is not None:
                self._capture_new_image(
                    summary,
                    test_name=test_name,
                    image_name=image_name,
                    cache_destination=current_cached_image,
                )
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
        matched_alternate = False
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
                    matched_alternate = True
                    break
            else:  # Loop completed - test still fails
                fail_msg += f"\n{msg_start} and failed again for all images in:\n\t{Path(self.cache_dir, test_name_no_prefix)!s}"

        if matched_alternate and summary is not None:
            # The record must describe the baseline that actually matched: its error, its diff
            # and the report's baseline panel are all relative to `current_cached_image`, not to
            # candidate 0. Preserving it this late is safe because the only cache write that
            # could overwrite it below (`reset_only_failed`) is unreachable once `fail_msg` is
            # None, which it always is here.
            preserved_baseline = summary.preserve_baseline(current_cached_image, test_name, self.n_calls - 1)

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
                pending_error = RegressionError(fail_msg)

        if warn_msg and pending_error is None:
            # Skipped when a hard failure is pending (fail_msg set, not reset_only_failed): the
            # original control flow raised before ever reaching this block in that case, and a
            # deferred raise must not change what gets saved to `failed_image_dir`.
            parent_dir: Literal["errors_as_warning", "warning"] = "errors_as_warning" if image_dirname.is_dir() else "warning"
            if self.failed_image_dir is not None:
                self._save_failed_test_images(parent_dir, plotter, image_name, cache_image_path=current_cached_image)
            warnings.warn(warn_msg, stacklevel=2)

        if summary is not None:
            # Reporting must never change a test's outcome nor skip the cleanup below, so any
            # failure raised while recording is downgraded to a warning.
            with _summary_capture_failures_are_warnings():
                summary.capture(
                    test_name=test_name,
                    image_name=image_name,
                    call_index=self.n_calls - 1,
                    baseline_source=preserved_baseline,
                    generated_source=_get_generated_image_path(
                        parent=cast("Path", self.generated_image_dir),
                        image_name=image_name,
                        generate_subdirs=self.generate_subdirs,
                        env_info=self.env_info,
                    ),
                    cache_destination=current_cached_image,
                    baseline_existed=preserved_baseline is not None,
                    cache_write_reason=self._cache_write_reason(baseline_existed=preserved_baseline is not None, failed=bool(fail_msg)),
                    error=None,
                    error_threshold=allowed_error,
                    warning_threshold=allowed_warning,
                    high_variance_test=self.high_variance_test,
                    matched_alternate=matched_alternate,
                    matched_baseline=str(current_cached_image),
                    candidate_baselines=[str(path) for path in cached_image_paths],
                    image_format=self.image_format,
                    env_info=str(self.env_info),
                )

        if pending_error is not None:
            remove_plotter_close_callback()
            raise pending_error

    def _allowed_thresholds(self) -> tuple[float, float]:
        """
        Return the ``(error, warning)`` thresholds this comparison is judged against.

        A high-variance test is judged against its own, far looser pair. Resolved in one place
        so that every caller - the comparison itself and the summary records describing it -
        cannot disagree about which pair applies.
        """
        if self.high_variance_test:
            return self.var_error_value, self.var_warning_value
        return self.error_value, self.warning_value

    def _capture_new_image(
        self,
        summary: SummarySession,
        *,
        test_name: str,
        image_name: str,
        cache_destination: Path,
    ) -> None:
        """
        Record a rendered image that has no baseline in the cache, as a ``new`` card.

        Both callers arrive here having just written the render to ``generated_image_dir`` and
        with nothing to compare it against: the missing-baseline failure, and the
        ``--allow_unused_generated`` early return. Without this the image would vanish from the
        report entirely, with nothing to say it was ever rendered; recorded as ``new``, the
        reader can approve it into the cache instead.

        Reporting must never change a test's outcome nor skip the cleanup that follows, so any
        failure raised while recording is downgraded to a warning.
        """
        error_threshold, warning_threshold = self._allowed_thresholds()
        with _summary_capture_failures_are_warnings():
            summary.capture(
                test_name=test_name,
                image_name=image_name,
                call_index=self.n_calls - 1,
                baseline_source=None,
                generated_source=_get_generated_image_path(
                    parent=cast("Path", self.generated_image_dir),
                    image_name=image_name,
                    generate_subdirs=self.generate_subdirs,
                    env_info=self.env_info,
                ),
                cache_destination=cache_destination,
                baseline_existed=False,
                cache_write_reason=None,
                error=None,
                error_threshold=error_threshold,
                warning_threshold=warning_threshold,
                high_variance_test=self.high_variance_test,
                image_format=self.image_format,
                env_info=str(self.env_info),
            )

    def _candidate_baselines(self, image_name: str) -> list[Path]:
        """
        Return every cached image this one may be compared against, best candidate first.

        A test's baselines are either the single flat ``<cache_dir>/<image_name>`` or, when
        ``<cache_dir>/<stem>/`` exists, every image inside it - the subdirectory wins outright
        when both are present. Candidates are paths, not guarantees: neither the flat file nor
        the default path standing in for an empty subdirectory need exist, and it is for the
        caller to decide what a missing baseline means. Resolving this in one place is what
        lets a comparison that never runs - a skipped test - report the same baseline the
        comparison would have used.
        """
        image_dirname = Path(self.cache_dir, Path(image_name).stem)
        if not image_dirname.is_dir():
            return [Path(self.cache_dir, image_name)]
        # An empty dir yields the default expected image path
        return _get_file_paths(image_dirname, ext=self.image_format) or [image_dirname / f"{self.env_info}.{self.image_format}"]

    def _skip_reason(self) -> str:
        """Return the flag responsible for skipping this image comparison."""
        if self.skip:
            return "skip"
        if self.ignore_image_cache:
            return "ignore_image_cache"
        if os.name == "nt" and self.windows_skip_image_cache:
            return "windows_skip_image_cache"
        if platform.system() == "Darwin" and self.macos_skip_image_cache:
            return "macos_skip_image_cache"
        return "skip"

    def _cache_write_reason(self, *, baseline_existed: bool, failed: bool) -> CacheWriteReason | None:
        """Return which policy wrote this image to the cache during this run, if any."""
        if self.add_missing_images and not baseline_existed:
            return "add_missing_images"
        if self.reset_image_cache and not self.reset_only_failed:
            return "reset_image_cache"
        if self.reset_only_failed and failed:
            return "reset_only_failed"
        return None

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
    """
    Return the path a generated render is written to, creating its parent directory.

    The ``mkdir`` below is intentional and load-bearing, not a stray side effect: callers pass a
    path whose directory may not exist yet - under ``--generate_subdirs`` a per-image
    subdirectory never does - and the summary report's auto-provisioned generated directory
    relies on it too. Do not remove it as a "cleanup"; the writes that follow would fail.
    """
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
def pytest_terminal_summary(terminalreporter: TerminalReporter, exitstatus: int, config: pytest.Config) -> None:  # noqa: ARG001
    """Execute after the whole test run completes."""
    if hasattr(config, "workerinput"):
        # on an pytest-xdist worker node, exit early
        return

    try:
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
    finally:
        # The report is written here and no later: `pytest_unconfigure` wipes the records
        # directory and the preserved baselines inside it. In a `finally` so that the
        # unused-cache exit above still leaves a report behind - it is the report that would
        # show which images are unused.
        #
        # Nothing in here may raise. `pytest.exit`'s `Exit` may be propagating through this
        # `finally`, and an exception raised here would replace it, turning a TESTS_FAILED
        # exit into an internal error. `_write_summary_report` already degrades a failed
        # write to a warning line; this suppression covers the two things left over - the
        # enablement lookup and the terminal write itself - for which there is by definition
        # no way left to report anything.
        with contextlib.suppress(Exception):
            if _summary_html_enabled(config):
                _write_summary_report(config, terminalreporter)

    VISITED_CACHED_IMAGE_NAMES.clear()
    SKIPPED_CACHED_IMAGE_NAMES.clear()


def _summary_version(lookup: Callable[[], object]) -> str:
    """Return the version ``lookup`` reports, or ``"unknown"`` when it cannot be read."""
    try:
        return str(lookup())
    except Exception:  # noqa: BLE001 - one unreadable version must not cost the whole report
        return "unknown"


def _summary_report_metadata(config: pytest.Config, cache_dir: Path | None) -> dict[str, str]:
    """Describe this run for the report header, tolerating a version that cannot be read."""
    return {
        "Generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Run id": _pyvista_run_id(config),
        "Cache directory": str(cache_dir),
        "Python": _summary_version(platform.python_version),
        "PyVista": _summary_version(lambda: pyvista.__version__),
        "VTK": _summary_version(lambda: vtkmodules.__version__),
        "pytest": _summary_version(lambda: pytest.__version__),
    }


def _write_summary_report(config: pytest.Config, terminalreporter: TerminalReporter) -> None:
    """
    Combine every worker's records into one HTML report and name it on the terminal.

    The report only observes a run, so writing it must never break one: an unwritable or
    full report directory, or a read-only filesystem, degrades to a warning line here and
    changes neither the exit status nor the rest of the terminal summary.
    """
    records_dir = getattr(config, PYVISTA_SUMMARY_RECORDS_DIRNAME, None)
    if records_dir is None:
        # No records directory was ever created, so this run captured nothing: the report was
        # switched on through an ini option while `--doc_mode` was in force, which is not
        # supported and is deliberately inert rather than an error.
        return

    try:
        cache_dir = _get_option_from_config_or_ini(config, "image_cache_dir", is_dir=True)
        path = write_report(
            read_records(Path(records_dir)),
            _summary_report_dir(config),
            run_id=_pyvista_run_id(config),
            metadata=_summary_report_metadata(config, cache_dir),
            embed=bool(_get_option_from_config_or_ini(config, "summary_html_embed")),
        )
    except Exception as error:  # noqa: BLE001 - deliberately total: writing the report may not break the run
        # `str` rather than `repr` because an OSError names the offending path only in `str`.
        message, warning = f"pytest-pyvista WARNING: could not write the image summary report: {type(error).__name__}: {error}", True
    else:
        message, warning = f"pytest-pyvista image summary report: {path}", False

    terminalreporter.ensure_newline()
    terminalreporter.write_line(message, yellow=warning)


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


DEFAULT_SUMMARY_HTML_DIR = "image_test_report"
DEFAULT_SUMMARY_HTML_MAX_IMAGE_SIZE = 400
DEFAULT_SUMMARY_HTML_FULL_SIZE = "failing"


def _summary_html_enabled(pytestconfig: pytest.Config) -> bool:
    """Return True if the HTML summary report should be generated."""
    if _get_option_from_config_or_ini(pytestconfig, "summary_html"):
        return True
    # An empty string (e.g. `--summary_html_dir=` or a blank ini value) must not enable the
    # report - only a genuinely set directory should.
    return bool(_get_option_from_config_or_ini(pytestconfig, "summary_html_dir"))


def _summary_report_dir(pytestconfig: pytest.Config) -> Path:
    """
    Return the directory holding the report page and its images.

    One resolution serves both writers: the image store fills ``images/`` during the run and
    the page is written here at the end of it. Were they to disagree, the page would silently
    reference images that are not beside it.
    """
    return pytestconfig.rootpath / str(_get_option_from_config_or_ini(pytestconfig, "summary_html_dir") or DEFAULT_SUMMARY_HTML_DIR)


def _summary_html_statuses(pytestconfig: pytest.Config) -> tuple[str, ...]:
    """Return the statuses to write to the summary report."""
    value = _get_option_from_config_or_ini(pytestconfig, "summary_html_include")
    if not value:
        return ALL_STATUSES
    statuses = tuple(part.strip() for part in str(value).split(",") if part.strip())
    unknown = [status for status in statuses if status not in ALL_STATUSES]
    if unknown:
        msg = f"--summary_html_include: unknown status {unknown[0]!r}. Choose from: {', '.join(ALL_STATUSES)}"
        raise pytest.UsageError(msg)
    return statuses


def _pyvista_run_id(config: pytest.Config) -> str:
    """
    Return the id identifying this whole test run, minting it once on first use.

    Every record of a run carries this id, so it must be stored on the config rather than
    minted per caller: xdist workers receive it through ``workerinput``, and the master
    reads back the same value here whether or not xdist is in use.
    """
    run_id = getattr(config, "pyvista_run_id", None)
    if run_id is None:
        run_id = str(uuid.uuid4())
        config.pyvista_run_id = run_id  # type: ignore[attr-defined]
    return str(run_id)


def _make_summary_session(pytestconfig: pytest.Config) -> SummarySession | None:
    """Build (once) the SummarySession for this run, or None when the report is off."""
    if not _summary_html_enabled(pytestconfig):
        return None

    existing = getattr(pytestconfig, "_pyvista_summary_session", None)
    if existing is not None:
        return cast("SummarySession", existing)

    from pytest_pyvista.summary.session import SummarySession  # noqa: PLC0415
    from pytest_pyvista.summary.store import ReportImageStore  # noqa: PLC0415

    report_dir = _summary_report_dir(pytestconfig)
    max_size = int(_get_option_from_config_or_ini(pytestconfig, "summary_html_max_image_size") or DEFAULT_SUMMARY_HTML_MAX_IMAGE_SIZE)
    full_size = str(_get_option_from_config_or_ini(pytestconfig, "summary_html_full_size") or DEFAULT_SUMMARY_HTML_FULL_SIZE)

    worker_input = getattr(pytestconfig, "workerinput", None)
    records_dir = Path(worker_input["pyvista_records_dir"]) if worker_input else Path(getattr(pytestconfig, PYVISTA_SUMMARY_RECORDS_DIRNAME))
    session = SummarySession(
        run_id=worker_input["pyvista_run_id"] if worker_input else _pyvista_run_id(pytestconfig),
        records_dir=records_dir,
        store=ReportImageStore(report_dir, max_image_size=max_size, full_size=full_size),  # type: ignore[arg-type]
        worker_id=worker_input["workerid"] if worker_input else "master",
        statuses=_summary_html_statuses(pytestconfig),
        cache_dir=cast("Path", _get_option_from_config_or_ini(pytestconfig, "image_cache_dir", is_dir=True)),
    )
    pytestconfig._pyvista_summary_session = session  # type: ignore[attr-defined]  # noqa: SLF001
    return session


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
        if pytestconfig.getoption("ignore_image_cache"):
            return

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
        # The outer suppression covers listing the directory itself, which can fail (e.g. on
        # permissions) and must not propagate out of pytest_configure.
        with contextlib.suppress(OSError):
            for item in newdir.iterdir():
                # Suppress per-item too, so one failure (e.g. a locked file) doesn't abort the
                # rest of the cleanup - this dir can contain subdirectories (e.g. the summary
                # report's preserved-baseline copies), which plain unlink() cannot remove.
                with contextlib.suppress(OSError):
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
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

    # Ini-configured summary options are simply inactive under --doc_mode (not an error); the
    # explicit CLI combination `--summary_html --doc_mode` is already rejected by the loop above.
    if not doc_mode and _summary_html_enabled(config):
        # Validate eagerly so a typo fails the run rather than the report.
        _summary_html_statuses(config)
        if is_master:
            _make_config_cache_dir(config, PYVISTA_SUMMARY_RECORDS_DIRNAME, clean=True)
            _pyvista_run_id(config)

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
        if run_id := getattr(node.config, "pyvista_run_id", None):
            node.workerinput["pyvista_run_id"] = run_id
            node.workerinput["pyvista_records_dir"] = str(getattr(node.config, PYVISTA_SUMMARY_RECORDS_DIRNAME))


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
    VerifyImageCache.summary_session = _make_summary_session(pytestconfig)

    cache_dir = cast("Path", _get_option_from_config_or_ini(pytestconfig, "image_cache_dir", is_dir=True))
    gen_dir = _get_option_from_config_or_ini(pytestconfig, "generated_image_dir", is_dir=True)
    if gen_dir is None and _summary_html_enabled(pytestconfig):
        # The report needs a generated render to copy, and the copy must outlive the run: an
        # approvals.json exported from this report names it as the source, and
        # `pytest-pyvista-approve` reads it back afterwards. The pytest cache is wiped by
        # `pytest_unconfigure`, so keep the renders beside the report instead - that directory
        # is the artifact users keep, zip and upload from CI, so the sources belong with it.
        # Created here rather than left to `_get_generated_image_path`'s first write, so that
        # `VerifyImageCache.__init__` does not warn about a directory the plugin provisioned.
        gen_dir = _summary_report_dir(pytestconfig) / "generated"
        try:
            gen_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            # Only this one subdirectory failed; the report directory around it may be perfectly
            # healthy, in which case the run still writes a report that looks entirely normal
            # while its manifest names sources in the pytest cache that `pytest_unconfigure`
            # deletes on the way out. Nothing downstream can explain that to the reader - the
            # approve CLI sees only a missing file, and `_ensure_dir_exists` cannot warn about a
            # directory that does now exist - so the degrade has to be announced here. A raise
            # would error this fixture and fail every test, which the report must never do.
            fallback: Path | None = None
            with contextlib.suppress(Exception):
                # `config.cache` is None under `-p no:cacheprovider`, so even the fallback can
                # fail. Losing the renders is a degraded report; raising is a broken run.
                fallback = _make_config_cache_dir(pytestconfig, PYVISTA_GENERATED_IMAGE_CACHE_DIRNAME)
            destination = (
                f"They are being written to {fallback} instead, which is removed when the run finishes"
                if fallback
                else "They cannot be written at all"
            )
            warnings.warn(
                f"pytest-pyvista could not create {gen_dir} for this run's generated images: {type(error).__name__}: {error}. "
                f"{destination}, so the image summary report is still written but approvals exported from it cannot be "
                f"applied. Set --generated_image_dir to a writable location to keep the renders the approvals need.",
                # 1, not 2: the caller is pytest's fixture machinery, and naming
                # `_pytest/fixtures.py` as the source of a pytest-pyvista warning helps nobody.
                stacklevel=1,
            )
            gen_dir = fallback
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


@pytest.fixture(autouse=True)
def _close_plotters_clear_trame_servers(pytestconfig: pytest.Config) -> Generator[None, None, None]:
    """
    Cleanup fixture.

    This teardown fixture serves mutltiple purposes:
    - closing all plotters,
    - clearing trame servers registry (to prevent test collision),
    - forcing garbage collection.
    """
    yield

    try:
        from trame.app.core import AVAILABLE_SERVERS  # noqa: PLC0415
    except ImportError:
        ...
    else:
        AVAILABLE_SERVERS.clear()

    if pytestconfig.getini("pyvista_close_all"):
        pyvista.close_all()
        gc.collect()


_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

if _APPLE_SILICON:
    from Foundation import NSAutoreleasePool


@pytest.fixture(autouse=True)
def _pyvista_macos_autorelease() -> Generator[None, None, None]:
    """Drain the Cocoa autorelease pool between tests on macOS Apple Silicon."""
    if not _APPLE_SILICON:
        yield
        return
    pool = NSAutoreleasePool.alloc().init()
    yield
    del pool


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
        for dirname in [
            PYVISTA_FAILED_IMAGE_CACHE_DIRNAME,
            PYVISTA_GENERATED_IMAGE_CACHE_DIRNAME,
            PYVISTA_IMAGE_NAMES_CACHE_DIRNAME,
            PYVISTA_SUMMARY_RECORDS_DIRNAME,
        ]:
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
