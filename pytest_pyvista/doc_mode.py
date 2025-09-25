"""Test the images generated from building the documentation."""

from __future__ import annotations

from functools import partial
import logging
import multiprocessing
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Literal
from typing import cast
from typing import overload
import warnings

from PIL import Image
import pytest
import pyvista as pv

from .pytest_pyvista import DEFAULT_ERROR_THRESHOLD
from .pytest_pyvista import DEFAULT_WARNING_THRESHOLD
from .pytest_pyvista import PYVISTA_FAILED_IMAGE_CACHE_DIRNAME
from .pytest_pyvista import PYVISTA_GENERATED_IMAGE_CACHE_DIRNAME
from .pytest_pyvista import _AllowedImageFormats
from .pytest_pyvista import _check_compare_fail
from .pytest_pyvista import _ensure_dir_exists
from .pytest_pyvista import _EnvInfo
from .pytest_pyvista import _get_file_paths
from .pytest_pyvista import _get_generated_image_path
from .pytest_pyvista import _get_option_from_config_or_ini
from .pytest_pyvista import _is_master
from .pytest_pyvista import _make_config_cache_dir
from .pytest_pyvista import _paths_from_strings
from .pytest_pyvista import _test_compare_images
from .pytest_pyvista import _validate_image_cache_dir  # noqa: F401

TEST_CASE_NAME = "_pytest_pyvista_test_case"
TEST_CASE_NAME_VTKSZ_FILE_SIZE = "_pytest_pyvista_test_case_vtksz"

logger = logging.getLogger("pytest-pyvista")
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

multiprocessing.set_start_method("spawn", force=True)


class _VtkszFileSizeTestCase:
    _max_vtksz_file_size: int | None

    @classmethod
    def init_from_config(cls, config: pytest.Config) -> None:
        max_file_size = _get_option_from_config_or_ini(config, "max_vtksz_file_size")
        cls._max_vtksz_file_size = None if max_file_size is None else int(max_file_size)

    def __init__(self, test_name: str, input_path: Path, max_vtksz_file_size: int) -> None:
        self.test_name = test_name
        self.input_path = input_path
        self.max_vtksz_file_size = max_vtksz_file_size


class _DocVerifyImageCache:
    doc_images_dir: Path
    image_cache_dir: Path
    generated_image_dir: Path
    failed_image_dir: Path
    generate_subdirs: bool
    image_format: _AllowedImageFormats
    max_image_size: int | None
    include_vtksz: bool
    _verbose: bool = False
    _terminalreporter: pytest.TerminalReporter = None

    @classmethod
    def init_from_config(cls, config: pytest.Config) -> None:
        def require_existing_dir(option: str) -> Path:
            """Fetch a required directory option and ensure it's valid."""
            path = _get_option_from_config_or_ini(config, option, is_dir=True)
            if path is None:
                msg = f"{option!r} must be specified when using --doc_mode"
                raise ValueError(msg)
            if not path.is_dir():
                msg = f"{option!r} must be a valid directory. Got:\n{path}."
                raise ValueError(msg)
            return path

        def optional_dir_else_cache(option: str, dirname: str) -> Path:
            """Fetch an optional directory option or save to cache if missing."""
            path = _get_option_from_config_or_ini(config, option, is_dir=True)
            if path is None:
                # Save to cache
                return _make_config_cache_dir(config, dirname)
            return path

        cls.doc_images_dir = require_existing_dir("doc_images_dir")
        cls.image_cache_dir = cast("Path", _get_option_from_config_or_ini(config, "image_cache_dir", is_dir=True))
        _ensure_dir_exists(cls.image_cache_dir, "cache image dir")

        cls.generated_image_dir = optional_dir_else_cache("generated_image_dir", dirname=PYVISTA_GENERATED_IMAGE_CACHE_DIRNAME)
        cls.failed_image_dir = optional_dir_else_cache("failed_image_dir", dirname=PYVISTA_FAILED_IMAGE_CACHE_DIRNAME)

        cls.image_format = cast("_AllowedImageFormats", _get_option_from_config_or_ini(config, "image_format"))
        cls.max_image_size = cast("int | None", _get_option_from_config_or_ini(config, "max_image_size"))
        cls.generate_subdirs = bool(_get_option_from_config_or_ini(config, "generate_subdirs"))

        cls.include_vtksz = bool(_get_option_from_config_or_ini(config, "include_vtksz"))

        cls._verbose = config.option.verbose
        cls._terminalreporter = config.pluginmanager.get_plugin("terminalreporter")

        if not any(cls.doc_images_dir.iterdir()) and not any(cls.image_cache_dir.iterdir()):
            msg = (
                "No doc images or cache images found. The doc images dir:\n"
                f"  {cls.doc_images_dir}\n"
                "and image cache dir:\n"
                f"  {cls.image_cache_dir}\n"
                "cannot both be empty."
            )
            raise RuntimeError(msg)

    def __init__(
        self, test_name: str, *, docs_image_path: Path | None, cached_image_path: Path | None, env_info: str | _EnvInfo, input_path: Path | None
    ) -> None:
        self.test_name = test_name
        self.test_image_path = docs_image_path
        self.cached_image_path = cached_image_path
        self.env_info = env_info
        self.input_path = input_path


def _flatten_path(path: Path) -> Path:
    return Path("_".join(path.parts))


def _preprocess_all_images_for_test_cases(num_workers: int = 1) -> tuple[list[Path], list[Path], list[Path], list[Path]]:  # process test images
    pp = partial(
        _preprocess_build_images,
        _DocVerifyImageCache.doc_images_dir,
        _DocVerifyImageCache.generated_image_dir,
        image_format=_DocVerifyImageCache.image_format,
        generate_subdirs=_DocVerifyImageCache.generate_subdirs,
        return_input_paths=True,
        num_workers=num_workers,
    )
    # preprocess png, jpg, gif images
    input_paths: list[Path]
    test_image_paths: list[Path]
    input_paths, test_image_paths = pp(vtksz=False)  # type:ignore[assignment]

    vtksz_input_paths: list[Path] = []
    vtksz_test_image_paths: list[Path] = []
    if _DocVerifyImageCache.include_vtksz:
        # preprocess interactive vtksz files
        vtksz_input_paths, vtksz_test_image_paths = pp(vtksz=True)  # type:ignore[assignment]

    return input_paths, test_image_paths, vtksz_input_paths, vtksz_test_image_paths


@overload
def _preprocess_build_images(
    build_images_dir: Path,
    output_dir: Path,
    *,
    image_format: _AllowedImageFormats = "png",
    generate_subdirs: bool = False,
    vtksz: bool = False,
    return_input_paths: Literal[False] = False,
    num_workers: int = ...,
) -> list[Path]: ...
@overload
def _preprocess_build_images(
    build_images_dir: Path,
    output_dir: Path,
    *,
    image_format: _AllowedImageFormats = "png",
    generate_subdirs: bool = False,
    vtksz: bool = False,
    return_input_paths: Literal[True],
    num_workers: int = ...,
) -> tuple[list[Path], list[Path]]: ...
def _preprocess_build_images(  # noqa: PLR0913
    build_images_dir: Path,
    output_dir: Path,
    *,
    image_format: _AllowedImageFormats = "png",
    generate_subdirs: bool = False,
    vtksz: bool = False,
    return_input_paths: bool = False,
    num_workers: int = 1,
) -> list[Path] | tuple[list[Path], list[Path]]:
    """
    Read images from the build dir, resize them, and save to a flat output dir.

    All JPG, PNG and GIF files from the build are included, and are saved to
    the desired image format.

    """

    def _get_output_path(input_path: Path, relative_to: Path) -> Path:
        # input image from the docs may come from a nested directory,
        # so we flatten the file's relative path
        output_file_name = _flatten_path(input_path.relative_to(relative_to))
        output_file_name = output_file_name.with_suffix("." + image_format)
        return _get_generated_image_path(
            parent=output_dir, image_name=output_file_name, generate_subdirs=generate_subdirs, env_info=_EnvInfo(), vtksz=vtksz
        )

    def _preprocess_input_paths(input_paths: list[Path], relative_to: Path) -> list[Path]:
        output_paths: list[Path] = []
        for input_path in input_paths:
            output_dir.mkdir(exist_ok=True)
            output_path = _get_output_path(input_path, relative_to=relative_to)
            output_paths.append(output_path)
            _preprocess_image(input_path, output_path)
        return output_paths

    def _get_output(input_paths: list[Path], output_paths: list[Path]) -> list[Path] | tuple[list[Path], list[Path]]:
        if return_input_paths:
            return input_paths, output_paths
        return output_paths

    if vtksz:
        vtksz_paths = _get_file_paths(build_images_dir, ext="vtksz")
        # Print a general statement before processing.
        msg = f"[pyvista] Preprocessing {len(vtksz_paths)} vtksz files. This may take several minutes..."
        _DocVerifyImageCache._terminalreporter.write_line(msg, bold=True)  # noqa: SLF001
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            html_paths = _vtksz_to_html_files(vtksz_paths, tmppath)
            _render_all_html(html_paths, tmppath, num_workers=num_workers)
            input_paths = _get_file_paths(tmppath, ext="png")
            output_paths = _preprocess_input_paths(input_paths, relative_to=tmppath)
        return _get_output(vtksz_paths, output_paths)

    input_png = _get_file_paths(build_images_dir, ext="png")
    input_gif = _get_file_paths(build_images_dir, ext="gif")
    input_jpg = _get_file_paths(build_images_dir, ext="jpg")
    input_paths = input_png + input_gif + input_jpg
    output_paths = _preprocess_input_paths(input_paths, relative_to=build_images_dir)
    return _get_output(input_paths, output_paths)


def _preprocess_image(input_path: Path, output_path: Path) -> None:
    # Resize image based on plotter window size and save to output
    with Image.open(input_path) as im:
        im = im.convert("RGB") if im.mode != "RGB" else im  # noqa: PLW2901
        max_size = _DocVerifyImageCache.max_image_size if _DocVerifyImageCache.max_image_size is not None else max(*pv.global_theme.window_size)
        im.thumbnail(size=(max_size, max_size))
        im.save(output_path, quality="keep") if im.format == "JPEG" else im.save(output_path)


def _vtksz_to_html_files(vtksz_files: list[Path], output_dir: Path) -> list[Path]:
    from trame_vtk.tools.vtksz2html import embed_data_to_viewer_file  # noqa: PLC0415

    output_paths: list[Path] = []
    for path in vtksz_files:
        if _DocVerifyImageCache._verbose:  # noqa: SLF001
            msg = f"Converting {path.name}"
            logger.info(msg)

        with path.open("rb") as file:
            data = file.read()
        output_path = Path(output_dir) / f"{path.stem}.html"
        output_paths.append(output_path)
        embed_data_to_viewer_file(data, output_path)
    return output_paths


def _html_screenshots(html_files: list[Path], output_dir: Path, verbose: bool = False) -> list[Path]:  # noqa: FBT001, FBT002
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    output_paths: list[Path] = []
    width, height = pv.global_theme.window_size
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": width, "height": height})
        page = context.new_page()

        for html_file in html_files:
            if verbose:
                msg = f"Rendering {html_file.name}"
                logger.info(msg)

            output_path = output_dir / f"{html_file.stem}.png"
            page.goto(f"file://{html_file}")
            page.screenshot(path=output_path)
            output_paths.append(output_path)

        browser.close()
    return output_paths


def _render_all_html(
    html_files: list[Path],
    output_dir: Path,
    *,
    num_workers: int = 1,
) -> list[Path]:
    """Dispatch rendering across multiple processes."""
    verbose = _DocVerifyImageCache._verbose  # noqa: SLF001
    if num_workers == 1:
        return _html_screenshots(html_files, output_dir, verbose)

    def _split_batches(files: list[Path], n: int) -> list[list[Path]]:
        """Split a list of files into n roughly equal contiguous batches."""
        k, m = divmod(len(files), n)
        return [files[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]

    batches = _split_batches(html_files, num_workers)
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_nested = pool.starmap(
            _html_screenshots,
            [(batch, output_dir, verbose) for batch in batches],
        )

    # Flatten list of lists
    return [p for sublist in results_nested for p in sublist]


def _generate_test_cases(input_paths: list[Path], test_image_paths: list[Path], *, vtksz: bool = False) -> list[_DocVerifyImageCache]:  # noqa: C901
    """
    Generate a list of image test cases.

    This function:
        (1) Generates DocVerifyImageCache objects from build images (input_paths and test_image_paths)
        (2) Generates DocVerifyImageCache objects from cache images
        (3) Merges the two lists together and returns separate test cases to
            comparing all docs images to all cached images
    """
    test_cases_dict: dict = {}

    def add_to_dict(filepath: Path, key: str, input_path: Path | None = None) -> None:
        # Function for stuffing image paths into a dict.
        # We use a dict to allow for any entry to be made based on image path alone.
        # This way, we can defer checking for any mismatch between the cached and docs
        # images to test time.
        nonlocal test_cases_dict
        test_name = filepath.stem
        try:
            test_cases_dict[test_name]
        except KeyError:
            test_cases_dict[test_name] = {}
        test_cases_dict[test_name].setdefault(key, filepath)
        if input_path:
            test_cases_dict[test_name]["input_path"] = input_path

    for input_path, test_path in zip(input_paths, test_image_paths):
        add_to_dict(test_path.parent if _DocVerifyImageCache.generate_subdirs else test_path, "docs", input_path=input_path)  # type: ignore[func-returns-value]

    # process cached images
    cache_dir = _DocVerifyImageCache.image_cache_dir
    cached_image_paths = _get_file_paths(cache_dir, ext=_DocVerifyImageCache.image_format)
    for path in cached_image_paths:
        # Check if we have a single image or a dir with multiple images
        rel = path.relative_to(cache_dir)
        parts = rel.parts
        if len(parts) > 1:  # means it's nested
            # Use the first subdir as the test input instead of the image path
            first_subdir = parts[0]  # one dir down from base
            add_to_dict(cache_dir / first_subdir, "cached")
        else:
            add_to_dict(path, "cached")

    # flatten dict
    test_cases_list = []
    for test_name, content in sorted(test_cases_dict.items()):
        doc = content.get("docs", None)
        cache = content.get("cached", None)
        input_path = content.get("input_path", None)
        test_case = _DocVerifyImageCache(
            test_name=test_name, docs_image_path=doc, cached_image_path=cache, env_info=_EnvInfo(), input_path=input_path
        )

        input_is_vtksz = input_path is not None and input_path.suffix == ".vtksz"
        cache_is_from_vtksz = cache is not None and cache.stem.endswith("_vtksz")

        # Interactive test cases from vtksz files are mutually exclusive with regular image file tests
        if input_is_vtksz or cache_is_from_vtksz:
            if vtksz:
                # We have a vtksz test case AND vtksz cases were requested, so keep it
                test_cases_list.append(test_case)
            elif not _DocVerifyImageCache.include_vtksz:
                # We have a vtksz test cases, but no vtksz cases were requested AND they should not exist
                # This is likely an unused cache image and should be included so that an error is raised later
                test_cases_list.append(test_case)
            # Else skip the test case
        elif not vtksz:
            # Keep regular image cases (not vtksz files) otherwise
            test_cases_list.append(test_case)

    return test_cases_list


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate parametrized tests."""
    if TEST_CASE_NAME in metafunc.fixturenames:
        # Get paths that were generated by master
        paths = metafunc.config.paths if _is_master(metafunc.config) else metafunc.config.workerinput["paths"]

        input_paths = _paths_from_strings(paths["input_paths"])
        test_image_paths = _paths_from_strings(paths["test_image_paths"])
        vtksz_input_paths = _paths_from_strings(paths["vtksz_input_paths"])
        vtksz_test_image_paths = _paths_from_strings(paths["vtksz_test_image_paths"])

        test_cases: list[_DocVerifyImageCache] = _generate_test_cases(input_paths, test_image_paths)
        if _DocVerifyImageCache.include_vtksz:
            test_cases_vtksz: list[_DocVerifyImageCache] = _generate_test_cases(vtksz_input_paths, vtksz_test_image_paths, vtksz=True)
            test_cases.extend(test_cases_vtksz)

        ids = [case.test_name for case in test_cases]
        metafunc.parametrize(TEST_CASE_NAME, test_cases, ids=ids)

    if TEST_CASE_NAME_VTKSZ_FILE_SIZE in metafunc.fixturenames:
        if (max_vtksz_file_size := _VtkszFileSizeTestCase._max_vtksz_file_size) is None:  # noqa: SLF001
            metafunc.parametrize(TEST_CASE_NAME_VTKSZ_FILE_SIZE, [])
            return

        # Generate a separate test case for each vtksz file
        vtksz_files = _get_file_paths(_DocVerifyImageCache.doc_images_dir, ext="vtksz")

        test_cases_ = [_VtkszFileSizeTestCase(test_name=file.stem, input_path=file, max_vtksz_file_size=max_vtksz_file_size) for file in vtksz_files]
        ids = [case.test_name for case in test_cases_]
        metafunc.parametrize(TEST_CASE_NAME_VTKSZ_FILE_SIZE, test_cases_, ids=ids)


def _save_failed_test_image(source_path: Path, category: Literal["warnings", "errors", "errors_as_warnings"]) -> None:
    """Save test image from cache or build to the failed image dir."""
    _DocVerifyImageCache.failed_image_dir.mkdir(exist_ok=True)

    if source_path.is_relative_to(_DocVerifyImageCache.image_cache_dir):
        rel = source_path.relative_to(_DocVerifyImageCache.image_cache_dir)
        dest_relative_dir = Path("from_cache") / rel.parent
    else:
        rel = source_path.relative_to(_DocVerifyImageCache.generated_image_dir)
        dest_relative_dir = Path("from_build") / rel.parent

    dest_dir = _DocVerifyImageCache.failed_image_dir / category / dest_relative_dir
    dest_dir.mkdir(exist_ok=True, parents=True)
    dest_path = dest_dir / source_path.name
    copy_method = shutil.copytree if source_path.is_dir() else shutil.copy
    copy_method(source_path, dest_path)


@pytest.fixture
def doc_verify_image_cache(request: pytest.FixtureRequest) -> _DocVerifyImageCache:
    """Fixture to allow users to mutate test cases before they run."""
    test_case: _DocVerifyImageCache = request.node.callspec.params[TEST_CASE_NAME]
    request.config.hook.pytest_pyvista_doc_mode_hook(doc_verify_image_cache=test_case, request=request)
    return test_case


@pytest.fixture
def max_vtksz_file_size(request: pytest.FixtureRequest) -> _VtkszFileSizeTestCase:
    """Fixture to allow users to mutate test cases before they run."""
    test_case: _VtkszFileSizeTestCase = request.node.callspec.params[TEST_CASE_NAME_VTKSZ_FILE_SIZE]
    request.config.hook.pytest_pyvista_max_vtksz_file_size_hook(test_case=test_case, request=request)
    return test_case


@pytest.mark.usefixtures("_validate_image_cache_dir")
def test_images(_pytest_pyvista_test_case: _DocVerifyImageCache, doc_verify_image_cache: _DocVerifyImageCache) -> None:  # noqa: PT019, ARG001
    """Compare generated image with cached image."""
    test_case = _pytest_pyvista_test_case
    fail_msg, fail_source = _test_both_images_exist(
        filename=test_case.test_name, docs_image_path=test_case.test_image_path, cached_image_path=test_case.cached_image_path
    )
    if fail_msg:
        _save_failed_test_image(cast("Path", fail_source), "errors")
        pytest.fail(fail_msg)

    cached_image_path = cast("Path", test_case.cached_image_path)
    cached_input_is_file = cached_image_path.is_file()
    cached_image_paths = [cached_image_path] if cached_input_is_file else _get_file_paths(cached_image_path, ext=_DocVerifyImageCache.image_format)
    cached_image_paths = cast("list[Path]", cached_image_paths)
    current_cached_image_path = cached_image_paths[0]

    # Ensure test path is an image
    test_image_path = cast("Path", test_case.test_image_path)
    test_image_path = test_image_path if test_image_path.is_file() else _get_file_paths(test_image_path, ext=_DocVerifyImageCache.image_format)[0]
    if test_case.generate_subdirs:
        # Need to update the filename in case it's been modified by a plugin hook
        new_path = test_image_path.with_stem(str(test_case.env_info))
        if not new_path.is_file():
            test_image_path.rename(new_path)
            test_image_path = new_path

    warn_msg, fail_msg = _test_compare_images(
        test_name=test_case.test_name,
        test_image=test_image_path,
        cached_image=current_cached_image_path,
        allowed_error=DEFAULT_ERROR_THRESHOLD,
        allowed_warning=DEFAULT_WARNING_THRESHOLD,
    )

    # Try again and compare with other cached images
    if fail_msg and len(cached_image_paths) > 1:
        # Compare build image to other known valid versions
        msg_start = "This test has multiple cached images. It initially failed (as above)"
        for path in cached_image_paths[1:]:
            error = pv.compare_images(pv.read(test_image_path), pv.read(path))
            if _check_compare_fail(test_case.test_name, error, allowed_error=DEFAULT_ERROR_THRESHOLD) is None:
                # Convert failure into a warning
                warn_msg = fail_msg + (f"\n{msg_start} but passed when compared to:\n\t{path}")
                fail_msg = None
                current_cached_image_path = path
                break
        else:  # Loop completed - test still fails
            fail_msg += f"\n{msg_start} and failed again for all images in:\n\t{_DocVerifyImageCache.image_cache_dir / test_case.test_name!s}"

    if fail_msg:
        _save_failed_test_image(test_image_path, "errors")
        # Save all cached images since they all failed
        for path in cached_image_paths:
            _save_failed_test_image(path, "errors")
        pytest.fail(fail_msg)

    if warn_msg:
        parent_dir: Literal["errors_as_warnings", "warnings"] = "warnings" if cached_input_is_file else "errors_as_warnings"
        _save_failed_test_image(test_image_path, parent_dir)
        _save_failed_test_image(current_cached_image_path, parent_dir)
        warnings.warn(warn_msg, stacklevel=2)


def _test_both_images_exist(filename: str, docs_image_path: Path | None, cached_image_path: Path | None) -> tuple[str | None, Path | None]:
    def has_no_images(path: Path | None) -> bool:
        return path is None or (path.is_dir() and len(_get_file_paths(path, ext=_DocVerifyImageCache.image_format)) == 0)

    build_has_no_images = has_no_images(docs_image_path)
    cache_has_no_images = has_no_images(cached_image_path)

    if build_has_no_images or cache_has_no_images:
        if build_has_no_images:
            source_path = cached_image_path
            exists = "cache"
            missing = "docs build"
            exists_path = cached_image_path
            missing_path = _DocVerifyImageCache.doc_images_dir
        else:
            source_path = docs_image_path
            exists = "docs build"
            missing = "cache"
            exists_path = _DocVerifyImageCache.doc_images_dir
            missing_path = _DocVerifyImageCache.image_cache_dir

        msg = (
            f"Test setup failed for test image:\n"
            f"\t{filename}\n"
            f"The image exists in the {exists} directory:\n"
            f"\t{exists_path}\n"
            f"but is missing from the {missing} directory:\n"
            f"\t{missing_path}\n"
        )
        return msg, source_path
    return None, None


def test_vtksz_file_size(_pytest_pyvista_test_case_vtksz: _VtkszFileSizeTestCase, max_vtksz_file_size: _VtkszFileSizeTestCase) -> None:  # noqa: PT019, ARG001
    """Test vtksz file size is less than max allowed."""
    test_case = _pytest_pyvista_test_case_vtksz
    vtksz_file = test_case.input_path
    assert vtksz_file.is_file()  # noqa: S101
    size_bytes = vtksz_file.stat().st_size
    size_megabytes = round(size_bytes / 100_000) / 10.0
    if size_megabytes > test_case.max_vtksz_file_size:
        msg = (
            f"The interactive plot file is too large:"
            f"\n\t{vtksz_file}\n"
            f"Its size is {size_megabytes} MB, but must be less than {test_case.max_vtksz_file_size} MB."
            f"\nConsider reducing the complexity of the plot or forcing it to be static."
        )
        pytest.fail(msg)
