"""Test the images generated from building the documentation."""

from __future__ import annotations

import glob
import os
from pathlib import Path
import shutil
import tempfile
from typing import ClassVar
from typing import Literal
from typing import NamedTuple
from typing import cast
import warnings

from PIL import Image
import pytest
import pyvista as pv

from .pytest_pyvista import _ensure_dir_exists
from .pytest_pyvista import _get_option_from_config_or_ini

MAX_IMAGE_DIM = 400  # pixels


class _DocTestInfo:
    doc_images_dir: Path
    doc_image_cache_dir: Path
    doc_generated_image_dir: Path
    doc_failed_image_dir: Path
    _tempdirs: ClassVar[list[tempfile.TemporaryDirectory]] = []

    @classmethod
    def init_dirs(cls, config: pytest.Config) -> None:
        must_be_specified_template = "{!r} must be specified when using --doc_mode"
        must_be_valid_template = "{!r} must be a valid directory. Got:\n{}."

        option = "doc_images_dir"
        doc_images_dir = _get_option_from_config_or_ini(config, option, is_dir=True)
        if doc_images_dir is None:
            raise ValueError(must_be_specified_template.format(option))
        if not doc_images_dir.is_dir():
            raise ValueError(must_be_valid_template.format(option, doc_images_dir))
        cls.doc_images_dir = doc_images_dir

        option = "doc_image_cache_dir"
        doc_image_cache_dir = _get_option_from_config_or_ini(config, option, is_dir=True)
        if doc_image_cache_dir is None:
            raise ValueError(must_be_specified_template.format(option))
        if not doc_image_cache_dir.is_dir():
            raise ValueError(must_be_valid_template.format(option, doc_image_cache_dir))
        cls.doc_image_cache_dir = doc_image_cache_dir

        doc_generated_image_dir = _get_option_from_config_or_ini(config, "doc_generated_image_dir", is_dir=True)
        if doc_generated_image_dir is None:
            # create a temp dir and keep it around until test session ends
            tempdir = tempfile.TemporaryDirectory(prefix="pytest_doc_generated_image_dir")
            cls._tempdirs.append(tempdir)
            path = Path(tempdir.name)
        else:
            path = Path(doc_generated_image_dir)
        cls.doc_generated_image_dir = path

        doc_failed_image_dir = _get_option_from_config_or_ini(config, "doc_failed_image_dir", is_dir=True)
        if doc_failed_image_dir is None:
            tempdir = tempfile.TemporaryDirectory(prefix="pytest_doc_failed_image_dir")
            cls._tempdirs.append(tempdir)
            path = Path(tempdir.name)
        else:
            path = doc_failed_image_dir
        cls.doc_failed_image_dir = path


class _TestCaseTuple(NamedTuple):
    test_name: str
    docs_image_path: str
    cached_image_path: str


def _get_file_paths(dir_: str, ext: str) -> list[str]:
    """Get all paths of files with a specific extension inside a directory tree."""
    pattern = str(Path(dir_) / "**" / ("*." + ext))
    return sorted(glob.glob(pattern, recursive=True))  # noqa: PTH207


def _flatten_path(path: str) -> str:
    return "_".join(os.path.split(path))[1:]


def _preprocess_build_images(build_images_dir: str, output_dir: str) -> list[str]:
    """
    Read images from the build dir, resize them, and save as JPG to a flat output dir.

    All PNG and GIF files from the build are included, and are saved as JPG.

    """
    input_png = _get_file_paths(build_images_dir, ext="png")
    input_gif = _get_file_paths(build_images_dir, ext="gif")
    input_jpg = _get_file_paths(build_images_dir, ext="jpg")
    output_paths = []
    Path(output_dir).mkdir(exist_ok=True)
    for input_path in input_png + input_gif + input_jpg:
        # input image from the docs may come from a nested directory,
        # so we flatten the file's relative path
        output_file_name = _flatten_path(os.path.relpath(input_path, build_images_dir))
        output_file_name = str(Path(output_file_name).with_suffix(".jpg"))
        output_path = str(Path(output_dir) / output_file_name)
        output_paths.append(output_path)

        # Ensure image size is max 400x400 and save to output
        with Image.open(input_path) as im:
            im = im.convert("RGB") if im.mode != "RGB" else im  # noqa: PLW2901
            if not (im.size[0] <= MAX_IMAGE_DIM and im.size[1] <= MAX_IMAGE_DIM):
                im.thumbnail(size=(MAX_IMAGE_DIM, MAX_IMAGE_DIM))
            im.save(output_path, quality="keep") if im.format == "JPEG" else im.save(output_path)

    return output_paths


def _generate_test_cases() -> list[_TestCaseTuple]:
    """
    Generate a list of image test cases.

    This function:
        (1) Generates a list of test images from the docs
        (2) Generates a list of cached images
        (3) Merges the two lists together and returns separate test cases to
            comparing all docs images to all cached images
    """
    test_cases_dict: dict = {}

    def add_to_dict(filepath: str, key: str) -> None:
        # Function for stuffing image paths into a dict.
        # We use a dict to allow for any entry to be made based on image path alone.
        # This way, we can defer checking for any mismatch between the cached and docs
        # images to test time.
        nonlocal test_cases_dict
        test_name = Path(filepath).stem
        try:
            test_cases_dict[test_name]
        except KeyError:
            test_cases_dict[test_name] = {}
        test_cases_dict[test_name].setdefault(key, filepath)

    # process test images
    test_image_paths = _preprocess_build_images(str(_DocTestInfo.doc_images_dir), str(_DocTestInfo.doc_generated_image_dir))
    [add_to_dict(path, "docs") for path in test_image_paths]  # type: ignore[func-returns-value]

    # process cached images
    cache_dir = Path(_DocTestInfo.doc_image_cache_dir)
    cached_image_paths = _get_file_paths(str(cache_dir), ext="jpg")
    for path in cached_image_paths:
        # Check if we have a single image or a dir with multiple images
        rel = Path(path).relative_to(cache_dir)
        parts = rel.parts
        if len(parts) > 1:  # means it's nested
            # Use the first subdir as the test input instead of the image path
            first_subdir = parts[0]  # one dir down from base
            add_to_dict(str(cache_dir / first_subdir), "cached")
        else:
            add_to_dict(path, "cached")

    # flatten dict
    test_cases_list = []
    for test_name, content in sorted(test_cases_dict.items()):
        doc = content.get("docs", None)
        cache = content.get("cached", None)
        test_case = _TestCaseTuple(
            test_name=test_name,
            docs_image_path=doc,
            cached_image_path=cache,
        )
        test_cases_list.append(test_case)

    return test_cases_list


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate parametrized tests."""
    if "test_case" in metafunc.fixturenames:
        # Generate a separate test case for each image being tested
        test_cases = _generate_test_cases()
        ids = [case.test_name for case in test_cases]
        metafunc.parametrize("test_case", test_cases, ids=ids)


def _save_failed_test_image(source_path: str, category: Literal["warnings", "errors", "errors_as_warnings"]) -> None:
    """Save test image from cache or build to the failed image dir."""
    _ensure_dir_exists(_DocTestInfo.doc_failed_image_dir, msg_name="doc failed image dir")

    parent_dir = Path(category)
    if Path(source_path).is_relative_to(_DocTestInfo.doc_image_cache_dir):
        rel = Path(source_path).relative_to(_DocTestInfo.doc_image_cache_dir)
        dest_relative_dir = Path("from_cache") / rel.parent
    else:
        dest_relative_dir = Path("from_build")

    Path(_DocTestInfo.doc_failed_image_dir).mkdir(exist_ok=True)
    Path(_DocTestInfo.doc_failed_image_dir, parent_dir).mkdir(exist_ok=True)
    dest_dir = Path(_DocTestInfo.doc_failed_image_dir) / parent_dir / dest_relative_dir
    dest_dir.mkdir(exist_ok=True, parents=True)
    dest_path = Path(dest_dir, Path(source_path).name)
    shutil.copy(source_path, dest_path)


def test_static_images(test_case: _TestCaseTuple) -> None:
    """Compare generated image with cached image."""
    _warn_cached_image_path(test_case.cached_image_path)
    fail_msg, fail_source = _test_both_images_exist(*test_case)
    if fail_msg:
        _save_failed_test_image(cast("str", fail_source), "errors")
        pytest.fail(fail_msg)

    cached_image_paths = (
        [test_case.cached_image_path] if Path(test_case.cached_image_path).is_file() else _get_file_paths(test_case.cached_image_path, ext="jpg")
    )
    current_cached_image_path = cached_image_paths[0]

    warn_msg, fail_msg = _test_compare_images(
        test_name=test_case.test_name, docs_image_path=test_case.docs_image_path, cached_image_path=current_cached_image_path
    )

    # Try again and compare with other cached images
    if fail_msg and len(cached_image_paths) > 1:
        # Compare build image to other known valid versions
        msg_start = "This test has multiple cached images. It initially failed (as above)"
        for path in cached_image_paths:
            error = pv.compare_images(pv.read(test_case.docs_image_path), pv.read(path))
            if _check_compare_fail(test_case.test_name, error) is None:
                # Convert failure into a warning
                warn_msg = fail_msg + (f"\n{msg_start} but passed when compared to:\n\t{path}")
                fail_msg = None
                current_cached_image_path = path
                break
        else:  # Loop completed - test still fails
            fail_msg += f"\n{msg_start} and failed again for all images in:\n\t{Path(_DocTestInfo.doc_image_cache_dir, test_case.test_name)!s}"

    if fail_msg:
        _save_failed_test_image(test_case.docs_image_path, "errors")
        # Save all cached images since they all failed
        for path in cached_image_paths:
            _save_failed_test_image(path, "errors")
        pytest.fail(fail_msg)

    if warn_msg:
        parent_dir: Literal["errors_as_warnings", "warnings"] = "errors_as_warnings" if Path(test_case.cached_image_path).is_dir() else "warnings"
        _save_failed_test_image(test_case.docs_image_path, parent_dir)
        _save_failed_test_image(current_cached_image_path, parent_dir)
        warnings.warn(warn_msg, stacklevel=2)


def _test_both_images_exist(filename: str, docs_image_path: str, cached_image_path: str) -> tuple[str | None, str | None]:
    if docs_image_path is None or cached_image_path is None:
        if docs_image_path is None:
            source_path = cached_image_path
            exists = "cache"
            missing = "docs build"
            exists_path = cached_image_path
            missing_path = _DocTestInfo.doc_images_dir
        else:
            source_path = docs_image_path
            exists = "docs build"
            missing = "cache"
            exists_path = _DocTestInfo.doc_images_dir
            missing_path = _DocTestInfo.doc_image_cache_dir

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


def _warn_cached_image_path(cached_image_path: str) -> None:
    """Warn if a subdir is used with only one cached image."""
    if cached_image_path is not None and Path(cached_image_path).is_dir():
        cached_images = _get_file_paths(cached_image_path, ext="jpg")
        if len(cached_images) == 1:
            cache_dir = _DocTestInfo.doc_image_cache_dir
            rel_path = Path(cache_dir.name) / Path(cached_images[0]).relative_to(cache_dir)
            msg = (
                "Cached image sub-directory only contains a single image.\n"
                f"Move the cached image {rel_path.as_posix()!r} directly to the cached image dir {cache_dir.name!r}\n"
                f"or include more than one image in the sub-directory."
            )
            warnings.warn(msg, stacklevel=2)


def _test_compare_images(test_name: str, docs_image_path: str, cached_image_path: str) -> tuple[str | None, str | None]:
    try:
        docs_image = cast("pv.ImageData", pv.read(docs_image_path))
        cached_image = cast("pv.ImageData", pv.read(cached_image_path))

        # Check if test should fail or warn
        error = pv.compare_images(docs_image, cached_image)
        fail_msg = _check_compare_fail(test_name, error)
        warn_msg = _check_compare_warn(test_name, error)
    except RuntimeError as e:
        warn_msg = None
        fail_msg = repr(e)
    return warn_msg, fail_msg


def _check_compare_fail(filename: str, error_: float, allowed_error: float = 500.0) -> str | None:
    if error_ > allowed_error:
        return f"{filename} Exceeded image regression error of {allowed_error} with an image error equal to: {error_}"
    return None


def _check_compare_warn(filename: str, error_: float, allowed_warning: float = 200.0) -> str | None:
    if error_ > allowed_warning:
        return f"{filename} Exceeded image regression warning of {allowed_warning} with an image error of {error_}"
    return None
