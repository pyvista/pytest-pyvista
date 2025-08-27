"""Test the images generated from building the documentation."""

from __future__ import annotations

import glob
import os
from pathlib import Path
import shutil
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
    flaky_test_cases: list[str]

    @classmethod
    def init_dirs(cls, config: pytest.Config) -> None:
        must_be_specified = "{!r} must be specified when using --doc_mode"
        option = "doc_images_dir"
        doc_images_dir = _get_option_from_config_or_ini(config, option, is_dir=True)
        if doc_images_dir is None:
            raise ValueError(must_be_specified.format(option))
        if not doc_images_dir.is_dir():
            msg = f"'doc_images_dir' {doc_images_dir} must be a valid directory."
            raise ValueError(msg)
        cls.doc_images_dir = doc_images_dir

        option = "doc_image_cache_dir"
        doc_image_cache_dir = _get_option_from_config_or_ini(config, option, is_dir=True)
        if doc_image_cache_dir is None:
            raise ValueError(must_be_specified.format(option))
        if not doc_image_cache_dir.is_dir():
            msg = f"'doc_image_cache_dir' {doc_image_cache_dir} must be a valid directory."
            raise ValueError(msg)
        cls.doc_image_cache_dir = doc_image_cache_dir

        doc_generated_image_dir = _get_option_from_config_or_ini(config, "doc_generated_image_dir", is_dir=True)
        if doc_generated_image_dir is None:
            # TODO: make tempdir and clean it up post-test
            path = Path("_generated")
            path.mkdir(exist_ok=True)
        else:
            _ensure_dir_exists(doc_generated_image_dir, msg_name="doc generated image dir")
            path = Path(doc_generated_image_dir)
        cls.doc_generated_image_dir = path

        doc_failed_image_dir = _get_option_from_config_or_ini(config, "doc_failed_image_dir", is_dir=True)
        if doc_failed_image_dir is None:
            # TODO: make tempdir and clean it up post-test
            path = Path("_failed")
            path.mkdir(exist_ok=True)
        else:
            _ensure_dir_exists(doc_failed_image_dir, msg_name="doc failed image dir")
            path = doc_failed_image_dir
        cls.doc_failed_image_dir = path

        cls.flaky_test_cases = [path.name for path in cls.doc_image_cache_dir.iterdir() if path.is_dir()]


class _TestCaseTuple(NamedTuple):
    test_name: str
    docs_image_path: str
    cached_image_path: str


def _get_file_paths(dir_: str, ext: str) -> list[str]:
    """Get all paths of files with a specific extension inside a directory tree."""
    pattern = str(Path(dir_) / "**" / ("*." + ext))
    return glob.glob(pattern, recursive=True)  # noqa: PTH207


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
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for input_path in input_png + input_gif + input_jpg:
        # input image from the docs may come from a nested directory,
        # so we flatten the file's relative path
        output_file_name = _flatten_path(os.path.relpath(input_path, build_images_dir))
        output_file_name = Path(output_file_name).with_suffix(".jpg")
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
    [add_to_dict(path, "docs") for path in test_image_paths]

    # process cached images
    cached_image_paths = _get_file_paths(str(_DocTestInfo.doc_image_cache_dir), ext="jpg")
    [add_to_dict(path, "cached") for path in cached_image_paths]

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


def _save_failed_test_image(source_path: str, category: Literal["warnings", "errors", "flaky"]) -> None:
    """Save test image from cache or build to the failed image dir."""
    parent_dir = Path(category)
    dest_dirname = "from_cache" if Path(source_path).parent == Path(_DocTestInfo.doc_images_dir) else "from_build"
    Path(_DocTestInfo.doc_failed_image_dir).mkdir(exist_ok=True)
    Path(_DocTestInfo.doc_failed_image_dir, parent_dir).mkdir(exist_ok=True)
    dest_dir = Path(_DocTestInfo.doc_failed_image_dir, parent_dir, dest_dirname)
    dest_dir.mkdir(exist_ok=True)
    dest_path = Path(dest_dir, Path(source_path).name)
    shutil.copy(source_path, dest_path)


def test_static_images(test_case: _TestCaseTuple) -> None:
    """Compare generated image with cached image."""
    fail_msg, fail_source = _test_both_images_exist(*test_case)
    if fail_msg:
        _save_failed_test_image(fail_source, "errors")
        pytest.fail(fail_msg)

    warn_msg, fail_msg = _test_compare_images(*test_case)
    if fail_msg:
        _save_failed_test_image(test_case.docs_image_path, "errors")
        _save_failed_test_image(test_case.cached_image_path, "errors")
        pytest.fail(fail_msg)

    if warn_msg:
        parent_dir: Literal["flaky", "warnings"] = "flaky" if Path(test_case.cached_image_path).stem in _DocTestInfo.flaky_test_cases else "warnings"
        _save_failed_test_image(test_case.docs_image_path, parent_dir)
        _save_failed_test_image(test_case.cached_image_path, parent_dir)
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


def _test_compare_images(test_name: str, docs_image_path: str, cached_image_path: str) -> tuple[str | None, str | None]:
    try:
        docs_image = cast("pv.ImageData", pv.read(docs_image_path))
        cached_image = cast("pv.ImageData", pv.read(cached_image_path))

        # Check if test should fail or warn
        error = pv.compare_images(docs_image, cached_image)
        fail_msg = _check_compare_fail(test_name, error)
        warn_msg = _check_compare_warn(test_name, error)
        # Check if test case is flaky test
        if fail_msg and test_name in _DocTestInfo.flaky_test_cases:
            # Compare build image to other known valid versions
            success_path = _is_false_positive(test_name, docs_image)
            if success_path:
                # Convert failure into a warning
                warn_msg = fail_msg + (f"\nTHIS IS A FLAKY TEST. It initially failed (as above) but passed when compared to:\n\t{success_path}")
                fail_msg = None
            else:
                # Test still fails
                fail_msg += (
                    "\nTHIS IS A FLAKY TEST. It initially failed (as above) and failed again "
                    f"for all images in \n\t{Path(_DocTestInfo.flaky_test_cases, test_name)!s}."
                )
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


def _is_false_positive(test_name: str, docs_image: pv.ImageData) -> str | None:
    """Compare against other image in the flaky image dir."""
    paths = _get_file_paths(str(Path(FLAKY_IMAGE_DIR, test_name)), "jpg")
    for path in paths:
        error = pv.compare_images(docs_image, pv.read(path))
        if _check_compare_fail(test_name, error) is None:
            return path
    return None
