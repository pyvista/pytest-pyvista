"""Test the --doc_mode option."""

from __future__ import annotations

import pytest

from pytest_pyvista.pytest_doc_images import _preprocess_build_images
from tests.test_pyvista import make_cached_images


def test_verify_image_cache(pytester: pytest.Pytester) -> None:
    """Test regular usage of the `verify_image_cache` fixture."""
    cache = "doc_image_cache_dir"
    images = "doc_images_dir"
    make_cached_images(pytester.path, cache)
    make_cached_images(pytester.path, images)
    _preprocess_build_images(str(pytester.path / cache), str(pytester.path / cache))

    result = pytester.runpytest("--doc_mode")
    assert result.ret == pytest.ExitCode.INTERNAL_ERROR
    result.stderr.fnmatch_lines(["*ValueError: 'doc_images_dir' must be specified when using --doc_mode"])

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images)
    assert result.ret == pytest.ExitCode.INTERNAL_ERROR
    result.stderr.fnmatch_lines(["*ValueError: 'doc_image_cache_dir' must be specified when using --doc_mode"])

    result = pytester.runpytest("--doc_mode", "--doc_images_dir", images, "--doc_image_cache_dir", cache)
    assert result.ret == pytest.ExitCode.OK
