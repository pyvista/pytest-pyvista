# -*- coding: utf-8 -*-

import io
import os
import pathlib
import platform
import re
import time
import warnings

from PIL import Image
import imageio
import numpy as np
import pytest
import vtk

import pyvista
from pyvista import examples
from pyvista._vtk import VTK9
from pyvista.core.errors import DeprecationError
from pyvista.plotting import system_supports_plotting
from pyvista.plotting.plotting import SUPPORTED_FORMATS
from pyvista.utilities.misc import can_create_mpl_figure


def pytest_addoption(parser):
    group = parser.getgroup('pyvista')
    group.addoption(
        '--foo',
        action='store',
        dest='dest_foo',
        default='2022',
        help='Set the value for the fixture "bar".'
    )

    parser.addini('HELLO', 'Dummy pytest.ini setting')


# Image regression warning/error thresholds for releases after 9.0.1
# TODO: once we have a stable release for VTK, remove these.
HIGH_VARIANCE_TESTS = {
    'test_add_title',
    'test_export_gltf',  # image cache created with 9.0.20210612.dev0
    'test_import_gltf',  # image cache created with 9.0.20210612.dev0
    'test_opacity_by_array_direct',  # VTK regression 9.0.1 --> 9.1.0
    'test_opacity_by_array_user_transform',
    'test_pbr',
    'test_plot',
    'test_set_environment_texture_cubemap',
    'test_set_viewup',
}

# these images vary between Windows when using OSMesa and Linux/MacOS
# and will not be verified
WINDOWS_SKIP_IMAGE_CACHE = {
    'test_array_volume_rendering',
    'test_closing_and_mem_cleanup',
    'test_cmap_list',
    'test_collision_plot',
    'test_enable_stereo_render',
    'test_multi_block_plot',
    'test_multi_plot_scalars',  # flaky
    'test_plot',
    'test_plot_add_scalar_bar',
    'test_plot_cell_data',
    'test_plot_complex_value',
    'test_plot_composite_bool',
    'test_plot_composite_lookup_table',
    'test_plot_composite_poly_component_nested_multiblock',
    'test_plot_composite_poly_scalars_cell',
    'test_plot_composite_preference_cell',
    'test_plot_helper_two_volumes',
    'test_plot_helper_volume',
    'test_plot_string_array',
    'test_plotter_lookup_table',
    'test_rectlinear_edge_case',
    'test_scalars_by_name',
    'test_user_annotations_scalar_bar_volume',
    'test_volume_rendering_from_helper',
}

# these images vary between Linux/Windows and MacOS
# and will not be verified for MacOS
MACOS_SKIP_IMAGE_CACHE = {
    'test_plot',
    'test_plot_show_grid_with_mesh',
    'test_property',
}


@pytest.fixture()
def multicomp_poly():
    """Create a dataset with vector values on points and cells."""
    data = pyvista.Plane()

    vector_values_points = np.empty((data.n_points, 3))
    vector_values_points[:, 0] = np.arange(data.n_points)
    vector_values_points[:, 1] = np.arange(data.n_points)[::-1]
    vector_values_points[:, 2] = 0

    vector_values_cells = np.empty((data.n_cells, 3))
    vector_values_cells[:, 0] = np.arange(data.n_cells)
    vector_values_cells[:, 1] = np.arange(data.n_cells)[::-1]
    vector_values_cells[:, 2] = 0

    data['vector_values_points'] = vector_values_points
    data['vector_values_cells'] = vector_values_cells
    return data


# this must be a session fixture to ensure this runs before any other test
@pytest.fixture(scope="session", autouse=True)
def get_cmd_opt(pytestconfig):
    VerifyImageCache.reset_image_cache = pytestconfig.getoption('reset_image_cache')
    VerifyImageCache.ignore_image_cache = pytestconfig.getoption('ignore_image_cache')
    VerifyImageCache.fail_extra_image_cache = pytestconfig.getoption('fail_extra_image_cache')


class VerifyImageCache:
    """Control image caching for testing.

    Image cache files are names according to ``test_name``.
    Multiple calls to an instance of this class will append
    `_X` to the name after the first one.  That is, files
    ``{test_name}``, ``{test_name}_1``, and ``{test_name}_2``
    will be saved if called 3 times.

    Parameters
    ----------
    test_name : str
        Name of test to save.  Sets name of image cache file.

    """

    reset_image_cache = False
    ignore_image_cache = False
    fail_extra_image_cache = False

    def __init__(
        self,
        test_name,
        *,
        cache_dir=None,
        error_value=500,
        warning_value=200,
        var_error_value=1000,
        var_warning_value=1000,
    ):
        self.test_name = test_name

        if cache_dir is None:
            # Reset image cache with new images
            this_path = pathlib.Path(__file__).parent.absolute()
            self.cache_dir = os.path.join(this_path, 'image_cache')
        else:
            self.cache_dir = cache_dir

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

        if self.test_name in HIGH_VARIANCE_TESTS:
            allowed_error = self.var_error_value
            allowed_warning = self.var_warning_value
        else:
            allowed_error = self.error_value
            allowed_warning = self.warning_value

        # some tests fail when on Windows with OSMesa
        if os.name == 'nt' and self.test_name in WINDOWS_SKIP_IMAGE_CACHE:
            return
        # high variation for MacOS
        if platform.system() == 'Darwin' and self.test_name in MACOS_SKIP_IMAGE_CACHE:
            return

        # cached image name
        image_filename = os.path.join(self.cache_dir, test_name[5:] + '.png')

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
                f'{test_name} Exceeded image regression error of '
                f'{allowed_error} with an image error of '
                f'{error}'
            )
        if error > allowed_warning:
            warnings.warn(
                f'{test_name} Exceeded image regression warning of '
                f'{allowed_warning} with an image error of '
                f'{error}'
            )
            
@pytest.fixture(autouse=True)
def verify_image_cache(request):
    verify_image_cache = VerifyImageCache(request.node.name)
    pyvista.global_theme.before_close_callback = verify_image_cache
    return verify_image_cache