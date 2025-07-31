==============
pytest-pyvista
==============
.. image:: https://img.shields.io/pypi/v/pytest-pyvista.svg?color=orange&label=pypi&logo=python&logoColor=white
    :target: https://pypi.org/project/pytest-pyvista
    :alt: PyPI version

.. image:: https://img.shields.io/conda/vn/conda-forge/pytest-pyvista?color=orange&label=conda-forge&logo=conda-forge&logoColor=white
    :target: https://anaconda.org/conda-forge/pytest-pyvista
    :alt: conda-forge version

.. image:: https://img.shields.io/pypi/pyversions/pytest-pyvista.svg?color=orange&logo=python&label=python&logoColor=white
    :target: https://pypi.org/project/pytest-pyvista
    :alt: Python versions

.. image:: https://github.com/pyvista/pytest-pyvista/actions/workflows/ci_cd.yml/badge.svg
    :target: https://github.com/pyvista/pytest-pyvista/actions/workflows/ci_cd.yml
    :alt: GitHub Actions: Unit Testing and Deployment

Plugin to test PyVista plot outputs.

----

This `pytest`_ plugin was generated with `Cookiecutter`_ along with
`@hackebrot`_'s `cookiecutter-pytest-plugin`_ template.


Features
--------
This plugin facilitates the comparison of the images produced by `PyVista`. It
generates a cache of images from the tests, using the `PyVista` plotting
function in its first execution. Then, further executions will compare its
results against this cache, so if there are any changes in the code that break
the image generation, the comparison against the cache will notice it. Note
that there is an error tolerance in the comparison, so minor differences won't
fail.


Requirements
------------
You must have a Python version >= 3.9, as well as PyVista installed
in your environment.

pyvista version >=0.37.0 and vtk version >=9.0.0 required.

Installation
------------
You can install "pytest-pyvista" via `pip`_ from `PyPI`_

.. code:: bash 

  pip install pytest-pyvista

Alternatively, you can also install via `conda`_ or `mamba`_ from `conda-forge`_

.. code:: bash

  mamba install -c conda-forge pytest-pyvista

Usage
-----
Once installed, you only need to use the command `pl.show()` in your test. The
plugin will automatically manage the cache generation if it does not exist, and
the image comparison itself. Make sure you enable `pv.OFF_SCREEN` when loading
PyVista, so the `pl.show()` doesn't pop up any window while testing.  By default,
the verify_image_cache fixture should be used for each test for image comparison:

.. code-block:: python

    import pyvista as pv

    pv.OFF_SCREEN = True


    def test_succeeds(verify_image_cache):
        pl = pyvista.Plotter()
        pl.add_mesh(pyvista.Sphere(), show_edges=True)
        pl.show()


If most tests utilize this functionality, possibly restricted to a module,
a wrapped version could be used

.. code-block:: python

    @pytest.fixture(autouse=True)
    def wrapped_verify_image_cache(verify_image_cache):
        return verify_image_cache


If you need to use any flag inside the tests, you can modify the
`verify_image_cache` object in the test

.. code-block:: python

    import pyvista as pv

    pv.OFF_SCREEN = True


    def test_succeeds(verify_image_cache):
        verify_image_cache.windows_skip_image_cache = True
        pl = pyvista.Plotter()
        pl.add_mesh(pyvista.Sphere(), show_edges=True)
        pl.show()


Global flags
------------
These are the flags you can use when calling ``pytest`` in the command line:

* ``--reset_image_cache`` creates a new image for each test in
  ``tests/plotting/test_plotting.py`` and is not recommended except for
  testing or for potentially a major or minor release. 

* You can use ``--ignore_image_cache`` if you want to
  temporarily ignore regression testing, e.g. on a particular CI action.
  
* ``--generated_image_dir <DIR>`` dumps all generated test images into the provided
  directory, relative to `pytest root path <https://docs.pytest.org/en/latest/reference/reference.html#pytest.Config.rootpath>`.
  This will override any configuration, see below.

* ``--failed_image_dir <DIR>`` dumps copies of cached and generated test images when
  there is a warning or error raised. This directory is useful for reviewing test
  failures. It is relative to `pytest root path <https://docs.pytest.org/en/latest/reference/reference.html#pytest.Config.rootpath>`.
  This will override any configuration, see below.

* ``--add_missing_images`` adds any missing images from the test run to the cache.

* ``--image_cache_dir <DIR>`` sets the image cache directory, relative to `pytest root path <https://docs.pytest.org/en/latest/reference/reference.html#pytest.Config.rootpath>`.
  This will override any configuration, see below.

* ``--reset_only_failed`` reset the image cache of the failed tests only.

* Use ``--allow_unused_generated`` to prevent an error from being raised when a
  test image is generated but not used. A test image is considered "used" if it has a
  corresponding cached image to compare against, or is used to reset or update the
  cache (e.g. if using ``--add_missing_images``). Otherwise, an error is raised by
  default.

* ``--disallow_unused_cache`` report test failure if there are any images in the cache
  which are not compared to any generated images.

* Use ``--allow_useless_fixture`` to prevent test failure when the ``verify_image_cache``
  fixture is used but no images are generated. If no images are generated (i.e. there are
  no calls made to ``Plotter.show()`` or ``mesh.plot()``), then these tests will fail
  by default. Set this CLI flag to allow this globally, or use the test-specific flag
  by the same name below to configure this on a per-test basis.

Test specific flags
-------------------
These are attributes of `verify_image_cache`. You can set them as ``True`` if needed
in the beginning of your test function.

* ``high_variance_test``: If necessary, the threshold for determining if a test
  will pass or not is incremented to another predetermined threshold. This is
  currently done due to the use of an unstable version of VTK, in stable
  versions this shouldn't be necessary.

* ``windows_skip_image_cache``: For test where the plotting in Windows is different
  from MacOS/Linux.

* ``macos_skip_image_cache``: For test where the plotting in MacOS is different
  from Windows/Linux.

* ``skip``: If you have a test that plots a figure, but you don't want to compare
  its output against the cache, you can skip it with this flag.

* ``allow_useless_fixture``: Set this flag to ``True`` to prevent test failure when the
  ``verify_image_cache`` fixture is used but no images are generated. The value of this
  flag takes precedence over the global flag by the same name (see above).

Configuration
-------------
If using ``pyproject.toml`` or any other 
`pytest configuration <https://docs.pytest.org/en/latest/reference/customize.html>`_
section, consider configuring your test directory location to
avoid passing command line arguments when calling ``pytest``, for example in
``pyproject.toml``:

.. code-block:: toml

    [tool.pytest.ini_options]
    image_cache_dir = "tests/plotting/image_cache"

Additionally, to configure the directory that will contain the generated test images:

.. code-block:: toml

    [tool.pytest.ini_options]
    generated_image_dir = "generated_images"

Similarly, configure the directory that will contain any failed test images:

.. code-block:: toml

    [tool.pytest.ini_options]
    failed_image_dir = "failed_images"

Note that these directories are relative to `pytest root path <https://docs.pytest.org/en/latest/reference/reference.html#pytest.Config.rootpath>`.

Contributing
------------
Contributions are always welcome. Tests can be run with `tox`_, please ensure
the coverage at least stays the same before you submit a pull request.

License
-------
Distributed under the terms of the `MIT`_ license, ``pytest-pyvista`` is free
and open source software.


Issues
------
If you encounter any problems, please `file an issue`_ along with a detailed
description.

.. _`Cookiecutter`: https://github.com/audreyr/cookiecutter
.. _`@hackebrot`: https://github.com/hackebrot
.. _`MIT`: http://opensource.org/licenses/MIT
.. _`BSD-3`: http://opensource.org/licenses/BSD-3-Clause
.. _`GNU GPL v3.0`: http://www.gnu.org/licenses/gpl-3.0.txt
.. _`Apache Software License 2.0`: http://www.apache.org/licenses/LICENSE-2.0
.. _`cookiecutter-pytest-plugin`: https://github.com/pytest-dev/cookiecutter-pytest-plugin
.. _`file an issue`: https://github.com/pyvista/pytest-pyvista/issues
.. _`pytest`: https://github.com/pytest-dev/pytest
.. _`tox`: https://tox.readthedocs.io/en/latest/
.. _`pip`: https://pypi.org/project/pip/
.. _`PyPI`: https://pypi.org/project
.. _`conda`: https://github.com/conda/conda
.. _`mamba`: https://github.com/mamba-org/mamba
.. _`conda-forge`: https://anaconda.org/conda-forge/pytest-pyvista
