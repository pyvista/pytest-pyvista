==============
pytest-pyvista
==============

.. image:: https://img.shields.io/pypi/v/pytest-pyvista.svg
    :target: https://pypi.org/project/pytest-pyvista
    :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/pytest-pyvista.svg
    :target: https://pypi.org/project/pytest-pyvista
    :alt: Python versions

.. image:: https://ci.appveyor.com/api/projects/status/github/pyvista/pytest-pyvista?branch=master
    :target: https://ci.appveyor.com/project/pyvista/pytest-pyvista/branch/master
    :alt: See Build Status on AppVeyor

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
You must have a Python version greater than 3.7, as well as PyVista installed
in your environment.


Installation
------------

You can install "pytest-pyvista" via `pip`_ from `PyPI`_::

    $ pip install pytest-pyvista


Usage
-----
Once installed, you only need to use the command `pl.show()` in your test. The
plugin will automatically manage the cache generation if it does not exist, and
the image comparison itself. Make sure you enable `pv.OFF_SCREEN` when loading
PyVista, so the `pl.show()` doesn't pop up any window while testing::

    import pyvista as pv
    pv.OFF_SCREEN = True
    def test_succeeds():
        pl = pyvista.Plotter()
        pl.add_mesh(pyvista.Sphere(), show_edges=True)
        pl.show()


If you need to use any flag inside the tests, you need to use the
`verify_image_cache` fixture as a parameter to the test::


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

* You can use ``--ignore_image_cache`` if you are running on Linux and want to
  temporarily ignore regression testing. Realize that regression testing will
  still occur on our CI testing.

* When using ``--fail_extra_image_cache`` if there is an extra image in the
  cache, it will report as an error.

Test specific flags
-------------------
These are attributes of `verify_image_cache`. You can set them as `True` if needed in the beginning of your test function.

* high_variance_tests: If necessary, the threshold for determining if a test
  will pass or not is incremented to another predetermined threshold. This is
  currently done due to the use of an unstable version of VTK, in stable
  versions this shouldn't be necessary.

* windows_skip_image_cache: For test where the plotting in Windows is different
  from MacOS/Linux.

* macos_skip_image_cache: For test where the plotting in MacOS is different
  from Windows/Linux.

* skip: If you have a test that plots a figure, but you don't want to compare
  its output against the cache, you can skip it with this flag.


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
