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

Plugin to test PyVista plot outputs

----

This `pytest`_ plugin was generated with `Cookiecutter`_ along with `@hackebrot`_'s `cookiecutter-pytest-plugin`_ template.


Features
--------

* TODO

Global flags
--------
`--reset_image_cache` 

Running ``--reset_image_cache`` creates a new image for each test in
``tests/plotting/test_plotting.py`` and is not recommended except for
testing or for potentially a major or minor release. 



ignore_image_cache 

You can use ``--ignore_image_cache`` if you’re running on Linux and want to
temporarily ignore regression testing. Realize that regression testing
will still occur on our CI testing.

fail_extra_image_cache 

Test specific flags
--------
These are attributes of `verify_image_cache`. You can set them as `True` if needed in the beginning of your test.

- high_variance_tests:  If necessary, the threshold for determining if a test is passing or not is 
incremented to another predetermined threshold. This is currently done due to the use of an unstable 
version of VTK, in stable versions this shouldn't be necessary.

- windows_skip_image_cache: For test where the plotting in Windows is different from MacOS/Linux.

- macos_skip_image_cache: For test where the plotting in MacOS is different from Windows/Linux.


Requirements
------------

* TODO


Installation
------------

You can install "pytest-pyvista" via `pip`_ from `PyPI`_::

    $ pip install pytest-pyvista


Usage
-----

* TODO

Contributing
------------
Contributions are very welcome. Tests can be run with `tox`_, please ensure
the coverage at least stays the same before you submit a pull request.

License
-------

Distributed under the terms of the `MIT`_ license, "pytest-pyvista" is free and open source software


Issues
------

If you encounter any problems, please `file an issue`_ along with a detailed description.

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
