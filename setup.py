#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os

from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()


setup(
    name="pytest-pyvista",
    version="0.1.dev0",
    author="The PyVista Developers",
    author_email="info@pyvista.org",
    maintainer="The PyVista Developers",
    maintainer_email="info@pyvista.org",
    license="MIT",
    url="https://github.com/pyvista/pytest-pyvista",
    description="Plugin to test PyVista plot outputs",
    long_description=read("README.rst"),
    long_description_content_type="text/x-rst",
    py_modules=["pytest_pyvista"],
    python_requires=">=3.7",
    install_requires=["pytest>=3.5.0"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: Pytest",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    entry_points={
        # Location of the plugin file, in this case ./pytest_pyvista/pytest_pyvista.py
        "pytest11": [
            "pyvista = pytest_pyvista.pytest_pyvista",
        ],
    },
)
