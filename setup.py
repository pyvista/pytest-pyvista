#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding='utf-8').read()


setup(
    name='pytest-pyvista',
    version='0.1.0',
    author='The PyVista Developers',
    author_email='alex.fernandezluces@ansys.com',
    maintainer='The PyVista Developers',
    maintainer_email='info@pyvista.org',
    license='MIT',
    url='https://github.com/pyvista/pytest-pyvista',
    description='Plugin to test PyVista plot outputs',
    long_description=read('README.rst'),
    py_modules=['pytest_pyvista'],
    python_requires='>=3.5',
    install_requires=['pytest>=3.5.0'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Framework :: Pytest',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Testing',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    entry_points={
        'pytest11': [
            'pyvista = pytest_pyvista',
        ],
    },
)
