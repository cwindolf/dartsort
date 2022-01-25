#!/usr/bin/env python
from setuptools import setup
from Cython.Build import cythonize


setup(
    name="spike_psvae",
    version="0.1",
    packages=["spike_psvae"],
    ext_modules=cythonize("spike_psvae/jisotonic5.pyx")
)
