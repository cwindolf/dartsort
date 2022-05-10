#!/usr/bin/env python
from setuptools import setup

try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [
            "spike_psvae/jisotonic5.pyx",
            "spike_psvae/ibme_fast_raster.pyx",
            "spike_psvae/denoise_temporal_decrease.pyx",
        ]
    )
except ImportError:
    ext_modules = None


setup(
    name="spike_psvae",
    version="0.1",
    packages=["spike_psvae"],
    ext_modules=ext_modules,
)
