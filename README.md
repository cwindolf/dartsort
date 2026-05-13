[![ci](https://github.com/cwindolf/dartsort/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cwindolf/dartsort/actions/)
[![coveralls](https://coveralls.io/repos/github/cwindolf/dartsort/badge.svg?branch=main)](https://coveralls.io/github/cwindolf/dartsort)
[![Zenodo DOI](https://zenodo.org/badge/421108722.svg)](https://doi.org/10.5281/zenodo.16943074)
[![pypi: dartsort](https://img.shields.io/pypi/v/dartsort?label=pypi:%20dartsort)](https://pypi.org/p/dartsort)


# dartsort


*dartsort* is a modular spike sorter built around a statistical clustering model and a new approach to probe motion.
It is also a toolkit of modules for building spike sorters or other analyses of electrophysiology data.


## :warning: work in progress :warning:

We do not currently recommend DARTsort for production spike sorting purposes.
Please feel free to open an issue or a discussion if you run into problems.


## Installation

### Installing into an existing environment

If you already have a Python environment with PyTorch working and you just want to get *dartsort* there, use

```sh
$ pip install dartsort
```

If you want to run the test suite or use `dartsort.vis`, you can install the optional dependencies with `pip install dartsort[full]`.

If you need a Python environment, expand the next section.

<details>
<summary><h3>Setting up a Python environment</h3></summary>

Otherwise, there are a few ways to get Python and PyTorch set up, including new tools like `uv`, but I find that a [`conda-forge`](https://conda-forge.org/)-based distribution is still the most reliable at installing the GPU dependencies which PyTorch needs (note: `conda-forge` is different from the non-free Anaconda).

You can use `conda-forge` to install Python, `dartsort`, and its dependencies as follows:

 - Follow the `conda-forge` installation instructions for your platform at [https://conda-forge.org/download/](https://conda-forge.org/download/)
 - Create an environment with
   ```sh
   $ mamba env create -f environment.yml
   ```
   This will create an environment called `dartsort`, but you can change the name by adding `-n othername`.
 - Activate the environment:
   ```sh
   $ mamba activate dartsort
   ```
 - Install `dartsort` and the rest of its dependencies by running the pip command [above](#installing-into-an-existing-environment)

</details>
