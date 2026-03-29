[![ci](https://github.com/cwindolf/dartsort/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cwindolf/dartsort/actions/)
[![coveralls](https://coveralls.io/repos/github/cwindolf/dartsort/badge.svg?branch=main)](https://coveralls.io/github/cwindolf/dartsort)
[![Zenodo DOI](https://zenodo.org/badge/421108722.svg)](https://doi.org/10.5281/zenodo.16943074)
[![pypi: dartsort](https://img.shields.io/pypi/v/dartsort?label=pypi:%20dartsort)](https://pypi.org/p/dartsort)

# dartsort

## :warning: Work in progress code repository

We do not currently recommend DARTsort for production spike sorting purposes. We are in the process of implementing a robust and documented pipeline in [`src/dartsort`](src/dartsort), and we will update this page accordingly.

A workflow described in our preprint (https://www.biorxiv.org/content/10.1101/2023.08.11.553023v1) is in [uhd_pipeline.py](scripts/uhd_pipeline.py), which is implemented using the legacy code in [`src/spike_psvae`](src/spike_psvae).


## Suggested install steps

If you already have a Python environment set up, you can install `dartsort` with

```bash
# get all dependencies (includes visualization and GPU packages); remove [gpu] if you don't have one
$ pip install dartsort[full,gpu]
```

If you don't already have Python and PyTorch 2 installed, we recommend doing this with the Miniforge distribution of `conda`. You can find info and installers for your platform [at Miniforge's GitHub repository](https://github.com/conda-forge/miniforge). After installing Miniforge, `conda` will be available on your computer for installing Python packages, as well as the newer and faster conda replacement tool `mamba`. We recommend using `mamba` instead of `conda` below, since the installation tends to be a lot faster with `mamba`. You can build a conda environment called `dartsort` with

```bash
$ mamba env create -f environment.yml
$ mamba activate dartsort
```

after cloning this repository. Next, visit https://pytorch.org/get-started/locally/ and follow the `PyTorch` install instructions for your specific OS and hardware needs.

Finish with the `pip` command above. Or, to work on the code, use:

```bash
(dartsort) $ pip install -e ./[full]
(dartsort) $ pytest tests/*
```
