[![badge](https://github.com/cwindolf/dartsort/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cwindolf/dartsort/actions/)
[![cov](https://coveralls.io/repos/cwindolf/dartsort/badge.png?branch=main)](https://coveralls.io/repos/cwindolf/dartsort/)

# dartsort

## :warning: Work in progress code repository

We do not currently recommend DARTsort for production spike sorting purposes. We are in the process of implementing a robust and documented pipeline in [`src/dartsort`](src/dartsort), and we will update this page accordingly.

A workflow described in our preprint (https://www.biorxiv.org/content/10.1101/2023.08.11.553023v1) is in [uhd_pipeline.py](scripts/uhd_pipeline.py), which is implemented using the legacy code in [`src/spike_psvae`](src/spike_psvae).


## Suggested install steps:

If you don't already have Python and PyTorch 2 installed, we recommend doing this with the Miniforge distribution of `conda`. You can find info and installers for your platform [at Miniforge's GitHub repository](https://github.com/conda-forge/miniforge). After installing Miniforge, `conda` will be available on your computer for installing Python packages, as well as the newer and faster conda replacement tool `mamba`. We recommend using `mamba` instead of `conda` below, since the installation tends to be a lot faster with `mamba`.

To install DARTsort, first clone this GitHub repository.

After cloning the repository, create and activate the `mamba`/`conda` environment from the configuration file provided as follows:

```bash
$ mamba env create -f environment.yml
$ mamba activate dartsort
```

Next, visit https://pytorch.org/get-started/locally/ and follow the `PyTorch` install instructions for your specific OS and hardware needs.
We also need to install `linear_operator` from the `gpytorch` channel.
For example, on a Linux workstation or cluster with NVIDIA GPUs available, one might use (dropping in `mamba` for `conda` commands):

```bash
# Example -- see https://pytorch.org/get-started/locally/ to find your platform's command.
(dartsort) $ mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 linear_operator -c pytorch -c nvidia -c gpytorch
```

Finally, install the remaining `pip` dependencies and `dartsort` itself:

```bash
(dartsort) $ pip install -r requirements-full.txt
(dartsort) $ pip install -e .
```

To enable DARTsort's default motion correction algorithm [DREDge](https://www.biorxiv.org/content/10.1101/2023.10.24.563768), clone [its GitHub repository](https://github.com/evarol/dredge), and then `cd dredge/` and install the DREDge package with `pip install -e .`.

Soon we will have a package on PyPI so that these last steps will be just a `pip install dartsort`.

To make sure everything is working:

```bash
$ (dartsort) pytest tests/*
```
