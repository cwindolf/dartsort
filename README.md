![example branch parameter](https://github.com/cwindolf/dartsort/actions/workflows/ci.yml/badge.svg?branch=main)

# dartsort

## :warning: Work in progress code repository

We do not currently recommend DARTsort for production spike sorting purposes. We are in the process of implementing a robust and documented pipeline in [`src/dartsort`](src/dartsort), and we will update this page accordingly.

A workflow described in our preprint (https://www.biorxiv.org/content/10.1101/2023.08.11.553023v1) is in [uhd_pipeline.py](scripts/uhd_pipeline.py), which is implemented using the legacy code in [`src/spike_psvae`](src/spike_psvae).


## Suggested install steps:

`mamba` is the recommended package manager for using DARTsort. It is a drop-in replacement for `conda` and can be installed from [here](https://mamba.readthedocs.io/en/latest/installation.html).

After cloning the repository, create and activate the `mamba`/`conda` environment from the configuration file provided as follows:

```bash
$ mamba env create -f environment.yml -n dartsort
$ mamba activate dartsort
```

Next, visit https://pytorch.org/get-started/locally/ and follow the `PyTorch` install instructions for your specific OS and hardware needs.
For example, on a Linux workstation or cluster with NVIDIA GPUs available, one might use (dropping in `mamba` for `conda` commands):

```bash
(dartsort) $ mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Finally, install the remaining `pip` dependencies and `dartsort` itself:

```bash
(dartsort) $ pip install -r requirements.txt
(dartsort) $ pip install -e .
```

Soon we will have a package on PyPI so that the last two steps will be just a `pip install dartsort`.

Make sure everything works:

```bash
$ (dartsort) pytest tests/*
```
