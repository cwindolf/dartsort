# dartsort

## Suggested install steps:

`mamba` is the recommended package manager for using DARTsort. It is a drop-in replacement for `conda` and can be installed from [here](https://mamba.readthedocs.io/en/latest/installation.html).

After cloning the repository, create the `mamba`/`conda` environment from the configuration file provided as follows:

```bash
$ mamba create -f environment.yml -n dartsort
```

Next, visit https://pytorch.org/get-started/locally/ and follow the `PyTorch` install instructions for your specific OS and hardware needs.

For example, on a Linux workstation or cluster with NVIDIA GPUs available, one might use (dropping in `mamba` for `conda` commands):

```bash
$ (dartsort) mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Finally, install the remaining `pip` dependencies:

```bash
$ (dartsort) pip install -r requirements.txt
```

Make sure everything works:

```bash
$ (dartsort) pytest tests/*
```
