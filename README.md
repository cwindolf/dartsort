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

If you want to run the test suite or use `dartsort.vis`, you can install the optional dependencies with `pip install dartsort[test,vis]`.

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


## Usage

### As a Python function

*dartsort* can be run from inside Python with:

```python
import dartsort

dartsort.dartsort(recording, output_dir)

# or, to set configuration options, something like...
dartsort_result = dartsort.dartsort(
    recording,
    output_dir,
    cfg=dartsort.DARTsortUserConfig(
        preprocessing="ibllike",
        work_in_tmpdir=True,
        
    )
)
```

**Please** read the [important configuration details section](#important-configuration-details) below.
Some of them, like preprocessing, are not set by default and need your input! (This could change.)

Here, `recording` is a [SpikeInterface](SpikeInterface) recording object.
SpikeInterface can read every electrophysiology data format that I've encountered and many I haven't; see

`output_dir` is the folder where *dartsort* will save its output.

Once you've run *dartsort*, you might want to check out [the outputs and exporting](#outputs-and-exporting) section below.

### Important configuration details

Before running *dartsort*, please be aware of the following important configuration options.

 - `preprocessing`: *dartsort* won't touch your data by default (`preprocessing="none"`), leaving you free to implement your own preprocessing in SpikeInterface or otherwise, and therefore *dartsort* will explode if you don't set this flag and leave your data in its original raw state (for instance, the raw `int16` data off the probe).
    - For a cheap but sensible default, try `preprocessing="ibllikecmr"`, which applies a pipeline similar to that of [the IBL](iblsorting) but with global median common referencing instead of their spatial highpass filter. `preprocessing="ibllike"` will use their spatial highpass filter.
 - `do_motion_estimation=True` by default, and you may like to disable it if you know for a fact that there is (say) less than 5 microns of total drift in your recording, or if you have handled this in your own preprocessing (which is discouraged, since *dartsort* has its own approach.)
 - `work_in_tmpdir` and `copy_recording_to_tmpdir` can be helpful in some cases where slow network drives are involved.

### Outputs and exporting

The `dartsort_result = dartsort(...)` function returns a dictionary `dartsort_result` containing a `dartsort.DARTsortSorting` object under the key `sorting = dartsort_result["sorting"]`.
This object has all the spike train data attached (as arrays under property names `.times_samples` and `.times_seconds` for spike times in samples and seconds, `.labels` for unit labels, and many others; `print(sorting)` to see some more).

If you already ran *dartsort* and want to load the output spike trains, use `dartsort.load(output_dir)` to get the `DARTsortSorting` object.

This object can also export itself to other formats:
 - For a SpikeInterface `NumpySorting` object, use `sorting.to_numpy_sorting()`
 - For a [Pynapple][Pynapple] `TsGroup`, use `sorting.to_tsgroup()`
 - To export to Phy, we currently suggest bridging through SpikeInterface. Start with `sorting.to_numpy_sorting()` and follow the instructions [in SpikeInterface's documentation](https://spikeinterface.readthedocs.io/en/stable/modules/exporters.html#exporting-to-phy) for first creating a `SortingAnalyzer` and then exporting that to Phy.
 - For a simple dictionary of numpy arrays, use `dict = sorting.spike_feature_dict`.
 - For pandas, use `sorting.to_pandas()`.

*dartsort* also saves motion information, returned as `dartsort_result["motion"]` or loaded after the fact as `dartsort.try_load_motion_info(output_dir)`.

The data is saved to `output_dir` in the following files:
 - `dartsort_sorting.npz`: This NPZ file contains the final spike train under the keys `times_samples`, `channels`, and `labels`.
 - `matching1.h5`: This HDF5 file contains spike features and other data from the last matching step. Amplitudes, localizations, and other features live in here; use `h5ls` on the command line to see what's in there. Be aware that the `labels` dataset in this HDF5 is not the same as what's saved in the `dartsort_sorting.npz`.
 - `motion_info.pkl` is a pickled `MotionInfo` object.
 - There may be a models/ folder containing PyTorch weights files with modeling quantities (for instance, featurization SVD bases, localization neural nets, Gaussian mixture model parameters). If you want to load these up, feel free to reach out for help.

### Visualization

To make some basic visualizations of the sorting result with matplotlib, try:

```python
import dartsort.vis as dartvis

# gather outputs from dartsort
dartsort_result = dartsort(recording, output_dir, ...)
sorting = dartsort_result["sorting"]
motion = dartsort_result["motion"]

# or, if you already ran it
sorting = dartsort.load(output_dir)
motion = dartsort.try_load_motion_info(output_dir)

dartvis.visualize_sorting(
    recording,
    sorting,
    vis_save_dir,
    motion=motion,
    make_unit_summaries=False,
)
```

Set `make_unit_summaries=True` to create a summary plot for each unit.

### Command-line interface

Try running 

```sh
$ dartsort -h
```

on your command line to see usage instructions; parameters can be configured on the command line or read from a TOML file.

## Troubleshooting

Please let us know if you run into any issues.
If you feel that the issue is a software bug, feel free to open an issue or a discussion on GitHub.
If it's more of a data-related or methodology thing, feel free to use the email on my GitHub [profile](https://github.com/cwindolf).

## References

[SpikeInterface]: https://spikeinterface.readthedocs.io
[iblsorting]: https://figshare.com/articles/online_resource/Spike_sorting_pipeline_for_the_International_Brain_Laboratory/19705522
[Pynapple]: https://pynapple.org/
