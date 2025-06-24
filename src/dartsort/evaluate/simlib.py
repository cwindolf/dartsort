from pathlib import Path


import h5py
import numpy as np
import probeinterface
from scipy.spatial.distance import cdist
from spikeinterface.core import NumpySorting
import torch

try:
    from importlib.resources import files
except ImportError:
    try:
        from importlib_resources import files
    except ImportError:
        raise ValueError("Need python>=3.10 or pip install importlib_resources.")

from ..transform import WaveformPipeline
from ..util.internal_config import FeaturizationConfig, WaveformConfig
from ..util.data_util import subsample_waveforms, yield_chunks


default_temporal_kernel_npy = files("dartsort.pretrained")
default_temporal_kernel_npy = default_temporal_kernel_npy.joinpath("default_temporal_kernel.npy")
default_temporal_kernel_npy = Path(str(default_temporal_kernel_npy))


# -- spike train sims


def refractory_poisson_spike_train(
    rate_hz,
    duration_samples,
    seed: int | np.random.Generator = 0,
    refractory_samples=40,
    trough_offset_samples=42,
    spike_length_samples=121,
    sampling_frequency=30000.0,
    overestimation=1.5,
):
    """Sample a refractory Poisson spike train

    Arguments
    ---------
    rate : float
        Spikes / second, well, except it'll be slower due to refractoriness.
    duration : float
    """
    rg = np.random.default_rng(seed)

    seconds_per_sample = 1.0 / sampling_frequency
    refractory_s = refractory_samples * seconds_per_sample
    duration_s = duration_samples * seconds_per_sample

    # overestimate the number of spikes needed
    overest_count = int(duration_s * rate_hz * overestimation)
    overest_count = max(10, overest_count)

    # generate interspike intervals
    intervals = rg.exponential(scale=1.0 / rate_hz, size=overest_count)
    intervals += refractory_s
    intervals_samples = np.floor(intervals * sampling_frequency).astype(int)

    # determine spike times and restrict to ones which we can actually
    # add into / read from a recording with this duration and trough offset
    spike_samples = np.cumsum(intervals_samples)
    max_spike_time = duration_samples - (spike_length_samples - trough_offset_samples)
    # check that we overestimated enough
    assert spike_samples.max() > max_spike_time
    valid = spike_samples == spike_samples.clip(trough_offset_samples, max_spike_time)
    spike_samples = spike_samples[valid]
    assert spike_samples.size

    return spike_samples


def piecewise_refractory_poisson_spike_train(rates, bins, binsize_samples, **kwargs):
    """
    Returns a spike train with variable firing rate using refractory_poisson_spike_train().

    :param rates: list of firing rates in Hz
    :param bins: bin starting samples (same shape as rates)
    :param binsize_samples: number of samples per bin
    :param **kwargs: kwargs to feed to refractory_poisson_spike_train()
    """
    st = []
    for rate, bin in zip(rates, bins):
        if rate < 0.1:
            continue
        binst = refractory_poisson_spike_train(rate, binsize_samples, **kwargs)
        st.append(bin + binst)
    st = np.concatenate(st)
    return st


def simulate_sorting(
    num_units,
    n_samples,
    firing_rates=None,
    rg: int | np.random.Generator = 0,
    nbefore: int = 42,
    spike_length_samples: int = 128,
    sampling_frequency=30_000.0,
    globally_refractory=False,
    refractory_samples=40,
):
    rg = np.random.default_rng(rg)

    # Default firing rates drawn uniformly from 1-10Hz
    if firing_rates is not None:
        assert firing_rates.shape[0] == num_units
    else:
        firing_rates = rg.uniform(1.0, 10.0, num_units)

    if not globally_refractory:
        spike_trains = [
            refractory_poisson_spike_train(
                firing_rates[i],
                n_samples,
                trough_offset_samples=nbefore,
                spike_length_samples=spike_length_samples,
                seed=rg,
                refractory_samples=refractory_samples,
                sampling_frequency=sampling_frequency,
            )
            for i in range(num_units)
        ]
        spike_times = np.concatenate(spike_trains)
        spike_labels = np.repeat(
            np.arange(num_units),
            np.array([spike_trains[i].shape[0] for i in range(num_units)]),
        )
    else:
        global_rate = np.sum(firing_rates)
        spike_times = refractory_poisson_spike_train(
            global_rate,
            n_samples,
            trough_offset_samples=nbefore,
            spike_length_samples=spike_length_samples,
            seed=rg,
            refractory_samples=refractory_samples,
            sampling_frequency=sampling_frequency,
        )
        unit_proportions = firing_rates / global_rate
        spike_labels = rg.choice(num_units, p=unit_proportions, size=spike_times.size)

    # order = np.argsort(spike_train)
    # spike_train = spike_train[order]
    # spike_labels = spike_labels[order]

    sorting = NumpySorting.from_samples_and_labels(
        [spike_times], [spike_labels], sampling_frequency=sampling_frequency
    )

    return sorting


# -- spatial utils


def generate_geom(
    num_columns=4,
    num_contact_per_column=96,
    xpitch=-16,
    ypitch=40,
    x_start=59,
    y_start=20,
    y_shift_per_column=(20, 0, 20, 0),
    sort=True,
    sort_x_down=True,
):
    """Defaults match NP1 geometry as returned by ibl-neuropixel."""
    p = probeinterface.generate_multi_columns_probe(
        num_columns=num_columns,
        num_contact_per_column=num_contact_per_column,
        xpitch=xpitch,
        ypitch=ypitch,
        y_shift_per_column=y_shift_per_column,
    )
    geom = p.contact_positions
    assert geom is not None
    geom[:, 0] += x_start
    geom[:, 1] += y_start
    if sort:
        if sort_x_down:
            order = np.lexsort((geom * [-1, 1]).T)
        else:
            order = np.lexsort(geom.T)
        geom = geom[order]
    return geom


def rbf_kernel_sqrt(geom, bandwidth=15.0, dtype="float32"):
    x = geom / (np.sqrt(2.0) * bandwidth)
    k = cdist(x, x, metric="sqeuclidean")
    np.negative(k, out=k)
    np.exp(k, out=k)
    vals, vecs = np.linalg.eigh(k)
    spatial_std = np.sqrt(vals, dtype=dtype)
    spatial_vt = np.ascontiguousarray(vecs.T, dtype=dtype)
    return spatial_std, spatial_vt



# -- sorting h5 helpers


def add_tpca_feature(h5_path, recording, rank=8):
    with h5py.File(h5_path, "r+", locking=False) as h5:
        geom = h5["geom"][:]
        channel_index = h5["channel_index"][:]
        channels, waveforms, weights = subsample_waveforms(h5=h5)
        gt_pipeline = WaveformPipeline.from_config(
            FeaturizationConfig(do_enforce_decrease=False, do_localization=False, tpca_rank=rank),
            WaveformConfig(),
            geom=geom,
            channel_index=channel_index,
        )
        gt_pipeline.fit(waveforms, channels, recording)
        models_dir = h5_path.parent / f"{h5_path.stem}_models"
        models_dir.mkdir(exist_ok=True)
        torch.save(gt_pipeline.state_dict(), models_dir / "featurization_pipeline.pt")

        wf_dset = h5["collisioncleaned_waveforms"]
        f_dset = h5.create_dataset(
            "collisioncleaned_tpca_features",
            shape=(wf_dset.shape[0], rank, wf_dset.shape[2]),
        )
        for sli, chunk in yield_chunks(h5["collisioncleaned_waveforms"], desc_prefix="PCA"):
            _, feats = gt_pipeline(chunk, h5["channels"][sli])
            f_dset[sli] = feats["collisioncleaned_tpca_features"]
