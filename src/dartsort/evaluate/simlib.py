from pathlib import Path
from typing import Literal, cast
from dataclasses import replace


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
        from importlib_resources import files  # pyright: ignore
    except ImportError:
        raise ValueError("Need python>=3.10 or pip install importlib_resources.")

from ..transform import WaveformPipeline
from ..util.internal_config import FeaturizationConfig, WaveformConfig
from ..util.data_util import subsample_waveforms, yield_chunks


default_temporal_kernel_npy = files("dartsort.pretrained")
default_temporal_kernel_npy = default_temporal_kernel_npy.joinpath(
    "default_temporal_kernel.npy"
)
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
    overestimation=2.0,
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
        if rate < 0.05:
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

    if firing_rates.ndim == 2:
        assert not globally_refractory
        assert int(sampling_frequency) == sampling_frequency
        assert firing_rates.shape[1] == num_units
        Tceil = np.ceil(n_samples / sampling_frequency)
        assert firing_rates.shape[0] == Tceil
        bins = np.arange(0, n_samples, sampling_frequency)
        assert firing_rates.shape[0] == bins.shape[0]

        spike_trains = [
            piecewise_refractory_poisson_spike_train(
                rates=firing_rates[i],
                bins=bins,
                binsize_samples=int(sampling_frequency),
                trough_offset_samples=nbefore,
                spike_length_samples=spike_length_samples,
                seed=rg,
                refractory_samples=refractory_samples,
                sampling_frequency=sampling_frequency,
            )
            for i in range(num_units)
        ]
        if num_units:
            spike_times = np.concatenate(spike_trains)
            spike_labels = np.repeat(
                np.arange(num_units),
                np.array([spike_trains[i].shape[0] for i in range(num_units)]),
            )
        else:
            spike_times = np.array([], dtype=np.int64)
            spike_labels = np.array([], dtype=np.int64)
    elif not globally_refractory:
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
        if num_units:
            spike_times = np.concatenate(spike_trains)
            spike_labels = np.repeat(
                np.arange(num_units),
                np.array([spike_trains[i].shape[0] for i in range(num_units)]),
            )
        else:
            spike_times = np.array([], dtype=np.int64)
            spike_labels = np.array([], dtype=np.int64)
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
    y_shift_per_column: Literal["stagger", "flat"]
    | tuple[float]
    | list[float] = "stagger",
    stagger: float = 20.0,
    sort=True,
    sort_x_down=True,
):
    """Defaults match NP1 geometry as returned by ibl-neuropixel."""
    if y_shift_per_column == "stagger" and num_columns == 1:
        y_shift_per_column = [0.0]
    elif y_shift_per_column == "stagger":
        num_columns_even_half = 1 + (num_columns // 2)
        y_shift_per_column = [stagger, 0.0] * num_columns_even_half
        y_shift_per_column = y_shift_per_column[:num_columns]
    elif y_shift_per_column == "flat" or y_shift_per_column is None:
        y_shift_per_column = [0.0] * num_columns
    else:
        assert isinstance(y_shift_per_column, (tuple, list, np.ndarray))
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
    if sort_x_down:
        order = np.lexsort((geom * [-1, 1]).T)
    elif sort:
        order = np.lexsort(geom.T)
    else:
        order = slice(None)
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


# collidedness special cased in sims
default_sim_featurization_cfg = FeaturizationConfig(
    do_enforce_decrease=False, additional_com_localization=True, save_collidedness=False
)


def add_features(h5_path, recording, featurization_cfg):
    with h5py.File(h5_path, "r+", locking=False) as h5:
        geom = cast(h5py.Dataset, h5["geom"])[:]
        channel_index = cast(h5py.Dataset, h5["channel_index"])[:]
        waveforms, fixed_properties = subsample_waveforms(h5=h5)
        if not len(waveforms):
            return
        featurization_cfg = replace(featurization_cfg, do_localization=len(geom) > 1)
        gt_pipeline = WaveformPipeline.from_config(
            featurization_cfg,
            WaveformConfig(),
            geom=geom,
            channel_index=channel_index,
            sampling_frequency=recording.sampling_frequency,
        )
        gt_pipeline.fit(recording, waveforms, **fixed_properties)
        models_dir = h5_path.parent / f"{h5_path.stem}_models"
        models_dir.mkdir(exist_ok=True)
        torch.save(gt_pipeline.state_dict(), models_dir / "featurization_pipeline.pt")

        wf_dset = cast(h5py.Dataset, h5["collisioncleaned_waveforms"])
        n = wf_dset.shape[0]
        f_dsets = {
            sd.name: h5.create_dataset(
                sd.name, shape=(n, *sd.shape_per_spike), dtype=sd.dtype
            )
            for sd in gt_pipeline.spike_datasets()
        }
        for sli, chunk in yield_chunks(
            h5["collisioncleaned_waveforms"], desc_prefix="Featurize"
        ):
            _, feats = gt_pipeline(
                chunk, channels=cast(h5py.Dataset, h5["channels"])[sli]
            )
            for k, v in feats.items():
                f_dsets[k][sli] = v.numpy(force=True)


def simulate_twostate_switching(
    rg: np.random.Generator,
    n_bins: int,
    state_affinity: float,
    n_units: int,
    min_fr: float,
    down_max_fr: float,
    up_min_fr: float,
    max_fr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate population firing rates in bins

    There are two latent states (random) and three neural populations.
    One is up in state 0, down state 1, other is reverse, and pop 3 doesn't
    care.

    Algorithm:
     - Simulate states
     - Simulate state affinities (0, 1, 2=don't care)
     - Draw per-neuron firing rates in each state according to affinity
     - Do a one-hot matmul to gather the final firing rates
    """
    states = rg.binomial(n=1, p=0.5, size=n_bins).astype(np.int32)
    states_onehot = np.zeros((n_bins, 2))
    states_onehot[np.arange(n_bins), states] = 1.0

    assert 2 * state_affinity < 1
    affinity_p = np.array([state_affinity, state_affinity, 1 - 2 * state_affinity])
    affinities = rg.choice(3, size=n_units, p=affinity_p)

    frs_by_state = np.zeros((n_units, 2))
    for aff in range(3):
        in_aff = np.flatnonzero(affinities == aff)
        n_aff = in_aff.size

        if aff == 2:
            frs_by_state[in_aff, :] = rg.uniform(size=n_aff, low=min_fr, high=max_fr)
        elif aff <= 1:
            frs_by_state[in_aff, aff] = rg.uniform(
                size=n_aff, low=up_min_fr, high=max_fr
            )
            frs_by_state[in_aff, 1 - aff] = rg.uniform(
                size=n_aff, low=min_fr, high=down_max_fr
            )
        else:
            assert False

    return states, states_onehot @ frs_by_state.T
